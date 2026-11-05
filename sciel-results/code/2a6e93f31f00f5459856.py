import itertools
import random
import time
import math
import numpy as np

# ---------------------------------------------------------------------------
# Global precomputation for the fixed task v=19, k=7, t=4
# ---------------------------------------------------------------------------
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)          # 3876

# Generate all 7-subsets (blocks)
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)        # 50388

# For each block, the list of 4-subset indices it covers
block_subset_indices = []
for block in all_blocks:
    idxs = [subset_to_idx[sub] for sub in itertools.combinations(block, T)]
    block_subset_indices.append(idxs)

# For each 4-subset, the list of blocks that cover it
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
for b, idxs in enumerate(block_subset_indices):
    for idx in idxs:
        blocks_containing_subset[idx].append(b)

# Frequency of each 4-subset (how many blocks cover it)
subset_frequencies = np.array([len(blocks_containing_subset[i]) for i in range(N_SUBSETS)], dtype=np.float64)

# Build a NumPy bitset: rows = blocks, columns = 4-subsets (as int64 bitset)
# Each block covers 35 subsets; store as a list of int64 words per block
N_WORDS = (N_SUBSETS + 63) // 64   # 61 words
block_bitsets = np.zeros((N_BLOCKS, N_WORDS), dtype=np.uint64)
for b, idxs in enumerate(block_subset_indices):
    for idx in idxs:
        block_bitsets[b, idx >> 6] |= (np.uint64(1) << (idx & 63))

# Precompute a per-block "value" array for weighted greedy (inverse frequency)
# We'll build this on the fly per restart to save memory/time
# Instead, precompute the sum of weights each block would give if all subsets uncovered
# Not needed; we compute gain dynamically.

# For fast gain: for each block, we have the list of subset indices.
# We can precompute a flat array of all (block, subset) pairs and use numpy.
# But gain depends on which subsets are currently uncovered, so we keep it dynamic.

# ---------------------------------------------------------------------------
# Helper: build a weighted-greedy cover
# ---------------------------------------------------------------------------
def _greedy_cover(weight_mode, rng, deadline):
    """
    Run a weighted greedy set cover.
    weight_mode: 0 = inverse frequency, 1 = inverse frequency^2, 2 = random
    Returns list of block indices.
    """
    if weight_mode == 0:
        weights = 1.0 / subset_frequencies
    elif weight_mode == 1:
        weights = 1.0 / (subset_frequencies * subset_frequencies)
    else:
        weights = np.ones(N_SUBSETS) * 0.1 + rng.random(N_SUBSETS)

    uncovered = np.ones(N_SUBSETS, dtype=bool)
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    chosen = []

    while uncovered.any():
        # Compute gain for each block: sum of weights of uncovered subsets it covers
        # Use the precomputed block_bitsets and a vectorized approach
        # For each block, gain = sum over its 35 subsets of weight if uncovered
        # We precompute per-block the list of indices; use that.
        best_idx = -1
        best_gain = -1.0
        # Vectorize: for each block, gain = sum(weights[idx] for idx in block_subset_indices[b] if uncovered[idx])
        # Precompute a flat structure: for each block, the indices are contiguous in a flat array
        # Build flat arrays once (done below at module level if not present)

        # To keep it simple and reasonably fast, iterate but with numpy per-block
        # This is the bottleneck; we optimize by precomputing per-block weight sums
        # Actually, let's precompute for each block the list as a numpy array.
        for b in range(N_BLOCKS):
            idxs = block_subset_indices_np[b]
            if idxs.size == 0:
                continue
            g = weights[idxs][uncovered[idxs]].sum()
            if g > best_gain:
                best_gain = g
                best_idx = b
            if time.time() > deadline:
                break
        if best_idx < 0:
            break
        chosen.append(best_idx)
        # Update uncovered and counts
        idxs = block_subset_indices_np[best_idx]
        counts[idxs] += 1
        uncovered[idxs] = False
        if time.time() > deadline:
            break

    return chosen

# Precompute block_subset_indices as numpy arrays for vectorized access
block_subset_indices_np = [np.array(idxs, dtype=np.int32) for idxs in block_subset_indices]


def _local_search(blocks, deadline):
    """
    Local search: repeatedly try to remove redundant blocks, then try 1-exchange
    (remove one, add one) and 2-exchange (remove one, add two) to reduce size.
    Returns list of block indices (the best found).
    """
    blocks = list(blocks)
    block_set = set(blocks)
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    for b in blocks:
        counts[block_subset_indices_np[b]] += 1

    best = list(blocks)
    best_len = len(blocks)

    def is_redundant(b):
        idxs = block_subset_indices_np[b]
        return (counts[idxs] > 1).all()

    def remove(b):
        nonlocal block_set
        idxs = block_subset_indices_np[b]
        counts[idxs] -= 1
        block_set.discard(b)
        blocks.remove(b)

    def add(b):
        nonlocal block_set
        idxs = block_subset_indices_np[b]
        counts[idxs] += 1
        block_set.add(b)
        blocks.append(b)

    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline:
        # Pass 1: remove redundant blocks
        removed_any = False
        for b in list(blocks):
            if time.time() > deadline:
                break
            if is_redundant(b):
                remove(b)
                removed_any = True
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best = list(blocks)
        if removed_any:
            no_improve = 0
            continue

        # Pass 2: try 1-exchange (remove one, add one that covers the newly-uncovered)
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            b = blocks[i]
            idxs = block_subset_indices_np[b]
            # subsets that would become uncovered
            newly_uncovered = idxs[counts[idxs] == 1]
            if newly_uncovered.size == 0:
                # redundant, should have been caught; remove
                remove(b)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best = list(blocks)
                improved = True
                break
            if newly_uncovered.size > 6:
                continue  # too many to cover with one block
            # find a block (not in set, not b) covering all newly_uncovered
            # candidate = intersection of blocks_containing_subset for each newly_uncovered
            cand = None
            for su in newly_uncovered:
                lst = blocks_containing_subset[su]
                if cand is None:
                    cand = set(lst)
                else:
                    cand &= set(lst)
                if not cand:
                    break
            if cand:
                cand.discard(b)
                cand -= block_set
                if cand:
                    # pick the candidate that covers the most currently-uncovered... none uncovered
                    # just pick one that is "central": covers many subsets with count==1? 
                    # Simplest: pick the one minimizing its own redundancy (most useful)
                    # We'll just pick the first; or the one covering most of its subsets with count==1
                    best_j = None
                    best_score = -1
                    for j in list(cand)[:30]:
                        idxs_j = block_subset_indices_np[j]
                        score = (counts[idxs_j] == 1).sum()
                        if score > best_score:
                            best_score = score
                            best_j = j
                    if best_j is not None:
                        remove(b)
                        add(best_j)
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best = list(blocks)
                        improved = True
                        break
        if improved:
            no_improve = 0
            continue

        # Pass 3: try 2-exchange (remove one, add two)
        improved = False
        if no_improve > 30:
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                b = blocks[i]
                idxs = block_subset_indices_np[b]
                newly_uncovered = idxs[counts[idxs] == 1]
                if newly_uncovered.size == 0:
                    remove(b)
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best = list(blocks)
                    improved = True
                    break
                if newly_uncovered.size > 8:
                    continue
                # try to find two blocks j1, j2 (not in set, not b) covering all newly_uncovered
                # candidate pairs: for each su in newly_uncovered, blocks covering it
                # We'll enumerate: pick a block j1 covering newly_uncovered[0], then find j2 covering the rest
                su0 = newly_uncovered[0]
                for j1 in blocks_containing_subset[su0][:40]:
                    if j1 == b or j1 in block_set:
                        continue
                    # subsets still uncovered after adding j1
                    idxs_j1 = block_subset_indices_np[j1]
                    still = [su for su in newly_uncovered if su not in set(idxs_j1)]
                    if not still:
                        # j1 alone covers all -> 1-exchange, but we're in 2-exchange; still valid
                        remove(b)
                        add(j1)
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best = list(blocks)
                        improved = True
                        break
                    # find j2 covering all 'still'
                    cand2 = None
                    for su in still:
                        lst = blocks_containing_subset[su]
                        if cand2 is None:
                            cand2 = set(lst)
                        else:
                            cand2 &= set(lst)
                        if not cand2:
                            break
                    if cand2:
                        cand2.discard(b)
                        cand2.discard(j1)
                        cand2 -= block_set
                        if cand2:
                            j2 = next(iter(cand2))
                            remove(b)
                            add(j1)
                            add(j2)
                            if len(blocks) < best_len:
                                best_len = len(blocks)
                                best = list(blocks)
                            improved = True
                            break
                if improved:
                    break
        if improved:
            no_improve = 0
            continue

        # No improvement: perturb
        no_improve += 1
        if no_improve > max_no_improve:
            # add a random block, then let next iteration prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in block_set:
                add(c)
            no_improve = 0
            if len(blocks) > best_len + 10:
                # too bloated, reset to best
                blocks = list(best)
                block_set = set(blocks)
                counts = np.zeros(N_SUBSETS, dtype=np.int32)
                for b in blocks:
                    counts[block_subset_indices_np[b]] += 1
                no_improve = 0

    # Final prune
    for b in list(blocks):
        if is_redundant(b):
            remove(b)
    if len(blocks) < best_len:
        best = list(blocks)
    return best


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill

    rng = random.Random(12345)
    np.random.seed(12345)

    best_blocks = None
    best_len = 10**9

    # Multi-start with different weightings
    # We have limited time; do a few greedy restarts then local search each.
    # Time budget: spend ~ (deadline - start - 20) on greedy+local search across restarts.
    # Each greedy is slow in pure Python; let's do 2-3 restarts.

    n_restarts = 3
    per_restart_budget = (deadline - start_time - 25) / n_restarts

    for r in range(n_restarts):
        rs_deadline = start_time + 25 + (r + 1) * per_restart_budget
        rs_deadline = min(rs_deadline, deadline - 5)
        weight_mode = r % 3
        greedy = _greedy_cover(weight_mode, rng, rs_deadline)
        if not greedy:
            continue
        ls = _local_search(greedy, deadline - 3)
        if len(ls) < best_len:
            best_len = len(ls)
            best_blocks = ls
        if time.time() > deadline - 10:
            break

    if best_blocks is None:
        # fallback: a single greedy
        greedy = _greedy_cover(0, rng, deadline - 5)
        best_blocks = greedy

    # Final polish: one more local search on the best
    if time.time() < deadline - 3:
        final = _local_search(best_blocks, deadline - 1)
        if len(final) < best_len:
            best_blocks = final

    return [list(all_blocks[i]) for i in best_blocks]
