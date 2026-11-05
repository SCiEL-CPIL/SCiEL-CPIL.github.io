import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    # Precompute all t-subsets and k-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Map each k-subset to the indices of the t-subsets it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = [t_sub_to_idx[t_sub] for t_sub in itertools.combinations(k_sub, t)]
        k_sub_t_idxs.append(t_idxs)

    # Map each t-subset to the indices of the k-subsets that cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed, strategy='standard'):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.uint8)
        blocks_idx = []
        used_blocks = set()

        # Precompute gain for each block (initially all t-subsets are uncovered)
        gain = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        # Mark used blocks as having 0 gain
        # We'll maintain a list of available blocks

        while True:
            if time.time() >= deadline - 10.0:
                break

            if not uncovered.any():
                break

            if strategy == 'most_constrained':
                # Find uncovered t-subset with fewest available candidates
                best_s_idx = -1
                min_candidates = float('inf')
                # Sample a few to find a constrained one
                for _ in range(30):
                    s_idx = rng.randint(0, n_t - 1)
                    if not uncovered[s_idx]:
                        continue
                    avail = [b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks]
                    if len(avail) < min_candidates:
                        min_candidates = len(avail)
                        best_s_idx = s_idx
                    if min_candidates <= 1:
                        break
                if best_s_idx == -1:
                    # Fallback to random uncovered
                    while True:
                        s_idx = rng.randint(0, n_t - 1)
                        if uncovered[s_idx]:
                            break
                else:
                    s_idx = best_s_idx
            elif strategy == 'random':
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if uncovered[s_idx]:
                        break
            else:
                # Standard greedy: pick block with max gain
                max_gain = int(np.max(gain))
                if max_gain <= 0:
                    break
                # Find all blocks with max gain
                candidates = np.where(gain == max_gain)[0]
                # Randomly pick one to break ties
                s_idx = -1  # Not used in standard, but we pick a block directly
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
                blocks_idx.append(b_idx)
                used_blocks.add(b_idx)
                # Update gains
                for t_idx in k_sub_t_idxs[b_idx]:
                    if uncovered[t_idx]:
                        uncovered[t_idx] = 0
                        # Decrement gain for all blocks covering this t-subset
                        for b in t_sub_to_k_idxs[t_idx]:
                            if b not in used_blocks:
                                gain[b] -= 1
                continue

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            # Vectorized gain calculation
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                t_idxs = k_sub_t_idxs[b_idx]
                gain_val = int(uncovered[t_idxs].sum())
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            for t_idx in k_sub_t_idxs[best_block_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
                    for b in t_sub_to_k_idxs[t_idx]:
                        if b not in used_blocks:
                            gain[b] -= 1

        return blocks_idx

    def reduce_blocks(blocks_idx, max_passes=15):
        if not blocks_idx:
            return blocks_idx

        blocks_idx = list(blocks_idx)

        for _ in range(max_passes):
            if time.time() >= deadline - 5.0:
                break

            # Build cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                np.add.at(cover_count, k_sub_t_idxs[b_idx], 1)

            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    # This block is redundant
                    changed = True
                    np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                else:
                    new_blocks.append(b_idx)

            blocks_idx = new_blocks
            if not changed:
                break

        return blocks_idx

    def local_search(blocks_idx, max_iter=2000):
        if not blocks_idx:
            return blocks_idx

        blocks_idx = list(blocks_idx)

        # Initial cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            np.add.at(cover_count, k_sub_t_idxs[b_idx], 1)

        rng = random.Random(42)

        for _ in range(max_iter):
            if time.time() >= deadline - 5.0:
                break

            improved = False

            # 1. Try deletions (multiple passes within one iteration)
            for i in range(len(blocks_idx)):
                b_idx = blocks_idx[i]
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    # Remove block
                    np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                    blocks_idx.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # 2. Try swaps
            i = rng.randint(0, len(blocks_idx) - 1)
            b_idx = blocks_idx[i]

            # Find essential t-subsets for this block
            essential = [t_idx for t_idx in k_sub_t_idxs[b_idx] if cover_count[t_idx] == 1]

            if not essential:
                # This block is not essential for anything, delete it
                np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                blocks_idx.pop(i)
                improved = True
                continue

            # Find candidates that cover all essential t-subsets
            candidates = set(t_sub_to_k_idxs[essential[0]])
            for t_idx in essential[1:]:
                candidates &= set(t_sub_to_k_idxs[t_idx])

            # Remove current block and already used blocks
            candidates.discard(b_idx)
            candidates -= set(blocks_idx)

            if not candidates:
                continue

            # Pick a candidate that maximizes "benefit"
            # Benefit = number of t-subsets in candidate that are currently covered only by b_idx
            # (i.e., will become uncovered if b_idx is removed, but are also in candidate)
            # Actually, we want to minimize the number of t-subsets that become uncovered.
            # Since we require covering all essential, the net change is:
            # -1 for t in b_idx \ candidate
            # +1 for t in candidate \ b_idx
            # We want to maximize the number of t in candidate that are currently covered by >1 blocks
            # (so that adding candidate increases redundancy where it's most needed)
            # Or more simply: maximize the number of t in candidate that are currently count 1
            # but are NOT in b_idx? No, those are already covered by b_idx.
            # Let's try: maximize the sum of cover_count[t] for t in candidate
            # This favors blocks that add redundancy to already-covered t-subsets.

            best_replace = -1
            best_score = -1

            # Limit candidates for speed
            if len(candidates) > 100:
                candidates = set(rng.sample(list(candidates), 100))

            essential_set = set(essential)

            for cand_idx in candidates:
                cand_t_idxs = k_sub_t_idxs[cand_idx]
                # Check if it covers all essential
                if not essential_set.issubset(set(cand_t_idxs)):
                    continue

                # Score: sum of cover_count for t in candidate
                # This measures how much redundancy we're adding
                score = int(cover_count[cand_t_idxs].sum())
                # Also add a small bonus for covering more t-subsets
                score += len(cand_t_idxs) * 0.1

                if score > best_score:
                    best_score = score
                    best_replace = cand_idx

            if best_replace != -1:
                # Perform swap
                np.subtract.at(cover_count, k_sub_t_idxs[b_idx], 1)
                np.add.at(cover_count, k_sub_t_idxs[best_replace], 1)
                blocks_idx[i] = best_replace
                improved = True

            if not improved:
                # No improvement in this iteration, break to avoid infinite loop
                break

        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy (standard)
    blocks_idx = greedy_cover(0, 'standard')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Most Constrained Greedy
    blocks_idx = greedy_cover(1, 'most_constrained')
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx and len(blocks_idx) < best_len:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 3. Randomized Restarts
    for seed in range(2, 300):
        if time.time() >= deadline - 40.0:
            break
        strategy = 'most_constrained' if seed % 2 == 0 else 'random'
        blocks_idx = greedy_cover(seed, strategy)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 4. Local Search on Best
    if best_blocks and time.time() < deadline - 30.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 5. Second Local Search Pass
    if best_blocks and time.time() < deadline - 15.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 6. Final Reduction
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
