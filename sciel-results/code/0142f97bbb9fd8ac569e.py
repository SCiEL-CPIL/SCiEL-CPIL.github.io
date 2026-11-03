import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return selected

    final_counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        final_counts[block_covers[i]] += 1

    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant: all its covered subsets have count > 1
            if np.all(final_counts[block_covers[i]] > 1):
                selected.remove(i)
                final_counts[block_covers[i]] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=300):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected

    # Build coverage counts
    counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        counts[block_covers[i]] += 1

    best_selected = set(selected)
    best_size = len(best_selected)

    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break

        if not best_selected:
            break

        # Try to remove a random block
        block_to_remove = random.choice(list(best_selected))

        # Find uncovered subsets if we remove this block
        covers_b = block_covers[block_to_remove]
        uncovered_if_removed = covers_b[counts[covers_b] <= 1]

        if len(uncovered_if_removed) == 0:
            # Block is redundant, remove it
            best_selected.remove(block_to_remove)
            counts[covers_b] -= 1
            best_size -= 1
            continue

        # Try to find a replacement block that covers all uncovered_if_removed
        # Use the most constrained uncovered subset
        min_count = float('inf')
        target_sub = -1
        for s in uncovered_if_removed:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break

        if target_sub == -1:
            continue

        candidates = sub_to_blocks[target_sub]

        # Try candidates, prefer those that cover all uncovered subsets
        best_replacement = None
        best_gain = -1

        for b in candidates:
            if b in best_selected:
                continue

            # Check if this block covers all uncovered_if_removed
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break

            if not covers_all:
                continue

            # Count how many currently uncovered subsets this block would add
            gain = np.sum(counts[block_covers[b]] == 0)

            if gain > best_gain:
                best_gain = gain
                best_replacement = b

        if best_replacement is not None:
            # Perform the swap
            best_selected.remove(block_to_remove)
            counts[covers_b] -= 1
            best_selected.add(best_replacement)
            counts[block_covers[best_replacement]] += 1

    return best_selected

def _polish_pass(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500):
    """
    High-precision polish pass using a smooth surrogate for candidate selection.
    Uses log-sum-exp style soft-min to weight candidates by their constraint tightness.
    """
    if not selected or len(selected) < 2:
        return selected

    counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        counts[block_covers[i]] += 1

    best_selected = set(selected)
    best_size = len(best_selected)

    # Precompute subset counts for soft-min weighting
    subset_counts = np.array([len(lst) for lst in sub_to_blocks], dtype=np.float64)

    for iteration in range(max_iterations):
        if not _deadline_ok(deadline):
            break

        if not best_selected:
            break

        # Try to remove a random block
        block_to_remove = random.choice(list(best_selected))
        covers_b = block_covers[block_to_remove]
        uncovered_if_removed = covers_b[counts[covers_b] <= 1]

        if len(uncovered_if_removed) == 0:
            # Block is redundant, remove it
            best_selected.remove(block_to_remove)
            counts[covers_b] -= 1
            best_size -= 1
            continue

        # Use soft-min to select target subset with probability proportional to
        # exp(-alpha * count) where count is the number of covering blocks
        # This gives higher probability to more constrained subsets
        alpha = 2.0  # temperature parameter
        weights = np.exp(-alpha * subset_counts[uncovered_if_removed])
        weights /= weights.sum()

        target_sub = np.random.choice(uncovered_if_removed, p=weights)
        candidates = sub_to_blocks[target_sub]

        if not candidates:
            continue

        # Score candidates using a smooth surrogate:
        # For each candidate, compute a soft coverage score
        best_replacement = None
        best_score = -float('inf')

        for b in candidates:
            if b in best_selected:
                continue

            # Check if this block covers all uncovered_if_removed
            covers_all = True
            for s in uncovered_if_removed:
                if s not in block_covers[b]:
                    covers_all = False
                    break

            if not covers_all:
                continue

            # Smooth score: primary = covers all critical, secondary = reduces waste
            # Use log-sum-exp style aggregation of coverage benefits
            block_cov = block_covers[b]
            # How many currently uncovered subsets does this block cover?
            new_coverage = np.sum(counts[block_cov] == 0)
            # How many already-covered subsets does this block add to (waste)?
            waste = np.sum((counts[block_cov] > 0) & (counts[block_cov] <= 1))

            # Score: prefer more new coverage, less waste
            # Smooth combination with temperature
            temp = 1.0
            score = new_coverage / temp - 0.5 * waste / temp

            if score > best_score:
                best_score = score
                best_replacement = b

        if best_replacement is not None:
            # Perform the swap
            best_selected.remove(block_to_remove)
            counts[covers_b] -= 1
            best_selected.add(best_replacement)
            counts[block_covers[best_replacement]] += 1

    return best_selected

def _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, max_steps=40000):
    """
    WalkSAT-style local search for set cover.
    """
    n_blocks = len(block_covers)

    # Initialize with greedy solution
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()

    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break

        # Find the uncovered subset with fewest candidate blocks
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        min_count = float('inf')
        target_sub = -1
        for s in uncovered_indices:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break

        if target_sub == -1:
            break

        candidates = sub_to_blocks[target_sub]

        # Pick candidate with highest gain (vectorized)
        best_block = -1
        best_gain = 0
        for b in candidates:
            gain = np.sum(uncovered_mask[block_covers[b]])
            if gain > best_gain:
                best_gain = gain
                best_block = b

        if best_block != -1:
            selected.add(best_block)
            uncovered_mask[block_covers[best_block]] = False

    if not _deadline_ok(deadline):
        return selected

    # Prune redundant blocks
    selected = _prune_redundant(selected, block_covers, n_t, deadline)

    if not _deadline_ok(deadline):
        return selected

    # Build coverage counts
    counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        counts[block_covers[i]] += 1

    best_selected = set(selected)
    best_score = len(best_selected)

    # WalkSAT parameters
    p_flip = 0.45

    for step in range(max_steps):
        if not _deadline_ok(deadline):
            break

        # Check if all covered
        if not np.any(counts == 0):
            # All covered, try to reduce size
            current_size = len(best_selected)

            if current_size > 1:
                b = random.choice(list(best_selected))
                covers_b = block_covers[b]
                would_uncover = np.any(counts[covers_b] <= 1)

                if not would_uncover:
                    # Safe removal
                    best_selected.remove(b)
                    counts[covers_b] -= 1
                    best_score = len(best_selected)
                    continue
            else:
                break

        # Do a flip
        b = random.randrange(n_blocks)

        if b in best_selected:
            # Consider unselecting b
            covers_b = block_covers[b]
            would_uncover = covers_b[counts[covers_b] <= 1]

            if len(would_uncover) == 0:
                # Safe to unselect
                best_selected.remove(b)
                counts[covers_b] -= 1
                best_score = len(best_selected)
            else:
                # Try to fix by selecting a block covering the most constrained uncovered subset
                if random.random() < (1 - p_flip):
                    min_c = float('inf')
                    target = -1
                    for s in would_uncover:
                        c = len(sub_to_blocks[s])
                        if c < min_c:
                            min_c = c
                            target = s
                            if c == 1:
                                break

                    if target != -1:
                        cands = sub_to_blocks[target]
                        best_cand = -1
                        best_cover = -1
                        for cb in cands:
                            if cb in best_selected:
                                continue
                            cov = np.sum(np.isin(block_covers[cb], would_uncover))
                            if cov > best_cover:
                                best_cover = cov
                                best_cand = cb

                        if best_cand != -1 and best_cover > 0:
                            # Do the swap
                            best_selected.remove(b)
                            counts[covers_b] -= 1
                            best_selected.add(best_cand)
                            counts[block_covers[best_cand]] += 1
        else:
            # Consider selecting b
            covers_b = block_covers[b]
            helps = np.sum(counts[covers_b] == 0)

            if helps > 0:
                best_selected.add(b)
                counts[covers_b] += 1
            elif random.random() < 0.05:
                best_selected.add(b)
                counts[covers_b] += 1

    # Final prune
    best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)

    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()

    while np.any(uncovered_mask):
        if not _deadline_ok(deadline):
            break

        # Find the uncovered subset with fewest candidate blocks
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        min_count = float('inf')
        target_sub = -1
        for s in uncovered_indices:
            c = len(sub_to_blocks[s])
            if c < min_count:
                min_count = c
                target_sub = s
                if c == 1:
                    break

        if target_sub == -1:
            break

        candidates = sub_to_blocks[target_sub]

        # Calculate gains for all candidates (vectorized)
        gains = []
        for b in candidates:
            gain = np.sum(uncovered_mask[block_covers[b]])
            gains.append((gain, b))

        gains.sort(reverse=True)

        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]

        selected.add(best_block)
        uncovered_mask[block_covers[best_block]] = False

    return selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin

    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}

    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []

    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(np.array(covers, dtype=np.int32))

    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)

    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)

    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')

    num_restarts = 10

    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break

        # Randomized greedy
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline,
                                  randomize=(restart > 0), top_k=10)

        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)

        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)

        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)

        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)

    # 4. WalkSAT-style local search
    if best_selected is not None and _deadline_ok(deadline):
        walksat_result = _walksat_cover(block_covers, n_t, sub_to_blocks, deadline, max_steps=35000)
        if walksat_result is not None and len(walksat_result) < best_size:
            best_size = len(walksat_result)
            best_selected = walksat_result

    # 5. High-precision polish pass with smooth surrogate
    if best_selected is not None and _deadline_ok(deadline):
        polished = _polish_pass(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=400)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished

    # 6. Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)

    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []

    return result
