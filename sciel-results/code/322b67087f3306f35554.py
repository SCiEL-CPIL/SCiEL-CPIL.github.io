import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Build bitset for each block: a Python int where bit i is 1 if block covers subset i
    block_bitsets = [0] * n_blocks
    for b_idx in range(n_blocks):
        bits = 0
        for u in block_to_subsets[b_idx]:
            bits |= (1 << u)
        block_bitsets[b_idx] = bits

    # For fast "is subset u in block b" check, precompute a set for each block
    block_subset_sets = [set(block_to_subsets[b]) for b in range(n_blocks)]

    def greedy_weighted(seed_val, alpha=1.0, beta=0.0):
        """
        Weighted greedy with adaptive subset weights (IPF-inspired).
        alpha: weight for uncovered subsets (high = prioritize uncovered)
        beta: weight for already-covered subsets (high = penalize redundancy)
        Score for block b = sum over subsets u in b of:
            w[u] where w[u] = alpha if u uncovered, beta if u covered
        This encourages blocks that cover many uncovered subsets while
        penalizing those that overlap heavily with already-covered ones.
        """
        rng = random.Random(seed_val if seed_val is not None else 0)
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        # Dynamic weights: start uniform, update via IPF-like scheme
        # weight[u] represents how "important" it is to cover subset u
        weights = np.ones(n_subsets, dtype=np.float64)
        chosen = []
        used = np.zeros(n_blocks, dtype=np.bool_)

        # Precompute block-subset incidence as numpy arrays for speed
        # block_to_subsets is list of lists; convert to array of arrays for vectorized ops
        # Actually, for weighted scoring we need per-block sum of weights over its subsets
        # Let's compute scores vectorized when possible

        while True:
            if time.time() > deadline - 1.0:
                break
            if not uncovered.any():
                break

            # Compute score for each unused block
            # score[b] = sum_{u in block_to_subsets[b]} weights[u]
            # But we want to emphasize uncovered: use alpha*weight if uncovered, beta*weight if covered
            # Simpler: just use weights directly, and update weights to reflect coverage
            scores = np.zeros(n_blocks, dtype=np.float64)
            for b in range(n_blocks):
                if used[b]:
                    continue
                # Sum weights over subsets in this block
                # block_to_subsets[b] is a list of indices
                scores[b] = weights[block_to_subsets[b]].sum()

            max_score = scores[~used].max() if (~used).any() else 0
            if max_score <= 0:
                break

            # Randomized selection: pick from top candidates
            if seed_val is not None and rng.random() < 0.2:
                threshold = max_score * 0.75
                candidates = np.where((scores >= threshold) & (~used))[0]
                if len(candidates) > 0:
                    # Weighted random choice: probability proportional to score
                    probs = scores[candidates]
                    probs = probs / probs.sum()
                    b_star = int(rng.choice(candidates, p=probs))
                else:
                    unused = np.where(~used)[0]
                    b_star = int(unused[np.argmax(scores[unused])])
            else:
                unused = np.where(~used)[0]
                b_star = int(unused[np.argmax(scores[unused])])

            chosen.append(b_star)
            used[b_star] = True

            # Update weights: IPF-like adjustment
            # Increase weight of uncovered subsets (they're still needed)
            # Decrease weight of covered subsets (less urgent)
            for u in block_to_subsets[b_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    # This subset is now covered; reduce its weight
                    weights[u] *= beta  # beta < 1 reduces weight
                else:
                    # Already covered; keep low weight
                    pass

            # Periodically renormalize and boost uncovered
            if len(chosen) % 10 == 0:
                # Boost uncovered subsets to maintain entropy
                for u in range(n_subsets):
                    if uncovered[u]:
                        weights[u] = max(weights[u], alpha)
                    else:
                        weights[u] = min(weights[u], beta * alpha)

        return chosen

    def greedy_cover_fast(seed_val):
        """Fast greedy using numpy for gain tracking. Returns list of block indices."""
        rng = random.Random(seed_val if seed_val is not None else 0)
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        gains = np.array([len(block_to_subsets[b]) for b in range(n_blocks)], dtype=np.int32)
        chosen = []
        used = np.zeros(n_blocks, dtype=np.bool_)

        while True:
            if time.time() > deadline - 1.0:
                break
            if not uncovered.any():
                break

            max_gain = gains[~used].max() if (~used).any() else 0
            if max_gain <= 0:
                break

            if seed_val is not None and rng.random() < 0.25:
                threshold = max_gain * 0.7
                candidates = np.where((gains >= threshold) & (~used))[0]
                if len(candidates) > 0:
                    b_star = int(rng.choice(candidates))
                else:
                    candidates = np.where(~used)[0]
                    b_star = int(candidates[np.argmax(gains[candidates])])
            else:
                unused = np.where(~used)[0]
                b_star = int(unused[np.argmax(gains[unused])])

            chosen.append(b_star)
            used[b_star] = True

            for u in block_to_subsets[b_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1

            if not uncovered.any():
                break

        return chosen

    def remove_redundant_fast(block_indices):
        """Remove redundant blocks using bitset operations for speed."""
        if len(block_indices) <= 1:
            return list(block_indices)

        remaining = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(remaining)):
                b = remaining[i]
                # Check if block b is redundant: all its subsets covered by other blocks
                # Use bitset: other_union = OR of all other blocks' bitsets
                # b is redundant if (block_bitsets[b] & ~other_union) == 0
                other_union = 0
                for j in range(len(remaining)):
                    if j != i:
                        other_union |= block_bitsets[remaining[j]]
                if (block_bitsets[b] & ~other_union) == 0:
                    remaining.pop(i)
                    changed = True
                    break

        return remaining

    def local_search_fast(block_indices):
        """Fast local search using bitsets for redundancy and swap checks."""
        if len(block_indices) <= 1:
            return list(block_indices)

        remaining = list(block_indices)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(remaining)):
                if time.time() > deadline - 3.0:
                    break
                b_remove = remaining[i]
                # Check if b_remove can be removed using bitsets
                other_union = 0
                for j in range(len(remaining)):
                    if j != i:
                        other_union |= block_bitsets[remaining[j]]
                if (block_bitsets[b_remove] & ~other_union) == 0:
                    remaining.pop(i)
                    improved = True
                    break
                else:
                    # Find critical subsets: those only covered by b_remove
                    critical_mask = block_bitsets[b_remove] & ~other_union
                    if critical_mask == 0:
                        continue
                    # Find a candidate block that covers all critical subsets
                    # Get list of critical subset indices
                    critical_list = []
                    temp = critical_mask
                    while temp:
                        u = temp & (-temp)
                        u_idx = u.bit_length() - 1
                        critical_list.append(u_idx)
                        temp &= temp - 1

                    if not critical_list:
                        continue

                    best_candidate = -1
                    best_extra = -1
                    # Candidates: blocks that cover the first critical subset
                    for b_add in subset_to_blocks[critical_list[0]]:
                        if b_add == b_remove:
                            continue
                        if b_add in remaining:
                            continue
                        # Check if b_add covers all critical subsets
                        if (block_bitsets[b_add] & critical_mask) != critical_mask:
                            continue
                        # Count extra coverage: subsets covered by b_add but not by (remaining - b_remove)
                        extra_mask = block_bitsets[b_add] & ~other_union
                        extra = extra_mask.bit_count() if hasattr(extra_mask, 'bit_count') else bin(extra_mask).count('1')
                        if extra > best_extra:
                            best_extra = extra
                            best_candidate = b_add
                    if best_candidate >= 0:
                        remaining[i] = best_candidate
                        improved = True
                        break

        return remaining

    # Run multiple strategies
    best_blocks = None
    best_count = float('inf')

    # Strategy 1: Fast greedy (absolute gain)
    chosen = greedy_cover_fast(0)
    chosen = remove_redundant_fast(chosen)
    chosen = local_search_fast(chosen)
    best_blocks = chosen
    best_count = len(chosen)

    # Strategy 2: Weighted greedy with different parameters (IPF-inspired)
    if time.time() < deadline - 20.0:
        for alpha, beta in [(1.0, 0.5), (1.0, 0.3), (1.5, 0.5), (2.0, 0.7)]:
            if time.time() > deadline - 10.0:
                break
            chosen = greedy_weighted(0, alpha=alpha, beta=beta)
            if len(chosen) >= best_count + 3:
                continue
            chosen = remove_redundant_fast(chosen)
            chosen = local_search_fast(chosen)
            if len(chosen) < best_count:
                best_count = len(chosen)
                best_blocks = chosen

    # Strategy 3: Randomized restarts with fast greedy
    for seed_val in range(1, 25):
        if time.time() > deadline - 8.0:
            break
        chosen = greedy_cover_fast(seed_val)
        if len(chosen) >= best_count + 3:
            continue
        chosen = remove_redundant_fast(chosen)
        chosen = local_search_fast(chosen)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    # Strategy 4: More weighted greedy with random seeds
    for seed_val in range(1, 15):
        if time.time() > deadline - 5.0:
            break
        alpha = 1.0 + 0.5 * (seed_val % 3)
        beta = 0.3 + 0.2 * (seed_val % 2)
        chosen = greedy_weighted(seed_val, alpha=alpha, beta=beta)
        if len(chosen) >= best_count + 3:
            continue
        chosen = remove_redundant_fast(chosen)
        chosen = local_search_fast(chosen)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    return [list(all_blocks[b]) for b in best_blocks]
