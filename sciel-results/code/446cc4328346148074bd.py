import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Map from block tuple to its index in all_k_subsets
    block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}

    # C(k,t) - number of t-subsets in a k-block
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover_matrix: for each k-subset, the indices of t-subsets it covers
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[triple]

    def fast_greedy_cover(seed):
        """Vectorized whole-block greedy set cover."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any():
            # Vectorized gain: for each candidate block, count uncovered triples it covers
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False

        return blocks

    def greedy_packing(seed, lambda_max=1):
        """Greedy packing: select k-subsets maximizing coverage while limiting overlap."""
        rng = random.Random(seed)
        coverage = np.zeros(n_t, dtype=np.int32)
        blocks = []
        used = np.zeros(n_k, dtype=bool)

        while True:
            if time.time() > deadline - 20:
                break

            # Find blocks that don't violate lambda_max
            # A block is valid if for all its t-subsets, coverage < lambda_max
            # We want to pick the block with max new coverage (count of t-subsets with coverage < lambda_max)
            gains = np.sum(coverage[cover_matrix] < lambda_max, axis=1)
            gains[used] = 0
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            blocks.append(all_k_subsets[chosen_idx])
            used[chosen_idx] = True
            coverage[cover_matrix[chosen_idx]] += 1

        return blocks

    def local_search_reduce(blocks):
        """Reduce block count: remove redundant blocks, then try block replacements."""
        if not blocks:
            return blocks

        blocks = list(blocks)

        def build_coverage(bks):
            coverage = np.zeros(n_t, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage

        # Phase 1: Remove redundant blocks
        coverage = build_coverage(blocks)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage(blocks)
                changed = True

        # Phase 2: Block replacement (1-exchange)
        rng2 = random.Random(123)
        max_replace_rounds = 100
        attempts_per_block = 50

        for round_num in range(max_replace_rounds):
            if time.time() > deadline - 10:
                break
            
            coverage = build_coverage(blocks)
            improved = False
            order = list(range(len(blocks)))
            rng2.shuffle(order)

            for i in order:
                if time.time() > deadline - 10:
                    break

                b_old = blocks[i]
                idx_old = block_to_k_idx[b_old]
                old_tris = cover_matrix[idx_old]

                critical_mask = (coverage[old_tris] == 1)
                if not np.any(critical_mask):
                    coverage[old_tris] -= 1
                    blocks.pop(i)
                    improved = True
                    break

                critical_triples = old_tris[critical_mask]
                must_points = set()
                for ti in critical_triples:
                    must_points.update(all_t_subsets[ti])

                if len(must_points) > k:
                    continue

                available_points = [p for p in range(v) if p not in must_points]
                needed = k - len(must_points)

                best_new_block = None
                best_score = -1

                for _ in range(attempts_per_block):
                    if time.time() > deadline - 10:
                        break
                    if needed > 0:
                        extra = rng2.sample(available_points, needed)
                    else:
                        extra = []
                    
                    candidate = tuple(sorted(list(must_points) + extra))
                    if candidate in block_to_k_idx:
                        idx_new = block_to_k_idx[candidate]
                        new_tris = cover_matrix[idx_new]
                        
                        valid = True
                        for ti in critical_triples:
                            tri = all_t_subsets[ti]
                            if not all(p in candidate for p in tri):
                                valid = False
                                break
                        if not valid:
                            continue

                        score = np.sum(coverage[new_tris] > 1)
                        if score > best_score:
                            best_score = score
                            best_new_block = candidate

                if best_new_block is not None and best_new_block != b_old:
                    idx_new = block_to_k_idx[best_new_block]
                    new_tris = cover_matrix[idx_new]
                    coverage[old_tris] -= 1
                    coverage[new_tris] += 1
                    blocks[i] = best_new_block
                    improved = True
                    break

            if not improved:
                break

        return blocks

    def complete_cover_from_packing(blocks):
        """Complete a partial cover (from packing) into a full cover using greedy."""
        coverage = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            idx = block_to_k_idx[b]
            coverage[cover_matrix[idx]] += 1
        uncovered = coverage == 0
        if not uncovered.any():
            return blocks[:]
        
        rng = random.Random(999)
        extended = list(blocks)
        while uncovered.any():
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            extended.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
        return extended

    best_blocks = None
    best_len = float('inf')

    # Try both greedy set cover and greedy packing approaches
    seeds_greedy = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
                    7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
    seeds_packing = [77, 88, 99, 111, 222, 333, 444, 555, 666, 777,
                     888, 999, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]

    for seed in seeds_greedy:
        if time.time() > deadline - 15:
            break
        blocks = fast_greedy_cover(seed)
        if not blocks:
            continue
        refined = local_search_reduce(blocks)
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = refined[:]
            if best_len <= 76:
                break

    for seed in seeds_packing:
        if time.time() > deadline - 15:
            break
        # Try packing with lambda_max=1 and lambda_max=2
        for lam in [1, 2]:
            if time.time() > deadline - 15:
                break
            packed = greedy_packing(seed, lam)
            completed = complete_cover_from_packing(packed)
            refined = local_search_reduce(completed)
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined[:]
                if best_len <= 76:
                    break
        if best_len <= 76:
            break

    if best_blocks is not None:
        return [list(b) for b in best_blocks]
    return []
