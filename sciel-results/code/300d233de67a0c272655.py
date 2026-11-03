import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140.0  # Safety margin for 150s limit

    # Precompute all t-subsets
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build block_cover_lists as NumPy arrays of t-set indices
    block_cover_lists = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_lists.append(np.array(indices, dtype=np.int32))

    # Build t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_list in enumerate(block_cover_lists):
        for ts_idx in cover_list:
            t_set_to_blocks[ts_idx].append(b_idx)

    best_solution = None
    best_size = float('inf')

    # Strategies for the greedy constructor
    # We will run a few different greedy variants and keep the best
    strategies = [
        (42, 1.0, 0.1),
        (101, 2.0, 0.2),
        (102, 1.5, 0.15),
        (201, 3.0, 0.3),
        (301, 5.0, 0.5),
    ]

    for seed, lambda_init, lambda_step in strategies:
        if time.time() > deadline - 30.0:
            break

        sol = _lagrangian_greedy_cover(
            v, k, t, t_set_index, all_blocks, block_cover_lists,
            num_t_sets, t_set_to_blocks, deadline,
            rng_seed=seed, lambda_init=lambda_init, lambda_step=lambda_step
        )
        if sol:
            # Quick redundancy removal
            sol = _remove_redundant_blocks(sol, block_cover_lists, all_blocks, deadline)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol

    if best_solution is None:
        return []

    # Final polish with stronger local search
    best_solution = _local_improve(best_solution, block_cover_lists, all_blocks, t_set_index, v, k, t, deadline)

    return [list(b) for b in best_solution]

def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_lists,
                              num_t_sets, t_set_to_blocks, deadline,
                              rng_seed=42, lambda_init=1.0, lambda_step=0.1):
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []
    w = np.full(num_t_sets, lambda_init, dtype=np.float64)

    while uncovered:
        if time.time() > deadline - 5.0:
            return None

        # Find the block that covers the most "weighted" uncovered t-sets
        best_block_idx = -1
        max_score = -1.0
        
        # Optimization: Instead of checking all blocks, we can prioritize blocks covering rare t-sets
        # But for simplicity and speed, we scan all blocks.
        # To speed up, we can precompute the sum of weights for each block?
        # w changes every step, so we can't precompute.
        # However, we can use NumPy to sum w[indices] for all blocks in parallel?
        # block_cover_lists is a list of arrays.
        # We can't easily vectorize this without a dense matrix or sparse matrix.
        # Given num_blocks ~ 75k and num_t_sets ~ 11k, a dense matrix is 75k * 11k * 8 bytes ~ 6.6 GB. Too big.
        # So we stick to the loop but optimize it.
        
        # We can skip blocks that don't cover any uncovered t-set?
        # We don't have a fast way to check that without a set intersection.
        
        # Let's just loop.
        for b_idx in range(num_blocks):
            indices = block_cover_lists[b_idx]
            # Only sum weights for uncovered t-sets?
            # w is dense. indices are sparse.
            # sum(w[indices]) is fast in NumPy.
            score = w[indices].sum()
            if score > max_score:
                max_score = score
                best_block_idx = b_idx

        if best_block_idx == -1:
            break

        selected_blocks.append(all_blocks[best_block_idx])

        # Update uncovered
        covered_indices = block_cover_lists[best_block_idx]
        for idx in covered_indices:
            uncovered.discard(idx)

        # Update weights
        if uncovered:
            uncovered_list = list(uncovered)
            w[uncovered_list] *= (1.0 + lambda_step)
        
        covered_list = covered_indices.tolist()
        w[covered_list] *= (1.0 - lambda_step * 0.5)
        w = np.maximum(w, 0.1)

    return selected_blocks

def _remove_redundant_blocks(blocks, block_cover_lists, all_blocks, deadline):
    if not blocks:
        return blocks

    all_blocks_set = {frozenset(b): i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for block in blocks:
        block_set = frozenset(block)
        if block_set in all_blocks_set:
            current_block_indices.append(all_blocks_set[block_set])

    if len(current_block_indices) != len(blocks):
        return blocks

    max_idx = 0
    for bl in block_cover_lists:
        if len(bl) > 0:
            max_idx = max(max_idx, np.max(bl))
    num_t_sets = max_idx + 1
    
    if num_t_sets == 0:
        return blocks

    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_block_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1

    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        
        for i in range(len(current_block_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            
            b_idx = current_block_indices[i]
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_block_indices.pop(i)
                improved = True
                break

    if not current_block_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_block_indices]

def _local_improve(blocks, block_cover_lists, all_blocks, t_set_index, v, k, t, deadline):
    if not blocks:
        return blocks
    
    block_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    current_indices = [block_to_idx[frozenset(b)] for b in blocks]
    
    num_t_sets = len(t_set_index)
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1
    
    rng = random.Random(42)
    
    # Phase 1: 1-Exchange (Remove 1, Add 1 or 0)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n = len(current_indices)
        
        # Try removing one block
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            
            b_idx = current_indices[i]
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_indices.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-exchange: remove i, add j
        # Randomize order to avoid bias
        indices_order = list(range(len(current_indices)))
        rng.shuffle(indices_order)
        
        for i in indices_order:
            if time.time() > deadline - 5.0:
                break
            
            b_idx_i = current_indices[i]
            
            # Temporarily remove i
            for idx in block_cover_lists[b_idx_i]:
                coverage[idx] -= 1
                
            # Find uncovered t-sets
            uncovered_indices = np.where(coverage == 0)[0]
            
            if len(uncovered_indices) == 0:
                # Should have been caught above
                current_indices.pop(i)
                improved = True
                break
            
            # Find best block to cover uncovered_indices
            best_j = -1
            max_covered = -1
            # Candidate blocks are those covering any of the uncovered indices
            candidates = set()
            for u_idx in uncovered_indices:
                candidates.update(t_set_to_blocks[u_idx])
            
            for j in candidates:
                if j == b_idx_i:
                    continue
                # Check if j covers all uncovered_indices
                # Efficient check: count how many uncovered_indices are in block_cover_lists[j]
                # We can use np.intersect1d or a loop since uncovered_indices is small
                block_j_indices = block_cover_lists[j]
                # Count intersection
                # Since uncovered_indices is small, convert to set for O(1) lookup?
                # Or just loop.
                covered_count = 0
                u_set = set(uncovered_indices.tolist())
                for idx in block_j_indices:
                    if idx in u_set:
                        covered_count += 1
                        if covered_count == len(uncovered_indices):
                            break
                if covered_count == len(uncovered_indices):
                    # j covers all uncovered
                    # Prefer blocks that cover more *other* t-sets? No, we just need validity.
                    # To minimize future issues, maybe pick one with high coverage of currently covered sets?
                    # For now, just pick the first one.
                    best_j = j
                    break
            
            if best_j != -1:
                # Add best_j
                for idx in block_cover_lists[best_j]:
                    coverage[idx] += 1
                current_indices.pop(i)
                current_indices.append(best_j)
                improved = True
                break
            else:
                # Restore coverage
                for idx in block_cover_lists[b_idx_i]:
                    coverage[idx] += 1

    # Phase 2: 2-Exchange (Remove 2, Add <2)
    # This is more expensive, so we do it if time permits
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            n = len(current_indices)
            if n < 2:
                break
            
            # Try random pairs
            # To be fast, we don't iterate all pairs, but sample them
            num_attempts = 1000
            for _ in range(num_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                i, j = rng.sample(range(n), 2)
                b_idx_i = current_indices[i]
                b_idx_j = current_indices[j]
                
                # Temporarily remove i and j
                for idx in block_cover_lists[b_idx_i]:
                    coverage[idx] -= 1
                for idx in block_cover_lists[b_idx_j]:
                    coverage[idx] -= 1
                
                uncovered_indices = np.where(coverage == 0)[0]
                
                if len(uncovered_indices) == 0:
                    # Remove both
                    # Order matters for pop
                    if i > j:
                        i, j = j, i
                    current_indices.pop(i)
                    current_indices.pop(j)
                    improved = True
                    break
                
                # Try to cover uncovered_indices with 1 block
                candidates = set()
                for u_idx in uncovered_indices:
                    candidates.update(t_set_to_blocks[u_idx])
                
                found_one = False
                for c in candidates:
                    if c == b_idx_i or c == b_idx_j:
                        continue
                    block_c_indices = block_cover_lists[c]
                    u_set = set(uncovered_indices.tolist())
                    covered_count = 0
                    for idx in block_c_indices:
                        if idx in u_set:
                            covered_count += 1
                            if covered_count == len(uncovered_indices):
                                break
                    if covered_count == len(uncovered_indices):
                        # Add c
                        for idx in block_cover_lists[c]:
                            coverage[idx] += 1
                        # Remove i and j
                        if i > j:
                            i, j = j, i
                        current_indices.pop(i)
                        current_indices.pop(j)
                        current_indices.append(c)
                        improved = True
                        found_one = True
                        break
                
                if found_one:
                    break
                
                # If not found 1 block, try to find 2 blocks?
                # That would be net 0 change. We can do this to escape local minima.
                # But it's slower. Let's skip for now to save time.
                
                # Restore coverage
                for idx in block_cover_lists[b_idx_i]:
                    coverage[idx] += 1
                for idx in block_cover_lists[b_idx_j]:
                    coverage[idx] += 1

    if not current_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_indices]
