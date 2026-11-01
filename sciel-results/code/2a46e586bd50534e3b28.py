import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # Build structures: block_to_ts[j] = list of t-subset indices covered by block j
    # ts_to_blocks[i] = list of block indices covering t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
    
    # Convert to numpy arrays for fast operations
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    # Precompute bitmask for each block for fast intersection
    block_bitmasks = [0] * n_k_blocks
    for i, ts_list in enumerate(block_to_ts):
        mask = 0
        for idx in ts_list:
            mask |= (1 << idx)
        block_bitmasks[i] = mask

    full_mask = (1 << n_t_subsets) - 1

    def run_greedy(seed=None):
        """Vectorized greedy set cover with random tie-breaking."""
        if seed is not None:
            random.seed(seed)
        
        gain = np.full(n_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t_subsets, True, dtype=bool)
        is_selected = np.full(n_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 2.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            # Identify newly covered t-subsets
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            # Vectorized gain update
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=n_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def compute_coverage_counts(blocks_indices):
        """Compute coverage counts for a list of block indices."""
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            for ts_idx in block_to_ts[b_idx]:
                counts[ts_idx] += 1
        return counts

    def local_search_remove(blocks_indices, coverage_counts):
        """Remove redundant blocks iteratively."""
        if time.time() > deadline - 1.0:
            return blocks_indices, coverage_counts
        
        changed = True
        while changed:
            if time.time() > deadline - 1.0:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                essential = False
                for ts_idx in block_to_ts[b_idx]:
                    if coverage_counts[ts_idx] == 1:
                        essential = True
                        break
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    for ts_idx in block_to_ts[b_idx]:
                        coverage_counts[ts_idx] -= 1
                changed = True
        
        return blocks_indices, coverage_counts

    def local_search_swap(blocks_indices, coverage_counts):
        """Try to swap blocks to reduce count: remove one, add one that covers more."""
        if time.time() > deadline - 1.0:
            return blocks_indices, coverage_counts
        
        improved = True
        while improved:
            if time.time() > deadline - 1.0:
                break
            improved = False
            
            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline - 1.0:
                    break
                
                # Temporarily remove b_idx
                for ts_idx in block_to_ts[b_idx]:
                    coverage_counts[ts_idx] -= 1
                
                # Find uncovered t-subsets
                uncovered_mask = 0
                for j in range(n_t_subsets):
                    if coverage_counts[j] == 0:
                        uncovered_mask |= (1 << j)
                
                if uncovered_mask == 0:
                    # Can remove b_idx entirely
                    blocks_indices = [b for j, b in enumerate(blocks_indices) if j != i]
                    improved = True
                    break
                
                # Try to find a block that covers the most uncovered
                best_new_idx = -1
                best_new_gain = 0
                for new_idx in range(n_k_blocks):
                    if new_idx == b_idx or new_idx in blocks_indices:
                        continue
                    gain = (uncovered_mask & block_bitmasks[new_idx]).bit_count()
                    if gain > best_new_gain:
                        best_new_gain = gain
                        best_new_idx = new_idx
                
                # Restore b_idx
                for ts_idx in block_to_ts[b_idx]:
                    coverage_counts[ts_idx] += 1
                
                # If we can cover all uncovered with one new block, do the swap
                if best_new_idx != -1 and best_new_gain >= uncovered_mask.bit_count():
                    # Remove old block
                    for ts_idx in block_to_ts[b_idx]:
                        coverage_counts[ts_idx] -= 1
                    
                    blocks_indices[i] = best_new_idx
                    
                    # Add new block
                    for ts_idx in block_to_ts[best_new_idx]:
                        coverage_counts[ts_idx] += 1
                    
                    improved = True
                    break
        
        return blocks_indices, coverage_counts

    def cover_holes_greedy(holes_mask):
        """Cover holes using greedy, returning new blocks added."""
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline - 2.0:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks

    def compute_coverage_mask(blocks_list):
        """Compute the coverage mask for a list of block indices."""
        mask = 0
        for b_idx in blocks_list:
            mask |= block_bitmasks[b_idx]
        return mask

    # Phase 1: Run multiple randomized greedy restarts to find a good initial solution
    best_blocks = None
    best_counts = None
    best_count = float('inf')
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        sol = run_greedy(seed=i)
        if sol is None:
            continue
        
        # Clean up with local search
        counts = compute_coverage_counts(sol)
        cleaned, cleaned_counts = local_search_remove(sol, counts)
        cleaned, cleaned_counts = local_search_swap(cleaned, cleaned_counts)
        
        if len(cleaned) < best_count:
            best_blocks = cleaned
            best_counts = cleaned_counts
            best_count = len(cleaned)
    
    if best_blocks is None:
        # Fallback: run greedy once more
        sol = run_greedy(seed=0)
        if sol is None:
            return []
        best_blocks = sol
        best_counts = compute_coverage_counts(sol)
        best_count = len(best_blocks)

    # Phase 2: Simulated Annealing-like perturbation and local search
    current_temp = 1.0
    cooling_rate = 0.95
    
    while time.time() < deadline - 5.0:
        # Create perturbation: remove 3-8 random blocks
        num_remove = min(len(best_blocks), random.randint(3, 8))
        if num_remove == 0:
            break
        
        to_remove = random.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in best_blocks if b not in remove_set]
        
        # Compute holes
        covered_mask = compute_coverage_mask(perturbed_blocks)
        holes_mask = full_mask & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes_greedy(holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        # Acceptance criterion: accept if better, or with probability exp(-delta/temp) if worse
        if new_count < best_count:
            # Definitely accept
            new_counts = compute_coverage_counts(new_total_blocks)
            cleaned_blocks, cleaned_counts = local_search_remove(new_total_blocks, new_counts)
            cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)
            
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
                current_temp = 1.0  # Reset temperature on improvement
            elif len(new_total_blocks) < best_count:
                best_blocks = new_total_blocks
                best_counts = new_counts
                best_count = len(best_blocks)
                current_temp = 1.0
        else:
            # Possibly accept worse solution to escape local minima
            delta = new_count - best_count
            if delta > 0 and random.random() < np.exp(-delta / max(current_temp, 0.01)):
                new_counts = compute_coverage_counts(new_total_blocks)
                cleaned_blocks, cleaned_counts = local_search_remove(new_total_blocks, new_counts)
                cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)
                
                if len(cleaned_blocks) < best_count:
                    best_blocks = cleaned_blocks
                    best_counts = cleaned_counts
                    best_count = len(best_blocks)
                    current_temp = 1.0
        
        current_temp *= cooling_rate
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        best_blocks, best_counts = local_search_remove(best_blocks, best_counts)
        best_blocks, best_counts = local_search_swap(best_blocks, best_counts)

    # Convert to output format
    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    
    return final_blocks
