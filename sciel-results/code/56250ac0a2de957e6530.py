import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Helper function for combinations count
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

    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Build incidence: block_index -> list of t_subset_indices
    # And t_subset_index -> list of block_indices
    block_to_ts = [[] for _ in range(num_k_blocks)]
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            block_to_ts[j].append(idx)
            ts_to_blocks[idx].append(j)
            
    # Precompute union of points for each t-subset for quick access? 
    # Actually, we need to check if a set of t-subsets can be covered by a k-block.
    # A set of t-subsets S can be covered by a k-block B iff union(S) subseteq B.
    # So, for a set of unique t-subsets U of a block, if |union(U)| <= k, 
    # there exists a block covering them. We need to find if there's a *better* block.
    
    # Helper to get union of points in a list of t-subset indices
    def get_union_points(ts_indices):
        union_set = set()
        for idx in ts_indices:
            union_set.update(all_t_subsets[idx])
        return union_set

    # State for greedy
    def run_targeted_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        # We will maintain a list of uncovered t-subset indices
        # Initially all are uncovered
        uncovered_indices = list(range(num_t_subsets))
        
        while uncovered_indices:
            if time.time() > deadline - 5.0:
                return None
            
            # Strategy: Pick a random uncovered t-subset, 
            # find blocks covering it, pick the one with max gain.
            # To speed up, we can pick the uncovered t-subset with the fewest covering blocks?
            # Or just random? Random is faster.
            
            # Optimization: If we pick a random uncovered ts, we can look at ts_to_blocks[ts_idx]
            # and filter for unselected blocks, then pick max gain.
            
            # Pick random uncovered ts
            ts_idx = random.choice(uncovered_indices)
            
            # Get candidate blocks covering this ts
            candidates = [b for b in ts_to_blocks[ts_idx] if not is_selected[b]]
            
            if not candidates:
                # Should not happen if problem is solvable, but safe fallback
                break
                
            # Find best candidate based on current gain
            best_block_idx = max(candidates, key=lambda x: gain[x])
            
            sol.append(best_block_idx)
            is_selected[best_block_idx] = True
            gain[best_block_idx] = -1
            
            # Update gains and uncovered list
            # We need to update gain for all blocks that share t-subsets with the selected block
            # And update uncovered_indices
            
            new_uncovered = []
            for ts_idx_cand in block_to_ts[best_block_idx]:
                if is_uncovered[ts_idx_cand]:
                    is_uncovered[ts_idx_cand] = False
                    # This ts is now covered. 
                    # For all blocks covering this ts, their gain decreases by 1
                    for b_idx in ts_to_blocks[ts_idx_cand]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
                else:
                    new_uncovered.append(ts_idx_cand)
            
            # Rebuild uncovered_indices? 
            # This is O(N) each step. 
            # Alternative: Just check is_uncovered when picking?
            # Let's stick to a simple list and filter occasionally or just use the array.
            # Actually, maintaining the list is O(N) per step, which is fine for ~300 steps.
            uncovered_indices = [i for i in range(num_t_subsets) if is_uncovered[i]]
            
        return sol

    def run_standard_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
                
            sol.append(max_idx)
            is_selected[max_idx] = True
            gain[max_idx] = -1
            
            for ts_idx in block_to_ts[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
                            
        return sol

    # 1. Initialization
    best_solution = None
    best_len = float('inf')
    
    # Run a few initializations
    init_time_limit = start_time + 20.0
    seeds = range(50)
    for seed in seeds:
        if time.time() > init_time_limit:
            break
        
        # Alternate between targeted and standard
        if seed % 2 == 0:
            sol = run_targeted_greedy(seed)
        else:
            sol = run_standard_greedy(seed)
            
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    if best_solution is None:
        # Fallback to simple standard greedy if timed out
        best_solution = run_standard_greedy(0)
        best_len = len(best_solution)

    # Local Search Optimization
    def optimize(blocks):
        # blocks is a list of block indices
        if not blocks:
            return blocks
            
        # Build cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] += 1
                
        # Set of selected block indices for O(1) lookup
        selected_set = set(blocks)
        
        # 1. Remove Redundant Blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                redundant = True
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    selected_set.discard(b_idx)
                    changed = True
                    break # Restart loop to be safe with indices
            
        # 2. 1-Exchange (Swap)
        # We want to replace a block B with B' such that B' covers all unique t-subsets of B.
        # Unique t-subsets of B are those with cover_count == 1.
        # If the union of points in these unique t-subsets has size <= k, 
        # there might be a block covering them.
        
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10.0:
                    return blocks
                    
                b_idx = blocks[i]
                unique_ts = [ts_idx for ts_idx in block_to_ts[b_idx] if cover_count[ts_idx] == 1]
                
                if not unique_ts:
                    continue
                
                # Check if union of unique_ts fits in a block
                union_pts = get_union_points(unique_ts)
                if len(union_pts) > k:
                    continue
                    
                # Find candidates that cover all unique_ts
                # A block B' covers all unique_ts if unique_ts subseteq block_to_ts[B']
                # We can find this by intersecting ts_to_blocks for each unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Remove current block and already selected blocks
                candidates.discard(b_idx)
                valid_candidates = [c for c in candidates if c not in selected_set]
                
                if not valid_candidates:
                    continue
                    
                # Pick a candidate. 
                # To improve, we might want one that covers *more* new stuff?
                # But since we are swapping, the size stays same.
                # Any valid candidate works to maintain validity.
                # However, a "better" candidate might have more unique coverage for future steps?
                # Let's just pick one randomly or the first one.
                new_b_idx = random.choice(valid_candidates)
                
                # Perform Swap
                # Update cover counts
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[new_b_idx]:
                    cover_count[ts_idx] += 1
                    
                blocks[i] = new_b_idx
                selected_set.discard(b_idx)
                selected_set.add(new_b_idx)
                
                improved = True
                # Restart local search after a swap? Or continue?
                # Continuing is faster, but might miss cascading improvements.
                # Let's break and restart the outer loop of 'improved'
                break
                
        # 3. 2-Exchange (Remove 2, Add 1)
        # This is more expensive. We try to remove two blocks i, j and add one block m.
        # Condition: All t-subsets uniquely covered by i or j must be covered by m.
        
        improved = True
        while improved:
            improved = False
            n_blocks = len(blocks)
            for i in range(n_blocks):
                if time.time() > deadline - 5.0:
                    return blocks
                    
                b_idx_i = blocks[i]
                unique_ts_i = [ts_idx for ts_idx in block_to_ts[b_idx_i] if cover_count[ts_idx] == 1]
                
                for j in range(i + 1, n_blocks):
                    if time.time() > deadline - 5.0:
                        return blocks
                        
                    b_idx_j = blocks[j]
                    unique_ts_j = [ts_idx for ts_idx in block_to_ts[b_idx_j] if cover_count[ts_idx] == 1]
                    
                    # Combine unique sets
                    # Note: A t-subset might be unique to i, unique to j, or covered by both (count 2).
                    # If count is 2, it's not unique to either.
                    # We need to cover all ts where count==1 AND (covered by i OR covered by j).
                    # Actually, if we remove i and j, any ts with count==1 that is covered by i or j becomes uncovered.
                    # So we need m to cover all such ts.
                    
                    union_unique_ts = set(unique_ts_i) | set(unique_ts_j)
                    
                    if not union_unique_ts:
                        continue
                        
                    # Check if union of points fits in k
                    union_pts = get_union_points(list(union_unique_ts))
                    if len(union_pts) > k:
                        continue
                        
                    # Find candidates covering all union_unique_ts
                    candidates = set(ts_to_blocks[list(union_unique_ts)[0]])
                    for ts_idx in list(union_unique_ts)[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                            
                    if not candidates:
                        continue
                        
                    candidates.discard(b_idx_i)
                    candidates.discard(b_idx_j)
                    valid_candidates = [c for c in candidates if c not in selected_set]
                    
                    if not valid_candidates:
                        continue
                        
                    new_b_idx = random.choice(valid_candidates)
                    
                    # Perform 2-Exchange
                    # Update cover counts
                    for ts_idx in block_to_ts[b_idx_i]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[b_idx_j]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[new_b_idx]:
                        cover_count[ts_idx] += 1
                        
                    # Verify validity (should be true by construction, but safe to check a few)
                    # Actually, we must ensure no ts has count 0.
                    # Since we only reduced counts for ts covered by i or j,
                    # and we added m which covers all unique ones,
                    # any ts covered by i or j but NOT unique (count >= 2) will have count >= 1 after removal?
                    # If count was 2, and we remove i, count becomes 1. If m also covers it, count becomes 2.
                    # If count was 2, and m does NOT cover it, count becomes 1. Still valid.
                    # So validity is guaranteed if m covers all unique ts.
                    
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(new_b_idx)
                    
                    selected_set.discard(b_idx_i)
                    selected_set.discard(b_idx_j)
                    selected_set.add(new_b_idx)
                    
                    improved = True
                    break # Restart 2-exchange loop
                    
                if improved:
                    break
                    
        return blocks

    # Main Loop
    while time.time() < deadline - 10.0:
        optimized = optimize(best_solution)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized
            
        # If no improvement, we might try a random restart or perturbation
        # To keep it simple and fast, we just stop if no improvement in a while?
        # Or we can try to break out of local minima by randomly removing a block and re-covering?
        # That's complex. Let's just iterate optimize() a few times or until time runs out.
        # Since optimize() includes swaps, it should converge quickly.
        
        # To avoid infinite loop if optimize() doesn't change anything,
        # we check if length changed.
        # Actually, optimize() modifies in place.
        # If len(optimized) == best_len, we might be stuck.
        # Let's try a few random restarts if we have time.
        
        if time.time() < deadline - 20.0:
            # Try a new initialization if we are stuck
            # Just pick a random seed
            new_seed = int(time.time() * 1000) % 10000
            new_sol = run_targeted_greedy(new_seed)
            if new_sol is not None:
                new_opt = optimize(new_sol)
                if len(new_opt) < best_len:
                    best_len = len(new_opt)
                    best_solution = new_opt
                # Even if not better, maybe it leads to better?
                # No, let's stick to the best found.

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[idx]) for idx in best_solution]
    
    return final_blocks

if __name__ == "__main__":
    # Test with C(19, 8, 5)
    blocks = construct(19, 8, 5)
    print(f"Number of blocks: {len(blocks)}")
