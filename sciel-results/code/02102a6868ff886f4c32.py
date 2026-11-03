import itertools
import time
import heapq
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Map each block to the list of t-subset indices it covers
    block_t_indices = []
    for block in k_blocks:
        covered = []
        for combo in itertools.combinations(block, t):
            covered.append(t_index[combo])
        block_t_indices.append(covered)

    # Map each t-subset to the list of block indices that cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_indices):
        for idx in covered:
            t_to_blocks[idx].append(i)

    best_solution = None
    best_size = float('inf')

    # Helper: Greedy Set Cover with Randomization
    def greedy_cover(seed):
        random.seed(seed)
        # current_gain[i] = number of currently uncovered t-subsets covered by block i
        current_gain = np.array([len(lst) for lst in block_t_indices], dtype=np.int32)
        selected = set()
        solution = []
        
        # Max-heap using negative gains
        # To handle randomization, we can just pick randomly from top candidates
        # or add noise. Let's add noise to the gain for heap ordering.
        # However, heap entries become stale. 
        # Simpler: Just use the heap, but if there are ties, the heap order is arbitrary.
        # To force different paths, we can slightly perturb the initial gains or the update logic.
        # Let's perturb the initial gains slightly.
        noise = np.random.uniform(0, 0.1, size=num_k)
        initial_gain = current_gain + noise
        
        heap = [(-initial_gain[i], i) for i in range(num_k)]
        heapq.heapify(heap)
        
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t

        while uncovered_count > 0:
            if time.time() > deadline - 5:
                return None # Timeout in this restart
            
            # Pop from heap until we find a valid (non-stale, non-selected) entry
            while heap:
                neg_gain, i = heapq.heappop(heap)
                if i in selected:
                    continue
                # Check if gain is still valid (approximate check, exact check is expensive)
                # Since we only decrement gains, -neg_gain is an upper bound on current_gain[i].
                # If -neg_gain != current_gain[i], it's stale.
                if -neg_gain != current_gain[i]:
                    continue
                # Valid candidate
                break
            else:
                break # Heap exhausted
                
            if -neg_gain <= 0:
                break # No gain, cannot cover more
                
            # Select block i
            selected.add(i)
            solution.append(i)
            
            # Find newly covered t-subsets
            newly_covered = []
            for s_idx in block_t_indices[i]:
                if uncovered[s_idx]:
                    newly_covered.append(s_idx)
                    
            if not newly_covered:
                continue # Should not happen if gain > 0, but safety
                
            # Mark them as covered
            for s_idx in newly_covered:
                uncovered[s_idx] = False
                uncovered_count -= 1
                
            # Update gains for all blocks that share these t-subsets
            for s_idx in newly_covered:
                for j in t_to_blocks[s_idx]:
                    if j in selected:
                        continue
                    current_gain[j] -= 1
                    # Push updated gain to heap
                    heapq.heappush(heap, (-current_gain[j], j))

        return solution

    # Helper: Remove Redundant Blocks
    def remove_redundant(solution):
        if not solution:
            return solution
            
        # Rebuild cover counts
        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in solution:
            for s_idx in block_t_indices[i]:
                cover_count[s_idx] += 1
                
        new_solution = []
        changed = True
        while changed:
            changed = False
            for i in solution:
                # Check if block i is essential (covers any t-subset with count 1)
                essential = False
                for s_idx in block_t_indices[i]:
                    if cover_count[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove it
                    for s_idx in block_t_indices[i]:
                        cover_count[s_idx] -= 1
                    changed = True
                else:
                    new_solution.append(i)
            solution = new_solution
            new_solution = []
            
        return solution

    # Helper: Local Search (Swap)
    def local_search(solution, max_iters=5):
        if not solution:
            return solution
            
        sol_set = set(solution)
        current_size = len(solution)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Iterate through a copy of the list to allow modification
            for idx, b in enumerate(list(solution)):
                if b not in sol_set:
                    continue # Should not happen
                    
                # Find unique t-subsets covered by b
                # We need cover counts for this specific step? 
                # Recalculating cover counts is expensive. 
                # Approximation: Assume the structure from last reduction is mostly stable.
                # But swaps change it.
                # Let's just do a simple check: 
                # If we remove b, which t-subsets become uncovered?
                # We need to know if any other block in solution covers them.
                
                # This local search is tricky without maintaining global counts.
                # Let's skip complex swapping and rely on multiple greedy restarts + reduction.
                # The reduction step is very effective.
                
            break # Disable complex local search for now to ensure speed and correctness
            
        return solution

    # Main Loop: Multiple Restarts
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        sol = greedy_cover(restart_count)
        
        if sol is None:
            break
            
        # Remove redundant blocks
        sol = remove_redundant(sol)
        
        # Simple Local Search: Try to swap blocks if it reduces size
        # We'll implement a simple 1-exchange reduction check
        # For each block, if it's redundant, remove it. (Already done)
        # Try swapping: For each block b, if we replace it with b' not in sol, does it allow removal?
        # This is O(N^2 * Cost). Might be too slow.
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            # If we hit a very good size, we can stop early?
            # Lower bound is 264. If we get close, keep trying.
            # if best_size <= 280: break # Optional early stop

    if best_solution is None:
        # Fallback: Return empty or a trivial cover if something went wrong
        # But greedy should always succeed.
        return []

    return [list(k_blocks[i]) for i in best_solution]
