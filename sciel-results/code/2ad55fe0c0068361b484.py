import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 3. Precompute block -> list of t_subset indices
    block_t_subsets = []
    for b in all_blocks:
        subs = []
        for ts in itertools.combinations(b, t):
            subs.append(t_subset_to_idx[ts])
        block_t_subsets.append(subs)
        
    # State for greedy
    is_uncovered = [True] * num_t_subsets
    num_uncovered = num_t_subsets
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Reset state
        is_uncovered = [True] * num_t_subsets
        num_uncovered = num_t_subsets
        
        # Initialize heap with randomized tie-breaking
        heap = []
        for i in range(num_blocks):
            heapq.heappush(heap, (-56, random.random(), i))
            
        solution = []
        
        while num_uncovered > 0:
            neg_gain, _, bid = heapq.heappop(heap)
            gain = -neg_gain
            
            # Verify gain
            real_gain = 0
            for ts_idx in block_t_subsets[bid]:
                if is_uncovered[ts_idx]:
                    real_gain += 1
            
            if real_gain != gain:
                if real_gain > 0:
                    heapq.heappush(heap, (-real_gain, random.random(), bid))
                continue
            
            if real_gain == 0:
                continue
            
            # Select this block
            solution.append(bid)
            
            # Update coverage
            for ts_idx in block_t_subsets[bid]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    num_uncovered -= 1
        
        # Post-optimization: Remove redundant blocks
        if num_uncovered > 0:
            continue  # Solution incomplete
            
        # Use coverage counts to efficiently check redundancy
        coverage_count = [0] * num_t_subsets
        for bid in solution:
            for ts_idx in block_t_subsets[bid]:
                coverage_count[ts_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(solution) - 1, -1, -1):
                if time.time() > deadline:
                    break
                    
                bid = solution[i]
                redundant = True
                for ts_idx in block_t_subsets[bid]:
                    if coverage_count[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    solution.pop(i)
                    for ts_idx in block_t_subsets[bid]:
                        coverage_count[ts_idx] -= 1
                    changed = True
            
            if time.time() > deadline:
                break
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = [all_blocks[i] for i in solution]
            
    if best_solution is None:
        return []
        
    return best_solution
