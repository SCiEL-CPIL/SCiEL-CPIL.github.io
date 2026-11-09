import itertools
import random

def construct(v, k, t):
    # 1. Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # 2. Generate all k-subsets and their bitmasks
    k_subs = list(itertools.combinations(range(v), k))
    block_data = [] # list of (block_tuple, mask_int)
    
    for block in k_subs:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))
        
    def greedy_once(rng):
        uncovered_mask = (1 << num_t_subs) - 1
        solution = []
        
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            for block, mask in block_data:
                gain = (uncovered_mask & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [(block, mask)]
                elif gain == best_gain and gain > 0:
                    best_candidates.append((block, mask))
                    
            if best_gain == 0:
                break
                
            chosen_block, chosen_mask = rng.choice(best_candidates)
            solution.append(chosen_block)
            uncovered_mask &= ~chosen_mask
            
        return solution

    def prune(solution):
        # Re-calculate counts from solution
        counts = [0] * num_t_subs
        for block in solution:
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                counts[idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(solution)):
                block = solution[i]
                # Tentatively remove
                for t_sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[t_sub]
                    counts[idx] -= 1
                    
                # Check if any count is 0
                if min(counts) > 0:
                    solution.pop(i)
                    changed = True
                    break # Restart loop
                else:
                    # Restore
                    for t_sub in itertools.combinations(block, t):
                        idx = t_sub_to_idx[t_sub]
                        counts[idx] += 1
                        
        return solution

    best_solution = None
    best_len = float('inf')
    
    # Run multiple iterations with different random seeds
    # 15 iterations should be fast enough
    for i in range(15):
        rng = random.Random(i)
        sol = greedy_once(rng)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol.copy()
            
    if best_solution is None:
        # Fallback to deterministic greedy if something went wrong
        rng = random.Random(0)
        best_solution = greedy_once(rng)

    best_solution = prune(best_solution)
    return [list(b) for b in best_solution]
