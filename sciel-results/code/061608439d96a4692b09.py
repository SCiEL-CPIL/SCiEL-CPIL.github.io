import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map them to indices
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Precompute masks for all k-subsets
    k_blocks = list(itertools.combinations(points, k))
    block_masks = []
    for b in k_blocks:
        mask = 0
        for s in itertools.combinations(b, t):
            mask |= (1 << t_to_idx[s])
        block_masks.append(mask)
    
    full_mask = (1 << num_t) - 1
    best_solution = None
    best_count = float('inf')
    
    # 3. Multiple Greedy Runs with Randomization
    while time.time() < deadline:
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            candidates = []
            # Iterate through all possible blocks to find the one covering most uncovered t-subsets
            for i, m in enumerate(block_masks):
                gain = (m & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if not candidates:
                # This shouldn't happen if a covering exists
                break
                
            # Random selection from the best candidates to diversify solutions
            idx = random.choice(candidates)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        # 4. Reduction: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            n = len(selected_indices)
            for i in range(n):
                others = 0
                for j in range(n):
                    if j != i:
                        others |= block_masks[selected_indices[j]]
                unique = block_masks[selected_indices[i]] & ~others
                if unique == 0:
                    selected_indices.pop(i)
                    changed = True
                    break
        
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_solution = [list(k_blocks[i]) for i in selected_indices]
            
            # If we found a very good solution, we might want to stop early?
            # Or continue to see if we can do better.
            # Given the time limit, continuing is better.
            
    if best_solution is None:
        # Fallback: return empty or a trivial invalid solution? 
        # The problem guarantees a solution exists. 
        # If we ran out of time before finishing the first run, we might have partial.
        # But the loop structure ensures we complete at least one run if we have time.
        # If deadline is hit during the first run's inner loop, selected_indices might be incomplete.
        # We should check if the current selected_indices form a valid covering before updating best.
        # Actually, the inner loop `while uncovered` ensures we only exit when uncovered is 0.
        # So selected_indices is always a valid covering when the inner loop finishes.
        
        # If the outer loop didn't complete even one iteration (extremely slow precomp?), 
        # we return whatever we have. But precomp is fast.
        pass
        
    return best_solution if best_solution is not None else []

