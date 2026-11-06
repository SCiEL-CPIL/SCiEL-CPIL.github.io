import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    max_target = 128
    
    # 1. Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        y = n
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(n)
    
    n_smooth = len(smooth)
    
    # 2. Build divisibility incompatibility matrix
    # incompat[i][j] = True if smooth[i] divides smooth[j] or vice versa (i != j)
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                incompat[i][j] = True
                incompat[j][i] = True
    
    # 3. Precompute for each index, the bitmask of incompatible indices
    # Use Python integers as bitsets
    incompat_mask = [0] * n_smooth
    for i in range(n_smooth):
        mask = 0
        for j in range(n_smooth):
            if i != j and incompat[i][j]:
                mask |= (1 << j)
        incompat_mask[i] = mask
    
    # 4. Find minimal antichain for each target using backtracking with pruning
    # State: (indices chosen so far as bitmask, current sum, next index to consider)
    # We want to minimize the number of elements
    
    best_repr = {}  # target -> list of smooth numbers
    best_count = {}  # target -> minimum count
    
    # We'll do iterative deepening on k
    for k in range(1, 31):
        if time.time() > deadline:
            break
        
        # DFS to find antichains of size k
        # Use bitmask for chosen set for fast compatibility check
        found_this_k = {}
        
        def dfs(start_idx, chosen_mask, chosen_list, current_sum):
            if time.time() > deadline:
                return
            
            if len(chosen_list) == k:
                s = current_sum
                if 1 <= s <= max_target and s not in best_count:
                    best_repr[s] = list(chosen_list)
                    best_count[s] = k
                    found_this_k[s] = True
                return
            
            remaining = k - len(chosen_list)
            
            for idx in range(start_idx, n_smooth):
                # Check if idx is compatible with all chosen
                if chosen_mask & incompat_mask[idx]:
                    continue
                
                new_sum = current_sum + smooth[idx]
                if new_sum > max_target:
                    # Since smooth is sorted, further idx will be larger
                    # But wait, smooth is not necessarily sorted by value in a way that helps
                    # Actually smooth IS sorted since we iterate 1..128
                    break
                
                # Pruning: even with the smallest possible remaining elements,
                # can we reach at most max_target?
                # Minimum sum of 'remaining' more elements starting from idx+1
                # This is expensive to compute, so skip for now
                
                chosen_list.append(smooth[idx])
                new_mask = chosen_mask | (1 << idx)
                dfs(idx + 1, new_mask, chosen_list, new_sum)
                chosen_list.pop()
            
            return
        
        dfs(0, 0, [], 0)
        
        # Check if all targets found
        if len(best_count) == max_target:
            break
    
    # 5. Construct final answer
    answers = []
    for n in range(1, max_target + 1):
        if n in best_repr:
            answers.append(best_repr[n])
        else:
            answers.append([])
    
    return answers
