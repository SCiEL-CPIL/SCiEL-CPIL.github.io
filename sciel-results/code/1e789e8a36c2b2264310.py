import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    n = len(smooth_numbers)
    
    # 2. Precompute divisibility matrix
    div = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and smooth_numbers[j] % smooth_numbers[i] == 0:
                div[i][j] = True
    
    # 3. Best representation per target
    best_repr = {}
    targets = set(range(1, 129))
    
    # Iterative deepening on k
    for k in range(1, 31):
        if not targets:
            break
        if time.time() > DEADLINE:
            break
        
        if k <= n:
            min_sum_k = sum(smooth_numbers[:k])
        else:
            min_sum_k = float('inf')
        if min_sum_k > 128:
            break
        
        # DFS with incremental antichain checking
        def dfs(pos, remaining, cur_sum, chosen_vals, chosen_idx):
            if time.time() > DEADLINE:
                return
            if remaining == 0:
                if cur_sum <= 128 and cur_sum in targets:
                    best_repr[cur_sum] = list(chosen_vals)
                    targets.discard(cur_sum)
                return
            if pos + remaining > n:
                return
            if cur_sum > 128:
                return
            
            for i in range(pos, n - remaining + 1):
                val = smooth_numbers[i]
                # Check antichain compatibility
                compatible = True
                for j in chosen_idx:
                    if div[i][j] or div[j][i]:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                new_sum = cur_sum + val
                if new_sum > 128:
                    continue
                
                chosen_vals.append(val)
                chosen_idx.append(i)
                dfs(i + 1, remaining - 1, new_sum, chosen_vals, chosen_idx)
                chosen_vals.pop()
                chosen_idx.pop()
                
                if not targets:
                    return
        
        dfs(0, k, 0, [], [])
    
    # 4. Fallback for missing targets
    if targets:
        for t in sorted(targets):
            if time.time() > DEADLINE:
                break
            for k in range(1, 31):
                if time.time() > DEADLINE:
                    break
                found = None
                
                def dfs_target(pos, remaining, cur_sum, chosen_vals, chosen_idx):
                    if time.time() > DEADLINE:
                        return None
                    if remaining == 0:
                        if cur_sum == t:
                            return list(chosen_vals)
                        return None
                    if pos + remaining > n:
                        return None
                    if cur_sum > t:
                        return None
                    if pos < n and cur_sum + remaining * smooth_numbers[pos] > t:
                        return None
                    
                    for i in range(pos, n - remaining + 1):
                        val = smooth_numbers[i]
                        compatible = True
                        for j in chosen_idx:
                            if div[i][j] or div[j][i]:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        new_sum = cur_sum + val
                        if new_sum > t:
                            continue
                        chosen_vals.append(val)
                        chosen_idx.append(i)
                        result = dfs_target(i + 1, remaining - 1, new_sum, chosen_vals, chosen_idx)
                        chosen_vals.pop()
                        chosen_idx.pop()
                        if result is not None:
                            return result
                    return None
                
                found = dfs_target(0, k, 0, [], [])
                if found is not None:
                    best_repr[t] = found
                    targets.discard(t)
                    break
    
    # 5. Construct answer
    answers = []
    for t in range(1, 129):
        if t in best_repr:
            answers.append(best_repr[t])
        else:
            answers.append([])
    
    return answers
