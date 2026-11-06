import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # target = 5/31
    t_num = 5
    t_den = 31
    
    best_solution = [7, 62, 434]
    best_max = 434
    
    for K in range(10, 500):
        if time.time() > deadline:
            break
        
        solution = None
        for N in range(3, 7):
            if solution:
                break
                
            def dfs(rem_num, rem_den, min_d, max_d, terms_left, current_denoms):
                nonlocal solution
                if solution:
                    return True
                if terms_left == 0:
                    if rem_num == 0:
                        solution = list(current_denoms)
                        return True
                    return False
                
                # Pruning
                # rem < terms_left / max_d
                # rem_num / rem_den < terms_left / max_d
                # rem_num * max_d < terms_left * rem_den
                if rem_num * max_d < terms_left * rem_den:
                    return False
                
                # rem > terms_left / min_d
                # rem_num / rem_den > terms_left / min_d
                # rem_num * min_d > terms_left * rem_den
                if rem_num * min_d > terms_left * rem_den:
                    return False
                
                # d must satisfy 1/d <= rem
                # 1/d <= rem_num / rem_den
                # rem_den <= d * rem_num
                # d >= rem_den / rem_num
                d_lo = max(min_d, (rem_den + rem_num - 1) // rem_num)
                
                for d in range(d_lo, max_d + 1):
                    if time.time() > deadline:
                        return True
                    
                    # new_rem = rem - 1/d
                    # new_rem = (rem_num * d - rem_den) / (rem_den * d)
                    new_num = rem_num * d - rem_den
                    if new_num < 0:
                        continue
                    
                    new_den = rem_den * d
                    
                    # Simplify? Not strictly necessary for correctness, but keeps numbers small.
                    # gcd calculation might be expensive.
                    # Let's skip gcd for now and see if it's fast enough.
                    
                    if dfs(new_num, new_den, d + 1, max_d, terms_left - 1, current_denoms + [d]):
                        return True
                return False
            
            dfs(t_num, t_den, 1, K, N, [])
            
        if solution:
            mx = max(solution)
            if mx < best_max:
                best_max = mx
                best_solution = solution
            break
            
    return best_solution
