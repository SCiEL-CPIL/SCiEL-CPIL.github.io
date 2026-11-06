import math
import time

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12! that are > 12
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    n_div = len(all_divisors)
    
    # Map each divisor to its prime exponent vector
    divisor_to_exp = {}
    for d in all_divisors:
        temp = d
        exp_vec = []
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            exp_vec.append(count)
        divisor_to_exp[d] = tuple(exp_vec)
    
    div_exp_list = [divisor_to_exp[d] for d in all_divisors]
    
    # Precompute suffix sum of exponents for pruning
    suffix_exp_sum = [[0, 0, 0, 0, 0]] * (n_div + 1)
    for i in range(n_div - 1, -1, -1):
        prev = suffix_exp_sum[i + 1]
        cur = div_exp_list[i]
        suffix_exp_sum[i] = [prev[j] + cur[j] for j in range(5)]
    
    # First, find any valid solution using greedy DFS (pick smallest fitting divisor)
    def greedy_dfs(rem_exp, min_idx, path):
        if rem_exp[0] == 0 and rem_exp[1] == 0 and rem_exp[2] == 0 and rem_exp[3] == 0 and rem_exp[4] == 0:
            return path[:]
        
        if min_idx >= n_div:
            return None
        
        for i in range(min_idx, n_div):
            d = all_divisors[i]
            exp_vec = div_exp_list[i]
            
            can_apply = True
            new_rem_exp = [0, 0, 0, 0, 0]
            for j in range(5):
                if rem_exp[j] < exp_vec[j]:
                    can_apply = False
                    break
                new_rem_exp[j] = rem_exp[j] - exp_vec[j]
            
            if not can_apply:
                continue
            
            path.append(d)
            result = greedy_dfs(tuple(new_rem_exp), i + 1, path)
            if result is not None:
                return result
            path.pop()
        
        return None
    
    initial_sol = greedy_dfs(tuple(exponents), 0, [])
    if initial_sol is not None:
        best_max = max(initial_sol)
        best_solution = initial_sol
    else:
        best_max = T12
        best_solution = [T12]
    
    # Now do branch and bound to improve
    def dfs(rem_exp, min_idx, current_max, path):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        # Check if we're done
        if rem_exp[0] == 0 and rem_exp[1] == 0 and rem_exp[2] == 0 and rem_exp[3] == 0 and rem_exp[4] == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = path[:]
            return
        
        if min_idx >= n_div:
            return
        
        # Pruning: if the remaining exponents cannot be satisfied by available divisors
        for j in range(5):
            if rem_exp[j] > suffix_exp_sum[min_idx][j]:
                return
        
        # Pruning: if current_max >= best_max, we can't improve
        if current_max >= best_max:
            return
        
        # Try each divisor from min_idx onwards
        for i in range(min_idx, n_div):
            d = all_divisors[i]
            exp_vec = div_exp_list[i]
            
            # If this divisor is larger than or equal to best_max, skip
            if d >= best_max:
                continue
            
            # Check if we can apply this divisor
            can_apply = True
            new_rem_exp = [0, 0, 0, 0, 0]
            for j in range(5):
                if rem_exp[j] < exp_vec[j]:
                    can_apply = False
                    break
                new_rem_exp[j] = rem_exp[j] - exp_vec[j]
            
            if not can_apply:
                continue
            
            new_current_max = max(current_max, d)
            path.append(d)
            dfs(tuple(new_rem_exp), i + 1, new_current_max, path)
            path.pop()
            
            if time.time() > deadline:
                return
    
    dfs(tuple(exponents), 0, 0, [])
    
    return best_solution
