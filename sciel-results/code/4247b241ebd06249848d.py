import math
import time

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    primes = [2, 3, 5, 7, 11]
    max_exp = [10, 5, 2, 1, 1]
    target_pf = tuple(max_exp)
    
    # Precompute prime factorization for all integers up to some bound
    # We'll compute on the fly but cache
    pf_cache = {}
    
    def get_pf(n):
        if n in pf_cache:
            return pf_cache[n]
        if n == 1:
            pf = (0, 0, 0, 0, 0)
        else:
            pf_list = [0, 0, 0, 0, 0]
            temp = n
            for i, p in enumerate(primes):
                while temp % p == 0:
                    pf_list[i] += 1
                    temp //= p
            if temp != 1:
                pf = None  # has prime factor > 11
            else:
                pf = tuple(pf_list)
        pf_cache[n] = pf
        return pf
    
    # Generate all divisors of 12! greater than 12
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(max_exp[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Map each valid divisor to its pf
    div_pf = {}
    for d in valid_divs:
        div_pf[d] = get_pf(d)
    
    # Index: for each divisor, its position in valid_divs
    # We'll use binary search for "next divisors after x"
    
    import bisect
    
    def min_product_after(x, m):
        """Product of m smallest divisors in valid_divs strictly > x. None if not enough."""
        if m == 0:
            return 1
        idx = bisect.bisect_right(valid_divs, x)
        if idx + m > len(valid_divs):
            return None
        prod = 1
        for j in range(idx, idx + m):
            prod *= valid_divs[j]
        return prod
    
    best_max_val = N
    best_factors = [N]
    
    def dfs(last_factor, rem_pf, count_left, path, current_max):
        nonlocal best_max_val, best_factors
        
        if time.time() - start_time > 85:
            return
        
        if count_left == 0:
            if rem_pf == (0, 0, 0, 0, 0):
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(path)
            return
        
        if current_max >= best_max_val:
            return
        
        # Compute remaining target value
        rem_target = 1
        for i in range(5):
            rem_target *= primes[i] ** rem_pf[i]
        
        # Min product check
        min_prod = min_product_after(last_factor, count_left)
        if min_prod is None:
            return
        if rem_target < min_prod:
            return
        
        # Upper bound for next factor: x * (x+1) * ... * (x + count_left - 1) <= rem_target
        # So x is at most roughly rem_target^(1/count_left)
        if count_left == 1:
            x = rem_target
            if x > last_factor and x > 12 and x < best_max_val:
                pf_x = get_pf(x)
                if pf_x is not None and pf_x == rem_pf:
                    path.append(x)
                    dfs(x, (0,0,0,0,0), 0, path, x)
                    path.pop()
            return
        
        # Upper bound for x
        # x * (x+1) * ... * (x + count_left - 1) <= rem_target
        # x^count_left <= rem_target roughly
        x_upper = int(rem_target ** (1.0 / count_left)) + count_left + 5
        x_upper = min(x_upper, best_max_val - 1)
        x_upper = min(x_upper, rem_target // 1)  # x < rem_target
        
        # Iterate through valid divisors in range (last_factor, x_upper]
        idx_start = bisect.bisect_right(valid_divs, last_factor)
        idx_end = bisect.bisect_right(valid_divs, x_upper)
        
        for idx in range(idx_start, idx_end):
            if time.time() - start_time > 85:
                return
            
            x = valid_divs[idx]
            if x >= best_max_val:
                break
            
            pf_x = div_pf[x]
            if pf_x is None:
                continue
            
            # Check pf_x <= rem_pf
            can_divide = True
            for i in range(5):
                if pf_x[i] > rem_pf[i]:
                    can_divide = False
                    break
            if not can_divide:
                continue
            
            new_rem_pf = tuple(rem_pf[i] - pf_x[i] for i in range(5))
            new_count = count_left - 1
            new_rem_target = rem_target // x
            
            if new_count == 1:
                # Last factor must be exactly new_rem_target
                last_x = new_rem_target
                if last_x > x and last_x > 12 and last_x < best_max_val:
                    pf_last = get_pf(last_x)
                    if pf_last is not None and pf_last == new_rem_pf:
                        path.append(x)
                        path.append(last_x)
                        dfs(last_x, (0,0,0,0,0), 0, path, last_x)
                        path.pop()
                        path.pop()
                continue
            
            # Pruning: min product for remaining
            min_prod_next = min_product_after(x, new_count)
            if min_prod_next is None:
                continue
            if new_rem_target < min_prod_next:
                continue
            
            path.append(x)
            dfs(x, new_rem_pf, new_count, path, x)
            path.pop()
    
    # Try k from larger to smaller (more factors → smaller max typically)
    # But k=1 gives max=12!, so start with reasonable k
    # The optimal k is probably 4-7
    for k in range(10, 0, -1):
        if time.time() - start_time > 80:
            break
        dfs(12, target_pf, k, [], 12)
        if best_max_val <= 25:
            break
    
    if not best_factors or best_factors == [N]:
        # Fallback: try a few specific decompositions
        # 12! = 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21 * 22 * 23 * 24 / (13*14*...*24 / 12!)
        # Actually just return [N] as valid
        best_factors = [N]
    
    return best_factors
