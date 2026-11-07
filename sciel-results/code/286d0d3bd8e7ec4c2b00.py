import math
import time

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n):
    k = n
    d = 2
    kernel = 1
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                kernel *= d
        d += 1 if d == 2 else 2
    if k > 1:
        kernel *= k
    return kernel

def _validate(lst, n=210, max_window=10210, max_size=100):
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)

def construct():
    t0 = time.time()
    deadline = t0 + 80.0
    
    n = P['n']
    best_val = float('inf')
    best_list = None
    
    # 1. Try single element
    # x = 210 * t^2 > 210
    # t=2 -> 840. val = 630.
    x_single = 210 * 4
    if x_single <= P['maximum_window']:
        if x_single - n < best_val:
            best_val = x_single - n
            best_list = [x_single]
            
    # 2. Try pairs and triples (and quadruples) with max element < 240
    # We know best pair is 30 (max 240).
    # We want to find max < 240.
    # Search M from 211 to 239.
    
    # Precompute sqf for 211..240
    sqf_map = {}
    for i in range(211, 241):
        sqf_map[i] = _squarefree_kernel(i)
        
    target_sqf = _squarefree_kernel(210) # 210
    
    # Search for subsets of size 2, 3, 4 in range [211, M]
    # We iterate M upwards to find the minimal max.
    for M in range(211, 240):
        if time.time() > deadline:
            break
            
        nums = list(range(211, M + 1))
        
        # Size 2
        found = False
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                x1 = nums[i]
                x2 = nums[j]
                if sqf_map[x1] ^ sqf_map[x2] == target_sqf:
                    val = M - n
                    if val < best_val:
                        best_val = val
                        best_list = [x1, x2]
                    found = True
                    break
            if found:
                break
        
        if found:
            # Since we iterate M upwards, the first M with a solution is the minimal M for pairs.
            # But we should also check triples for the same M?
            # No, if a pair exists for M, max is M.
            # If a triple exists for M, max is M.
            # We want minimal max.
            # So if we find ANY solution for M, we can stop?
            # Wait. We want to minimize max.
            # If we find a solution with max M, can we find one with max M-1?
            # We are iterating M from 211.
            # So the first M we find a solution for is the global minimum max.
            # So we can return immediately.
            if _validate(best_list):
                return [int(x) for x in best_list]
                
        # Size 3
        found = False
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    x1 = nums[i]
                    x2 = nums[j]
                    x3 = nums[k]
                    if sqf_map[x1] ^ sqf_map[x2] ^ sqf_map[x3] == target_sqf:
                        val = M - n
                        if val < best_val:
                            best_val = val
                            best_list = [x1, x2, x3]
                        found = True
                        break
                if found:
                    break
            if found:
                break
                
        if found:
            if _validate(best_list):
                return [int(x) for x in best_list]

        # Size 4
        found = False
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    for l in range(k + 1, len(nums)):
                        x1 = nums[i]
                        x2 = nums[j]
                        x3 = nums[k]
                        x4 = nums[l]
                        if sqf_map[x1] ^ sqf_map[x2] ^ sqf_map[x3] ^ sqf_map[x4] == target_sqf:
                            val = M - n
                            if val < best_val:
                                best_val = val
                                best_list = [x1, x2, x3, x4]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
                
        if found:
            if _validate(best_list):
                return [int(x) for x in best_list]
                
    # If no solution found with max < 240, use the best pair found by algebraic search (which is 30)
    # We can just run the algebraic search for pairs to be sure, or hardcode the known best if time is tight.
    # But let's run a quick pair search for t up to 20.
    
    pair_found = False
    for t in range(1, 20):
        M = 210 * t * t
        sqrt_M = math.isqrt(M)
        if sqrt_M * sqrt_M != M:
            sqrt_M += 1
        # Check divisors near sqrt
        a = math.isqrt(M)
        while a > 210:
            if M % a == 0:
                b = M // a
                if b > 210 and b != a:
                    val = max(a, b) - 210
                    if val < best_val:
                        best_val = val
                        best_list = [a, b]
                    pair_found = True
                    break
            a -= 1
            
    if best_list is None:
        return [840]
        
    return [int(x) for x in best_list]
