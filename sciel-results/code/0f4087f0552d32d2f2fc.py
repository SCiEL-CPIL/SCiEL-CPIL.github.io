import math
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Leave some margin for 60s limit
    
    # Calculate LCM(1..30)
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    D = 1
    for i in range(1, 31):
        D = lcm(D, i)
        
    # Generate sums for first 15 terms
    # We store the integer numerators
    sums1 = [0]
    for i in range(1, 16):
        val = D // i
        new_sums = []
        for s in sums1:
            new_sums.append(s - val)
            new_sums.append(s)
            new_sums.append(s + val)
        sums1 = new_sums
        
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Generate sums for last 15 terms
    sums2 = [0]
    for i in range(16, 31):
        val = D // i
        new_sums = []
        for s in sums2:
            new_sums.append(s - val)
            new_sums.append(s)
            new_sums.append(s + val)
        sums2 = new_sums
        
    if time.time() > deadline:
        return [0] * 28 + [1, -1]

    # Sort sums2 for binary search
    sums2.sort()
    
    min_abs_num = float('inf')
    best_n1 = None
    best_n2 = None
    
    # For each sum in sums1, find best match in sums2
    # We want n1 + n2 to be close to 0 but not 0
    for n1 in sums1:
        target = -n1
        # Find insertion point in sums2
        import bisect
        idx = bisect.bisect_left(sums2, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(sums2):
            candidates.append(sums2[idx])
        if idx > 0:
            candidates.append(sums2[idx-1])
            
        # Also check neighbors of candidates in case the closest one sums to 0?
        # Actually, if sums2[idx] == target, sum is 0. We need next closest.
        # So we should check a small window around idx.
        # Let's check idx-2 to idx+2 to be safe against 0-sums.
        start_idx = max(0, idx - 2)
        end_idx = min(len(sums2), idx + 3)
        
        for j in range(start_idx, end_idx):
            n2 = sums2[j]
            total = n1 + n2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < min_abs_num:
                min_abs_num = abs_total
                best_n1 = n1
                best_n2 = n2
                
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    if best_n1 is None:
        # Fallback
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    # Find coeffs for best_n1 in first half
    coeffs1 = None
    # We need to search for the specific numerator
    # We can do a DFS or iterate again. Since 3^15 is large, we should be smart.
    # But we only need ONE solution.
    # We can just iterate through the generation process and check if we can form best_n1.
    # This is effectively a subset sum problem with -1,0,1.
    # Given the density of sums, it's likely fast to find one.
    
    def find_coeffs(D, start, end, target):
        # Recursive search might be too slow if unlucky.
        # Let's use a set of reachable sums with their coeffs?
        # Or just backtrack.
        # Since we know best_n1 is reachable, we can try to find it.
        # A simple DFS with pruning?
        # Actually, we can just re-generate and store the path for the first time we see the target?
        # No, that requires storing all paths.
        
        # Let's try a simple recursive search.
        # 15 steps. Branching factor 3.
        # In the worst case, it visits all nodes.
        # But we can prune if the remaining capacity can't reach the target.
        
        # Capacity of remaining terms i..end is sum(D/j for j in i..end)
        # Precompute suffix sums of capacities
        caps = [0] * (end + 2)
        for i in range(end, start - 1, -1):
            caps[i] = caps[i+1] + D // i
            
        def dfs(i, current_sum, coeffs):
            if time.time() > deadline:
                return None
            if i > end:
                if current_sum == target:
                    return coeffs[:]
                return None
            
            # Pruning
            diff = target - current_sum
            if abs(diff) > caps[i]:
                return None
                
            # Try -1, 0, 1
            val = D // i
            coeffs.append(-1)
            res = dfs(i+1, current_sum - val, coeffs)
            if res: return res
            coeffs.pop()
            
            coeffs.append(0)
            res = dfs(i+1, current_sum, coeffs)
            if res: return res
            coeffs.pop()
            
            coeffs.append(1)
            res = dfs(i+1, current_sum + val, coeffs)
            if res: return res
            coeffs.pop()
            
            return None
        
        return dfs(start, 0, [])
        
    c1 = find_coeffs(D, 1, 15, best_n1)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    c2 = find_coeffs(D, 16, 30, best_n2)
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
        
    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]
        
    return c1 + c2
