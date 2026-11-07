import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of all divisors
    LCM = 1
    for d in divisors:
        LCM = LCM * d // gcd(LCM, d)
    # LCM = 30030
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Sort divisors by rarity (larger divisors have fewer multiples in a window)
    # This helps prune the search tree earlier
    indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
    orig_indices = [d[0] for d in indexed_divs]
    sorted_divs = [d[1] for d in indexed_divs]
    
    # For each span from minimum possible (9) upward, search for a valid window
    for span in range(9, LCM + 1):
        if time.time() > deadline:
            break
            
        # For each possible window start L in [1, LCM]
        for L in range(1, LCM + 1):
            if time.time() > deadline:
                break
            
            # Quick check: does each divisor have at least one multiple in [L, L+span]?
            valid = True
            for d in sorted_divs:
                k = (L + d - 1) // d
                val = k * d
                if val > L + span:
                    valid = False
                    break
            if not valid:
                continue
            
            # Generate candidate lists for each divisor
            candidates = []
            for d in sorted_divs:
                k = (L + d - 1) // d
                val = k * d
                cand = []
                while val <= L + span:
                    cand.append(val)
                    val += d
                candidates.append(cand)
            
            # Backtracking search for distinct assignment
            # Sort by number of candidates (ascending) for better pruning
            order = list(range(n))
            order.sort(key=lambda i: len(candidates[i]))
            
            assignment = [0] * n
            used = set()
            
            def dfs(idx):
                if idx == n:
                    return True
                orig_idx = order[idx]
                for v in candidates[orig_idx]:
                    if v not in used:
                        used.add(v)
                        assignment[orig_idx] = v
                        if dfs(idx + 1):
                            return True
                        used.remove(v)
                        assignment[orig_idx] = 0
                return False
            
            if dfs(0):
                # Reconstruct in original order
                result = [0] * n
                for i in range(n):
                    result[orig_indices[i]] = assignment[i]
                return [int(x) for x in result]
    
    # Fallback: return a valid (though not optimal) solution
    # Use greedy: for each divisor, pick the smallest multiple >= 1
    # This won't be optimal but ensures validity
    result = []
    used = set()
    for d in divisors:
        k = 1
        while True:
            val = k * d
            if val not in used:
                result.append(int(val))
                used.add(val)
                break
            k += 1
    return result
