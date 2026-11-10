import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    deadline = start_time + 85
    
    # Start from minimum possible span
    for W in range(9, 200):
        if time.time() > deadline:
            break
            
        # Precompute valid residues for this W?
        # Condition: (-L) % d_i <= W
        
        for L in range(LCM):
            if time.time() > deadline:
                break
                
            # Quick filter
            valid = True
            # For each divisor, check if there is at least one multiple in [L, L+W]
            # This is equivalent to (-L) % d_i <= W
            for d in divisors:
                if (-L) % d > W:
                    valid = False
                    break
            if not valid:
                continue
                
            # Generate candidates
            # candidates[i] is list of offsets r in [0, W] such that (L+r) % divisors[i] == 0
            candidates = []
            for d in divisors:
                r0 = (-L) % d
                # r = r0, r0+d, ... <= W
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                candidates.append(cands)
                
            # Backtracking
            # Sort indices by length of candidates (MRV)
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = [False] * (W + 1)
            choices = [-1] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for r in candidates[i]:
                    if not used[r]:
                        used[r] = True
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        used[r] = False
                        choices[i] = -1
                return False
                
            if backtrack(0):
                # Construct result
                result = []
                for i in range(n):
                    result.append(L + choices[i])
                # Verify
                # (Optional, but good for safety)
                return [int(x) for x in result]
                
    # Fallback
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
