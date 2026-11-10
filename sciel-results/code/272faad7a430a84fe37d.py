import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    timeout = 85
    
    # Precompute nothing complex, just loop
    
    for W in range(1, LCM):
        if time.time() - start_time > timeout:
            break
            
        for L in range(1, LCM + 1):
            # Check if all divisors have at least one multiple in [L, L+W]
            valid_L = True
            available = []
            for d in divisors:
                # Find first multiple >= L
                k = (L + d - 1) // d
                first = k * d
                if first > L + W:
                    valid_L = False
                    break
                # Collect multiples
                mults = []
                x = first
                while x <= L + W:
                    mults.append(x)
                    x += d
                available.append(mults)
            
            if not valid_L:
                continue
            
            # Solve matching
            # Sort indices by number of available options (MRV)
            indices = sorted(range(n), key=lambda i: len(available[i]))
            used = set()
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
                for val in available[i]:
                    if val not in used:
                        used.add(val)
                        choices[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                result = [int(choices[i]) for i in range(n)]
                # Verify
                if len(set(result)) == n and all(0 < x <= 1000000 for x in result) and all(result[i] % divisors[i] == 0 for i in range(n)):
                    return result
                    
    # Fallback
    return [int(x) for x in divisors]
