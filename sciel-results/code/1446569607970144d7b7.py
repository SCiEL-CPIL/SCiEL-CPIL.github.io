import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # We need 10 distinct integers, so the range must be at least 9.
    # Search for minimal window size W
    for W in range(9, 300):
        if time.time() - start_time > timeout:
            break
            
        # Check all possible starting positions L modulo LCM
        for L in range(1, LCM + 1):
            if time.time() - start_time > timeout:
                break
                
            start = L
            end = L + W
            
            # Find available multiples for each divisor in [start, end]
            available = []
            valid = True
            # Sort indices by constraint tightness later, but first collect
            # Actually, collecting in original order is fine, we sort indices later
            
            for d in divisors:
                k = (start + d - 1) // d
                first = k * d
                if first > end:
                    valid = False
                    break
                multiples = []
                x = first
                while x <= end:
                    multiples.append(x)
                    x += d
                available.append(multiples)
                
            if not valid:
                continue
            
            # Check if we can assign distinct values
            # Sort by list length (most constrained first)
            indices = list(range(n))
            indices.sort(key=lambda i: len(available[i]))
            
            used = set()
            choices = [0] * n
            
            # Recursive backtracking
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
                result = [choices[i] for i in range(n)]
                # Verify just in case
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            continue
                    return result
                # If verification fails (should not happen), continue searching?
                # The logic above ensures validity.
                
    # Fallback if no solution found in time or range
    # This should not happen for W up to 300
    # Return a valid but possibly suboptimal solution
    # Using the LCM multiple strategy:
    # Pick multiples of LCM? No, they are not distinct enough or aligned.
    # Just return the divisors themselves? No, they might not be distinct or within range?
    # Divisors are distinct. Max is 39, Min is 6. Range 33.
    # But are they valid? Yes.
    # However, we want to minimize range.
    # If the loop failed, we return a known valid solution.
    # A simple valid solution:
    # For each d, pick the smallest multiple >= 1.
    # This might not be distinct.
    # Let's just return the divisors as a fallback, they are distinct and valid.
    return divisors[:]
