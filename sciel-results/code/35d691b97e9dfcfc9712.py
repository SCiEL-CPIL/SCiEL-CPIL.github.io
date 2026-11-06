import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Two-pointer approach:
    # We want the longest subarray [s, e] such that no pair (x, y) with x*y % MOD == 1 exists within [s, e].
    # A pair exists if there is an x in [s, e] such that inv[x] is also in [s, e].
    # Self-inverse elements (1 and 1008) cannot be in any valid interval.
    
    best_len = 0
    best_start = 1
    
    s = 1
    for e in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Handle self-inverse: 1 and 1008 cannot be in any valid interval
        if e == 1 or e == 1008:
            s = e + 1
            continue
            
        # Handle distinct pair: if inv[e] is in [s, e-1], move s past it
        ie = inv[e]
        if ie >= s and ie < e:
            s = ie + 1
            
        # Now [s, e] should be valid
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    # Fast verification using numpy
    def is_valid_interval(start, length):
        end = start + length - 1
        if start < 1 or end > N or length <= 0:
            return False
        # Check all pairs (x, y) with x <= y in [start, end]
        arr = np.arange(start, end + 1)
        # For each x, check if inv[x] is in the interval
        # Self-inverse check: x*x % MOD == 1
        # Distinct pair check: inv[x] in [start, end] and inv[x] != x
        for x in range(start, end + 1):
            if (x * x) % MOD == 1:
                return False
            ix = inv[x]
            if ix != x and start <= ix <= end:
                return False
        return True
    
    if best_len == 0:
        # Fallback: find any valid single element
        for i in range(2, N):
            if i != 1008 and (i * i) % MOD != 1:
                return [i, 1]
        return [2, 1]
    
    if not is_valid_interval(best_start, best_len):
        # If verification fails, do a brute-force search for the best valid interval
        # This should not happen, but it's a safety net
        best_len = 0
        best_start = 1
        for s2 in range(1, N + 1):
            if time.time() > deadline - 5:
                break
            for e2 in range(s2, N + 1):
                if time.time() > deadline - 5:
                    break
                if e2 == 1 or e2 == 1008 or s2 == 1:
                    if e2 >= s2 and (e2 == 1 or e2 == 1008):
                        continue
                # Quick check: if s2 == 1, skip (1 is self-inverse)
                if s2 == 1 or e2 == 1008:
                    continue
                valid = True
                for x in range(s2, e2 + 1):
                    if (x * x) % MOD == 1:
                        valid = False
                        break
                    ix = inv[x]
                    if ix != x and s2 <= ix <= e2:
                        valid = False
                        break
                if valid:
                    length = e2 - s2 + 1
                    if length > best_len:
                        best_len = length
                        best_start = s2
                # Early termination: if we can't beat best_len, skip
                if e2 - s2 + 1 <= best_len and e2 - s2 + 1 + (N - e2) <= best_len:
                    break
    
    return [best_start, best_len]
