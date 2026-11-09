import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 30
    
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into 5 groups: 4 of size 7, 1 of size 2
    # G1: 0-6 (7 items)
    # G2: 7-13 (7 items)
    # G3: 14-20 (7 items)
    # G4: 21-27 (7 items)
    # G5: 28-29 (2 items)
    
    def get_group_sums(start, end):
        """
        Generate all sums and coefficient tuples for indices [start, end)
        """
        sums = []
        group_weights = weights[start:end]
        size = end - start
        # Use itertools.product for -1, 0, 1
        for coeffs in itertools.product([-1, 0, 1], repeat=size):
            s = 0
            for c, w in zip(coeffs, group_weights):
                s += c * w
            sums.append((s, coeffs))
        return sums
    
    G1 = get_group_sums(0, 7)
    G2 = get_group_sums(7, 14)
    G3 = get_group_sums(14, 21)
    G4 = get_group_sums(21, 28)
    G5 = get_group_sums(28, 30)
    
    # Combine G1 and G2 into S12
    S12 = []
    for s1, c1 in G1:
        for s2, c2 in G2:
            S12.append((s1 + s2, c1, c2))
            
    # Combine G3 and G4 into S34
    S34 = []
    for s3, c3 in G3:
        for s4, c4 in G4:
            S34.append((s3 + s4, c3, c4))
            
    # Sort S12 and S34 by sum
    S12.sort(key=lambda x: x[0])
    S34.sort(key=lambda x: x[0])
    
    len_S12 = len(S12)
    len_S34 = len(S34)
    
    best_abs = None
    best_coeffs = None
    best_val = None
    
    # Iterate over G5
    for s5, c5 in G5:
        target = -s5
        
        # Two pointer sweep
        i = 0
        j = len_S34 - 1
        
        # We need to find min |S12[i][0] + S34[j][0] - target| != 0
        # The sweep visits O(N+M) points.
        
        while i < len_S12 and j >= 0:
            current_sum = S12[i][0] + S34[j][0]
            diff = current_sum - target
            
            if diff == 0:
                # Sum is zero, which is forbidden.
                # Check neighbors (i+1, j) and (i, j-1)
                candidates = []
                
                if i + 1 < len_S12:
                    s_next = S12[i+1][0] + S34[j][0]
                    d_next = s_next - target
                    if d_next != 0:
                        candidates.append((abs(d_next), d_next, i+1, j))
                        
                if j - 1 >= 0:
                    s_prev = S12[i][0] + S34[j-1][0]
                    d_prev = s_prev - target
                    if d_prev != 0:
                        candidates.append((abs(d_prev), d_prev, i, j-1))
                        
                if candidates:
                    min_abs, min_val, ni, nj = min(candidates, key=lambda x: x[0])
                    if best_abs is None or min_abs < best_abs:
                        best_abs = min_abs
                        best_val = min_val
                        best_coeffs = S12[ni][1] + S12[ni][2] + S34[nj][1] + S34[nj][2] + c5
                
                # Move pointers to continue search?
                # If we found 0, the next best is likely nearby.
                # But we should continue the sweep to ensure we don't miss a better one elsewhere?
                # Actually, the global min for this target is likely at the crossing.
                # If 0 is at the crossing, the second best is adjacent.
                # We can break or continue.
                # To be safe, we can continue, but we already captured the local min.
                # Let's move i and j to continue the sweep.
                i += 1
                j -= 1
            elif diff < 0:
                # current_sum < target
                # Check current
                if diff != 0: # always true here
                    if best_abs is None or abs(diff) < best_abs:
                        best_abs = abs(diff)
                        best_val = diff
                        best_coeffs = S12[i][1] + S12[i][2] + S34[j][1] + S34[j][2] + c5
                # Increase sum
                i += 1
            else: # diff > 0
                # current_sum > target
                # Check current
                if best_abs is None or abs(diff) < best_abs:
                    best_abs = abs(diff)
                    best_val = diff
                    best_coeffs = S12[i][1] + S12[i][2] + S34[j][1] + S34[j][2] + c5
                # Decrease sum
                j -= 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    if best_coeffs is None:
        # Fallback
        fallback = [0] * 30
        fallback[1] = 1
        fallback[2] = -1
        return fallback
        
    return best_coeffs
