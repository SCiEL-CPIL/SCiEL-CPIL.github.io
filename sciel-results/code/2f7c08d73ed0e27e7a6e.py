import math
import time
import sys
from collections import defaultdict
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 10000000
    
    # 1. Sieve for sigma(n)
    # sigma[n] = sum of divisors of n
    # Using a modified sieve of Eratosthenes
    # Initialize sigma with 1 for all? No, sigma(n) = sum_{d|n} d.
    # We can do: for i in 1..N: for j in i, 2i, 3i...: sigma[j] += i
    # This is O(N log N) operations. For N=10^7, N log N is ~10^7 * 16 = 1.6 * 10^8.
    # In Python, this might be too slow (1.6e8 operations).
    # Let's check speed. 10^8 simple ops in Python takes ~5-10 seconds.
    # We have 85s budget. It should be fine.
    
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    if time.time() - t0 > DEADLINE:
        return [11, 6] # Fallback

    # 2. Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
        
    if time.time() - t0 > DEADLINE:
        return [11, 6] # Fallback

    # 3. Search for best pair
    best_val = float('inf')
    best_pair = [11, 6] # Fallback: sigma(11)=12, sigma(6)=12. 11/6 = 1.833. |1.833-1.5| = 0.333.
    
    # We want to minimize |a/b - 1.5|
    # Equivalent to minimizing |2a - 3b| / (2b)
    
    # Iterate through groups
    for s, vals in groups.items():
        if len(vals) < 2:
            continue
            
        # Sort vals to enable binary search
        vals.sort()
        
        # For each b in vals, find a in vals closest to 1.5 * b
        for b in vals:
            target_a = 1.5 * b
            
            # Binary search for target_a in vals
            idx = bisect.bisect_left(vals, target_a)
            
            # Check idx and idx-1
            # Also check idx+1 just in case? No, bisect_left gives first >= target.
            # So idx is first >= target, idx-1 is last < target.
            
            candidates = []
            if idx < len(vals):
                candidates.append(vals[idx])
            if idx > 0:
                candidates.append(vals[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                if a < 1 or a > N:
                    continue
                if b < 1 or b > N:
                    continue
                    
                # Calculate error
                # |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # This is invalid per task description "raw value of exactly 0 cannot be scored"
                    # But if we find 0, it's the best mathematical answer, but invalid.
                    # We should skip it or look for next best?
                    # The task says "A raw value of exactly 0 cannot be scored and counts as invalid."
                    # So we must avoid a/b = 1.5 exactly.
                    continue
                    
                val = num / (2.0 * b)
                
                if val < best_val:
                    best_val = val
                    best_pair = [a, b]
                    
        # Early termination?
        # If best_val is very small, we might stop.
        # But we don't know if there's a better one.
        # Given the structure, small sigma values have small n.
        # Large sigma values have large n.
        # The error |2a - 3b| / 2b is roughly 1/b for fixed integer difference.
        # So larger b generally allows smaller error.
        # So we should check larger sigma groups (larger n) later?
        # The iteration order of groups.items() is arbitrary.
        # It's better to iterate b in increasing order of magnitude?
        # Actually, if we find a pair with error E, any pair with b < 1/(2E) * min_diff (where min_diff >= 1) cannot be better.
        # min_diff is at least 1.
        # So if 1/(2b) > E, i.e., b < 1/(2E), we can't improve.
        # Since we want to maximize b to minimize 1/b, we should check large b first?
        # But groups are by sigma. Large sigma usually means large n.
        
        # Let's add a time check
        if time.time() - t0 > DEADLINE - 5:
            break

    return [int(best_pair[0]), int(best_pair[1])]
