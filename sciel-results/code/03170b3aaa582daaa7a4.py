import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit
    
    N = 10000000
    
    # Sieve for sigma(n)
    # sigma[1] = 1
    # For prime p, sigma(p) = 1+p
    # Linear sieve
    sigma = [0] * (N + 1)
    lp = [0] * (N + 1)      # largest prime factor
    p_pow = [0] * (N + 1)   # highest power of lp[i] dividing i
    primes = []
    
    sigma[1] = 1
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            p_pow[i] = i
        
        for j in primes:
            if i * j > N:
                break
            lp[i * j] = j
            if j == lp[i]:
                p_pow[i * j] = p_pow[i] * j
                sigma[i * j] = sigma[i] + p_pow[i * j]
                break
            else:
                p_pow[i * j] = j
                sigma[i * j] = sigma[i] * (1 + j)
        
        if time.time() > deadline:
            break
            
    # If we broke early due to time, we might not have full sigma array.
    # But the loop usually finishes well before 50s.
    # Let's assume it finishes. If not, we handle partial data? 
    # Better to just let it run if it's fast enough.
    # 10^7 iterations in linear sieve in Python might take ~5-10s.
    
    # Group numbers by sigma value
    # We only care about groups with size >= 2
    groups = {}
    for n in range(1, N + 1):
        s = sigma[n]
        if s == 0: continue # Unvisited, shouldn't happen if loop finished
        
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
            
    if time.time() > deadline:
        # Return a trivial valid pair if found, or fallback
        # We need a valid pair. 
        # If we haven't found any, we might have issues.
        # But sigma(1)=1, sigma(2)=3, sigma(3)=4...
        # There are many amicable-like or sigma-equal pairs.
        # e.g. sigma(1)=1 (only 1), sigma(2)=3 (only 2? no, sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12...
        # Wait, sigma(1)=1. Is there another n with sigma(n)=1? No.
        # sigma(2)=3. Another? No.
        # sigma(3)=4. Another? No.
        # sigma(4)=7. Another? No.
        # sigma(5)=6. Another? sigma(5)=6. Is there another? 
        # sigma(6)=12.
        # Let's check small ones.
        # 1: 1
        # 2: 3
        # 3: 4
        # 4: 7
        # 5: 6
        # 6: 12
        # 7: 8
        # 8: 15
        # 9: 13
        # 10: 18
        # 11: 12 -> sigma(6)=12, sigma(11)=12. Pair (6, 11).
        # So (6, 11) is a valid pair.
        
        # If we run out of time, return [6, 11]
        return [6, 11]

    best_a, best_b = 6, 11
    best_err = abs(2*6 - 3*11) # 12 - 33 = 21
    
    # Iterate through groups
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        nums.sort()
        k = len(nums)
        
        # For each b in nums, find a closest to 1.5 * b
        for i in range(k):
            b = nums[i]
            target = 1.5 * b
            
            # Find insertion point in nums
            pos = bisect.bisect_left(nums, target)
            
            # Check candidates around pos
            # pos can be 0 to k
            # Candidates: pos-1, pos, pos+1 (if valid and distinct from b)
            
            # We want to minimize |2a - 3b|
            # Equivalent to |a - 1.5b|
            
            for p in [pos - 1, pos, pos + 1]:
                if 0 <= p < k:
                    a = nums[p]
                    if a == b:
                        continue
                    err = abs(2 * a - 3 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        if best_err == 0:
                            return [best_a, best_b]
                            
        if time.time() > deadline:
            break

    return [best_a, best_b]
