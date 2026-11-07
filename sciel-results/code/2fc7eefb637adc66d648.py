import numpy as np
import bisect
import time
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 85.0 # Safety margin for 90s kill
    
    N = 10_000_000
    
    # 1. Sieve for sigma
    # sigma[n] = sum of divisors of n
    # Using NumPy for speed
    # int32 is sufficient for sigma(n) up to 10^7 (max ~4.5e7)
    sigma = np.zeros(N + 1, dtype=np.int32)
    
    # The loop for d in 1..N
    # This is the bottleneck.
    # We can check time every 1000 iterations of d?
    # Or just let it run. If it exceeds, we return current best?
    # But we need the full sigma array to group.
    # If we can't finish sieve, we can't do the grouping.
    # Fallback: Return a known good pair if sieve is too slow?
    # Let's assume NumPy sieve finishes in < 10s.
    
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 100000 == 0:
            if time.time() > deadline:
                # Fallback
                # Use a known good pair from seed or small search
                # Seed score ~4e6 implies error ~2.5e-7
                # We can try to find a good pair in smaller range?
                # Or just return 11, 6?
                # 11/6 error is 1/3. Score 3.
                # That's much worse than 4e6.
                # If we time out here, we return invalid or low score.
                # Let's try to be robust.
                # However, if we are here, we have partial sigma.
                # We cannot group partial sigma correctly.
                # So we must return a hardcoded good pair.
                # Let's find a pair with sigma(a)=sigma(b) and a/b close to 1.5
                # 11, 6: sigma(11)=12, sigma(6)=12. 11/6 = 1.833. |1.833-1.5|=0.333.
                # 14, 9: sigma(14)=24, sigma(9)=13. No.
                # 15, 10: sigma(15)=24, sigma(10)=18. No.
                # 18, 12: sigma(18)=39, sigma(12)=28. No.
                # 20, 15: sigma(20)=42, sigma(15)=24. No.
                # 21, 14: sigma(21)=32, sigma(14)=24. No.
                # 22, 15: sigma(22)=36, sigma(15)=24. No.
                # 24, 16: sigma(24)=60, sigma(16)=31. No.
                # 27, 18: sigma(27)=40, sigma(18)=39. No.
                # 28, 19: sigma(28)=56, sigma(19)=20. No.
                # 30, 20: sigma(30)=72, sigma(20)=42. No.
                # 33, 22: sigma(33)=48, sigma(22)=36. No.
                # 34, 23: sigma(34)=54, sigma(23)=24. No.
                # 35, 24: sigma(35)=48, sigma(24)=60. No.
                # 36, 24: sigma(36)=91, sigma(24)=60. No.
                # 38, 25: sigma(38)=60, sigma(25)=31. No.
                # 39, 26: sigma(39)=56, sigma(26)=42. No.
                # 40, 27: sigma(40)=90, sigma(27)=40. No.
                # 42, 28: sigma(42)=96, sigma(28)=56. No.
                # 44, 29: sigma(44)=84, sigma(29)=30. No.
                # 45, 30: sigma(45)=78, sigma(30)=72. No.
                # 46, 31: sigma(46)=72, sigma(31)=32. No.
                # 48, 32: sigma(48)=124, sigma(32)=63. No.
                # 50, 33: sigma(50)=93, sigma(33)=48. No.
                # 51, 34: sigma(51)=72, sigma(34)=54. No.
                # 52, 35: sigma(52)=98, sigma(35)=48. No.
                # 54, 36: sigma(54)=120, sigma(36)=91. No.
                # 55, 37: sigma(55)=72, sigma(37)=38. No.
                # 56, 38: sigma(56)=120, sigma(38)=60. No.
                # 57, 39: sigma(57)=80, sigma(39)=56. No.
                # 58, 40: sigma(58)=90, sigma(40)=90. YES.
                # a=58, b=40. sigma(58)=90, sigma(40)=90.
                # 58/40 = 2.9 / 2 = 1.45.
                # |1.45 - 1.5| = 0.05.
                # Score = 1/0.05 = 20.
                # This is much better than 11,6 (score 3).
                return [58, 40]
                
    # 2. Group
    groups = defaultdict(list)
    # Convert numpy array to list for faster indexing in the loop?
    # sigma is int32 array. Accessing sigma[i] is fast.
    for i in range(1, N + 1):
        groups[int(sigma[i])].append(i)
        
    if time.time() > deadline:
        return [58, 40] # Fallback
        
    # 3. Find best pair
    best_a, best_b = 58, 40
    best_num = abs(2 * 58 - 3 * 40) # |116 - 120| = 4
    best_den = 2 * 40 # 80
    # error = 4/80 = 0.05. Score 20.
    
    bisect_left = bisect.bisect_left
    
    # Iterate through groups
    for lst in groups.values():
        if time.time() > deadline:
            break
        l_len = len(lst)
        if l_len < 2:
            continue
            
        # lst is sorted
        for b in lst:
            target = 1.5 * b
            # bisect requires list or array-like. lst is list of ints.
            idx = bisect_left(lst, target)
            
            # Check neighbors
            candidates = []
            if idx < l_len:
                candidates.append(lst[idx])
            if idx > 0:
                candidates.append(lst[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                    
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue # Invalid per task rules
                    
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                # num * best_den < best_num * den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
