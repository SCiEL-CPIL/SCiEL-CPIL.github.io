import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    n = 256
    
    def is_valid_set(S):
        # S is a set of integers
        # Check: for all a < b <= c in S, a does not divide b+c
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            # b must be > a
            # c must be >= b
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # 1. Find the best interval [L, 256]
    best_L = 1
    best_size = 0
    best_set = set()
    
    # We expect L to be around 170. Check from 150 to 256 to be safe, or just 1 to 256.
    # 256 intervals is small.
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S = set(range(L, n + 1))
        # Quick check for interval: 
        # Only a=L is the "dangerous" small one? 
        # No, a can be any element in [L, 256].
        # But if a > L, it's "safer" usually.
        # However, full check is O(N^3) ~ 16M ops. 
        # 256 * 16M is too slow for full check.
        
        # Optimized Interval Check:
        # For interval [L, N], the condition fails if there exist a, b, c in [L, N] 
        # with a < b <= c and a | (b+c).
        # The most likely failure is with the smallest a = L.
        # If L does not divide any b+c, does a > L divide any?
        # If a > L, then a is larger, so b+c must be a multiple of a.
        # Since b,c >= L+1 > a is not necessarily true (b,c can be close to a).
        # Actually, if we verify a=L, we don't verify a=L+1.
        
        # Let's do a partial check:
        # Check a = L.
        valid = True
        min_sum = 2 * (L + 1)
        max_sum = 2 * n
        # Check if any multiple of L in [min_sum, max_sum] is representable.
        # A sum S is representable by b,c in [L+1, N] with b<=c iff:
        # 2(L+1) <= S <= 2N  AND  S can be written as b+c.
        # Actually, any S in [2(L+1), 2N] is representable EXCEPT potentially the ends?
        # Min sum 2(L+1) -> (L+1, L+1)
        # Max sum 2N -> (N, N)
        # Any integer S in between is representable.
        # So if there is a multiple k*L in [2(L+1), 2N], the interval is INVALID.
        
        # Find if there is k such that 2(L+1) <= k*L <= 2N
        k_min = (2 * (L + 1) + L - 1) // L  # ceiling(2(L+1)/L)
        k_max = (2 * n) // L
        if k_min <= k_max:
            valid = False
        else:
            # If L is safe, are larger a's safe?
            # If a = L + 1.
            # Min sum for a=L+1 is 2(L+2).
            # If 2(L+2) > 2N, no sum exists.
            # If there is a multiple of L+1 in [2(L+2), 2N], it is a violation.
            # We should probably just run the full check for the candidate L if the interval is "promising".
            # Given N=256, full check is ~0.1s.
            # We only need to check L near the optimal (170).
            # Let's check all L, but skip full check if the simple "L check" fails?
            # No, if L check fails, interval is invalid.
            # If L check passes, we must verify larger a's.
            
            # For speed, let's assume if L passes the "no multiple" check, 
            # and L is large enough (L > N/2), larger a's are likely safe.
            # If L <= N/2, it's risky.
            
            # Let's just do the full check for L in range(150, 257) to be safe.
            if L >= 150:
                S_test = set(range(L, n + 1))
                if is_valid_set(S_test):
                    size = len(S_test)
                    if size > best_size:
                        best_size = size
                        best_set = S_test
                        best_L = L
            else:
                # For L < 150, the "L check" is likely to fail or be borderline.
                # We can skip full check if k_min <= k_max (already did).
                # If k_min > k_max, it's a candidate.
                # But L<150 means L is small, many multiples.
                # Actually, if L < 86, 3L < 258 < 512. 
                # The condition "no multiple in [2L+2, 512]" is very hard to satisfy for small L.
                # Example L=100. 300 in [202, 512]. 300 = 101+199? 101>100. Yes. Invalid.
                # So most L < 170 are invalid intervals.
                pass

    # Current best is likely [171, 256] size 86.
    # Let's verify best_set is not empty.
    if not best_set:
        # Fallback to seed
        return list(range(192, 257))

    # 2. Greedy Expansion
    # Try to add elements from best_L - 1 down to 1
    current_set = set(best_set)
    
    # Sort candidates
    candidates = list(range(best_L - 1, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Tentatively add x
        test_set = set(current_set)
        test_set.add(x)
        
        # Check validity.
        # Optimization: We only need to check constraints where a = x.
        # Because x is the new minimum (or close to it, if we skipped some).
        # Actually, if we skipped some, x might not be the minimum.
        # But x is smaller than all existing elements in current_set?
        # current_set contains elements from [best_L, 256] plus some added.
        # The added elements were smaller than best_L.
        # So x is smaller than all elements in current_set.
        # Thus, for any a in current_set, a > x.
        # The condition is a < b <= c.
        # If a is from current_set, b and c must be > a > x.
        # So x is never b or c when a is in current_set.
        # x is only a.
        # So we only need to check:
        # For all b, c in current_set with b < c (or b<=c), x does not divide b+c.
        
        valid = True
        c_list = list(current_set)
        for i, b in enumerate(c_list):
            for k in range(i, len(c_list)): # b <= c
                c = c_list[k]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            current_set = test_set

    # 3. Local Search / Improvement Phase
    # Try to swap elements or add more via swaps.
    # We will iterate a few times trying to add any element not in current_set.
    # If we can't add directly, we look for 1-for-1 swaps that might enable future additions.
    # To keep it simple and fast, we will just try to add any remaining elements
    # in a random order or different order, and see if we can beat the current size.
    
    best_found_set = set(current_set)
    best_found_size = len(current_set)
    
    # Try a few restarts with different greedy orders or random perturbations
    import random
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Randomize the candidate order slightly
        cand = list(range(best_L - 1, 0, -1))
        random.shuffle(cand)
        
        test_set = set(best_set)
        
        for x in cand:
            if time.time() > deadline:
                break
            test_set.add(x)
            
            valid = True
            c_list = list(test_set)
            # Optimization: x is likely small. Check x as 'a'.
            # Also check if x as 'b' or 'c' causes issues with existing 'a's.
            # Since we are adding small numbers, x is likely the smallest.
            # But if we shuffled, we might add a large one first? No, cand is 1..L-1.
            # So x is always smaller than elements in best_set.
            # So x is always the new minimum or close to it.
            # Wait, if we add 5, then 10, 10 is not smaller than 5.
            # So we must do a full check or a smart check.
            # Full check is O(N^3). N is max 256. 256^3 is 16M.
            # 10 restarts * 256 candidates * 16M ops = 40B ops. Too slow.
            
            # We need a faster validity check for incremental addition.
            # If we add x to S, we only need to check triples involving x.
            # Triples (a,b,c) where x is a, b, or c.
            # 1. x=a: x | b+c for b,c in S, b,c > x.
            # 2. x=b: a | x+c for a,c in S, a < x <= c.
            # 3. x=c: a | b+x for a,b in S, a < b <= x.
            
            valid = True
            S_list = list(test_set - {x})
            
            # Case 1: x is 'a'
            for i, b in enumerate(S_list):
                if b <= x: continue
                for k in range(i, len(S_list)):
                    c = S_list[k]
                    if c < b: continue
                    if (b + c) % x == 0:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # Case 2: x is 'b'
                for a in S_list:
                    if a >= x: continue
                    for c in S_list:
                        if c < x: continue
                        if (x + c) % a == 0:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                # Case 3: x is 'c'
                for a in S_list:
                    for b in S_list:
                        if a >= b: continue
                        if b > x: continue
                        if (b + x) % a == 0:
                            valid = False
                            break
                    if not valid: break
            
            if not valid:
                test_set.remove(x)

        if len(test_set) > best_found_size:
            best_found_set = test_set
            best_found_size = len(test_set)
            
    return list(best_found_set)

# EVOLVE-BLOCK-END
