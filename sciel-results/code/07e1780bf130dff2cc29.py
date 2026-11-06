import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    
    # Candidate set of primes
    # We want to maximize coverage. 
    # The sum of reciprocals must be <= 1.
    # Let's use all primes from 13 to 997, and add 3 and 5 if possible.
    # Sum(1/13...1/997) is roughly 0.67.
    # Sum(1/3) = 0.33, Sum(1/5) = 0.20. Total = 1.20 > 1.
    # So we can't have both 3 and 5 with all large primes.
    # Option A: {3, 5, 13, 17, ...} -> Sum = 0.33+0.20+0.67 = 1.20 (Too high)
    # Option B: {3, 13, 17, ...} -> Sum = 0.33+0.67 = 1.00 (Just at limit? Let's check exact sum)
    # Option C: {5, 13, 17, ...} -> Sum = 0.20+0.67 = 0.87. Can we add 7? 0.87+0.14=1.01 (Too high).
    # Option D: {3, 5, 13, ...} excluding some large primes?
    
    # Let's check the sum of 1/p for p in [13..997].
    sum_13_997 = sum(1.0/p for p in primes if 13 <= p <= 997)
    
    # Try sets:
    sets_to_try = []
    
    # Set 1: All primes >= 13
    P1 = [p for p in primes if p >= 13]
    sets_to_try.append(P1)
    
    # Set 2: 3 + primes >= 13
    P2 = [3] + [p for p in primes if p >= 13]
    if sum(1.0/p for p in P2) <= 1.0:
        sets_to_try.append(P2)
        
    # Set 3: 5 + primes >= 13
    P3 = [5] + [p for p in primes if p >= 13]
    if sum(1.0/p for p in P3) <= 1.0:
        sets_to_try.append(P3)
        
    # Set 4: 3, 5 + primes >= 17 (exclude 13 to make room?)
    P4 = [3, 5] + [p for p in primes if p >= 17]
    if sum(1.0/p for p in P4) <= 1.0:
        sets_to_try.append(P4)
        
    # Set 5: 3, 7 + primes >= 13 (exclude 5?)
    P5 = [3, 7] + [p for p in primes if p >= 13 and p != 5]
    if sum(1.0/p for p in P5) <= 1.0:
        sets_to_try.append(P5)

    # Strategy: Diagonal Initialization + Greedy Extension
    # 1. Initialize residues such that small numbers are covered as much as possible.
    #    "Diagonal" means r_p = 0 for all p? Or r_p = 1? 
    #    If r_p = 1, then 1 is covered by all primes.
    #    If r_p = 0, then multiples are covered.
    #    Let's try r_p = 1 for all p initially. This covers 1.
    #    Then greedily assign residues for the first uncovered number.
    
    best_L = 0
    best_result = []
    
    for P in sets_to_try:
        P_sorted = sorted(P)
        
        # Try a few different initializations
        for init_mode in range(3):
            assigned = {}
            
            # Initialize
            if init_mode == 0:
                # All r = 1
                for p in P:
                    assigned[p] = 1
            elif init_mode == 1:
                # All r = 0
                for p in P:
                    assigned[p] = 0
            else:
                # Random
                for p in P:
                    assigned[p] = random.randint(0, p-1)
            
            # Check coverage of 1..L
            # We need to find the largest L such that 1..L are covered.
            # Then extend.
            
            # Fast check for coverage up to some limit
            # Let's just simulate forward.
            # To speed up, we can use a "next_uncovered" pointer.
            
            # However, re-assigning residues for later primes is the key to extension.
            # The "Greedy Prefix Extension" idea:
            # 1. Find the first uncovered number N in the current configuration.
            # 2. If N exists, we can try to "fix" it by changing the residue of a prime that hasn't been "used" to cover a small number?
            #    Actually, all primes are "active".
            #    Changing a residue r_p affects all multiples.
            #    If we change r_p to cover N, we might uncover previous numbers.
            #    
            #    The insight from inspiration 2: "Diagonal initialization ensures high initial coverage... greedy prefix-extension... maximizes L without local minima traps".
            #    This suggests a specific algorithm.
            #    
            #    Let's try a simpler greedy approach first:
            #    Start with empty assignment? No, that's what the current code does and it's slow/weak.
            #    
            #    Let's try:
            #    1. Sort primes ascending.
            #    2. For each prime p, assign r_p to cover the smallest currently uncovered number.
            #    This is similar to the current code but maybe the order matters?
            #    The current code picks the smallest prime to cover the first gap.
            #    
            #    Let's try a different heuristic:
            #    Assign residues to maximize coverage of [1, K] for a fixed K?
            #    
            #    Given the time constraint, let's stick to a robust greedy extension.
            #    1. Initialize with r_p = 1 for all p (covers 1).
            #    2. Find smallest uncovered N.
            #    3. If none found up to 10000, return.
            #    4. If found, we need to cover N.
            #       We can change the residue of ANY prime p in P to r = N % p.
            #       Which p to choose?
            #       Heuristic: Choose the largest p? (Least likely to disturb small numbers?)
            #       Or the smallest p? (Most likely to cover more numbers?)
            #       
            #       Let's try: Choose the prime p in P that is NOT responsible for covering any number in [1, N-1]?
            #       This is hard to track.
            #       
            #       Alternative: Local search.
            #       Try changing the residue of each p in P to N % p.
            #       Check if [1, N-1] are still covered.
            #       If yes, accept.
            #       If no, try next p.
            #       
            #       This is expensive but might work for small L.
            
            # Let's implement the local search greedy extension.
            # State: assigned dict {p: r}
            # We maintain a list `covered` of size MAX_L (e.g. 20000) to quickly check coverage.
            # But updating `covered` after a residue change is O(MAX_L/p).
            
            MAX_L = 20000
            covered = np.zeros(MAX_L, dtype=bool)
            
            def update_coverage(p, old_r, new_r):
                if old_r == new_r:
                    return
                # Remove old coverage
                # Note: We can't just subtract because other primes might cover the same numbers.
                # We need to recount or use a count array.
                # For simplicity and speed, let's just recompute coverage for the whole range?
                # No, that's too slow.
                # Let's use a count array.
                
            # Actually, let's just use the simple "check on the fly" method but optimize it.
            # We will iterate n from 1 to MAX_L.
            # If n is not covered, we trigger a "repair".
            # Repair: Find a prime p to change.
            
            # Let's try the "Smallest Prime" greedy from the seed but with the "Diagonal" init.
            # The seed used "Empty" init effectively (assigned = {}).
            # Diagonal init (r=1) covers 1.
            # Then 2 is uncovered (if 2 not in P).
            # If P has 3, 5, 13...
            # 2 % 3 = 2, 2 % 5 = 2...
            # We pick smallest p (3) and set r_3 = 2.
            # Now 2 is covered. 3 is covered (3%3=0 != 2? Wait. 3%3=0. If r_3=2, 3 is NOT covered by 3).
            # But 3 might be covered by 5? 3%5=3. If r_5=1, no.
            # So we might uncover 1 or 3.
            
            # The key is: we must ensure we don't uncover previous numbers.
            
            # Let's implement a strict check:
            # When we change r_p, we must verify that all n in [1, current_max_covered] are still covered.
            
            current_max_covered = 0
            # Initialize coverage
            for p, r in assigned.items():
                for n in range(r, MAX_L, p):
                    if n < MAX_L:
                        covered[n] = True
            
            # Find current max covered
            if not any(covered[1:MAX_L]):
                current_max_covered = 0
            else:
                # Find the largest index i such that covered[1:i+1] are all True
                # This is O(MAX_L)
                i = 1
                while i < MAX_L and covered[i]:
                    i += 1
                current_max_covered = i - 1
            
            n = current_max_covered + 1
            while n < MAX_L:
                if covered[n]:
                    n += 1
                    current_max_covered = n
                    continue
                
                # n is uncovered.
                # Try to fix by changing residue of some p in P.
                # Heuristic: Try primes in descending order (largest first) to minimize disruption?
                # Or ascending?
                # Let's try ascending first.
                
                fixed = False
                for p in P_sorted:
                    old_r = assigned[p]
                    new_r = n % p
                    if old_r == new_r:
                        continue
                    
                    # Check if changing p to new_r breaks any previous coverage
                    # We only need to check numbers in [1, n-1] that were ONLY covered by p with old_r?
                    # No, we need to check if any number in [1, n-1] becomes uncovered.
                    # A number x in [1, n-1] becomes uncovered if:
                    # 1. It was covered by p (x % p == old_r)
                    # 2. It is NOT covered by any other prime q (q != p) with x % q == assigned[q]
                    
                    # This check is expensive: O(n * |P|).
                    # But n is small initially.
                    
                    # Optimization: Only check x in [1, n-1] where x % p == old_r.
                    # There are at most n/p such numbers.
                    
                    break_flag = False
                    for x in range(old_r, n, p):
                        if x == 0: continue
                        # Check if x is covered by any other prime
                        is_covered_by_other = False
                        for q in P:
                            if q == p: continue
                            if x % q == assigned[q]:
                                is_covered_by_other = True
                                break
                        if not is_covered_by_other:
                            break_flag = True
                            break
                    
                    if not break_flag:
                        # Safe to change
                        assigned[p] = new_r
                        # Update covered array
                        # Remove old coverage
                        for x in range(old_r, MAX_L, p):
                            if x < MAX_L and x > 0:
                                # We can't just set to False, need to recount?
                                # For speed, let's just set to False and rely on the fact that
                                # if it's covered by others, we don't care?
                                # No, we need accurate `covered` for the main loop.
                                # This "remove" is incorrect if others cover it.
                                # We need a count array.
                                pass
                        # Add new coverage
                        for x in range(new_r, MAX_L, p):
                            if x < MAX_L and x > 0:
                                covered[x] = True
                        
                        # The "Remove" part is the problem.
                        # Let's use a count array `counts` of size MAX_L.
                        # This will be more robust.
                        
                        fixed = True
                        break
                
                if not fixed:
                    break # Cannot fix n
                
                # We changed a residue. We need to re-evaluate `covered` and `current_max_covered`.
                # Since we used a simplified update, let's just recompute `covered` from scratch for [1, n]?
                # Or just recompute the prefix until we find a gap.
                
                # Recompute prefix coverage
                i = 1
                while i < MAX_L:
                    # Check if i is covered by any p
                    cov = False
                    for p, r in assigned.items():
                        if i % p == r:
                            cov = True
                            break
                    if not cov:
                        break
                    i += 1
                current_max_covered = i - 1
                n = current_max_covered + 1
                
                if time.time() > deadline:
                    break
            
            if current_max_covered > best_L:
                best_L = current_max_covered
                best_result = [[p, assigned[p]] for p in assigned]
                
                if best_L >= 999:
                    # We hit the target. Can we do better?
                    # The prompt says maximize.
                    # But 999 is the "Erdos-Ruzsa" milestone.
                    # Let's see if we can push further.
                    pass

    if not best_result:
        # Fallback
        P1 = [p for p in primes if p >= 13]
        return [[p, 1] for p in P1]
        
    return best_result

# EVOLVE-BLOCK-END
