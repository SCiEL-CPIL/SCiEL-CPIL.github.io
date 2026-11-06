import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s kill

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        q = 3
        while q * q <= n:
            if n % q == 0:
                return False
            q += 2
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    prime_set = set(primes)

    max_L = 10000

    # ---------- Static Initialization with Known Good Covering ----------
    # Use a pre-computed set of (p, r) that covers a significant prefix.
    # This avoids the slow greedy construction which caused timeouts/invalid states.
    # We can use a simple heuristic: cover small numbers with small primes.
    # A known covering system for small ranges often uses:
    # 2: 0, 3: 0, 5: 0, 7: 0, 11: 0, 13: 0, 17: 0, 19: 0, 23: 0, 29: 0, 31: 0, 37: 0
    # Let's construct a valid set that sums to <= 1.
    # Sum of reciprocals of primes: 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322 + 0.0270
    # Sum is approx 1.55. This is too high. We need sum <= 1.
    
    # Let's pick a subset that sums to <= 1.
    # 1/2 + 1/3 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31
    # 0.5 + 0.333 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322
    # Sum approx 1.36. Still too high.
    
    # We need a tighter set.
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322
    # Sum approx 1.18. Still too high.
    
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434
    # Sum approx 1.16.
    
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526
    # Sum approx 1.12.
    
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588
    # Sum approx 1.06.
    
    # 1/2 + 1/5 + 1/7 + 1/11 + 1/13
    # 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769
    # Sum approx 1.01.
    
    # 1/2 + 1/5 + 1/7 + 1/11
    # 0.5 + 0.2 + 0.142 + 0.0909
    # Sum approx 0.93. This is safe.
    
    # Let's use these primes: 2, 5, 7, 11.
    # And maybe add a larger prime to cover gaps, if budget allows.
    # Remaining budget: 1 - 0.93 = 0.07.
    # 1/13 = 0.0769 (too big).
    # 1/17 = 0.0588 (fits).
    # 1/19 = 0.0526 (fits).
    # 1/23 = 0.0434 (fits).
    # 1/29 = 0.0344 (fits).
    # 1/31 = 0.0322 (fits).
    # 1/37 = 0.0270 (fits).
    # 1/41 = 0.0243 (fits).
    # 1/43 = 0.0232 (fits).
    # 1/47 = 0.0212 (fits).
    # 1/53 = 0.0188 (fits).
    # 1/59 = 0.0169 (fits).
    # 1/61 = 0.0163 (fits).
    # 1/67 = 0.0149 (fits).
    # 1/71 = 0.0140 (fits).
    # 1/73 = 0.0136 (fits).
    # 1/79 = 0.0126 (fits).
    # 1/83 = 0.0120 (fits).
    # 1/89 = 0.0112 (fits).
    # 1/97 = 0.0103 (fits).
    
    # Let's add primes 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97.
    # Sum check: 0.93 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322 + 0.0270 + 0.0243 + 0.0232 + 0.0212 + 0.0188 + 0.0169 + 0.0163 + 0.0149 + 0.0140 + 0.0136 + 0.0126 + 0.0120 + 0.0112 + 0.0103
    # This sum is clearly > 1. We need to be careful.
    
    # Let's just use the first few that fit.
    # Start with 2, 5, 7, 11. Sum = 0.933.
    # Add 17: 0.933 + 0.0588 = 0.992.
    # Add 19: 0.992 + 0.0526 = 1.044 (Too high).
    
    # So, primes: 2, 5, 7, 11, 17.
    # Residues: We want to cover as many small numbers as possible.
    # 2: 0 (covers evens)
    # 5: 1 (covers 1, 6, 11, 16, 21, 26, 31...)
    # 7: 2 (covers 2, 9, 16, 23, 30, 37...)
    # 11: 3 (covers 3, 14, 25, 36, 47...)
    # 17: 5 (covers 5, 22, 39, 56...)
    
    # Let's verify coverage for 1..L.
    # 1: 5%5=0? No, 1%5=1. Covered by 5.
    # 2: 2%7=2. Covered by 7.
    # 3: 3%11=3. Covered by 11.
    # 4: 4%2=0. Covered by 2.
    # 5: 5%17=5. Covered by 17.
    # 6: 6%5=1. Covered by 5.
    # 7: 7%7=0. Not covered by 7 (r=2). 7%2=1. 7%5=2. 7%11=7. 7%17=7.
    #   7 is not covered!
    #   We need to adjust residues.
    
    # Instead of guessing, let's use a simple heuristic that is guaranteed to work for small L:
    # Use primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97.
    # But sum of reciprocals is too high.
    
    # Let's use a greedy algorithm but with a very small max_L for construction (e.g., 100) to find a good set, then extend.
    # Or better: use a known covering set from literature or a simple one.
    # A simple valid set:
    # (2,0), (3,0), (5,0), (7,0), (11,0), (13,0), (17,0), (19,0), (23,0), (29,0), (31,0), (37,0), (41,0), (43,0), (47,0), (53,0), (59,0), (61,0), (67,0), (71,0), (73,0), (79,0), (83,0), (89,0), (97,0)
    # Sum of reciprocals is way over 1.
    
    # Let's stick to the safe set: 2, 5, 7, 11, 17.
    # And tune residues to cover 1..20.
    # 1: 5%5=1 (r=1)
    # 2: 7%7=2 (r=2)
    # 3: 11%11=3 (r=3)
    # 4: 2%2=0 (r=0)
    # 5: 17%17=5 (r=5)
    # 6: 5%5=1 (r=1)
    # 7: 7%7=0 (r=0) -> Change 7 to r=0?
    #    If 7 is r=0, covers 7, 14, 21...
    #    1: 5 (r=1)
    #    2: ? (2%2=0, 2%5=2, 2%7=2, 2%11=2, 2%17=2) -> Not covered.
    #    So 7 must cover 2? No, 2 is even, covered by 2.
    #    2 is covered by 2 (r=0).
    #    1 is covered by 5 (r=1).
    #    3 is covered by 11 (r=3).
    #    4 is covered by 2 (r=0).
    #    5 is covered by 17 (r=5).
    #    6 is covered by 5 (r=1).
    #    7 is covered by 7 (r=0).
    #    8 is covered by 2 (r=0).
    #    9 is covered by ? (9%5=4, 9%7=2, 9%11=9, 9%17=9) -> Not covered.
    #    We need to cover 9.
    #    9%11=9. If 11 is r=9, covers 9, 20, 31...
    #    But 3 was covered by 11 (r=3).
    #    We can't have two residues for the same prime.
    
    # This trial and error is inefficient.
    # Let's use a simple randomized local search to find a good initial set within the budget.
    # We will start with a random set of primes that fits the budget and then optimize.
    
    import random
    random.seed(42)
    
    # Generate a random valid set
    classes = []
    budget = 0.0
    available_primes = primes[:]
    random.shuffle(available_primes)
    
    for p in available_primes:
        cost = 1.0 / p
        if budget + cost <= 1.0:
            r = random.randint(0, p - 1)
            classes.append([p, r])
            budget += cost
    
    # Ensure we have at least a few primes
    if len(classes) < 5:
        # Fallback to a simple deterministic set
        classes = [[2, 0], [5, 1], [7, 2], [11, 3], [17, 5]]

    # ---------- Build coverage count array for local search ----------
    count = [0] * (max_L + 1)
    for p, r in classes:
        start = r if r >= 1 else p
        for n in range(start, max_L + 1, p):
            count[n] += 1

    def find_prefix_from(count, start):
        for n in range(start, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L

    best_L = find_prefix_from(count, 1)
    
    # If the random start is very bad (L < 10), force a better start
    if best_L < 10:
        classes = [[2, 0], [3, 0], [5, 0], [7, 0], [11, 0], [13, 0], [17, 0], [19, 0], [23, 0], [29, 0], [31, 0], [37, 0]]
        # Check budget
        b = sum(1.0/p for p,r in classes)
        if b > 1.0:
             # Remove largest primes until valid
             classes.sort(key=lambda x: x[0], reverse=True)
             while sum(1.0/p for p,r in classes) > 1.0:
                 classes.pop()
        
        count = [0] * (max_L + 1)
        for p, r in classes:
            start = r if r >= 1 else p
            for n in range(start, max_L + 1, p):
                count[n] += 1
        best_L = find_prefix_from(count, 1)

    # ---------- Local search: optimize residues (1-moves) ----------
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 10:
                break
            p, r = classes[i]
            best_r = r
            best_L_local = best_L
            for new_r in range(p):
                if new_r == r:
                    continue
                # Apply change
                start_old = r if r >= 1 else p
                start_new = new_r if new_r >= 1 else p
                for n in range(start_old, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_new, max_L + 1, p):
                    count[n] += 1
                # Evaluate new prefix
                L_local = best_L
                if best_L < max_L and count[best_L + 1] > 0:
                    L_local = find_prefix_from(count, best_L + 1)
                if L_local > best_L_local:
                    best_L_local = L_local
                    best_r = new_r
                # Revert
                for n in range(start_new, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_old, max_L + 1, p):
                    count[n] += 1
            if best_r != r:
                start_old = r if r >= 1 else p
                start_new = best_r if best_r >= 1 else p
                for n in range(start_old, max_L + 1, p):
                    count[n] -= 1
                for n in range(start_new, max_L + 1, p):
                    count[n] += 1
                classes[i] = [p, best_r]
                best_L = best_L_local
                improved = True
    
    # ---------- Try adding/removing primes to improve coverage ----------
    # If we have budget left, try adding a prime.
    # If we are stuck, try removing a prime and re-optimizing? (Riskier)
    # Let's just try adding primes if budget allows.
    used_primes = set(p for p, r in classes)
    for p in primes:
        if p in used_primes:
            continue
        cost = 1.0 / p
        current_budget = sum(1.0/pp for pp, rr in classes)
        if current_budget + cost <= 1.0:
            # Try adding p with best residue
            best_r = 0
            best_L_add = best_L
            for new_r in range(p):
                start_new = new_r if new_r >= 1 else p
                for n in range(start_new, max_L + 1, p):
                    count[n] += 1
                L_local = best_L
                if best_L < max_L and count[best_L + 1] > 0:
                    L_local = find_prefix_from(count, best_L + 1)
                if L_local > best_L_add:
                    best_L_add = L_local
                    best_r = new_r
                for n in range(start_new, max_L + 1, p):
                    count[n] -= 1
            
            if best_L_add > best_L:
                classes.append([p, best_r])
                best_L = best_L_add
                # Run local search again
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(len(classes)):
                        if time.time() > deadline - 5:
                            break
                        p_i, r_i = classes[i]
                        best_r_i = r_i
                        best_L_local = best_L
                        for new_r in range(p_i):
                            if new_r == r_i:
                                continue
                            start_old = r_i if r_i >= 1 else p_i
                            start_new = new_r if new_r >= 1 else p_i
                            for n in range(start_old, max_L + 1, p_i):
                                count[n] -= 1
                            for n in range(start_new, max_L + 1, p_i):
                                count[n] += 1
                            L_local = best_L
                            if best_L < max_L and count[best_L + 1] > 0:
                                L_local = find_prefix_from(count, best_L + 1)
                            if L_local > best_L_local:
                                best_L_local = L_local
                                best_r_i = new_r
                            for n in range(start_new, max_L + 1, p_i):
                                count[n] -= 1
                            for n in range(start_old, max_L + 1, p_i):
                                count[n] += 1
                        if best_r_i != r_i:
                            start_old = r_i if r_i >= 1 else p_i
                            start_new = best_r_i if best_r_i >= 1 else p_i
                            for n in range(start_old, max_L + 1, p_i):
                                count[n] -= 1
                            for n in range(start_new, max_L + 1, p_i):
                                count[n] += 1
                            classes[i] = [p_i, best_r_i]
                            best_L = best_L_local
                            improved = True
                break # Only add one prime at a time, then re-optimize

    # ---------- Optional: try adding a few more primes if budget remains ----------
    # (skip to keep within time; local search already optimizes existing set)

    return [[p, r] for p, r in classes]
