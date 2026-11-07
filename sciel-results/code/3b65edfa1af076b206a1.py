import time
from math import gcd

def construct():
    deadline = time.time() + 55.0
    start = time.time()
    
    # --- Precomputation ---
    # multiples_of[k] = list of m in [2,256] such that m % k == 0
    multiples_of = [[] for _ in range(257)]
    for k in range(1, 257):
        for m in range(k, 257, k):
            if m >= 2:
                multiples_of[k].append(m)
                
    # Precompute largest prime factor for ordering heuristics
    lpf = [0] * 257
    for i in range(2, 257):
        if lpf[i] == 0: # i is prime
            for j in range(i, 257, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    candidates = list(range(2, 257))

    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set (boolean array/list).
        Returns True if valid, False otherwise.
        """
        # We need to check for all b in S (where in_set[b] is True):
        # 1. a | b*c for some c in S, c != b
        #    This is equivalent to: (a / gcd(a,b)) | c
        # 2. b | a*c for some c in S, c != b
        #    This is equivalent to: (b / gcd(a,b)) | c
        
        for b in range(2, 257):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check 1: a | b*c => (a/g) | c
            k1 = a // g
            # We need c in S, c != b, such that c is a multiple of k1
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check 2: b | a*c => (b/g) | c
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
                    
        return True

    def greedy_construction(order):
        in_set = [False] * 257
        current_set = []
        for a in order:
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set

    best_set = []
    
    # --- Strategy 1: Deterministic Greedy with specific orderings ---
    # Order 1: Descending Largest Prime Factor (often good for avoiding divisibility)
    order_lpf_desc = sorted(candidates, key=lambda x: (-lpf[x], -x))
    S = greedy_construction(order_lpf_desc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 2: Descending Value
    order_desc = sorted(candidates, reverse=True)
    S = greedy_construction(order_desc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 3: Ascending Value
    order_asc = sorted(candidates)
    S = greedy_construction(order_asc)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # Order 4: Primes first (Primes are "safe" if they don't divide products of other primes? 
    # Actually p | q*r implies p|q or p|r. If p,q,r distinct primes, p never divides q*r.
    # So all primes > 128 can coexist. Primes < 128 might conflict with composites.
    # Let's try an order that prioritizes primes.
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]
    order_primes_first = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_construction(order_primes_first)
    if len(S) > len(best_set):
        best_set = S[:]
        
    # --- Strategy 2: Randomized Greedy Restarts ---
    # If time permits, run random restarts to find better structures.
    import random
    
    # We'll run restarts until deadline - 2s to allow final polishing.
    num_restarts = 0
    max_restarts = 2000 # Hard cap to prevent infinite loops if time check is slow
    
    while time.time() < deadline - 3.0 and num_restarts < max_restarts:
        # Create a randomized order with some bias
        # Bias: Higher LPF is slightly more likely to be earlier?
        # Or just pure shuffle.
        random.shuffle(candidates)
        S = greedy_construction(candidates)
        num_restarts += 1
        if len(S) > len(best_set):
            best_set = S[:]
            
        # Optional: Try a "weighted" shuffle where high LPF items have higher weight
        # This might find sets with more large primes/composites.
        # But pure shuffle is a good baseline. Let's stick to pure shuffle for now 
        # unless performance is very fast.
        
        # If we want more diversity, we can try swapping some pairs in the order
        # But greedy is sensitive to order. Pure random covers the space broadly.

    # --- Strategy 3: Final Polishing (Add-all-possible) ---
    # The greedy process stops when no more elements from the *order* can be added.
    # But maybe an element rejected early can be added now? 
    # No, constraints are monotonic. If a was rejected because b,c in S, and S only grows, 
    # a remains rejected. 
    # HOWEVER, greedy only checks against the *current* set.
    # If greedy order was A, B, C... and B was rejected because of A, 
    # but later C is added, does B become valid? No, A is still in the set.
    # So if B was rejected, it stays rejected.
    # Wait, the `can_add` check is against the *current* set.
    # If I iterate through the order, and reject B, I never reconsider B.
    # So a single pass of greedy is complete for that order?
    # Yes, because the set only grows. If B is incompatible with {A}, it is incompatible with {A, C, ...}.
    # So why did I add "final check" in the previous code? 
    # Ah, the previous code did:
    #   for a in range(256, 1, -1): if can_add(a, in_set): add
    # This is essentially a second greedy pass with a fixed order (descending).
    # This CAN add elements if the first greedy pass (with random order) missed them?
    # No. If the first pass rejected `a` because of `b, c` in `S_current`, 
    # and `b, c` are still in `S_final` (which is a superset of `S_current`), 
    # then `a` is still rejected.
    # So a second pass with a different order won't help if the set only grows.
    # UNLESS... the first pass added elements that *blocked* `a`, but a different order 
    # would have added `a` and *blocked* those others.
    # But we are trying to *add* to the *existing* best_set. We cannot remove elements.
    # So if `a` is not in `best_set`, it was either:
    # 1. Rejected by `best_set` during the construction. In which case it is still rejected by `best_set`.
    # 2. Never considered? No, greedy considers all in the order.
    # So `best_set` is a maximal independent set for that specific greedy order?
    # Not necessarily maximal in the global sense, but no single element from the *remaining* 
    # candidates can be added to `best_set` without violating constraints?
    # Yes, that is true. If `a` was rejected during greedy construction with order `O`, 
    # it was rejected because some subset of `S_final` (the set at that time) made it invalid.
    # Since `S_final` contains that subset, `a` is still invalid for `S_final`.
    # Therefore, `best_set` is "maximal" in the sense that no element from 2..256 not in `best_set` 
    # can be added to it.
    # So the "final polishing" loop is redundant! It will find no new elements.
    # Let's verify this logic.
    # Greedy:
    # S = {}
    # for x in Order:
    #    if can_add(x, S): S.add(x)
    # If x was rejected, it is because there exist b, c in S (at that time) such that conflict.
    # S only grows. So b, c are in S_final.
    # So x is still in conflict with S_final.
    # Correct.
    # So we just return best_set.

    # One optimization: 
    # Sometimes, a better set size can be found by *swapping* elements.
    # E.g., remove x, add y, z, w.
    # This is a local search / simulated annealing territory.
    # Given the time limit, simple greedy restarts are usually sufficient for this size.
    
    # However, we can try one more thing: 
    # If we have time, try to find a pair (remove x, add y) that allows adding more?
    # This is expensive.
    
    # Let's just ensure the return type is correct.
    result = [int(x) for x in best_set]
    return result
