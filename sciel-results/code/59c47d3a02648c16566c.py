import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s/90s limits
    
    N = 256
    all_numbers = list(range(1, N + 1))
    
    # Precompute divisibility check
    # (a*b) % (a+b) == 0  is the forbidden condition
    # We want to maximize the set S such that for all distinct a,b in S, (a*b) % (a+b) != 0
    
    # Initial set: all odd numbers (known valid base)
    # Let's verify: if a and b are odd, a+b is even. 
    # a*b is odd. 
    # An odd number cannot be divisible by an even number (unless it's 0, which it's not).
    # So all odd numbers form a valid set.
    # Size: 128.
    
    # Can we add even numbers?
    # If we add an even number e, we must check against all current members.
    # If current set is all odds, e + odd is odd. 
    # e * odd is even.
    # (e*odd) % (e+odd) == 0 ?
    
    # Let's try a greedy approach to add numbers.
    # Start with all odds.
    S = set(range(1, N + 1, 2))
    
    # Candidate numbers to add: evens
    candidates = list(range(2, N + 1, 2))
    
    # Shuffle candidates to try different orders? Or sort by some heuristic?
    # Smaller numbers might be "harder" to fit or "easier"?
    # Let's just try adding them one by one if they fit.
    # We can also try removing some odds to fit more evens?
    # But 128 is already half of 256. Can we exceed 128?
    
    # Let's check if we can add ANY even number to the set of all odds.
    # e.g., try 2.
    # S = {1, 3, 5, ...}
    # Check 2 with 1: (2*1) % (2+1) = 2 % 3 != 0. OK.
    # Check 2 with 3: (2*3) % (2+3) = 6 % 5 != 0. OK.
    # Check 2 with 5: (2*5) % (2+5) = 10 % 7 != 0. OK.
    # Check 2 with 7: (2*7) % (2+7) = 14 % 9 != 0. OK.
    # Check 2 with 9: (2*9) % (2+9) = 18 % 11 != 0. OK.
    # Check 2 with 11: (2*11) % (2+11) = 22 % 13 != 0. OK.
    # Check 2 with 13: (2*13) % (2+13) = 26 % 15 != 0. OK.
    # Check 2 with 15: (2*15) % (2+15) = 30 % 17 != 0. OK.
    # ...
    # Is there an odd number o such that (2*o) % (2+o) == 0?
    # 2o = k(2+o) => 2o - ko = 2k => o(2-k) = 2k => o = 2k / (2-k)
    # If k=1, o = 2/1 = 2 (not odd).
    # If k=0, o = 0 (not in range).
    # If k > 2, denominator negative.
    # So 2 can be added to all odds!
    
    # Let's try adding all evens greedily to the set of odds.
    # But adding one even might conflict with another even.
    # If we add 2, can we add 4?
    # Check 2 and 4: (2*4) % (2+4) = 8 % 6 != 0. OK.
    # Check 4 with odds:
    # 4*1 % 5 = 4 != 0
    # 4*3 % 7 = 12 % 7 = 5 != 0
    # 4*5 % 9 = 20 % 9 = 2 != 0
    # 4*7 % 11 = 28 % 11 = 6 != 0
    # 4*9 % 13 = 36 % 13 = 10 != 0
    # 4*11 % 15 = 44 % 15 = 14 != 0
    # 4*13 % 17 = 52 % 17 = 1 != 0
    # 4*15 % 19 = 60 % 19 = 3 != 0
    # It seems many evens can be added.
    
    # Let's implement a greedy adder.
    # Start with S = all odds.
    # For each even e in shuffled order:
    #   If e is compatible with all x in S, add e to S.
    
    # However, the order might matter. If we add a "bad" even early, it might block others.
    # Or if we add a "good" even, it doesn't block.
    # Let's try multiple random shuffles and keep the best set.
    
    best_set = set(S)
    best_size = len(S)
    
    # Also, maybe we can remove some odds to add more evens?
    # But odds are very "compatible" with each other.
    # Evens are the ones causing conflicts.
    # So keeping all odds seems safe.
    
    # Let's run a few iterations of greedy adding evens with different shuffles.
    evens = list(range(2, N + 1, 2))
    
    for iteration in range(20):
        if time.time() > deadline:
            break
            
        current_set = set(range(1, N + 1, 2))
        candidates = list(evens)
        random.shuffle(candidates)
        
        for e in candidates:
            if time.time() > deadline:
                break
            compatible = True
            for x in current_set:
                if (e * x) % (e + x) == 0:
                    compatible = False
                    break
            if compatible:
                current_set.add(e)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we reached 256, we can stop
            if best_size == N:
                break
    
    # Now, can we do better by removing some odds and adding more evens?
    # Or swapping?
    # Let's try a local search: 
    # 1. Try to add any remaining number.
    # 2. Try to remove one number and add two others? (Too slow)
    # 3. Try to swap an odd with an even if it allows adding more?
    
    # Let's just try to add any remaining numbers to best_set.
    remaining = [x for x in all_numbers if x not in best_set]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        compatible = True
        for y in best_set:
            if (x * y) % (x + y) == 0:
                compatible = False
                break
        if compatible:
            best_set.add(x)
    
    # One more pass to see if we can add more
    remaining = [x for x in all_numbers if x not in best_set]
    for x in remaining:
        if time.time() > deadline:
            break
        compatible = True
        for y in best_set:
            if (x * y) % (x + y) == 0:
                compatible = False
                break
        if compatible:
            best_set.add(x)
    
    return list(best_set)
