import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Max possible sum: 12 * 256 = 3072. Let's assume max size 12 for precomputation.
    # If we find size 13, we might need to extend, but 12 is a strong target.
    MAX_SUM = 3200
    MAX_SIZE = 12
    
    # Precompute divisors for all numbers up to MAX_SUM
    # divisors[i] = list of divisors of i
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            divisors[j].append(i)
            
    # Precompute multiples? Or just use the divisors list.
    # To check if a in S divides b: check if a is in divisors[b].
    # To check if b in S divides a: check if b is in divisors[a].
    # Since we add new sums, we need to check new sums against old sums.
    
    def is_valid_addition(x, current_sums):
        """
        Checks if adding x to the set with current_sums results in a valid set of sums.
        Returns True if valid, False otherwise.
        Also returns the new sums if valid? No, just bool for speed in inner loop.
        We will compute new_sums separately if needed.
        """
        # New sums are current_sums and current_sums + x
        # 1. Check for duplicates
        # If (current_sums + x) intersects current_sums, duplicate.
        # Also x must not be in current_sums.
        # And x + s must not be in current_sums for any s in current_sums.
        
        # Efficient duplicate check:
        # A duplicate exists if there is s1, s2 in current_sums such that s1 + x = s2.
        # i.e., s2 - s1 = x.
        # Or x = s1 - s2.
        # Simpler: just generate the set and check size?
        # |new_sums| should be 2 * |current_sums| + 1.
        
        # Let's just do the divisibility check first, it's more restrictive and uses precomputed data.
        # But we need the set of new sums to check against.
        
        # Optimization: 
        # The new sums are {x} U {s+x for s in current_sums} U current_sums.
        # We need to ensure no a, b in new_sums with a|b or b|a.
        
        # 1. Check x against current_sums
        # x must not divide any s in current_sums, and no s in current_sums must divide x.
        for s in current_sums:
            if x > s:
                if x % s == 0: return False
            else:
                if s % x == 0: return False
        
        # 2. Check (s+x) against current_sums
        for s in current_sums:
            new_s = s + x
            # new_s vs s (already checked x vs s? No, new_s vs s is (s+x) vs s)
            # (s+x) is always > s, so s divides (s+x) iff (s+x) % s == 0 iff x % s == 0.
            # We already checked x % s == 0 in step 1? 
            # Step 1 checked x vs s. If x % s == 0, then s | x, so s | (s+x). 
            # So if step 1 passed, s does not divide (s+x).
            
            # Now check new_s against other t in current_sums (t != s)
            # We need to check if any t in current_sums divides new_s, or new_s divides t.
            # Since new_s > t is not always true, we check both.
            # But we can use the divisor precomputation.
            # Check if any t in current_sums is a divisor of new_s.
            # And if new_s is a divisor of any t in current_sums.
            
            # To do this fast, we need to check if the set `current_sums` intersects `divisors[new_s]`.
            # And if `new_s` is in `divisors[t]` for any t in current_sums? 
            # That is equivalent to checking if any multiple of new_s is in current_sums.
            
            # Let's just iterate current_sums for now, it's O(N). N=2^k.
            # If k=10, N=1024. 1024 * 1024 = 1M ops per check. 
            # With 12 depth, this might be too slow if branching factor is high.
            
            # Let's optimize:
            # Convert current_sums to a set (it is a set).
            # For new_s, check if any d in divisors[new_s] is in current_sums.
            for d in divisors[new_s]:
                if d in current_sums:
                    # d divides new_s.
                    # We must ensure d != new_s (impossible since d < new_s usually, unless d=new_s which means 1*new_s, but d is from divisors list)
                    # Wait, if d == new_s, that means new_s divides itself, which is fine? 
                    # No, the rule is "no two DISTINCT resulting values". 
                    # So a|b is bad if a != b.
                    # divisors[new_s] includes new_s itself.
                    if d != new_s:
                        return False
            
            # Check if new_s divides any t in current_sums.
            # i.e., is there a t in current_sums such that t % new_s == 0?
            # This is harder with precomputed divisors. 
            # We can check: for each multiple m of new_s, is m in current_sums?
            # Max multiple is MAX_SUM.
            for m in range(new_s, MAX_SUM + 1, new_s):
                if m in current_sums:
                    if m != new_s:
                        return False
        
        # 3. Check (s1+x) against (s2+x)
        # (s1+x) and (s2+x) are distinct if s1 != s2.
        # Do they divide each other?
        # Let a = s1+x, b = s2+x.
        # This is the most complex part.
        # We need to check all pairs (s1, s2) in current_sums.
        # This is O(N^2).
        # For N=1024, 1M checks.
        # Inside check: a % b == 0 or b % a == 0.
        # This might be too slow for deep backtracking.
        
        # Alternative: 
        # If we find that this is too slow, we can restrict the search to smaller sizes or use a faster heuristic.
        
        # For now, let's implement the full check but limit the branching.
        
        # Actually, let's just do the full check in a separate function to keep it clean, 
        # but we will rely on the fact that we only call this for promising candidates.
        
        return True

    def full_validity_check(new_sums):
        """
        Checks if a set of sums is valid (distinct and divisibility incomparable).
        """
        sums_list = list(new_sums)
        n = len(sums_list)
        # Sort to help with divisibility? Not strictly necessary for set intersection logic, 
        # but for the O(N^2) check, maybe.
        
        # 1. Distinctness is guaranteed by set construction.
        
        # 2. Divisibility
        for i in range(n):
            a = sums_list[i]
            # Check if any other b divides a
            for d in divisors[a]:
                if d != a and d in new_sums:
                    return False
            # Check if a divides any other b
            # Iterate multiples
            for m in range(a + a, MAX_SUM + 1, a):
                if m in new_sums:
                    return False
        return True

    # Backtracking Search
    best_set = []
    best_size = 0
    
    # Initial candidates: 1 to 256
    # Heuristic: Try numbers that are "large" first? 
    # Or numbers that are primes?
    # Let's try a mix.
    
    def backtrack(current_set, current_sums, candidates, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        if depth >= MAX_SIZE:
            return
            
        if not candidates:
            return
            
        # Try each candidate
        # To improve speed, we can limit the number of candidates we try at each step
        # or use a heuristic to order them.
        
        # Heuristic: Prefer candidates that are larger?
        # Let's shuffle candidates to avoid bias, but keep the best ones.
        # Actually, let's just iterate through a subset of candidates.
        
        # To make it faster, we will only try the first K candidates in a randomized order.
        K = 20
        if len(candidates) > K:
            # Pick K random candidates
            chosen = random.sample(candidates, K)
        else:
            chosen = candidates
            
        random.shuffle(chosen)
        
        for x in chosen:
            if time.time() > deadline:
                return
                
            # Quick check: x not in current_sums
            if x in current_sums:
                continue
                
            # Generate new sums
            new_sums = set(current_sums)
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(s + x)
                
            # Check size (distinctness)
            if len(new_sums) != 2 * len(current_sums) + 1:
                continue
                
            # Check divisibility
            # Use the optimized check
            valid = True
            # Check x against current_sums
            for s in current_sums:
                if x > s:
                    if x % s == 0:
                        valid = False
                        break
                else:
                    if s % x == 0:
                        valid = False
                        break
            if not valid:
                continue
                
            # Check (s+x) against current_sums
            if valid:
                for s in current_sums:
                    ns = s + x
                    # Check if any d in divisors[ns] is in current_sums
                    for d in divisors[ns]:
                        if d != ns and d in current_sums:
                            valid = False
                            break
                    if not valid:
                        break
                    # Check if any multiple of ns is in current_sums
                    if valid:
                        for m in range(ns + ns, MAX_SUM + 1, ns):
                            if m in current_sums:
                                valid = False
                                break
                    if not valid:
                        break
            if not valid:
                continue
                
            # Check (s1+x) against (s2+x)
            # This is the expensive part.
            # We only need to check if any pair divides.
            # Let's just do the full check on new_sums if we pass the quick checks.
            # It might be faster to just call full_validity_check(new_sums) here.
            # But full_validity_check is O(N * log N) roughly due to divisors.
            
            if full_validity_check(new_sums):
                # Valid!
                new_candidates = [c for c in candidates if c != x]
                backtrack(current_set + [x], new_sums, new_candidates, depth + 1)
                
    # Start search
    # Initial candidates
    candidates = list(range(1, 257))
    
    # To help the search, let's try a few different initial orderings
    # 1. Descending
    # 2. Primes first
    # 3. Random
    
    strategies = [
        sorted(candidates, reverse=True),
        sorted(candidates, key=lambda x: (x % 2 == 0, x)), # Odd first
        candidates.copy()
    ]
    
    for strat in strategies:
        if time.time() > deadline:
            break
        random.shuffle(strat)
        backtrack([], set(), strat, 0)
        
    # If we didn't find anything big, try a greedy approach as fallback
    if best_size < 5:
        greedy_set = []
        greedy_sums = set()
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            new_sums = set(greedy_sums)
            new_sums.add(x)
            for s in greedy_sums:
                new_sums.add(s + x)
            if len(new_sums) != 2 * len(greedy_sums) + 1:
                continue
            if full_validity_check(new_sums):
                greedy_set.append(x)
                greedy_sums = new_sums
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(greedy_set)
            
    return [int(x) for x in best_set]
