import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # The goal is to minimize M = max(P(a), P(b)).
    # This is equivalent to finding the smallest M such that there exist
    # a, b with a+b=target, and all prime factors of a and b are <= M.
    # Such numbers are called M-smooth.
    
    # We will test increasing sets of primes.
    # Primes to consider: 2, 3, 5, 7, 11, 13, ...
    
    def generate_smooth_numbers(primes, limit):
        """
        Generates all numbers <= limit whose prime factors are in 'primes'.
        Returns a set of these numbers.
        """
        smooths = {1}
        # Use a queue-like approach or recursive generation
        # Since limit is ~1e5, the number of smooth numbers is small for small primes.
        
        # We can use a simple BFS/DFS from 1
        # To avoid duplicates and ensure efficiency, we can use a set.
        # However, for small primes, the count is manageable.
        
        # A more efficient way:
        # Start with 1.
        # For each prime p, multiply existing numbers by p as long as <= limit.
        # This is similar to the sieve but only for specific primes.
        
        # Let's use a list to store current smooth numbers and expand
        current = [1]
        all_smooths = set([1])
        
        # We need to be careful with the order to not miss combinations.
        # Standard approach:
        # For each prime p in primes:
        #   For each x in current_smooths (sorted):
        #      y = x * p
        #      while y <= limit:
        #         add y to set
        #         y *= p
        #   Update current_smooths to include new ones?
        #   Actually, it's easier to just generate all combinations.
        
        # Let's use a recursive generator or iterative expansion.
        # Given the small limit, a simple iterative expansion over the set is fine.
        
        # Initialize with 1
        smooth_set = {1}
        
        # We process primes one by one.
        # If we have a set S of numbers using primes P1..Pk,
        # and we add prime p_{k+1}, the new set is S union {x * p_{k+1}^j | x in S, x * p_{k+1}^j <= limit}
        
        # This can grow large, but for small primes and limit 1e5, it's small.
        
        for p in primes:
            new_smooths = set(smooth_set)
            # For each number in the current smooth set, multiply by powers of p
            # We need to iterate over the *old* smooth set to generate new ones
            # and add them to the new set.
            
            # To be efficient, we should iterate over the current smooth_set
            # and for each x, multiply by p, p^2, etc.
            
            # Note: We must not use the newly added numbers to generate further multiples of p
            # in the same step? Actually, if we do x*p, then (x*p)*p = x*p^2, it works.
            # But if we iterate over a set that is being modified, it's tricky.
            
            # Better: collect new candidates
            candidates = []
            for x in smooth_set:
                val = x * p
                while val <= limit:
                    candidates.append(val)
                    val *= p
            
            # Add all candidates to the set
            for c in candidates:
                new_smooths.add(c)
            
            smooth_set = new_smooths
            
            # Early termination if set size explodes (shouldn't for small primes)
            if len(smooth_set) > 100000:
                break
                
        return smooth_set

    # List of primes to try in order
    # We start with the smallest primes because they allow more combinations.
    # If a solution exists with primes {2,3,5}, it also exists with {2,3,5,7}.
    # So we want the *minimal* set of primes that allows a solution?
    # No, we want to minimize the *largest* prime factor.
    # So we check M=2, M=3, M=5, M=7, etc.
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We need to check if there is a pair a,b such that a+b=target
    # and P(a) <= M, P(b) <= M.
    
    # We iterate M through the primes in primes_list.
    # For each M, we generate the set of M-smooth numbers.
    # Then we check if there is a collision.
    
    # Optimization: We can just generate the set once and check.
    # But generating the set for M=97 is expensive if we do it from scratch each time.
    # We can build incrementally.
    
    current_smooth_set = {1}
    
    for i, p in enumerate(primes_list):
        if time.time() > deadline:
            break
            
        # Add prime p to the set of allowed primes
        # Update current_smooth_set to include numbers with p as a factor
        
        # We need to generate new numbers by multiplying existing smooth numbers by powers of p
        # Note: The existing smooth numbers are (p-1)-smooth.
        # The new set will be p-smooth.
        
        new_numbers = []
        # We iterate over the *current* smooth set (which is (p-1)-smooth)
        # and multiply by p^k
        for x in current_smooth_set:
            val = x * p
            while val <= target:
                new_numbers.append(val)
                val *= p
        
        # Add new numbers to the set
        for n in new_numbers:
            current_smooth_set.add(n)
            
        # Now check if there is a pair a, b in current_smooth_set such that a + b = target
        # We can do this by iterating through the set and checking if target - a is in the set.
        # To avoid a+b and b+a, we can just check all a.
        
        found = False
        best_a = -1
        best_b = -1
        
        # Iterate through the set
        # To make it deterministic and potentially faster, we can sort? 
        # No, set lookup is O(1).
        
        for a in current_smooth_set:
            b = target - a
            if b in current_smooth_set:
                # Found a valid pair
                # We want to minimize max(P(a), P(b)). 
                # Since all numbers in the set are p-smooth, P(a)<=p and P(b)<=p.
                # So max(P(a),P(b)) <= p.
                # Since we are iterating p in increasing order, the first p we find is the minimum possible max prime factor.
                # Wait, is it possible that a smaller p worked but we missed it?
                # No, we are building the set incrementally. 
                # If a solution existed with primes {2,3}, it would have been found in the previous step.
                # So the first p that yields a solution gives the minimal M.
                
                # However, we might find multiple pairs for this p. 
                # The objective is to minimize max(P(a), P(b)). 
                # For a fixed p, all valid pairs have max(P(a), P(b)) <= p.
                # But some pairs might have max(P(a), P(b)) < p.
                # Example: p=5. Pair (2, 5) has max=5. Pair (2, 2) sum=4 != target.
                # If we find a pair where both a and b are actually 3-smooth, then the true max is 3.
                # But we would have found that in the p=3 step!
                # So if we reach p=5, it means no pair existed with max prime factor <= 3.
                # Therefore, any pair found at step p=5 must have at least one number with prime factor 5.
                # So max(P(a), P(b)) will be exactly 5 (or higher, but bounded by 5 in this set).
                # Actually, if a is 5-smooth and b is 5-smooth, max(P(a),P(b)) is <= 5.
                # Since no solution was found for p=3, it implies that for all pairs summing to target, at least one has a prime factor > 3.
                # So the minimal possible max prime factor is indeed the current p.
                
                # So we can return immediately upon finding the first valid pair for the smallest p.
                # Wait, what if there are multiple pairs for this p? 
                # They all have max(P(a),P(b)) <= p.
                # Since p is the smallest prime for which a solution exists, the optimal value is p.
                # Any pair found here is optimal.
                
                best_a = a
                best_b = b
                found = True
                break
                
        if found:
            return [best_a, best_b]

    # Fallback: If no small prime set worked, we need to find the pair with minimal max(P(a), P(b))
    # using a standard sieve. This is unlikely to be reached for N=123457 with primes up to 97,
    # but we must ensure correctness.
    
    # Standard Sieve for Largest Prime Factor
    lpf = [0] * (target + 1)
    lpf[1] = 1
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target // 2 + 1):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            if best_max_p == 3: # Theoretical min for odd sum > 2
                break
        if time.time() > deadline:
            break
            
    return [best_a, best_b]
