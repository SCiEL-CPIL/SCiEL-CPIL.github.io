"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin within 90s limit

    # Precompute primes in 2..256
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [a for a in range(2, 257) if is_prime(a)]
    prime_set = set(primes)
    all_nums = list(range(2, 257))

    # For each candidate composite, determine if it can coexist with all primes
    # Constraint: for distinct a,b,c in S, a does not divide b*c
    # If we include composite c along with primes p1, p2 (all distinct),
    # we need: c does not divide p1*p2, and p1 does not divide c*p2, and p2 does not divide c*p1
    
    # For p1 (prime) not dividing c*p2:
    #   p1 | c*p2 means p1 | c (since p1 is prime and p1 != p2)
    # So p1 does NOT divide c*p2 iff p1 does NOT divide c.
    
    # For c not dividing p1*p2:
    #   c | p1*p2. Since c is composite and p1,p2 are distinct primes,
    #   c | p1*p2 is possible if c = p1*p2, c = p1^2, c = p2^2, or c = 1 (not in range)
    #   Actually c could be p1*p2, or p1^2 (if p1^2 <= 256), etc.
    
    # So the constraint for including composite c with all primes is:
    # 1. No prime p in primes divides c (otherwise p | c*p' for any other prime p')
    #    Wait, that's wrong. If p | c, then p | c*p' for ANY p' != p.
    #    So if c is divisible by any prime p in our set, we have a violation.
    #    Therefore: c must NOT be divisible by ANY prime in our prime set.
    #    But c is in 2..256, so c has prime factors. If any of those prime factors
    #    are in our prime set, we get a violation.
    #    This means we can ONLY include composites whose prime factors are NOT in our prime set.
    #    But ALL primes in 2..256 are in our prime set. So NO composite can be included!
    
    # Wait, let me re-examine. If p | c, then for any other prime q != p in our set:
    #   p | c*q because p | c. So p divides c*q, which is a violation.
    # Therefore, if our set contains ALL primes in 2..256, then NO composite can be added.
    
    # But what if we REMOVE some primes to add composites?
    # Trade-off: remove k primes, add m composites. We need m > k for improvement.
    
    # Let's think about which composites are "compatible" with most primes.
    # A composite c is compatible with prime p iff p does not divide c.
    # So c is compatible with all primes EXCEPT those that divide c.
    # If c = p1^a * p2^b * ... * pk^m, then c is incompatible with primes p1, p2, ..., pk.
    # To include c, we must exclude all primes that divide c.
    # Cost: exclude k primes. Gain: add 1 composite.
    # Net change: 1 - k. This is negative unless k <= 0, which is impossible.
    
    # Wait, that's not quite right. When we exclude a prime p, we also need to check
    # that the remaining set is still valid. But the main point is:
    # Including c requires excluding all prime divisors of c.
    # The number of distinct prime divisors of c is at least 1 (since c >= 2 and composite means >= 4).
    # Actually, c = 4 = 2^2 has 1 distinct prime divisor.
    # c = 6 = 2*3 has 2 distinct prime divisors.
    # c = 30 = 2*3*5 has 3 distinct prime divisors.
    
    # For c = 4: exclude prime 2, add 4. Net: -1 + 1 = 0. No gain.
    #   But we also need to check: with 2 removed, does 4 cause issues with other pairs?
    #   4 | p*q for primes p,q in remaining set? 4 | p*q means p*q is divisible by 4.
    #   Since p,q are odd primes (2 is removed), p*q is odd, so 4 cannot divide p*q.
    #   Also, for prime p in remaining: p | 4*q? p is odd, 4*q is even, so p | 4*q iff p | q,
    #   which is false for distinct primes. And p | 4*p'? No, p is prime and doesn't divide 4*p'
    #   unless p | p', false.
    #   So removing 2 and adding 4 gives same size. Not an improvement.
    
    # For c = 8 = 2^3: same as c=4, exclude {2}, add 8. Same size.
    # For c = 9 = 3^2: exclude {3}, add 9. Same size.
    # For c = 25 = 5^2: exclude {5}, add 25. Same size.
    # For c = 49 = 7^2: exclude {7}, add 49. Same size.
    
    # So we can always swap one prime for a prime power of that prime, keeping the same size.
    # But we can't INCREASE the size this way.
    
    # What about including multiple composites? 
    # If we exclude prime p, we can include MULTIPLE composites that are all powers of p
    # (like 4, 8, 16, 32, 64, 128 for p=2; 9, 27, 81 for p=3; etc.)
    # But wait, we need to check constraints between these composites too.
    # If we have 4 and 8 both in the set, and some prime q:
    #   4 | 8*q? 8*q / 4 = 2q, yes! 4 divides 8*q. Violation!
    # So we can't have both 4 and 8 in the set.
    # In general, if a | b (both in set) and there's any third element c, then a | b*c. Violation.
    # So no two elements in the set can have one dividing the other!
    
    # This is a key insight: the set must be an antichain under divisibility.
    # Additionally, the "no a | b*c" constraint is stronger than just antichain.
    
    # Let me reconsider. The constraint is: for distinct a, b, c in S, a does not divide b*c.
    # This is equivalent to: for every pair (b,c) of distinct elements, and every a distinct from both,
    # a does not divide b*c.
    
    # If S is the set of all primes, this holds because no prime divides a product of two other distinct primes.
    
    # Can we do better than 55? The answer from the analysis above suggests we can't easily,
    # but let me try a greedy approach to find a larger set.
    
    # Strategy: Start with all primes, then try to add composites one by one,
    # removing conflicting primes as needed. Use a greedy/swap approach.
    
    # Actually, let's just try to find the maximum set using a smarter approach.
    # Given the time constraint, let's try a randomized/local search approach.
    
    import random
    
    def check_valid(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i:
                    continue
                b = S_list[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    def check_pairwise_conflict(a, b, S_without_a_and_b):
        """Check if adding a and b together to S_without_a_and_b causes a conflict."""
        for c in S_without_a_and_b:
            if (b * c) % a == 0 or (a * c) % b == 0:
                return True
        return False
    
    # Greedy approach: try to build a large set
    # Start with primes, then try to swap in composites if beneficial
    
    best_set = set(primes)
    best_size = len(best_set)
    
    # Try each composite: see if we can swap it in for one of its prime factors
    composites = [x for x in range(2, 257) if x not in prime_set]
    
    for c in composites:
        if time.time() > deadline - 10:
            break
        
        # Find prime divisors of c
        prime_divs = [p for p in primes if c % p == 0]
        if not prime_divs:
            continue
        
        # Try removing one prime divisor and adding c
        for p in prime_divs:
            if time.time() > deadline - 10:
                break
            
            new_set = best_set - {p}
            new_set.add(c)
            
            # Quick check: is this valid?
            # We need to verify the full constraint, but that's O(n^3)
            # For now, do a partial check
            if len(new_set) > best_size:
                # Only do full check if we're getting larger
                if check_valid(new_set):
                    best_size = len(new_set)
                    best_set = new_set
    
    # Try more aggressive swaps: remove multiple primes to add multiple composites
    # This is more complex; let's try a simpler local search
    
    # Local search: try swapping one element for another
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for c in composites:
            if time.time() > deadline - 5:
                break
            if c in best_set:
                continue
            for p in primes:
                if p not in best_set:
                    continue
                # Try swapping p for c
                new_set = best_set - {p}
                new_set.add(c)
                if len(new_set) != best_size:
                    continue
                # Same size swap - might enable future improvements
                # Skip for now, focus on size-increasing moves
                pass
    
    # Final verification
    if not check_valid(best_set):
        # Fallback to primes
        best_set = set(primes)
    
    result = sorted(best_set)
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
