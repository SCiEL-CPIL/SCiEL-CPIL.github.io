import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    TARGET = 123457
    
    # We'll search for the smallest B such that TARGET = a + b where both
    # a and b are B-smooth (all prime factors <= B)
    # The answer will be B (or the actual max prime factor of the pair)
    
    # Generate primes up to some limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(2000)
    
    # For a given set of primes, generate all smooth numbers <= TARGET
    # A number is smooth if all its prime factors are in the set
    
    def generate_smooth(prime_list, limit):
        """Generate all numbers <= limit whose prime factors are all in prime_list."""
        smooth = set([1])
        # Use BFS/DFS to generate
        # Start with 1, multiply by each prime
        result = [1]
        for p in prime_list:
            new = []
            for s in result:
                v = s
                while v * p <= limit:
                    v *= p
                    new.append(v)
            result.extend(new)
            # Remove duplicates and keep sorted
            result = sorted(set(result))
        return result
    
    # Try increasing sets of primes (smallest primes first)
    best_value = 643  # seed value
    best_pair = (1, 123456)
    
    # For each prefix of primes, generate smooth numbers and check
    for num_primes in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
        
        prime_subset = primes[:num_primes]
        max_p = prime_subset[-1]
        
        # If max_p >= best_value, we might still find a better pair
        # but the max prime factor of any smooth number using these primes
        # is at most max_p. We want to find a pair where both are smooth
        # using only primes <= max_p, giving max(P(a),P(b)) <= max_p.
        # But we need the actual P values, which could be less than max_p
        # if a or b don't use the largest prime.
        
        # Generate smooth numbers
        smooth_a = generate_smooth(prime_subset, TARGET - 1)  # a from 1 to TARGET-1
        smooth_set = set(smooth_a)
        
        # Check if there's a pair
        found = False
        best_in_this_round = max_p  # upper bound
        
        # For each smooth a, check if TARGET - a is also smooth
        # We want to minimize max(P(a), P(b))
        # P(a) is the largest prime factor of a, which is <= max_p
        
        # To be more precise, let's track the actual largest prime factor
        # For each smooth number, compute its largest prime factor
        
        def largest_prime_factor(n, prime_list):
            if n <= 1:
                return 1
            lp = 1
            for p in reversed(prime_list):
                if p > n:
                    break
                if n % p == 0:
                    lp = p
                    break
            # If we didn't find it, n might be prime itself (but it's smooth, so no)
            # Actually since n is smooth with primes in prime_list, the largest
            # prime factor must be in prime_list
            # Let's do it properly
            temp = n
            lp = 1
            for p in prime_list:
                if p * p > temp:
                    break
                while temp % p == 0:
                    lp = p
                    temp //= p
            if temp > 1:
                # temp should be in prime_list since n is smooth
                lp = max(lp, temp)
            return lp
        
        # Compute P for each smooth number
        P_map = {}
        for s in smooth_a:
            P_map[s] = largest_prime_factor(s, prime_subset)
        
        # Now search for pairs
        for a in smooth_a:
            if time.time() > deadline:
                break
            b = TARGET - a
            if b in smooth_set:
                Pa = P_map[a]
                Pb = P_map[b]
                val = max(Pa, Pb)
                if val < best_value:
                    best_value = val
                    best_pair = (a, b)
                    found = True
        
        # If we found a pair with max prime factor = max_p, we can't do better
        # with this prime set. But we might find a pair with smaller max prime
        # factor using a subset. However, we're going in order of increasing primes,
        # so the first time we find a valid pair, max_p is the smallest possible
        # upper bound. But the actual max(P(a),P(b)) could be less than max_p.
        
        # Once we've found a valid pair, we can stop if max_p == best_value
        # because we can't get a smaller max prime factor with more primes
        if found and best_value <= max_p:
            # We found the optimal for this prime set
            # Can we do better with fewer primes? We already tried fewer.
            # So this is optimal.
            break
        if found:
            # We found a pair, but maybe with a larger prime set we could find
            # a pair with smaller max prime factor? No, that doesn't make sense.
            # With more primes, smooth numbers can have larger prime factors.
            # Actually, we want to MINIMIZE the max prime factor.
            # Using fewer primes means all prime factors are smaller.
            # So the first valid pair we find (with smallest prime set) gives
            # the smallest possible max prime factor.
            # Wait, not exactly. With primes {2,3,5}, a number could be 2*3*5=30,
            # P=5. With primes {2,3,5,7}, a number could be 7, P=7.
            # So using fewer primes constrains the max prime factor more.
            # The first time we find a valid decomposition, the max prime factor
            # of the pair is at most the largest prime in our set.
            # And since we're trying primes in order, this is the smallest possible.
            # But the actual P(a) or P(b) might be less than the largest prime in set.
            # E.g., if set is {2,3,5,7} but a=10, b=123447, P(a)=5, P(b)=?
            # We need to check actual values.
            # In any case, we should continue to see if we can find a pair
            # with smaller max(P(a),P(b)) using the same or more primes.
            # Actually no - with more primes, we have MORE smooth numbers, so
            # we might find a pair where both have small prime factors even
            # though the prime set includes large primes.
            # Example: with primes {2,3,5,7,11}, we could have a=2*3=6, b=123451
            # if 123451 is also smooth. P(6)=3, P(123451)=?
            # So we should keep searching.
            pass
    
    # If we didn't find anything better, return the best we have
    return list(best_pair)
