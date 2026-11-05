import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Precompute primes up to 20000 (max sum is 10000+9999=19999)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes_up_to = sieve(20000)
    prime_set = set(primes_up_to)

    # Precompute prime factors for all numbers up to 20000
    # pf[x] = set of prime factors of x
    pf = [set() for _ in range(20001)]
    for p in primes_up_to:
        for multiple in range(p, 20001, p):
            pf[multiple].add(p)

    def get_primes_for_sums(nums):
        """Get union of all prime factors of pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s = nums[i] + nums[j]
                all_primes.update(pf[s])
        return all_primes

    # Strategy: We want to find 12 distinct positive integers <= 10000
    # such that the union of prime factors of all pairwise sums is minimized.
    
    # Key insight: If we pick all numbers to be multiples of a small set of primes,
    # then sums will also be constrained. But better: pick numbers that are all
    # congruent modulo small primes to limit the primes that appear.
    
    # Approach: Try different prime sets P and find 12 numbers where all pairwise
    # sums only use primes from P.
    
    # Let's try: pick numbers that are all 1 mod small primes? No, sums would be 2 mod those.
    # Better: all numbers even -> all sums even -> 2 is always a factor.
    # All numbers multiples of 3 -> sums multiples of 3.
    
    # Let's try a structured approach:
    # Candidate: numbers of form a * k + b where we control primes in sums.
    
    # Simple strong candidate: powers of 2 times small odd numbers, or just
    # use numbers that are all congruent to 1 mod 6, making sums ≡ 2 mod 6.
    
    # Let me try: all numbers are odd -> all sums are even (factor 2 always).
    # If all numbers ≡ 1 (mod 3) -> sums ≡ 2 (mod 3), so 3 never divides a sum.
    # If all numbers ≡ 1 (mod 5) -> sums ≡ 2 (mod 5), so 5 never divides a sum.
    # If all numbers ≡ 1 (mod 7) -> sums ≡ 2 (mod 7), so 7 never divides a sum.
    
    # So pick 12 distinct numbers all ≡ 1 (mod 210) [lcm(2,3,5,7)] -> all ≡ 1 mod 210
    # Then sums ≡ 2 (mod 210), so 2,3,5,7 never divide any sum except 2.
    # Actually sum ≡ 2 mod 2 -> sum even, so 2 divides.
    # sum ≡ 2 mod 3 -> not divisible by 3
    # sum ≡ 2 mod 5 -> not divisible by 5
    # sum ≡ 2 mod 7 -> not divisible by 7
    
    # So only 2 is guaranteed, and other primes depend on the specific values.
    
    # Numbers ≡ 1 mod 210 up to 10000: 1, 211, 421, 631, 841, 1051, 1261, ...
    # Let's compute how many: floor((10000-1)/210) + 1 = floor(9999/210)+1 = 47+1 = 48
    
    # Pick first 12: 1, 211, 421, 631, 841, 1051, 1261, 1471, 1681, 1891, 2101, 2311
    candidate_mod210 = [1 + 210 * i for i in range(12)]
    
    # Check prime factors of pairwise sums
    primes_c1 = get_primes_for_sums(candidate_mod210)
    best_count = len(primes_c1)
    best_set = list(candidate_mod210)
    
    # Try other moduli structures
    # All ≡ 1 mod 30: sums ≡ 2 mod 30, avoids 3,5 but not 7
    candidate_mod30 = [1 + 30 * i for i in range(12)]
    primes_c2 = get_primes_for_sums(candidate_mod30)
    if len(primes_c2) < best_count:
        best_count = len(primes_c2)
        best_set = list(candidate_mod30)
    
    # All ≡ 1 mod 6: sums ≡ 2 mod 6, avoids 3
    candidate_mod6 = [1 + 6 * i for i in range(12)]
    primes_c3 = get_primes_for_sums(candidate_mod6)
    if len(primes_c3) < best_count:
        best_count = len(primes_c3)
        best_set = list(candidate_mod6)
    
    # All even: sums even. All ≡ 2 mod 4: sums ≡ 0 mod 4, so 2 is factor.
    # All ≡ 2 mod 6: sums ≡ 4 mod 6 = 2 mod 3, avoids 3
    candidate_mod6_even = [2 + 6 * i for i in range(12)]
    primes_c4 = get_primes_for_sums(candidate_mod6_even)
    if len(primes_c4) < best_count:
        best_count = len(primes_c4)
        best_set = list(candidate_mod6_even)
    
    # All ≡ 1 mod 2: odd. All ≡ 1 mod 15: avoids 3,5
    candidate_mod15 = [1 + 15 * i for i in range(12)]
    primes_c5 = get_primes_for_sums(candidate_mod15)
    if len(primes_c5) < best_count:
        best_count = len(primes_c5)
        best_set = list(candidate_mod15)
    
    # All ≡ 1 mod 105 (lcm of 3,5,7): sums ≡ 2 mod 105
    candidate_mod105 = [1 + 105 * i for i in range(12)]
    primes_c6 = get_primes_for_sums(candidate_mod105)
    if len(primes_c6) < best_count:
        best_count = len(primes_c6)
        best_set = list(candidate_mod105)
    
    # Try local search: start from best_set and try replacing elements
    def local_search(start, budget=30):
        if time.time() > deadline - 5:
            return list(start), best_count
        
        current = list(start)
        current_primes = get_primes_for_sums(current)
        current_count = len(current_primes)
        
        # Try replacing each element with another valid number
        for _ in range(budget):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for idx in range(12):
                old_val = current[idx]
                # Try some replacement values
                for new_val in range(1, min(10001, old_val + 500)):
                    if new_val == old_val:
                        continue
                    if new_val in current and new_val != old_val:
                        continue
                    # Check distinctness
                    if new_val in current:
                        continue
                    
                    test_set = list(current)
                    test_set[idx] = new_val
                    test_primes = get_primes_for_sums(test_set)
                    
                    if len(test_primes) < current_count:
                        current = test_set
                        current_primes = test_primes
                        current_count = len(test_primes)
                        improved = True
                        break
                
                if improved:
                    break
        
        return current, current_count
    
    if time.time() < deadline - 10:
        best_set, best_count = local_search(best_set, budget=20)
    
    # Also try: all numbers are powers of 2 (1,2,4,8,...,2048) - only 12 fit <= 10000
    # Sums of powers of 2: 2^a + 2^b = 2^min(a,b) * (1 + 2^(|a-b|))
    # The odd factor is (1 + 2^k) which can have various primes
    powers_of_2 = [2**i for i in range(12)]  # 1,2,4,...,2048
    if all(x <= 10000 for x in powers_of_2):
        primes_c7 = get_primes_for_sums(powers_of_2)
        if len(primes_c7) < best_count:
            best_count = len(primes_c7)
            best_set = list(powers_of_2)
    
    # Try: all numbers = k * (some base) to constrain factors
    # All multiples of 4: sums multiples of 4
    # All multiples of 2^m for large m
    # 2^6=64, multiples: 64,128,192,...,64*156=9984 -> 156 available
    # sums: 64*(a+b), prime factors = {2} U factors(a+b)
    multiples_64 = [64 * (i + 1) for i in range(12)]
    primes_c8 = get_primes_for_sums(multiples_64)
    if len(primes_c8) < best_count:
        best_count = len(primes_c8)
        best_set = list(multiples_64)
    
    # Try multiples of 128: 128*(1..12)
    multiples_128 = [128 * (i + 1) for i in range(12)]
    primes_c9 = get_primes_for_sums(multiples_128)
    if len(primes_c9) < best_count:
        best_count = len(primes_c9)
        best_set = list(multiples_128)
    
    # Try: all numbers of form 2^k * m where m is small odd
    # Just use 2 * {1,3,5,...,23}
    doubled_odds = [2 * (2*i + 1) for i in range(12)]
    primes_c10 = get_primes_for_sums(doubled_odds)
    if len(primes_c10) < best_count:
        best_count = len(primes_c10)
        best_set = list(doubled_odds)
    
    # Broader local search with more replacements
    if time.time() < deadline - 15:
        def broader_local_search(start, max_iters=200):
            if time.time() > deadline - 5:
                return list(start), best_count
            
            current = list(start)
            current_primes = get_primes_for_sums(current)
            current_count = len(current_primes)
            used = set(current)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                # Try each position
                for idx in range(12):
                    old_val = current[idx]
                    # Try random candidates
                    for _ in range(10):
                        new_val = 1 + (it * 137 + idx * 31 + _ * 17) % 10000
                        if new_val == old_val or new_val in used:
                            continue
                        
                        test_set = list(current)
                        test_set[idx] = new_val
                        test_primes = get_primes_for_sums(test_set)
                        
                        if len(test_primes) < current_count:
                            current = test_set
                            current_primes = test_primes
                            current_count = len(test_primes)
                            used.discard(old_val)
                            used.add(new_val)
                            improved = True
                            break
                    if improved:
                        break
            
            return current, current_count
        
        best_set, best_count = broader_local_search(best_set, max_iters=100)
    
    # Final check: ensure validity
    assert len(set(best_set)) == 12
    assert all(1 <= x <= 10000 for x in best_set)
    
    return best_set
