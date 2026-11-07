import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0


def construct():
    if time.time() > _DEADLINE:
        return [2]

    # The following set of 18 integers has been verified to have all distinct
    # subset products (including the empty product 1).
    # Structure: powers of 2 (2^0..2^4), powers of 3 (3^0, 3^1, 3^2), powers of 5 (5^0, 5^1, 5^2),
    # and remaining primes 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Total: 5 + 3 + 3 + 7 = 18.
    
    result = [2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 125, 7, 11, 13, 17, 19, 23, 29]
    
    # Wait, 125 is not in 2..40. 
    # Let's re-evaluate the set.
    # Range is 2..40.
    # Powers of 2: 2, 4, 8, 16, 32 (5 numbers)
    # Powers of 3: 3, 9, 27 (3 numbers)
    # Powers of 5: 5, 25 (2 numbers) -- 125 is too big.
    # Primes: 7, 11, 13, 17, 19, 23, 29, 31, 37 (9 numbers)
    # Total: 5 + 3 + 2 + 9 = 19?
    # Let's check if 19 is possible.
    # Numbers: 2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Are all subset products distinct?
    # Yes, by unique factorization, since each number is a pure power of a single prime,
    # and the primes are distinct.
    # The subset product corresponds to a tuple of exponents (e2, e3, e5, e7, e11, e13, e17, e19, e23, e29, e31, e37).
    # e2 can be 0..4 (from 1, 2, 4, 8, 16, 32 -> exponents 0,1,2,3,4,5? No, the set contains the numbers themselves).
    # Let's be precise.
    # The set S = {2^1, 2^2, 2^3, 2^4, 2^5, 3^1, 3^2, 3^3, 5^1, 5^2, 7^1, 11^1, 13^1, 17^1, 19^1, 23^1, 29^1, 31^1, 37^1}.
    # A subset product is determined by the sum of exponents for each prime.
    # For prime 2, the possible sums of exponents from the subset {1,2,3,4,5} are distinct?
    # No, subset sums of {1,2,3,4,5} are NOT all distinct. E.g., 1+4 = 5, 2+3 = 5.
    # So {2, 4, 8, 16, 32} does NOT have distinct subset products!
    # 2 * 16 = 32. Collision with {32}.
    
    # So we cannot have both 2 and 16 and 32?
    # If we have 2, 4, 8, 16, 32:
    # Subsets:
    # {2, 16} -> 32
    # {32} -> 32
    # Collision.
    
    # We need a set of powers of 2 such that all subset sums of exponents are distinct.
    # This is a "superincreasing" sequence or similar.
    # Powers of 2: 1, 2, 4, 8, 16... exponents 0, 1, 2, 3, 4...
    # If we pick numbers 2^a1, 2^a2, ..., 2^ak.
    # Subset products are 2^(sum of ai).
    # We need all subset sums of {a1, ..., ak} to be distinct.
    # This implies {a1, ..., ak} must be a set with distinct subset sums.
    # The smallest such set of size k is {1, 2, 4, ..., 2^(k-1)}.
    # For k=5, {1, 2, 4, 8, 16}. Sum max 31.
    # Numbers: 2^1=2, 2^2=4, 2^4=16, 2^8=256 (too big).
    # So for powers of 2, we can have at most:
    # Exponents: 1, 2, 4. (Sum 7). Numbers: 2, 4, 16.
    # Next exponent 8 -> 256 > 40.
    # So max 3 powers of 2: {2, 4, 16}.
    
    # Let's check {2, 4, 16}.
    # Subsets: 1, 2, 4, 8, 16, 32, 64, 128. Distinct.
    
    # Powers of 3: Exponents 1, 2, 4?
    # 3^1=3, 3^2=9, 3^4=81 (too big).
    # So max 2 powers of 3: {3, 9}.
    # Subsets of exponents {1, 2}: 0, 1, 2, 3. Distinct.
    
    # Powers of 5: Exponents 1, 2?
    # 5^1=5, 5^2=25, 5^4=625 (too big).
    # So max 2 powers of 5: {5, 25}.
    
    # Powers of 7: 7, 49 (too big). Max 1: {7}.
    
    # So we can have:
    # 2s: 2, 4, 16 (3)
    # 3s: 3, 9 (2)
    # 5s: 5, 25 (2)
    # 7: 7 (1)
    # 11: 11 (1)
    # 13: 13 (1)
    # 17: 17 (1)
    # 19: 19 (1)
    # 23: 23 (1)
    # 29: 29 (1)
    # 31: 31 (1)
    # 37: 37 (1)
    
    # Total: 3 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 16.
    
    # Can we do better?
    # Maybe mix primes?
    # If we use 6 (2*3), it introduces cross terms.
    # But 6 is just 2*3.
    # If we have 2, 3, 6:
    # Subsets: 1, 2, 3, 6, 6, 12, 18, 36.
    # {6} and {2,3} both give 6. Collision.
    # So we cannot have 2, 3, and 6 together.
    
    # What if we drop 2 and 3 and use 6?
    # 6, 12, 24?
    # 6=2*3, 12=2^2*3, 24=2^3*3.
    # Exponents of 2: 1, 2, 3. Subset sums: 0,1,2,3,3,4,5,6. Collision (1+2=3, 3=3).
    # So 6, 12, 24 is invalid.
    
    # It seems 16 is a strong candidate for the maximum.
    # The current program finds 16.
    # Is 17 possible?
    # We have 16 numbers: 2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # Leftovers in 2..40:
    # 6, 8, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 27, 28, 30, 33, 34, 35, 36, 38, 39, 40.
    
    # Can we add any of these?
    # 8: 2^3. We have 2 (2^1), 4 (2^2), 16 (2^4).
    # Exponents: 1, 2, 4. Add 3.
    # Set: {1, 2, 3, 4}. Subset sums: 0,1,2,3,3,4,5,6,7. Collision (1+2=3, 3=3).
    # So 8 cannot be added.
    
    # 27: 3^3. We have 3 (3^1), 9 (3^2).
    # Exponents: 1, 2. Add 3.
    # Set: {1, 2, 3}. Subset sums: 0,1,2,3,3,4,5,6. Collision (1+2=3, 3=3).
    # So 27 cannot be added.
    
    # 10: 2*5.
    # We have 2, 5.
    # 10 = 2*5. Collision with {2,5}.
    # So 10 cannot be added.
    
    # 15: 3*5. Collision with {3,5}.
    
    # 14: 2*7. Collision with {2,7}.
    
    # 18: 2*3^2. Collision with {2,9}.
    
    # 20: 2^2*5. Collision with {4,5}.
    
    # 21: 3*7. Collision with {3,7}.
    
    # 22: 2*11. Collision with {2,11}.
    
    # 24: 2^3*3.
    # We have 2, 4, 16, 3, 9.
    # 24 = 2^3 * 3.
    # Is 24 a subset product of {2, 4, 16, 3, 9, ...}?
    # 2 * 4 * 3 = 24. Yes. Collision.
    
    # 26: 2*13. Collision.
    
    # 28: 4*7. Collision.
    
    # 30: 2*3*5. Collision.
    
    # 33: 3*11. Collision.
    
    # 34: 2*17. Collision.
    
    # 35: 5*7. Collision.
    
    # 36: 4*9. Collision.
    
    # 38: 2*19. Collision.
    
    # 39: 3*13. Collision.
    
    # 40: 8*5. 8 is not in set.
    # 40 = 2^3 * 5.
    # Subset product?
    # 2 * 4 * 5 = 40. Yes. Collision.
    
    # So 16 seems to be the maximum.
    
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
