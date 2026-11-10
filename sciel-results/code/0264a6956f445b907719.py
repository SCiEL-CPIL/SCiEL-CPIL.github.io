"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The goal is to find the largest subset of distinct integers from 2 to 40
    # such that all 2^k subset products are distinct.
    
    # Mathematical Insight:
    # A set of numbers {a_1, ..., a_k} has distinct subset products if and only if
    # no non-empty subset of them has a product of 1 (which is impossible for positive integers > 1)
    # and, more importantly, no two distinct subsets have the same product.
    # This is equivalent to the exponent vectors of the numbers (with respect to prime bases) being
    # "linearly independent" over the integers in a specific sense (subset sums of vectors are unique).
    # 
    # If we choose numbers that are powers of distinct primes, say {p_1^{e_1}, p_2^{e_2}, ..., p_k^{e_k}},
    # then any subset product corresponds to a unique vector of exponents (k_1, ..., k_k) where k_i is 0 or e_i.
    # Since the primes are distinct, the prime factorization of any subset product is unique.
    # Thus, if two subset products are equal, their prime factorizations must be identical, implying the same subsets.
    # Therefore, a set of powers of distinct primes always has distinct subset products.
    
    # To maximize the size, we should pick as many numbers as possible.
    # The numbers must be distinct and in the range [2, 40].
    # We can pick multiple powers for the same prime as long as they are distinct integers.
    # Let's analyze which primes have multiple powers in [2, 40]:
    # Prime 2: 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32. (5 numbers)
    # Prime 3: 3^1=3, 3^2=9, 3^3=27. (3 numbers)
    # Prime 5: 5^1=5, 5^2=25. (2 numbers)
    # Prime 7: 7^1=7, 7^2=49 (>40). (1 number)
    # Prime 11: 11^1=11. (1 number)
    # ...
    
    # However, we must be careful. If we pick {2, 4}, the vectors are (1,0,...) and (2,0,...).
    # Subset sums of exponents: 0, 1, 2, 3. Distinct.
    # If we pick {2, 4, 8}, vectors (1,0), (2,0), (3,0). Subset sums: 0,1,2,3, 1+2=3 (collision with 3? No, 8 is 2^3, vector (3,0)).
    # Wait, the condition is that the *sum* of exponent vectors for any subset must be unique.
    # For {2, 4, 8}:
    # Subsets:
    # {} -> (0)
    # {2} -> (1)
    # {4} -> (2)
    # {8} -> (3)
    # {2,4} -> (3)  <-- Collision! 2*4 = 8. Product is 8. Subset {8} also gives product 8.
    # So we cannot pick 2, 4, and 8 together.
    # 
    # Let's re-evaluate. The condition is: no two distinct subsets have the same product.
    # If we have powers of a single prime p: p^{e_1}, p^{e_2}, ..., p^{e_m}.
    # A subset corresponds to a sum of a subset of {e_1, ..., e_m}.
    # We need all subset sums to be distinct.
    # This is a "distinct subset sum" problem for the exponents.
    # To maximize the number of exponents we can pick from {1, 2, 3, 4, 5} (for p=2), we should pick the largest possible set with distinct subset sums.
    # A greedy strategy for distinct subset sums is to pick powers of 2: 1, 2, 4. (Sums: 0,1,2,3, 1+2=3? No, 1,2,4 -> sums: 0,1,2,4, 1+2=3, 1+4=5, 2+4=6, 1+2+4=7. All distinct 0..7).
    # So for p=2, we can pick exponents {1, 2, 4} corresponding to numbers {2, 4, 16}.
    # Can we add more? Next exponent is 5. If we add 5, subset sums include 5, 1+5=6, 2+5=7, 4+5=9, etc.
    # Check collisions with existing {0,1,2,3,4,5,6,7}.
    # 5 is new. 1+5=6 (collision with 2+4=6? No, 2+4 corresponds to exponents 2 and 4, sum 6. 1+5 is sum 6. Collision!).
    # So {1,2,4,5} is invalid.
    # What about {1,2,4}? Size 3.
    # What about {1,3,4}? Sums: 0,1,3,4, 1+3=4 (collision).
    # What about {2,3,4}? Sums: 0,2,3,4, 2+3=5, 2+4=6, 3+4=7, 2+3+4=9. Distinct?
    # 0, 2, 3, 4, 5, 6, 7, 9. Yes, distinct. Size 3.
    # Can we get size 4 for p=2?
    # Max exponent is 5. We need 4 distinct exponents from {1,2,3,4,5} with distinct subset sums.
    # Try {1,2,3,4}: 1+2=3 (collision).
    # Try {1,2,3,5}: 1+2=3 (collision).
    # Try {1,2,4,5}: 1+4=5 (collision with 5? No, 5 is an element. 1+4=5, so subset {1,4} has sum 5, and subset {5} has sum 5. Collision).
    # Try {1,3,4,5}: 1+4=5 (collision).
    # Try {2,3,4,5}: 2+3=5 (collision).
    # It seems max size for p=2 is 3.
    
    # For p=3, exponents available: {1, 2, 3} (3, 9, 27).
    # Max size for distinct subset sums?
    # {1,2}: sums 0,1,2,3. Distinct. Size 2.
    # {1,2,3}: 1+2=3 (collision).
    # So max size for p=3 is 2. Numbers: {3, 9} or {3, 27} or {9, 27}?
    # {3, 9} -> exponents {1, 2}.
    # {9, 27} -> exponents {2, 3}. Sums: 0,2,3,5. Distinct.
    # {3, 27} -> exponents {1, 3}. Sums: 0,1,3,4. Distinct.
    # Any pair works.
    
    # For p=5, exponents: {1, 2} (5, 25). Size 2.
    # For p=7, exponents: {1} (7). Size 1.
    # For p=11, exponents: {1} (11). Size 1.
    # For p=13, exponents: {1} (13). Size 1.
    # For p=17, exponents: {1} (17). Size 1.
    # For p=19, exponents: {1} (19). Size 1.
    # For p=23, exponents: {1} (23). Size 1.
    # For p=29, exponents: {1} (29). Size 1.
    # For p=31, exponents: {1} (31). Size 1.
    # For p=37, exponents: {1} (37). Size 1.
    
    # We can combine these sets because the primes are distinct.
    # Total count:
    # p=2: 3 numbers (e.g., 2, 4, 16)
    # p=3: 2 numbers (e.g., 3, 9)
    # p=5: 2 numbers (5, 25)
    # p=7: 1 number (7)
    # p=11: 1 number (11)
    # p=13: 1 number (13)
    # p=17: 1 number (17)
    # p=19: 1 number (19)
    # p=23: 1 number (23)
    # p=29: 1 number (29)
    # p=31: 1 number (31)
    # p=37: 1 number (37)
    
    # Total = 3 + 2 + 2 + 1*8 = 15.
    
    # Can we do better?
    # We have 13 primes <= 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. (Wait, 41 is not <= 40).
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. (12 primes).
    # My count: 3+2+2 + 9*1 = 15.
    # Primes used: 2,3,5,7,11,13,17,19,23,29,31,37.
    # Wait, I listed 8 singletons after 5.
    # Primes > 5 and <= 40: 7, 11, 13, 17, 19, 23, 29, 31, 37. (9 primes).
    # So 3 (for 2) + 2 (for 3) + 2 (for 5) + 9 (for others) = 16.
    
    # Let's verify the set of 16:
    # S = {2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    # All are distinct integers in 2..40.
    # Subset products are distinct because:
    # - For each prime p, the exponents chosen from its powers have distinct subset sums.
    # - The primes are distinct, so the factorizations are unique.
    
    # Is it possible to get 17?
    # We need to extract more from some primes.
    # p=2 max size 3.
    # p=3 max size 2.
    # p=5 max size 2.
    # Others max size 1.
    # Total 16 seems to be the maximum if we restrict to prime powers.
    
    # Can we mix non-prime-powers?
    # E.g., 6 = 2*3.
    # If we add 6, its vector is (1,1,0,...).
    # We already have 2 (1,0,...) and 3 (0,1,...).
    # 2*3 = 6. So {2,3} product is 6. If we add 6, we have a collision with {2,3}.
    # So we cannot add 6 if 2 and 3 are present.
    # Generally, composite numbers introduce dependencies that reduce the maximum size.
    # Prime powers are the "safest" choice for maximizing cardinality while maintaining independence.
    
    # Therefore, 16 is likely the maximum.
    
    result = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
