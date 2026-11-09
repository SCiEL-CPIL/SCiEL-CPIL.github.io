from math import gcd
from itertools import combinations

def is_dissociated_reciprocal(subset):
    """
    Checks if all reciprocal subset sums are distinct.
    Equivalent to checking if the set of integers L/a_i is dissociated.
    """
    if not subset:
        return False
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for x in subset:
        L = lcm(L, x)
        
    # Convert to integers
    int_set = [L // x for x in subset]
    
    # Check for any non-trivial signed sum equal to 0
    # Since size is small (<= 13), we can check all subsets
    # Or use the property that if max element > sum of others, it's safe?
    # No, that's for positive sums. For signed sums, we need to be careful.
    # However, for prime powers, the "largest power dominates" argument holds for single prime.
    # For mixed primes, we can just iterate over all subsets and check if sum is 0.
    # Number of subsets 2^13 = 8192. Very fast.
    
    n = len(int_set)
    # We want to check if there exists S, T disjoint subsets such that sum(S) == sum(T)
    # Which is equivalent to sum(S \ T) == sum(T \ S) ? No.
    # sum(S) - sum(T) = 0.
    # This is equivalent to checking if any subset of {v_1, ..., v_n} with coefficients -1, 0, 1 sums to 0.
    # We can just compute all subset sums of the integer set and check for duplicates?
    # No, subset sums are positive.
    # We need to check if any two disjoint subsets have the same sum.
    # This is equivalent to checking if the set of all possible signed sums contains 0 (excluding the empty set).
    
    # Efficient way:
    # Compute all subset sums of int_set.
    # If any sum appears more than once, is it a problem?
    # If sum(A) == sum(B) and A != B, then sum(A \ B) == sum(B \ A).
    # Let C = A \ B and D = B \ A. C and D are disjoint.
    # sum(C) == sum(D).
    # This implies a relation $\sum_{c \in C} c - \sum_{d \in D} d = 0$.
    # So yes, distinct subset sums of the integer set implies dissociation.
    
    sums = [0]
    for v in int_set:
        new_sums = [s + v for s in sums]
        sums += new_sums
        
    # Check for duplicates
    # Since we include the empty sum (0), if 0 appears more than once, it's a problem.
    # But we only add positive numbers, so 0 only appears once (empty set).
    # We need to check if any other sum is duplicated.
    
    if len(sums) != len(set(sums)):
        return False
    return True

def construct():
    # Prime powers up to 22
    prime_powers = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19]
    
    # Try adding 1
    candidate_13 = [1] + prime_powers
    if is_dissociated_reciprocal(candidate_13):
        return candidate_13
        
    # Try other candidates?
    # Maybe swap a small prime power for a composite?
    # Unlikely to increase size if 12 is the limit.
    
    # Verify prime powers
    if is_dissociated_reciprocal(prime_powers):
        return prime_powers
        
    # Fallback
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
