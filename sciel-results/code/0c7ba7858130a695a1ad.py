"""
Program to construct a set of integers in [2, 40] with distinct subset products.
Approach: Decompose the problem by prime factors. For each prime, select a subset
of its powers such that the exponents have distinct subset sums. This ensures
multiplicative independence.
"""

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # For each prime, find the best subset of powers <= 40
    # such that exponents have distinct subset sums.
    # We can hardcode the optimal selections based on analysis.
    
    # Prime 2: Powers 2, 4, 8, 16, 32 (exponents 1, 2, 3, 4, 5)
    # Optimal exponents: {1, 2, 4} -> Numbers: 2, 4, 16
    # Check: Sums of {1,2,4} are 0..7, distinct.
    sel_2 = [2, 4, 16]
    
    # Prime 3: Powers 3, 9, 27 (exponents 1, 2, 3)
    # Optimal exponents: {1, 2} -> Numbers: 3, 9
    # Check: Sums of {1,2} are 0,1,2,3, distinct.
    sel_3 = [3, 9]
    
    # Prime 5: Powers 5, 25 (exponents 1, 2)
    # Optimal exponents: {1, 2} -> Numbers: 5, 25
    sel_5 = [5, 25]
    
    # Prime 7: Power 7
    sel_7 = [7]
    
    # Primes 11, 13, 17, 19, 23, 29, 31, 37
    # Only first power available.
    sel_others = [11, 13, 17, 19, 23, 29, 31, 37]
    
    result = sel_2 + sel_3 + sel_5 + sel_7 + sel_others
    
    # Verify distinctness of subset products (optional but good for safety)
    # Since we constructed it via independent prime exponents with distinct sums,
    # it is guaranteed.
    
    return result
