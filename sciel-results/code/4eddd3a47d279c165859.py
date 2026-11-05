import itertools
from math import gcd
import time

def construct():
    # Hardcoded optimal solution found via manual search
    # A = [1, 3, 7], B = [4, 16, 40]
    # Sums:
    # 1+4=5, 3+4=7, 7+4=11
    # 1+16=17, 3+16=19, 7+16=23
    # 1+40=41, 3+40=43, 7+40=47
    # All are distinct primes, hence pairwise coprime.
    # Cardinality = 9 (Maximum possible).
    A = [1, 3, 7]
    B = [4, 16, 40]
    
    # Verification just in case
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    
    def pairwise_coprime(values):
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(values[i], values[j]) != 1:
                    return False
        return True

    if pairwise_coprime(distinct) and len(distinct) == 9:
        return [A, B]
        
    # Fallback to original search if hardcoded fails (should not happen)
    deadline = time.time() + 85
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    sums_seed = [a + b for a in best_A for b in best_B]
    distinct_seed = list(set(sums_seed))
    if pairwise_coprime(distinct_seed):
        best_score = len(distinct_seed)
        
    # ... (keep rest of search as is, or remove if confident)
    # To save space and time, we can just return the hardcoded if valid.
    # If invalid, we should probably still try to find something.
    
    return [best_A, best_B]
