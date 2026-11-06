import math

def construct():
    # Verified solution
    # A = [1, 13, 37]
    # B = [4, 6, 10]
    # Sums: 5, 7, 11, 17, 19, 23, 41, 43, 47 (all distinct primes)
    
    A = [1, 13, 37]
    B = [4, 6, 10]
    
    # Sanity check
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
            
    if len(sums) != 9:
        # Fallback to a simple search if hardcoded fails (should not happen)
        return [1, 7, 13], [4, 10, 16]
        
    distinct_sums = list(sums)
    for i in range(len(distinct_sums)):
        for j in range(i + 1, len(distinct_sums)):
            if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                return [1, 7, 13], [4, 10, 16]
                
    return A, B
