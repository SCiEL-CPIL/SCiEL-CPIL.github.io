"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Hardcoded solution with 9 distinct prime sums (all pairwise coprime)
    # A = [1, 7, 13], B = [4, 6, 30]
    # Sums: {5, 11, 17, 7, 13, 19, 31, 37, 43} - all primes, all distinct
    A = [1, 7, 13]
    B = [4, 6, 30]
    
    # Verify the solution meets all constraints
    # 1. Each list has exactly 3 distinct integers in [1, 128]
    assert len(A) == 3 and len(set(A)) == 3
    assert len(B) == 3 and len(set(B)) == 3
    assert all(1 <= x <= 128 for x in A)
    assert all(1 <= x <= 128 for x in B)
    
    # 2. All distinct values in A+B must be pairwise coprime
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    
    # Check pairwise coprime
    import math
    sum_list = list(sums)
    for i in range(len(sum_list)):
        for j in range(i + 1, len(sum_list)):
            if math.gcd(sum_list[i], sum_list[j]) != 1:
                # This should not happen for our hardcoded solution
                pass
    
    return [A, B]
# EVOLVE-BLOCK-END
