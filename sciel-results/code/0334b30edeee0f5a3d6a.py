import math
import itertools
import time

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All distinct values in A+B are pairwise coprime.
    Maximizes the cardinality of A+B (target: 9).
    """
    
    # 1. Sieve of Eratosthenes for primes up to 256 (max sum 128+128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # 2. Build adjacency lists (neighbor sets) for each a in [1, 128]
    # neighbors[a] is a set of b in [1, 128] such that a+b is prime
    neighbors = [None] * 129
    for a in range(1, 129):
        nbrs = set()
        for b in range(1, 129):
            if is_prime[a + b]:
                nbrs.add(b)
        neighbors[a] = nbrs

    # 3. Search for A (size 3) such that the intersection of their neighbors has size >= 3
    # We iterate over combinations of A to find a valid B.
    # To ensure we find a solution quickly, we can bias towards smaller numbers
    # which tend to have more prime sum partners, or just iterate in order.
    
    # Time guard
    start_time = time.time()
    TIME_LIMIT = 85.0
    
    # Iterate over all combinations of 3 distinct elements for A
    # 128 choose 3 is ~350k, which is very fast.
    for A in itertools.combinations(range(1, 129), 3):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Calculate intersection of neighbors for a1, a2, a3
        common_neighbors = neighbors[A[0]] & neighbors[A[1]] & neighbors[A[2]]
        
        if len(common_neighbors) >= 3:
            # We found a valid A. Now pick any 3 distinct elements from common_neighbors for B.
            # Sort to be deterministic
            B_candidates = sorted(common_neighbors)
            B = B_candidates[:3]
            
            # Verify validity (safety check)
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            
            if len(distinct_sums) == 9:
                # Check pairwise coprimality (should be true as they are primes)
                valid = True
                ds_list = list(distinct_sums)
                for i in range(9):
                    for j in range(i + 1, 9):
                        if math.gcd(ds_list[i], ds_list[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return list(A), B
            
    # Fallback to a known valid solution if search fails or times out
    # A = [1, 13, 37], B = [4, 6, 10] -> Sums: 5,7,11,17,19,23,41,43,47 (all primes)
    return [1, 13, 37], [4, 6, 10]
