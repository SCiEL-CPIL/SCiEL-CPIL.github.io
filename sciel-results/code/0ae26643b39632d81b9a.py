"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Decompose the problem by prime factors.
    2. If we only select prime powers, the condition that all subset products are distinct 
       decomposes into independent conditions for each prime: the set of exponents for each 
       prime must have distinct subset sums.
    3. For each prime p in 2..40, find the available powers p^k <= 40.
    4. For each prime, find the largest subset of exponents {k} such that all subset sums of {k} are distinct.
    5. Combine the selected powers for all primes.
    """
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    selected_numbers = []
    
    for p in primes:
        # Find available powers p^k <= 40
        powers = []
        k = 1
        val = p
        while val <= 40:
            powers.append(k)
            k += 1
            val *= p
            
        # Find max subset of powers with distinct subset sums
        # Since the list of powers is small (max 5 for p=2), we can brute force
        max_size = 0
        best_subset = []
        
        n = len(powers)
        # Iterate over all subsets of powers
        for mask in range(1 << n):
            subset = [powers[i] for i in range(n) if (mask >> i) & 1]
            size = len(subset)
            if size <= max_size:
                continue
                
            # Check if subset sums are distinct
            sums = set()
            valid = True
            # Calculate all 2^size subset sums
            for smask in range(1 << size):
                s = 0
                for j in range(size):
                    if (smask >> j) & 1:
                        s += subset[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
                
            if valid:
                max_size = size
                best_subset = subset
                
        # Add the corresponding numbers to the result
        for k in best_subset:
            selected_numbers.append(p ** k)
            
    # Ensure types are correct and return
    return [int(x) for x in selected_numbers]
# EVOLVE-BLOCK-END
