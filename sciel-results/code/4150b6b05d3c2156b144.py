"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Add safe composites (powers with gaps > 1)
    # Powers of 2: 4 (2^2), 16 (2^4). Gap 2. Safe.
    # Powers of 3: 9 (3^2). 27 is 3^3, gap 1 from 9. Pick 9.
    # Powers of 5: 25 (5^2).
    # Powers of 7: 49 > 40.
    
    composites = [4, 16, 9, 25]
    
    result = primes + composites
    
    # Verify distinct subset products
    # This is a small set (16 items), so 2^16 = 65536 products.
    # We can verify quickly to be safe.
    from itertools import combinations
    
    def verify(nums):
        seen = set()
        for i in range(len(nums) + 1):
            for c in combinations(nums, i):
                prod = 1
                for x in c:
                    prod *= x
                if prod in seen:
                    return False
                seen.add(prod)
        return True
    
    if not verify(result):
        # Fallback to primes only if verification fails (should not happen)
        return primes
        
    return result
# EVOLVE-BLOCK-END
