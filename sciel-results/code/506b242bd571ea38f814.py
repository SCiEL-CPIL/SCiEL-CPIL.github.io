"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    result = []
    for p in primes:
        # Find max k such that p^(2^(k-1)) <= 40 is not quite right.
        # We want a superincreasing sequence of exponents 1, 2, 4, ...
        # Max exponent E_max = floor(log_p(40))
        # Max size k is such that 2^(k-1) <= E_max
        # So k = floor(log2(E_max)) + 1
        
        import math
        E_max = 0
        val = p
        while val <= P['n']:
            E_max += 1
            val *= p
            
        if E_max == 0:
            continue
            
        # Max k
        k = 1
        while (1 << (k-1)) <= E_max:
            k += 1
        k -= 1 # last valid k
        
        # Add p^(2^i) for i=0..k-1
        for i in range(k):
            exp = 1 << i
            num = p ** exp
            if num <= P['n']:
                result.append(num)
                
    return sorted(result)
# EVOLVE-BLOCK-END
