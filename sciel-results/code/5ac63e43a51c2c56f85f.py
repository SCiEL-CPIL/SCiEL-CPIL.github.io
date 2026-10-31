import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use prime weights to create a linear projection of the vector.
    # This prioritizes vectors based on a weighted sum of their coordinates,
    # which creates a different ordering than a cryptographic hash.
    # Primes are used to ensure that different coordinate patterns map to
    # distinct values with high probability (Schur's theorem related intuition).
    
    # Generate first n primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    if n > len(primes):
        # Fallback to simple powers of 2 if n is very large, or generate more
        # For now, assume n is small enough. If not, we can extend.
        # Let's just use a simple loop to generate more if needed, or modulo.
        # But standard cap set n is small.
        pass

    total = 0
    for i in range(n):
        # Ensure we have enough primes
        if i < len(primes):
            w = primes[i]
        else:
            # Fallback: use a large prime-like number
            w = 1000003 + i * 7 
        total += vector[i] * w
        
    return float(total)
# EVOLVE-BLOCK-END
