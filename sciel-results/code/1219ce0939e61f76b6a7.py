"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimal such n is 143 = 11 * 13.
    Verification:
    - phi(143) = phi(11) * phi(13) = 10 * 12 = 120
    - 120 is divisible by 120
    - No smaller n in [1, 142] has phi(n) divisible by 120
    
    This implementation uses a direct constructive search to verify the minimal n.
    """
    def phi(n):
        """Compute Euler's totient function."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    # Search for the smallest n
    # We know it's small, so we don't need to go very far, 
    # but we set a reasonable upper bound to be safe.
    for n in range(1, 10000):
        if phi(n) % 120 == 0:
            return n
            
    # Fallback (should not be reached)
    return 143
# EVOLVE-BLOCK-END
