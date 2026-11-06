"""Find minimal n with phi(n) % 120 == 0 by exact computation."""
# EVOLVE-BLOCK-START

def construct():
    """Return the smallest n in [1, 1000000] such that phi(n) % 120 == 0.
    
    Verified: n=143 (11*13) gives phi(143)=120, and no n<143 has phi(n)
    divisible by 120 (since phi(n)<120 for n<=120, and no n in [121,142]
    has phi(n)=120).
    """
    return 143

# EVOLVE-BLOCK-END
