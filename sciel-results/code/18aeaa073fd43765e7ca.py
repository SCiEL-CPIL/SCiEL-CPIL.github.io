"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need the smallest n such that phi(n) is divisible by 120.
    # Based on analysis, n=143 is a candidate (phi(143)=120).
    # We will compute phi(n) for n up to a limit to be sure.
    # 143 is small, so a limit of 1000 is safe and fast.
    limit = 1000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i: # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    # Fallback if not found (should not happen for this problem size)
    return 143
# EVOLVE-BLOCK-END
