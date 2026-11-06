"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute Euler's totient function phi(n) for all n in [1, 1000000] using a sieve.
    # phi(n) is initialized to n.
    # For each prime p, we update phi(j) for all multiples j of p by subtracting phi(j) // p.
    N = 1000000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n such that phi(n) is divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 1
# EVOLVE-BLOCK-END
