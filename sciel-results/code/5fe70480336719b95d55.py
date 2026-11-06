"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Precompute Euler's totient function for all integers up to 1,000,000 using a sieve.
    # This ensures we can find the smallest n with phi(n) divisible by 120 efficiently.
    N = 1000000
    # Initialize phi array where phi[i] = i for all i
    phi_arr = list(range(N + 1))
    
    # Sieve to compute totients
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_arr[j] -= phi_arr[j] // i
    
    # Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120
    for n in range(1, N + 1):
        if phi_arr[n] % 120 == 0:
            return n
    
    # Fallback (should not be reached)
    return 1
# EVOLVE-BLOCK-END
