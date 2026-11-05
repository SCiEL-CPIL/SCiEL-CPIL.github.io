"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 512
    
    # The set of divisors of a number M <= N is a valid subset because 
    # lcm(a, b) divides M, so lcm(a, b) <= M <= N.
    # To maximize the size, we choose M <= N with the maximum number of divisors.
    # For N=512, the maximum number of divisors is 24 (e.g., for M=504, 480, 420, 360).
    
    # Let's pick M=504.
    M = 504
    
    # Generate all divisors of M
    divisors = []
    for i in range(1, int(M**0.5) + 1):
        if M % i == 0:
            divisors.append(i)
            if i != M // i:
                divisors.append(M // i)
    
    # The divisors form a valid set.
    # We can try to see if we can add more numbers, but 24 is a strong lower bound.
    # Given the previous poor performance, this simple construction is a significant improvement.
    
    return divisors
# EVOLVE-BLOCK-END
