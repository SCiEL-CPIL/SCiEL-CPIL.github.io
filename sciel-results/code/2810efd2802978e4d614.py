"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Since the objective is OneMax (maximize number of 1s) and there are no
    constraints other than the length being N=100, the optimal solution
    is simply a string of all 1s.
    
    This is achieved in constant time using Python's arbitrary-precision
    integer arithmetic: create a mask of 2^N - 1 (all N bits set to 1)
    and convert to binary string.
    """
    # Create an integer with all N bits set to 1
    mask = (1 << N) - 1
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(mask)[2:]
    
    # Ensure the string is exactly N characters long
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
    
    return binary_string
# EVOLVE-BLOCK-END
