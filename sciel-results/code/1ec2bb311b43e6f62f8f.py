import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    Strategy:
    1. Define a theoretical continuous relaxation of the OneMax objective where 
       each bit $x_i \in [0, 1]$. The objective is $f(x) = \sum x_i$.
    2. The global maximum of this smooth surrogate is at $x_i = 1$ for all $i$.
    3. Project this continuous optimum onto the binary lattice $\{0, 1\}^{100}$ 
       using a deterministic bitwise OR operation. This acts as a final 
       high-precision polish step, ensuring exact integer representation.
    4. Convert the resulting integer to a binary string.
    
    An explicit deadline guard is implemented to ensure the function returns 
    the best result found (which is the global optimum) well before the 
    150s wall-clock limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5-second safety margin
    
    # Step 1 & 2: The continuous optimum is the vector of ones.
    # In integer arithmetic, this corresponds to an integer with the lowest N bits set.
    # mask = 2^N - 1
    mask = (1 << N) - 1
    
    # Step 3: Deterministic bitwise OR construction.
    # We model the construction as OR-ing a zero-initialized integer with the mask.
    # This leverages Python's arbitrary-precision integer arithmetic.
    constructed_val = 0 | mask
    
    # Check deadline (should always pass for constant-time ops, but required by directive)
    if time.time() > deadline:
        # If we somehow exceeded the deadline (impossible here), return the current best.
        # In a search context, 'constructed_val' would be the best found so far.
        pass
        
    # Step 4: Convert to binary string.
    # bin() returns a string like '0b111...1'. We strip the prefix.
    # The length of the resulting string is exactly N because the highest bit is set.
    result = bin(constructed_val)[2:]
    
    # Final check before return
    if time.time() > deadline:
        # Fallback safety net, though not expected to trigger
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END
