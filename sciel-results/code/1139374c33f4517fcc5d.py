"""Program for OneMax 100-bit direct binary construction.

Uses a constructive approach that explicitly assembles the string from
individual bits, validating length and content before returning.
"""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string that maximizes the number of ones.
    
    The OneMax objective is maximized when every bit is set to 1.
    We build the string explicitly to ensure correctness and clarity.
    
    Returns:
        str: A binary string of length N consisting entirely of '1's.
    """
    # Build the string by joining individual '1' characters
    bits = ['1' for _ in range(N)]
    result = ''.join(bits)
    
    # Optional: validate the result (defensive programming)
    assert len(result) == N, f"Expected length {N}, got {len(result)}"
    assert all(c == '1' for c in result), "Result contains non-'1' characters"
    
    return result
# EVOLVE-BLOCK-END
