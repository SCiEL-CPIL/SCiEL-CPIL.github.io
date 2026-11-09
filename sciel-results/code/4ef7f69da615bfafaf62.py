"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    Strategy:
    1. Create a 128-bit integer with all bits set to 1 (0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF).
       This is equivalent to (1 << 128) - 1.
    2. Extract the lower 100 bits. Since the integer is all ones, the lower 100 bits are also all ones.
    3. Format this integer as a binary string with zero-padding to 100 characters.
    
    This leverages arbitrary-precision integer arithmetic as a distinct computational pathway 
    from string repetition.
    
    The optimal solution is all ones: "1" * 100.
    Score: 100 (maximum possible).
    """
    # Create an integer with 128 bits set to 1
    # (1 << 128) - 1 is a 128-bit number with all bits set to 1
    full_int = (1 << 128) - 1
    
    # We need the lower 100 bits. 
    # Since full_int is all ones, full_int & ((1 << 100) - 1) is just (1 << 100) - 1.
    # However, to strictly follow the "extract lower 100 bits" logic even if the integer wasn't all ones:
    mask = (1 << N) - 1
    target_int = full_int & mask
    
    # Format as a binary string of length N (100), padding with zeros on the left.
    # Since target_int is 2^100 - 1, it has 100 bits, all set to 1.
    # format(target_int, '0100b') will produce '111...1' (100 times).
    return format(target_int, f'0{N}b')
# EVOLVE-BLOCK-END
