"""OneMax direct binary construction (100 bits)."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy: Use bytes translation to generate the optimal string of all ones.
    This bypasses standard string multiplication in favor of a byte-level operation,
    as per the directive to test this specific hypothesis.
    
    The global optimum for OneMax is a string of length N consisting entirely of '1's.
    """
    start_time = time.time()
    # Safety margin: return before the 120s limit (using 110s as a conservative guard)
    deadline = start_time + 110
    
    try:
        # Generate 100 null bytes
        buf = bytes(N)
        
        # Create a translation table mapping \x00 to \x31 ('1')
        # bytes.maketrans(from, to) creates a table for bytes.translate()
        trans_table = bytes.maketrans(b'\x00', b'1')
        
        # Apply translation: replaces all \x00 with '1'
        byte_string = buf.translate(trans_table)
        
        # Convert to a standard ASCII string
        result = byte_string.decode('ascii')
        
        # Verify length and content integrity (cheap check)
        if len(result) != N:
            raise ValueError("Construction failed: incorrect length")
            
        # Deadline check (should be instantaneous)
        if time.time() > deadline:
            # Fallback to simple multiplication if somehow delayed
            return "1" * N
            
        return result

    except Exception:
        # Fallback to the most basic, guaranteed correct method
        return "1" * N
# EVOLVE-BLOCK-END
