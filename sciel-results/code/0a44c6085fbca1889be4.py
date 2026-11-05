<<<<<<< SEARCH
def _verify_booleans_v2(network):
    """
    Vectorized verification using Python big integers.
    Each wire is a big int; bit m represents input m's value on that wire.
    Comparator (i,j): new_i = i & j, new_j = i | j (for 0/1 values).
    """
    total = 1 << N
    mask_all = (1 << total) - 1
    
    # Initialize
    wires = [0] * N
    for k in range(N):
        # Bit m of wire k = (m >> k) & 1
        # Build efficiently
        # For k=0: 010101... pattern
        # For k=1: 00110011...
        # For k=2: 00001111...
        # General: block of 2^k zeros, 2^k ones
        block = 1 << k
        # We can build this iteratively
        val = 0
        # Number of 1s in the full range
        for m in range(block, total, 2 * block):
            # Set bits from m to m+block-1
            bits_to_set = (1 << block) - 1
            val |= bits_to_set << m
        wires[k] = val
    
    # Apply comparators
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    # Check sorted
    for k in range(N - 1):
        if wires[k] & ~wires[k + 1] & mask_all:
            return False
    return True
=======
def _verify_booleans_v2(network):
    """
    Vectorized verification using Python big integers.
    Each wire is a big int; bit m represents input m's value on that wire.
    Comparator (i,j): new_i = i & j, new_j = i | j (for 0/1 values).
    """
    total = 1 << N
    mask_all = (1 << total) - 1
    
    # Initialize
    wires = [0] * N
    for k in range(N):
        # Bit m of wire k = (m >> k) & 1
        # Build efficiently
        block = 1 << k
        val = 0
        # Use a precomputed pattern or iterative shift
        # Pattern for bit k: 0...0 1...1 0...0 1...1
        # Period 2^(k+1).
        # We can construct this by shifting and ORing.
        # Alternatively, just iterate. For N=17, total=131072.
        # The loop runs total/2^(k+1) times.
        # For k=0, 65536 iterations. This is slow if done naively in Python.
        # Let's optimize the initialization.
        
        # A faster way to generate the pattern for bit k:
        # It consists of blocks of 2^k zeros and 2^k ones.
        # We can create a small pattern and repeat it.
        # But Python ints are immutable, so we must build the string or use shifts.
        # Let's stick to the shift method but optimize it.
        
        # Actually, for k=0, the pattern is 010101...
        # We can generate the integer for the full period and repeat it.
        # Period length P = 2^(k+1).
        # Pattern in one period: 0...0 (k+1 bits? No, k zeros, k ones? No.)
        # Bit k is 0 for indices 0..2^k-1, 1 for 2^k..2^(k+1)-1.
        # So in a period of 2^(k+1) bits, the lower 2^k bits are 0, upper 2^k are 1.
        # Value in one period: (2^(2^k) - 1) << 2^k ? No.
        # Bits are indexed from LSB (bit 0) to MSB.
        # Bit m is 1 if (m >> k) & 1 == 1.
        # So for m in [2^k, 2^(k+1)-1], bit is 1.
        # This corresponds to bits 2^k to 2^(k+1)-1 in the integer.
        # So the value for one period is ((1 << (1 << k)) - 1) << (1 << k).
        
        period_val = ((1 << (1 << k)) - 1) << (1 << k)
        period_len = 1 << (k + 1)
        
        # How many periods fit in total?
        num_periods = total // period_len
        
        # Build the full integer by repeating period_val
        val = 0
        # We can shift and OR, or use a string multiplication if we converted to binary string, 
        # but int shifting is likely faster for 131k bits.
        # Actually, repeating a bit pattern in a Python int is not trivial without a loop.
        # Let's use the loop but it might be slow for k=0.
        # For k=0, period_len=2, period_val = 1 << 1 = 2 (binary 10).
        # num_periods = 65536.
        # Loop 65536 times. This is acceptable in Python? ~0.01s?
        # 17 * 65536 ~ 1M operations. Might be ~0.1s.
        # We can optimize by building the string of bits.
        
        # Alternative: Generate the bit string.
        # For k, the bit is 1 if (m >> k) % 2 == 1.
        # Let's try to generate the int directly.
        
        # Efficient generation:
        # The pattern is periodic with period 2^(k+1).
        # We can create the int for the first period and then multiply by (2^period_len - 1) / (2^period_len - 1)? No.
        # Repeating a bit pattern B of length L in an int of length T*L:
        # B * (2^(L*(T-1)) + ... + 1) = B * (2^(L*T) - 1) / (2^L - 1)
        # This works if B is an integer.
        
        if num_periods > 0:
            # Geometric series sum: 1 + 2^L + 2^(2L) + ... + 2^((T-1)L)
            # = (2^(L*T) - 1) / (2^L - 1)
            # Since T*L = total, and L = period_len.
            # Note: 2^L - 1 is a mask of L ones.
            # (2^total - 1) // (2^period_len - 1)
            # This division is exact.
            multiplier = ((1 << total) - 1) // ((1 << period_len) - 1)
            val = period_val * multiplier
            
        wires[k] = val
    
    # Apply comparators
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    # Check sorted
    for k in range(N - 1):
        if wires[k] & ~wires[k + 1] & mask_all:
            return False
    return True
>>>>>>> REPLACE

