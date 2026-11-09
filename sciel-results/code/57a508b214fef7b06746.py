import numpy as np

N = 400
h = 1.0 / (2 * N)  # interval width: total length 1/2, so h = (1/2)/400 = 1/800

def autocorr_peak(f):
    """Compute max |autocorrelation| for step function f on [-1/4, 1/4]."""
    R = np.correlate(f, f, mode='full') * h
    return np.max(np.abs(R))

def generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1."""
    length = (1 << m) - 1
    # Use a primitive polynomial. For m=9, x^9 + x^5 + 1 is primitive.
    # Taps: 9 and 5 (1-indexed, LSB is 1).
    # Let's use a known primitive polynomial for m=9: 0x119 (x^9 + x^5 + 1) or similar.
    # Actually, let's just use a generic LFSR implementation.
    # Polynomial: x^9 + x^4 + x^3 + x + 1 is also common? Let's check a table.
    # For m=9, primitive polynomial: x^9 + x^5 + 1 (hex 0x024 if LSB is x^0? No, standard is x^m + ... + 1)
    # Let's use the polynomial x^9 + x^5 + 1.
    # Feedback tap at bit 9 (index 8 if 0-indexed LSB) and bit 5 (index 4).
    # Wait, standard LFSR: next_bit = XOR of tapped bits.
    # Let's use a reliable m-sequence generator.
    
    # Simpler: Use a known good seed for numpy random? No, that's random.
    # Let's implement LFSR.
    # Polynomial: 0x119 is for m=9? 
    # Let's use the polynomial x^9 + x^4 + x^3 + x + 1 (0x11B in some conventions).
    # To be safe, let's just use a long period PRNG but fixed seed, which is effectively a deterministic sequence, 
    # but the prompt says "do not reuse current designs". Current uses random. 
    # A specific LFSR is a deterministic mathematical construction.
    
    # Let's use m=9, length 511.
    # Primitive polynomial: x^9 + x^5 + 1.
    # Taps: 9, 5.
    # State: 9 bits.
    # Initial state: all 1s.
    
    state = (1 << 9) - 1  # 111111111
    seq = []
    for _ in range(length):
        # Output bit is LSB
        out = state & 1
        seq.append(1.0 if out == 1 else -1.0)
        # Next state
        # Feedback bit = XOR of bits at positions 9 and 5 (1-indexed from LSB, so indices 8 and 4)
        # Wait, polynomial x^9 + x^5 + 1 means feedback from x^9 and x^5 to x^0?
        # Standard Fibonacci LFSR:
        # new_bit = s[8] XOR s[4]  (if taps are 9 and 5)
        fb = ((state >> 8) ^ (state >> 4)) & 1
        state = (state >> 1) | (fb << 8)
        
    return np.array(seq)

def construct():
    # 1. Initialize with m-sequence (deterministic, good autocorrelation properties)
    m_seq = generate_m_sequence(9)
    f = m_seq[:N].copy()
    
    current_peak = autocorr_peak(f)
    
    # 2. Local Search (Hill Climbing)
    # Try to improve by flipping individual bits
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Flip bit i
            f[i] *= -1.0
            new_peak = autocorr_peak(f)
            
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                # Revert
                f[i] *= -1.0
                
    return f
