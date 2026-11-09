"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0
    
    def compute_score(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        # R[0] is the sum of squares, which is N for +/-1 sequences.
        # We sum the squares of the non-zero lags.
        return np.sum(R[1:]**2)

    def generate_m_sequence(n, poly, init_state=1):
        """
        Generate an m-sequence of length 2^n - 1.
        poly is an integer representing the taps of the LFSR (excluding the output bit).
        e.g. for n=8, poly=0x49 (1001001) means taps at 8 and 6? 
        Standard representation: polynomial x^n + ... 
        Let's use a known primitive polynomial for each n.
        """
        L = (1 << n) - 1
        seq = np.zeros(L, dtype=np.float64)
        state = init_state
        
        # We need to define the feedback taps.
        # For n=9, primitive poly x^9 + x^4 + 1 -> taps 9, 4. 
        # In LFSR implementation, if we shift left, the new bit is xor of bits at specific positions.
        # Let's use a standard table of primitive polynomials.
        # n=9: x^9 + x^4 + 1 (taps 9, 4) -> 0x011? No, bits.
        # Let's just use a few known good ones.
        
        # For simplicity and robustness, I will use a few hardcoded primitive polynomials for n=9 (511) and n=10 (1023).
        # n=9: x^9 + x^4 + 1. Taps: 9, 4.
        # n=10: x^10 + x^3 + 1. Taps: 10, 3.
        
        if n == 9:
            taps = [9, 4]
        elif n == 10:
            taps = [10, 3]
        elif n == 11:
            taps = [11, 9, 5, 2, 1] # x^11 + x^9 + x^5 + x^2 + 1
        else:
            taps = [n, 1] # Fallback, might not be primitive
            
        for i in range(L):
            seq[i] = 1.0 if (state & 1) == 0 else -1.0
            # LFSR step
            # Bit 0 is the output.
            # New bit is XOR of bits at (tap - 1) positions? 
            # Convention: If poly is x^n + c_{n-1}x^{n-1} + ... + c_0
            # Feedback is sum of bits at positions i where c_i=1.
            # Let's stick to a simple implementation:
            # State is n bits. Output is bit 0.
            # Next state: shift left, new MSB is feedback.
            # Feedback = XOR of bits at indices (tap-1) for each tap in taps.
            
            feedback = 0
            for t in taps:
                # The tap 't' corresponds to x^t. In the polynomial x^n + ..., 
                # the coefficient of x^{t-1} is what we check?
                # Standard LFSR: 
                # If poly is x^9 + x^4 + 1, taps are 9 and 4.
                # The bit fed back is XOR of bit 8 and bit 3 (0-indexed from LSB=0? or MSB=0?)
                # Let's assume standard Fibonacci LFSR.
                # Bit i of state.
                # If tap is k, we take bit k-1.
                feedback ^= (state >> (t - 1)) & 1
                
            state = ((state << 1) & ((1 << n) - 1)) | feedback
            
        return seq

    candidates = []
    
    # Generate m-sequences of different lengths
    # n=9 -> L=511
    # n=10 -> L=1023
    # n=11 -> L=2047
    
    for n in [9, 10, 11]:
        seq = generate_m_sequence(n, 0) # poly arg ignored in current gen
        
        # We need a window of length N=400.
        # Try different offsets in the periodic sequence.
        # Since it's periodic, we can just take slices.
        # To avoid boundary artifacts from the truncation itself, we might want to 
        # center the window or just take the first N.
        # The "boundary condition optimization" will fix the ends.
        
        # Try a few offsets to find the best "bulk" autocorrelation
        num_offsets = 10
        for off in range(num_offsets):
            # Create a longer sequence to allow wrapping if needed, or just modulo index
            # Since L > N for n=9,10,11, we can just take a slice.
            # But if we shift, we might wrap around.
            # Let's just take the first N for simplicity, as the m-sequence is stationary in stats.
            # Actually, let's try shifting the window within the period.
            
            # Create a view of the sequence repeated to ensure we have enough data
            # Or just use modulo indexing.
            a = np.array([seq[(off + i) % len(seq)] for i in range(N)])
            
            score = compute_score(a)
            candidates.append((score, a.copy(), n, off))
            
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    best_score, best_a, best_n, best_off = candidates[0]
    
    # Now, perform targeted optimization on the boundaries of the best candidate
    # The first and last 20 intervals are most affected by the truncation discontinuity.
    # We will do a greedy local search on these regions.
    
    a = best_a.copy()
    score = best_score
    
    # Define regions to optimize: [0, 20) and [380, 400)
    boundary_indices = list(range(0, 20)) + list(range(N - 20, N))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate through boundary indices
        for idx in boundary_indices:
            if (time.time() - start_time) > time_limit:
                break
            
            # Try flipping
            a[idx] *= -1.0
            new_score = compute_score(a)
            
            if new_score < score:
                score = new_score
                improved = True
            else:
                # Revert
                a[idx] *= -1.0
                
    # If time remains, expand to all indices (global greedy)
    if (time.time() - start_time) < time_limit:
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for idx in range(N):
                if (time.time() - start_time) > time_limit:
                    break
                
                a[idx] *= -1.0
                new_score = compute_score(a)
                
                if new_score < score:
                    score = new_score
                    improved = True
                else:
                    a[idx] *= -1.0
                    
    return a
# EVOLVE-BLOCK-END
