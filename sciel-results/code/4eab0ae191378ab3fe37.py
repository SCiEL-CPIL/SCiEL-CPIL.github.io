# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_bound(h):
    """
    Compute the upper bound on C3 for a signed step function h of length N.
    The bound is related to the maximum absolute value of the off-center autocorrelation.
    """
    n = len(h)
    if n == 0:
        return 0.0
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, size)
    power = F * np.conj(F)
    corr = np.fft.irfft(power, size)[:n]
    if n > 1:
        return np.max(np.abs(corr[1:]))
    return 0.0

def _lfsr_sequence(n, poly, init_state):
    """
    Generate an m-sequence using a Linear Feedback Shift Register (LFSR).
    poly: integer representing the polynomial (e.g., 0b100001001 for x^8+x^4+x^3+x^2+1)
    init_state: initial state (non-zero)
    Returns a sequence of +1 and -1 of length n.
    """
    # Determine degree from poly
    degree = poly.bit_length() - 1
    mask = (1 << degree) - 1
    
    state = init_state & mask
    seq = []
    
    for _ in range(n):
        # Output bit
        bit = state & 1
        seq.append(1.0 if bit else -1.0)
        
        # Feedback
        # The feedback bit is the XOR of the bits at the tap positions.
        # The taps are determined by the non-zero coefficients of the polynomial,
        # excluding the leading x^degree term.
        # For a polynomial x^d + sum a_i x^i, the feedback is sum a_i s_i.
        # In binary representation of poly, bit i is set if x^i is a term.
        # We need to XOR the bits of the state at positions i where poly has a 1.
        # Note: The state bits are s_0 ... s_{d-1}. The feedback is s_{d-1} XOR ...
        # Standard Galois or Fibonacci? Let's use Fibonacci.
        # Recurrence: s_{k} = sum_{i in taps} s_{k-i-1}
        # This is slightly confusing. Let's stick to a robust implementation.
        
        # A simpler way: 
        # The next bit is determined by the taps.
        # If poly is 0b100001001 (x^8+x^4+x^3+x^2+1), taps are at 4, 3, 2.
        # The new bit is s_4 XOR s_3 XOR s_2.
        # Then we shift left and insert the new bit at the LSB? Or shift right?
        # Let's shift right (LSB out) and insert new bit at MSB.
        
        # Calculate feedback
        fb = 0
        # Iterate through bits 0 to degree-1
        # If poly has bit i set, XOR state bit i
        # Wait, the standard mapping for LFSR with polynomial P(x) = x^d + ... + 1
        # is that the feedback is the XOR of the bits corresponding to the non-leading terms.
        # If we store state as s_0 (LSB) ... s_{d-1} (MSB), and output s_0,
        # then the new bit (becoming s_{d-1}) is XOR of s_i where x^i is in P(x) (i < d).
        
        for i in range(degree):
            if (poly >> i) & 1:
                fb ^= (state >> i) & 1
        
        # Shift right, insert fb at MSB
        state = (state >> 1) | (fb << (degree - 1))
        state &= mask

    return np.array(seq)

def construct():
    N = 400
    deadline = time.time() + 110
    
    # Use a known primitive polynomial for degree 8: x^8 + x^4 + x^3 + x^2 + 1
    # Binary: 1 0000 1001 -> 0x119? No.
    # x^8: bit 8
    # x^4: bit 4
    # x^3: bit 3
    # x^2: bit 2
    # 1: bit 0
    # So poly = 0b100001001 = 0x119
    poly = 0x119
    init_state = 1
    
    # Generate m-sequence of length 255 (2^8 - 1)
    # We need 400, so we generate 400 bits from the LFSR.
    h = _lfsr_sequence(N, poly, init_state)
    
    # Local Search Refinement
    # We will perform a greedy local search to minimize the bound.
    # Since N=400, a full pass of 400 flips is cheap.
    # We can do many passes.
    
    best_bound = _compute_bound(h)
    best_h = h.copy()
    
    # We use a simple hill climbing approach.
    # To speed up, we can try to flip bits that are part of the "worst" lags?
    # Or just iterate.
    
    # Optimization: We can estimate the change in bound for a flip without full FFT?
    # R_new(k) = R_old(k) - 2 * h_i * h_{i+k} (for positive lags) etc.
    # This is complex to track max |.| efficiently without updating the whole array.
    # Given N=400, full FFT is fast enough (~0.1ms).
    # 1000 passes * 400 flips * 0.1ms = 40 seconds.
    # So we can do ~1000 passes in 100s.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias? Or just sequential.
        # Sequential is fine for now.
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            
            new_bound = _compute_bound(h)
            
            if new_bound < best_bound:
                best_bound = new_bound
                best_h = h.copy()
                improved = True
            else:
                h[i] = old_val
                
        # If no improvement, we might be stuck.
        # We could add a perturbation step, but let's see if greedy is enough.
        # Given the m-sequence start, it should be close to optimal.
        if not improved:
            # Try a few random flips to escape local minimum
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            for _ in range(10):
                if time.time() > deadline:
                    break
                pos = rng.randint(N)
                old_val = h[pos]
                h[pos] = -old_val
                new_bound = _compute_bound(h)
                if new_bound < best_bound:
                    best_bound = new_bound
                    best_h = h.copy()
                    improved = True
                else:
                    h[pos] = old_val
            if not improved:
                break

    return best_h
# EVOLVE-BLOCK-END
