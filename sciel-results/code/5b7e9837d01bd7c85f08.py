"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Strategy: Generate LFSR (m-sequence) and map to {-1, 1}.
    # To ensure the score is not infinite due to zero sum, we add a bias.
    # The LFSR sequence has ideal periodic autocorrelation, but for linear 
    # autoconvolution with a bias, the performance depends on the bias.
    # We search for a bias that minimizes the score, effectively balancing 
    # the LFSR structure with the DC component required for the denominator.
    
    # Generate a maximal length sequence (m-sequence) of length N
    # N is not necessarily 2^m - 1. We generate a longer sequence and take N.
    # Primitive polynomial for GF(2) of degree 8: x^8 + x^4 + x^3 + x^2 + 1
    # Taps: 8, 4, 3, 2, 1 (indices 7, 3, 2, 1, 0 in 0-indexed)
    
    def generate_lfsr(length, taps, initial_state=1):
        """
        Generate LFSR sequence.
        taps: list of indices (0-indexed) where feedback is taken.
        initial_state: initial value of the register (1 bit).
        """
        # We need a register size that can generate at least 'length' bits.
        # Max length for degree d is 2^d - 1.
        # Find smallest d such that 2^d - 1 >= length
        d = 1
        while (1 << d) - 1 < length:
            d += 1
        
        # Use a known primitive polynomial for this degree if possible, 
        # otherwise use the provided taps (which must be valid for the degree).
        # For simplicity and robustness, we use a standard primitive poly for degree 8 or higher.
        # If N=400, d=9 (2^9-1=511).
        # Primitive poly for degree 9: x^9 + x^5 + 1 (taps 9, 5, 1 -> indices 8, 4, 0)
        # Let's use degree 9 taps [8, 4, 0]
        
        if d <= 8:
             # For small N, we can just use random or specific small polys
             # But N=400 requires d>=9
             pass
        
        # Standard primitive polynomial for degree 9: x^9 + x^5 + 1
        # Feedback is x^9 + x^5 + 1 = 0 => x^9 = x^5 + 1
        # In LFSR terms, new bit = bit[4] XOR bit[0] (if bits are indexed 0..8)
        # Let's define taps for degree d properly.
        # We'll use a fixed degree 9 polynomial for N=400.
        
        # Register size
        reg_size = d
        # Initial state
        state = [0] * reg_size
        state[0] = initial_state
        
        seq = []
        for _ in range(length):
            seq.append(state[0])
            # Calculate feedback
            # x^d = sum(x^{t_i})
            # For x^9 + x^5 + 1: x^9 = x^5 + 1
            # Indices: 8, 4, 0
            # new_bit = state[4] ^ state[0]
            # Shift left
            # state = [new_bit, state[0], state[1], ..., state[d-2]]
            
            if d == 9:
                new_bit = state[4] ^ state[0]
            elif d == 8:
                # x^8 + x^4 + x^3 + x^2 + 1
                new_bit = state[3] ^ state[2] ^ state[1] ^ state[0]
            elif d == 7:
                # x^7 + x^3 + 1
                new_bit = state[2] ^ state[0]
            elif d == 6:
                # x^6 + x^5 + 1
                new_bit = state[4] ^ state[0]
            elif d == 5:
                # x^5 + x^2 + 1
                new_bit = state[1] ^ state[0]
            elif d == 4:
                # x^4 + x^3 + 1
                new_bit = state[2] ^ state[0]
            elif d == 3:
                # x^3 + x + 1
                new_bit = state[0] ^ state[0] # x^3 = x
                # Actually x^3+x+1=0 => x^3 = x+1. 
                # Standard LFSR: new = x^(k-1) ... 
                # Let's just use a generic implementation for d>=5 to be safe.
                new_bit = state[0]
            else:
                new_bit = state[0]
                
            state = [new_bit] + state[:-1]
            
        return np.array(seq)

    # Generate LFSR sequence of length N
    lfsr_bits = generate_lfsr(N, taps=None, initial_state=1)
    
    # Map 0/1 to -1/1
    # 0 -> -1, 1 -> 1
    x_base = 2.0 * lfsr_bits - 1.0
    
    best_x = x_base.copy()
    best_score = _score(best_x)
    
    # The LFSR sequence has sum approx 0. 
    # Score is likely very high.
    # We try to add a bias c to x_base to increase the sum and potentially lower the score.
    # x = x_base + c
    # We search for c.
    
    rng = np.random.default_rng(42)
    
    # Coarse search for bias
    biases = np.linspace(-5, 5, 100)
    for c in biases:
        if time.time() > deadline:
            break
        x_candidate = x_base + c
        sc = _score(x_candidate)
        if sc < best_score:
            best_score = sc
            best_x = x_candidate.copy()
            
    # Fine search around best bias
    if best_score < np.inf:
        # Estimate best c from the coarse search
        # We can just do a local search on c
        # Reconstruct x_base from best_x? No, best_x = x_base + c_best
        # We can just perturb best_x.
        
        # Local search on the bias value
        # We need to know the current bias. 
        # Since x_base is fixed, we can just try adding small deltas to the current best_x.
        # But changing x_base elements individually is different from changing bias.
        # The strategy is "LFSR mapped to signs". Adding bias changes the "signed step function" 
        # into a "signed step function with offset".
        
        # Let's try refining the bias.
        # We don't know the exact c used, but we can infer it or just perturb.
        # Perturbing the whole vector by a constant delta is equivalent to changing bias.
        
        current_bias_delta = 0.1
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for delta in [current_bias_delta, -current_bias_delta]:
                x_new = best_x + delta
                sc_new = _score(x_new)
                if sc_new < best_score - 1e-10:
                    best_x = x_new
                    best_score = sc_new
                    improved = True
            if not improved:
                current_bias_delta /= 2.0
                if current_bias_delta < 1e-4:
                    break
            else:
                # If improved, keep the step size or reduce slightly? 
                # Usually keep if we want to explore, reduce if we want to converge.
                # Let's reduce slowly.
                current_bias_delta *= 0.9

    return best_x
# EVOLVE-BLOCK-END
