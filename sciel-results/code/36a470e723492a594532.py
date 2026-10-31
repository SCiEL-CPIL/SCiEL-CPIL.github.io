"""New design: LFSR-based signed sequences with local refinement."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution."""
    # Linear autoconvolution via FFT for efficiency
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * f.conj(), size)[:n]
    return np.max(np.abs(ac))

def _squared_integral(x):
    """Squared integral (sum of squares) of the sequence."""
    return np.sum(x * x)

def _score(x):
    """Compute the C3' score: max_abs_autoconv / squared_integral."""
    si = _squared_integral(x)
    if si == 0:
        return float('inf')
    return _autoconv_max_abs(x) / si

def _lfsr_sequence(n, poly_bits, seed_val):
    """Generate an LFSR sequence mapped to {-1, 1}.
    
    poly_bits: integer representing the feedback polynomial (e.g., 0b10011 for x^4+x+1)
    seed_val: initial register value
    """
    # Determine LFSR width from polynomial
    width = poly_bits.bit_length()
    # Feedback taps: positions where bits are set (excluding the MSB which is x^width)
    # Standard LFSR: new_bit = XOR of bits at tap positions
    # We'll use a Galois LFSR style
    reg = seed_val & ((1 << width) - 1)
    if reg == 0:
        reg = 1  # avoid all-zero state
    
    seq = []
    for i in range(n):
        # Extract bit (LSB or MSB depending on convention)
        bit = (reg >> (width - 1)) & 1  # MSB
        seq.append(1 if bit else -1)
        
        # Feedback
        # For a primitive polynomial, we need proper tap positions
        # Simple Fibonacci LFSR:
        # new_bit = XOR of selected bits
        # Let's use the polynomial to determine taps
        # poly_bits has width+1 bits, MSB is x^width
        # Taps are at positions where poly has 1 (excluding MSB)
        feedback = 0
        for j in range(width):
            if (poly_bits >> j) & 1:
                feedback ^= (reg >> j) & 1
        # Shift and insert
        reg = ((reg << 1) & ((1 << width) - 1)) | feedback
    
    return np.array(seq[:n], dtype=np.float64)

def _local_refine(x, max_iters=50, time_budget=30.0):
    """Local sign-flip optimization to reduce max abs autoconvolution."""
    x = x.copy()
    n = len(x)
    best_score = _score(x)
    start_time = time.time()
    
    # Precompute autoconvolution
    ac = _autoconv_full(x)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Try flipping each position and see if it reduces max abs ac
        improved = False
        # Sample positions to try (for speed)
        positions = np.random.choice(n, size=min(n, 200), replace=False)
        
        for idx in positions:
            if time.time() - start_time > time_budget:
                break
            
            # Flip sign at idx
            x[idx] = -x[idx]
            
            # Update autoconvolution efficiently
            # ac[k] = sum_{i} x[i] * x[i+k]
            # Flipping x[idx] changes ac by -2 * x[idx] * x[idx+k] for each k
            # But recomputing is simpler for correctness
            new_score = _score(x)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                x[idx] = -x[idx]  # revert
        
        if not improved:
            # Try random restarts of local search
            break
    
    return x, best_score

def _autoconv_full(x):
    """Full linear autoconvolution."""
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * f.conj(), size)
    return ac[:n]

def construct():
    """Construct signed step heights minimizing max abs autoconv / squared integral."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_x = None
    best_score = float('inf')
    
    # Try multiple LFSR configurations
    # Primitive polynomials for various widths
    # Width 8: x^8 + x^4 + x^3 + x^2 + 1 = 0b100011101 (285)
    # Width 10: x^10 + x^7 + x^5 + x^2 + 1 = 0b10010101001 (1205)
    # Width 12: x^12 + x^6 + x^4 + x + 1 = 0b1000010100101 (4357)
    # Width 15: x^15 + x + 1 = 0b1000000000000011 (32771)
    # Width 20: x^20 + x^3 + x^2 + x + 1 = 0b100000000000000011111 (1048595)
    
    lfsr_configs = [
        (8, 0b100011101),    # 285
        (10, 0b10010101001), # 1205
        (12, 0b1000010100101), # 4357
        (15, 0b1000000000000011), # 32771
        (20, 0b100000000000000011111), # 1048595
        (24, 0b10000000000000000000100101), # some primitive poly for 24
    ]
    
    np.random.seed(42)
    
    for width, poly in lfsr_configs:
        if time.time() > deadline - 40:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 40:
                break
            
            x = _lfsr_sequence(N, poly, seed + 1)
            
            # Local refinement
            x_refined, score = _local_refine(x, max_iters=30, time_budget=20.0)
            
            if score < best_score:
                best_score = score
                best_x = x_refined
            
            if time.time() > deadline - 60:
                break
    
    # If we haven't found anything good, use a simple alternating pattern
    if best_x is None:
        best_x = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64)
        best_score = _score(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
