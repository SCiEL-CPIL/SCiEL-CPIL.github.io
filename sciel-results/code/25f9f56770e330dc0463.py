"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def _pair_flip(h, max_passes=5, deadline=None):
    """Local optimization with pair flips: try flipping two elements at once."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(12345)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try single flips
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        
        # Try pair flips (random pairs)
        for _ in range(20):
            if deadline is not None and time.time() > deadline:
                break
            i, j = rng.choice(N, size=2, replace=False)
            orig_i, orig_j = h[i], h[j]
            h[i] = -orig_i
            h[j] = -orig_j
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = orig_i
                h[j] = orig_j
        
        if not improved:
            break
    return h, best_val

def _generate_msequence(k, length):
    """Generate an m-sequence (maximal length sequence) from a primitive polynomial over GF(2^k).
    
    Returns a sequence of +1/-1 values of given length.
    """
    # Primitive polynomials for small k
    # Represented as list of exponents where the coefficient is 1 (excluding x^k term)
    primitive_polys = {
        1: [0],       # x+1
        2: [0, 1],    # x^2+x+1
        3: [0, 1],    # x^3+x+1
        4: [0, 2],    # x^4+x+1
        5: [0, 2],    # x^5+x^2+1
        6: [0, 1, 3], # x^6+x^5+x^3+x^2+1
        7: [0, 1],    # x^7+x+1
        8: [0, 1, 4], # x^8+x^4+x^3+x^2+1
        9: [0, 1, 4], # x^9+x^4+x^3+x+1
        10: [0, 1, 3, 5], # x^10+x^9+x^7+x^6+1
    }
    
    if k not in primitive_polys:
        return np.ones(length)
    
    exponents = primitive_polys[k]
    # Build the feedback polynomial: x^k + sum(x^e for e in exponents)
    # For LFSR, the feedback taps are at positions corresponding to the polynomial
    
    # Initialize register with all 1s (non-zero state)
    reg = [1] * k
    
    seq = []
    for _ in range(length):
        # Output is the first bit of the register
        seq.append(1 if reg[0] == 1 else -1)
        
        # Compute feedback: XOR of bits at tap positions
        # The polynomial x^k + sum x^e means: new_bit = XOR of reg[i] for i in exponents
        # where the bit at position i corresponds to x^(k-1-i) in the register
        # Actually, let's use a simpler convention:
        # reg[0] is MSB, reg[k-1] is LSB
        # Feedback = XOR of reg[j] for j where (k-1-j) is in exponents
        # i.e., j = k-1-e for e in exponents
        
        feedback = 0
        for e in exponents:
            j = k - 1 - e
            if 0 <= j < k:
                feedback ^= reg[j]
        
        # Shift left and insert feedback at LSB
        reg = reg[1:] + [feedback]
    
    return np.array(seq, dtype=float)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Initialize with a raised cosine (Hanning) window
    x = np.arange(N)
    h_init = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    
    best_h = h_init.copy()
    best_val = _score(best_h)
    
    # Also try ones(N) as baseline
    h_ones = np.ones(N)
    val_ones = _score(h_ones)
    if val_ones < best_val:
        best_val = val_ones
        best_h = h_ones.copy()
    
    # Local optimization from the tapered window
    h_opt, val_opt = _local_sign_flip(h_init, max_passes=3, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Try m-sequence based initializations
    # m-sequences have flat autocorrelation: sidelobes are exactly -1 for non-zero lags
    # This gives a very low peak-to-average ratio structurally
    for k in range(2, 11):
        if time.time() > deadline:
            break
        
        mseq_len = (1 << k) - 1  # m-sequence period
        # Generate m-sequence and tile/truncate to N
        mseq = _generate_msequence(k, mseq_len)
        
        # Tile to get at least N elements
        if mseq_len >= N:
            h = mseq[:N]
        else:
            reps = (N + mseq_len - 1) // mseq_len
            h = np.tile(mseq, reps)[:N]
        
        # Ensure non-zero sum
        s = np.sum(h)
        if abs(s) < N * 0.05:
            # If sum is too small, try flipping the whole sequence
            if s < 0:
                h = -h
                s = -s
            if abs(s) < N * 0.05:
                # Add a bias: make one element +1 instead of -1
                min_idx = np.argmin(h)
                h[min_idx] = 1.0
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
        
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Local optimization from m-sequence
        h_opt, val_opt = _pair_flip(h, max_passes=4, deadline=deadline)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Try multiple random initializations with local optimization
    np.random.seed(42)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 5 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = np.random.randint(N//4, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            # Chirp-like pattern
            xx = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * xx * (xx+1) / (2*N)))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            # Smooth raised cosine with random phase
            phase = np.random.uniform(0, 2*np.pi)
            h = 0.5 * (1 + np.cos(2 * np.pi * x / N + phase))
        else:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, N//10 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = np.sign(h + 0.5)
        
        # Use pair-flip local search for better escape from local minima
        h_opt, val_opt = _pair_flip(h, max_passes=4, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Final polish: additional local search on the best solution
    if time.time() < deadline - 5:
        h_final, val_final = _pair_flip(best_h, max_passes=8, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h

# EVOLVE-BLOCK-END
