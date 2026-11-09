"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _m_sequence(prime, poly):
    """Generate a binary m-sequence of length prime^k - 1 where k = bit_length(poly)."""
    k = poly.bit_length()
    # State is k bits
    state = 1
    seq = []
    length = prime ** k - 1
    for _ in range(length):
        seq.append(state & 1)
        # Shift right
        new_bit = 0
        temp = state
        for i in range(k):
            # Check if bit i is set in poly
            if poly & (1 << i):
                new_bit ^= (temp >> i) & 1
        state = (state >> 1) | (new_bit << (k - 1))
    return np.array(seq, dtype=np.int8)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits in binary representation
        seq[i] = bin(i).count('1') % 2
    return seq

def _compute_c3_bound(x):
    """
    Compute the C3 upper bound for the given sequence x.
    The task mentions 'autoconvolution upper bound' and 'C3'.
    In the context of the third autocorrelation inequality (often related to the Welch bound or specific 
    combinatorial bounds for autocorrelation), C3 is typically the peak sidelobe of the autocorrelation 
    normalized by the main lobe, or a specific sum of autocorrelation values.
    
    Given the hint 'score is the upper bound on C3' and 'lower is better', and the previous finding 
    'minimizes the peak normalized autocorrelation, which is the C3 upper bound', 
    we assume C3 is the maximum absolute value of the non-zero lag autocorrelation 
    normalized by the main lobe (N).
    
    Autocorrelation R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    Normalized: R(k) / N
    C3 = max_{k != 0} |R(k)| / N
    """
    # Use FFT for efficient autocorrelation
    N_local = len(x)
    # Pad to next power of 2 for FFT efficiency
    size = 1
    while size < 2 * N_local:
        size *= 2
    
    # x is +/- 1
    # Autocorrelation via FFT:
    # R(k) = ifft(fft(x) * conj(fft(x)))
    # Note: This computes circular autocorrelation. We need linear autocorrelation.
    # Zero-padding handles this.
    
    fft_x = np.fft.fft(x, size)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real
    
    # The first N_local values are the autocorrelation for lags 0 to N_local-1
    # The rest are wrapped around or zero.
    # Specifically, acf[0:N_local] are R(0), R(1), ..., R(N_local-1)
    # R(0) is the main lobe (sum x[i]^2 = N_local)
    
    # We want max |R(k)| for k=1..N_local-1
    sidelobes = acf[1:N_local]
    max_sidelobe = np.max(np.abs(sidelobes))
    
    # Normalize by N
    return max_sidelobe / N_local

def _optimize_boundary(x, max_flips=20, time_limit=10):
    """
    Try flipping bits in the first and last 'max_flips' positions to minimize C3.
    Uses a simple local search / hill climbing.
    """
    start_time = time.time()
    current_c3 = _compute_c3_bound(x)
    best_x = x.copy()
    best_c3 = current_c3
    
    n = len(x)
    # Indices to consider for flipping: first 20 and last 20
    # But avoid overlapping if n is small (not the case here, n=400)
    indices = list(range(max_flips)) + list(range(n - max_flips, n))
    
    # Simple random restart local search
    # We have a tight time budget here, so keep it simple.
    # Try flipping each index individually, keep if it improves.
    # Repeat a few times.
    
    for _ in range(50): # 50 passes
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Shuffle indices to avoid order bias
        np.random.shuffle(indices)
        for idx in indices:
            # Flip
            best_x[idx] *= -1
            new_c3 = _compute_c3_bound(best_x)
            if new_c3 < best_c3:
                best_c3 = new_c3
                improved = True
            else:
                # Revert
                best_x[idx] *= -1
        
        if not improved:
            break
            
    return best_x, best_c3

def construct():
    """
    Construct a signed step function with 400 equal intervals.
    Returns a sequence of 400 values (+1 or -1).
    """
    start_time = time.time()
    deadline = 140.0 # 20s safety margin
    
    # 1. Generate candidate sequences
    # Strategy: m-sequences and Thue-Morse
    candidates = []
    
    # Thue-Morse
    tm = _thue_morse(N)
    # Convert 0/1 to -1/+1
    tm_signed = 2 * tm - 1
    candidates.append(tm_signed)
    
    # M-sequences for small primes
    # m-sequence length is 2^k - 1. We need to truncate or pad to 400.
    # Truncation is mentioned in the directive.
    # Common m-sequence generators:
    # k=6, poly=63 (111111) -> length 63
    # k=7, poly=127 (1111111) -> length 127
    # k=8, poly=255 -> length 255
    # k=9, poly=511 -> length 511 (truncate to 400)
    # k=10, poly=1023 -> length 1023 (truncate to 400)
    
    # Let's try a few known good m-sequence polynomials
    polys = [
        (6, 63),   # 111111
        (7, 127),  # 1111111
        (8, 255),  # 11111111
        (9, 511),  # 111111111
        (9, 109),  # 1101101 (primitive? need to check, but let's try)
        (9, 137),  # 10001001
        (10, 1023),# 1111111111
        (10, 101), # 1100101
    ]
    
    for k, poly in polys:
        if time.time() > deadline - 10:
            break
        try:
            seq = _m_sequence(2, poly) # Base 2 for binary m-seq? 
            # Wait, _m_sequence implementation above uses 'prime' parameter but logic is for base 2?
            # My _m_sequence implementation:
            # state is k bits.
            # new_bit calculation:
            # for i in range(k): if poly & (1 << i): new_bit ^= (temp >> i) & 1
            # This is standard LFSR feedback.
            # It generates a sequence of length 2^k - 1.
            
            if len(seq) >= N:
                # Truncate
                seq = seq[:N]
            else:
                # Repeat to fill? Or pad?
                # Directive says "truncated to 400 samples".
                # If shorter, we can repeat or just use what we have and pad with 0?
                # Signed step function implies +/-1.
                # Let's repeat.
                reps = int(np.ceil(N / len(seq)))
                seq = np.tile(seq, reps)[:N]
                
            signed_seq = 2 * seq - 1
            candidates.append(signed_seq)
        except Exception:
            pass

    # Also add a random sequence as a fallback
    rand_seq = np.random.choice([-1, 1], size=N)
    candidates.append(rand_seq)
    
    # 2. Evaluate candidates and pick the best
    best_seq = None
    best_score = float('inf')
    
    for seq in candidates:
        if time.time() > deadline - 5:
            break
        score = _compute_c3_bound(seq)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
            
    # 3. Boundary optimization on the best candidate
    if best_seq is not None and time.time() < deadline - 2:
        # Optimize boundary
        opt_seq, opt_score = _optimize_boundary(best_seq, max_flips=20, time_limit=10)
        if opt_score < best_score:
            best_seq = opt_seq
            best_score = opt_score
        else:
            # Try optimizing other top candidates if time permits?
            # For now, just return the best one.
            pass
            
    if best_seq is None:
        # Fallback
        return np.ones(N)
        
    return best_seq
# EVOLVE-BLOCK-END
