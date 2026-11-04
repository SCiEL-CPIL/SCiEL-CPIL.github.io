"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak(f):
    """Compute the maximum absolute value of the autoconvolution at non-zero lags.
    
    The autoconvolution (full linear convolution) of f with itself is:
        (f * f)[k] = sum_i f[i] * f[i + k]
    We exclude k=0 (the zero-lag self-term) and take max |value| over k != 0.
    """
    n = len(f)
    # Use FFT for efficient linear convolution
    size = 2 * n
    f_pad = np.zeros(size)
    f_pad[:n] = f
    F = np.fft.rfft(f_pad, size)
    conv = np.fft.irfft(F * F, size)
    # conv[0] is the zero-lag term (sum f^2)
    # For lags 1..n-1, the values are conv[1:n]
    if n > 1:
        return float(np.max(np.abs(conv[1:n])))
    return 0.0


def _m_sequence_init(n_total, seed=0):
    """
    Generate a ±1 sequence using an m-sequence (Maximum Length Sequence) approach.
    This provides a deterministic low-autocorrelation starting point.
    We use a simple LFSR-based construction or a Legendre sequence if appropriate.
    For N=400, we can use a subset or a composite construction.
    
    A simple approach: use a quadratic residue sequence (Legendre symbol) for a prime length,
    then pad or truncate to N. Or use a known good binary sequence pattern.
    
    Here we use a simple LFSR m-sequence of length 2^m - 1 and tile/truncate to N.
    """
    rng = np.random.default_rng(seed)
    
    # Try to find a good m-sequence length close to N or use a combination
    # m-sequence lengths are 2^m - 1: 3, 7, 15, 31, 63, 127, 255, 511
    # 255 is close to 400? No, 511 is closer.
    # Let's use 511 and take the first 400.
    
    # Generate m-sequence of length 511 using polynomial x^9 + x^4 + 1 (primitive)
    # This is a standard m-sequence.
    # Actually, let's just use a simple construction that is known to have low autocorrelation.
    # Legendre sequence for prime p: s[i] = 1 if i^2 mod p == 1, -1 otherwise (for i != 0)
    # For p=401 (prime), we can use the first 400.
    
    p = 401  # Prime
    f = np.zeros(n_total)
    for i in range(1, p):
        # Legendre symbol: 1 if quadratic residue, -1 if not
        # Use Euler's criterion
        if pow(i, (p-1)//2, p) == 1:
            f[i] = 1.0
        else:
            f[i] = -1.0
    f[0] = 1.0  # Convention
    
    # We need exactly n_total elements. If n_total < p, truncate.
    return f[:n_total]


def _sparse_init(n_active, n_total, seed):
    """Initialize a sparse step function with n_active non-zero entries."""
    rng = np.random.default_rng(seed)
    f = np.zeros(n_total)
    indices = rng.choice(n_total, size=n_active, replace=False)
    # Assign random signs
    signs = rng.choice([-1.0, 1.0], size=n_active)
    f[indices] = signs
    return f


def _local_search(f, n_iters, deadline):
    """Local search: try flipping or zeroing individual entries to reduce the peak."""
    current_score = _autoconv_peak(f)
    best_f = f.copy()
    best_score = current_score
    n = len(f)
    
    for _ in range(n_iters):
        if time.time() > deadline:
            break
        improved = False
        # Try flipping or zeroing each position
        for i in range(n):
            if time.time() > deadline:
                break
            old_val = f[i]
            # Try setting to 0
            if old_val != 0:
                f[i] = 0.0
                s = _autoconv_peak(f)
                if s < best_score:
                    best_score = s
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = old_val
            
            # Try flipping sign (if non-zero)
            if old_val != 0:
                f[i] = -old_val
                s = _autoconv_peak(f)
                if s < best_score:
                    best_score = s
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] = old_val
            
            # Try adding a new non-zero at a zero position
            if old_val == 0:
                for sign in [-1.0, 1.0]:
                    f[i] = sign
                    s = _autoconv_peak(f)
                    if s < best_score:
                        best_score = s
                        best_f = f.copy()
                        improved = True
                    else:
                        f[i] = 0.0
        
        if not improved:
            break
    
    return best_f, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # 1. Start with the m-sequence (Legendre) construction
    f_mseq = _m_sequence_init(N)
    score_mseq = _autoconv_peak(f_mseq)
    if score_mseq < best_score:
        best_score = score_mseq
        best_f = f_mseq.copy()
    
    # Local search on the m-sequence
    if time.time() < deadline:
        f_mseq, score_mseq = _local_search(f_mseq, n_iters=20, deadline=deadline)
        if score_mseq < best_score:
            best_score = score_mseq
            best_f = f_mseq.copy()
    
    # 2. Try different sparsity levels with random initialization
    for n_active in [10, 20, 30, 40, 50, 60, 80, 100]:
        if time.time() > deadline:
            break
        
        # Multiple random initializations for each sparsity
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            seed = 42 + r * 1000 + n_active
            f = _sparse_init(n_active, N, seed)
            score = _autoconv_peak(f)
            
            if score < best_score:
                best_score = score
                best_f = f.copy()
            
            # Local search to improve
            f, score = _local_search(f, n_iters=50, deadline=deadline)
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    if best_f is None:
        # Fallback: return ones (shouldn't happen)
        return np.ones(N)
    
    return best_f

# EVOLVE-BLOCK-END
