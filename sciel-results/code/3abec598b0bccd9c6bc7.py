"""Rewritten program for C3-prime task using FRFT-inspired phase optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct a signed step function (sequence of +1/-1) that minimizes
    the maximum absolute autoconvolution divided by squared integral.
    
    Approach: Use fractional Fourier transform (FRFT) concepts to design
    sequences with suppressed peak autoconvolution. We optimize over
    phase vectors in a fractional domain to achieve good autocorrelation
    properties.
    
    The sequence is constrained to be signed (+1 or -1), so we use a
    soft optimization with rounding, or directly search over sign patterns
    using gradient-free methods on the continuous relaxation.
    """
    # We'll work with a continuous relaxation first, then round to signs.
    # The autoconvolution of a sequence x is (x * x)[k] = sum_i x[i] x[i+k]
    # For zero-mean or unit-variance sequences, we want to minimize max_k |autoconv[k]|
    # relative to the squared integral (sum x)^2.
    
    # Since we need signed steps (+1/-1), let's use a smarter initialization
    # and local search.
    
    # Initialize with a pseudo-random sign pattern based on a chaotic/fractal seed
    # that has good spectral flatness properties
    rng = np.random.default_rng(42)
    
    # Start with a random sign sequence
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Compute the objective: max absolute autoconvolution / (sum x)^2
    # But sum x for a balanced sign sequence is small, so let's think carefully.
    # The task says "maximum absolute autoconvolution divided by squared integral"
    # Integral of a discrete sequence is its sum. Squared integral is (sum x)^2.
    # If sum x = 0, this is problematic. Let's assume we want to minimize
    # max_k |autoconv[k]| / (sum x)^2, but if sum x is near 0, we should
    # perhaps use a different normalization or ensure sum x is bounded away from 0.
    
    # Actually, re-reading: "minimizing the maximum absolute autoconvolution 
    # divided by squared integral". For a sequence of +1/-1, sum x can range 
    # from -N to N. To make the ratio well-defined and meaningful, let's
    # assume we want a sequence where the DC component is controlled.
    
    # Let's define the score as max_k |autocorr[k]| / (sum x)^2
    # and minimize it. We need sum x != 0.
    
    # Better approach: use a continuous variable y in [-1, 1] and optimize,
    # then project to signs.
    
    def autoconv_max(x):
        """Compute max absolute value of circular autoconvolution."""
        # Use FFT for efficient computation
        X = np.fft.fft(x, n=2*N)
        auto = np.fft.ifft(X * np.conj(X)).real[:N]
        return np.max(np.abs(auto))
    
    def objective(y):
        """Objective function for continuous relaxation y in [-1,1]."""
        s = np.sum(y)
        if abs(s) < 1e-6:
            return 1e10  # Penalize near-zero sum
        ac_max = autoconv_max(y)
        return ac_max / (s * s)
    
    # Start from a good initialization: use a chirp-like or PRN-like pattern
    # Let's try a few initializations and pick the best
    
    best_x = None
    best_score = np.inf
    
    # Initialization 1: Random
    for trial in range(5):
        rng_trial = np.random.default_rng(100 + trial)
        x_init = rng_trial.choice([-1.0, 1.0], size=N)
        s = np.sum(x_init)
        if abs(s) < N * 0.1:  # Ensure reasonable DC
            continue
        ac_max = autoconv_max(x_init)
        score = ac_max / (s * s)
        if score < best_score:
            best_score = score
            best_x = x_init.copy()
    
    # Initialization 2: Based on a fractional Fourier transform of a random phase
    # FRFT of order alpha rotates in time-frequency plane
    # For alpha = pi/4, it's like a chirp
    for alpha in [np.pi/4, np.pi/3, np.pi/2 * 0.7, np.pi/6]:
        t = np.arange(N)
        # Fractional chirp
        phase = alpha * t * t / (2 * N)
        # Create a signed sequence from the sign of the fractional chirp
        x_chirp = np.sign(np.sin(phase + np.pi/4))
        x_chirp[x_chirp == 0] = 1.0
        s = np.sum(x_chirp)
        if abs(s) < N * 0.1:
            continue
        ac_max = autoconv_max(x_chirp)
        score = ac_max / (s * s)
        if score < best_score:
            best_score = score
            best_x = x_chirp.copy()
    
    # Local search: flip individual elements to improve
    def local_search(x, max_iter=500):
        """Flip elements that reduce the objective."""
        x = x.copy()
        s = np.sum(x)
        ac_max = autoconv_max(x)
        current_score = ac_max / (s * s) if abs(s) > 1e-6 else 1e10
        
        for _ in range(max_iter):
            improved = False
            # Try flipping each position
            for i in range(N):
                x_old = x[i]
                x[i] = -x[i]
                s_new = s - 2 * x_old
                if abs(s_new) < 1e-6:
                    x[i] = x_old
                    continue
                ac_max_new = autoconv_max(x)
                new_score = ac_max_new / (s_new * s_new)
                if new_score < current_score * (1 - 1e-6):
                    current_score = new_score
                    s = s_new
                    ac_max = ac_max_new
                    improved = True
                else:
                    x[i] = x_old
            
            if not improved:
                break
        
        return x, current_score
    
    best_x, best_score = local_search(best_x, max_iter=200)
    
    return best_x
# EVOLVE-BLOCK-END
