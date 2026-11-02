"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(signs):
    """
    Compute the upper bound on C3 for a given signed step function.
    
    The task involves a signed step function with N intervals on [-1/4, 1/4].
    We need to minimize an autoconvolution-based upper bound.
    
    For a step function f(x) = sum_{i} s_i * 1_{[x_i, x_{i+1})}(x) where s_i in {-1, 1},
    the autoconvolution is (f * f)(t) = integral f(x) f(t-x) dx.
    
    The "third autocorrelation inequality" likely refers to bounding some integral
    involving f and its autoconvolution. A common quantity is:
        C3 = integral |f(x)|^3 dx or related to (f*f*f)
    
    But the task says "autoconvolution upper bound" and "upper bound on C3".
    
    Let's think about this more carefully. The paper mentions "autoconvolution upper bound"
    and the score is "the upper bound on C3". Lower is better.
    
    For a step function with values in {-1, 1} on N equal intervals of width w = 1/2 / N,
    the L2 norm squared is fixed: ||f||_2^2 = N * w = 1/2.
    
    The autoconvolution at lag 0 is (f*f)(0) = integral f(x)^2 dx = 1/2 (fixed).
    
    Perhaps C3 relates to the L1 norm of the autoconvolution, or the maximum of the
    autoconvolution, or some other functional.
    
    Given the hint that "greedy coordinate descent... converges to a local minimum" and
    the measured improvement from 0.0001235 to 0.003214, let's assume the score is
    something like the L1 norm of the autoconvolution minus its value at 0, or
    the integral of |f*f|, or similar.
    
    Actually, re-reading: "improve an autoconvolution upper bound" and "score is the
    upper bound on C3". This suggests we're minimizing an upper bound that involves
    the autoconvolution.
    
    A natural quantity: for the autocorrelation inequality, one might bound
    integral |f|^3 or use (f*f) in some way.
    
    Let me try: score = integral |f*f(t)| dt / (integral f^2)^2 or similar normalized quantity.
    
    Actually, let's try the most straightforward interpretation: 
    score = sum over all lags of |autoconvolution(lag)|, i.e., L1 norm of autoconvolution.
    
    For a step function, the autoconvolution at lag k*w is:
    (f*f)(k*w) = sum_i f(x_i) f(x_i + k*w) * w
    where the sum is over valid i.
    
    Let's compute: for signs s[0..N-1], the autoconvolution at lag d (in interval units) is:
    A[d] = w * sum_{i=0}^{N-1-d} s[i] * s[i+d]
    
    The L1 norm of the autoconvolution would be:
    sum_{d=-(N-1)}^{N-1} |A[d]| = 2 * sum_{d=1}^{N-1} |A[d]| + A[0]
    
    where A[0] = w * sum_i s[i]^2 = w * N = 1/2.
    
    So score = 0.5 + 2 * sum_{d=1}^{N-1} |w * sum_{i=0}^{N-1-d} s[i]*s[i+d]|
             = 0.5 + 2*w * sum_{d=1}^{N-1} |sum_{i=0}^{N-1-d} s[i]*s[i+d]|
    
    Since w = 1/(2N), this is:
    score = 0.5 + (1/N) * sum_{d=1}^{N-1} |sum_{i=0}^{N-1-d} s[i]*s[i+d]|
    
    This seems like a reasonable quantity to minimize. Let's go with this.
    """
    w = 1.0 / (2 * N)
    # Autoconvolution at lag 0 is fixed: w * N = 0.5
    score = 0.5
    for d in range(1, N):
        # Sum of s[i]*s[i+d] for i=0 to N-1-d
        c = np.sum(signs[:N-d] * signs[d:])
        score += 2 * w * abs(c)
    return score


def _compute_score_fast(signs, fft_cache=None):
    """
    Faster computation using FFT.
    
    The autoconvolution of a discrete sequence can be computed via FFT.
    For signs s[0..N-1], the circular autoconvolution is:
    circular_corr[d] = sum_i s[i] * s[(i+d) mod N]
    
    But we need linear autoconvolution:
    linear_corr[d] = sum_{i=0}^{N-1-d} s[i] * s[i+d] for d >= 0
    
    We can compute this via FFT by zero-padding.
    """
    # Pad to at least 2N for linear convolution
    nfft = 1 << (2 * N - 1).bit_length()
    f = np.fft.rfft(signs, n=nfft)
    corr = np.fft.irfft(f * np.conj(f), n=nfft)[:N]
    # corr[d] = sum_{i=0}^{N-1-d} s[i] * s[i+d]
    w = 1.0 / (2 * N)
    score = 0.5 + 2 * w * np.sum(np.abs(corr[1:]))
    return score


def construct():
    """
    Construct a signed step function (sequence of +1/-1) that minimizes
    the autoconvolution upper bound on C3.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with a random initialization
    rng = np.random.default_rng(42)
    signs = rng.choice([-1, 1], size=N).astype(float)
    
    best_score = _compute_score_fast(signs)
    best_signs = signs.copy()
    
    # Greedy coordinate descent: try flipping each position
    n_improvements = 0
    n_flips = 0
    
    while time.time() < deadline:
        improved = False
        # Try flipping each position in random order
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 5:
                break
            
            # Compute current contribution of position i
            # Flipping s[i] changes corr[d] for all d where i or i+d is in range
            # Specifically, corr[d] = sum_{j} s[j]*s[j+d]
            # Position i appears in corr[d] as s[i]*s[i+d] if i+d < N, 
            # and as s[i-d]*s[i] if i-d >= 0
            # So flipping s[i] changes corr[d] by -2*s[i]*s[i+d] for d where valid
            # and by -2*s[i-d]*s[i] for d where valid
            
            old_s = signs[i]
            signs[i] = -old_s
            
            new_score = _compute_score_fast(signs)
            
            if new_score < best_score - 1e-15:
                best_score = new_score
                best_signs = signs.copy()
                improved = True
                n_improvements += 1
                n_flips += 1
            else:
                # Revert
                signs[i] = old_s
        
        if not improved:
            # No improvement in full pass, try a few random restarts or perturbations
            # Try small random perturbations to escape local minima
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                # Flip 5 random positions
                perturb_indices = rng.choice(N, size=5, replace=False)
                for idx in perturb_indices:
                    signs[idx] *= -1
                perturbed_score = _compute_score_fast(signs)
                if perturbed_score < best_score:
                    best_score = perturbed_score
                    best_signs = signs.copy()
                    n_improvements += 1
                else:
                    # Revert perturbation
                    for idx in perturb_indices:
                        signs[idx] *= -1
    
    return best_signs
# EVOLVE-BLOCK-END
