"""Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution."""
import numpy as np
import time
import sys

N = 50

def _autoconv(h):
    """Compute autoconvolution of h with itself (full)."""
    # h has length N. Autoconvolution has length 2N-1.
    # Using FFT for efficiency.
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * H, n=n_fft)
    # The autoconvolution of h with h is ac[0:N] + ac[N:] ... 
    # Actually, let's think carefully.
    # If h = [h0, h1, ..., h_{N-1}], then (h*h)[k] = sum_j h_j h_{k-j}
    # The full convolution has indices from -(N-1) to (N-1), i.e., 2N-1 elements.
    # Using FFT: we compute circular convolution of length n_fft >= 2N-1.
    # h padded to n_fft: [h0, ..., h_{N-1}, 0, ..., 0]
    # Circular conv of length n_fft gives the linear conv in the first 2N-1 elements.
    # So ac[0:2*N-1] is the autoconvolution.
    return ac[:2*N-1]

def _score(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    ac = _autoconv(h)
    ac_max = np.max(np.abs(ac))
    if ac_max == 0:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _score_logsumexp(h, beta=100.0):
    """Smooth surrogate for the score using log-sum-exp for the max."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    ac = _autoconv(h)
    # Use log-sum-exp for the max of |ac|
    ac_abs = np.abs(ac)
    ac_max_approx = beta * np.log(np.sum(np.exp(ac_abs / beta)))
    if ac_max_approx == 0:
        return 0.0
    return l2_sq / (l1 * ac_max_approx)

def _random_nonneg(rng, n):
    """Generate a random nonnegative vector of length n."""
    return rng.random(n)

def _random_positive(rng, n):
    """Generate a random positive vector of length n (all entries > 0)."""
    return rng.random(n) + 0.01

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin before 150s kill
    
    best_score = 0.0
    best_h = np.ones(N) / N
    
    rng = np.random.default_rng(42)
    
    # First, evaluate the seed
    seed_score = _score(best_h)
    best_score = seed_score
    
    # Try some structured initializations
    # 1. Uniform
    # 2. Triangular
    # 3. Gaussian-like
    # 4. Random
    
    inits = []
    # Uniform
    inits.append(np.ones(N))
    # Triangular
    tri = np.abs(np.linspace(-1, 1, N))
    inits.append(tri / np.sum(tri))
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 5)
    inits.append(gauss / np.sum(gauss))
    # Random
    inits.append(_random_positive(rng, N))
    inits.append(_random_positive(rng, N))
    
    for h_init in inits:
        h_init = h_init / np.sum(h_init)  # Normalize
        s = _score(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Local optimization using gradient-free methods
    # We'll use a simple coordinate descent / pattern search approach
    # Since the objective is non-smooth (max in denominator), we use finite differences
    
    def _local_opt(h, max_iters=500, tol=1e-8):
        """Simple coordinate descent with line search."""
        h = h.copy().astype(float)
        current_score = _score(h)
        
        # Coordinate descent
        for it in range(max_iters):
            improved = False
            for i in range(N):
                # Try increasing and decreasing h[i]
                for delta in [0.1, 0.01, 0.001, -0.1, -0.01, -0.001]:
                    h_new = h.copy()
                    h_new[i] = max(0, h_new[i] + delta)
                    # Keep sum constant? No, the objective is scale-invariant in a specific way.
                    # Actually, let's check: if we scale h by c, l2_sq scales by c^2, l1 by c, ac by c^2.
                    # So score = c^2 / (c * c^2) = 1/c. So scaling matters!
                    # We want to maximize, so we want small scale? But h must be nonnegative.
                    # Actually, the problem says "nonnegative heights", so we can choose any scale.
                    # But wait, if we scale h by c, score becomes 1/c. So we want c as small as possible?
                    # But then ac_max also scales, so... let me recheck.
                    # l2_sq(h) = sum h_i^2, scales as c^2
                    # l1(h) = sum h_i, scales as c
                    # ac(h) = h*h, scales as c^2
                    # score = c^2 / (c * c^2) = 1/c
                    # So the score is inversely proportional to the scale. This means we can make the score arbitrarily large by making h very small!
                    # That can't be right. Let me re-read the problem.
                    # "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
                    # So it's ||h||_2^2 / (||h||_1 * ||h*h||_inf)
                    # With h scaled by c: c^2 / (c * c^2) = 1/c. Yes, it's 1/c.
                    # This suggests the problem is ill-posed unless there's a constraint I'm missing.
                    # Perhaps the heights are meant to be on equal intervals, so the L norms are discrete sums, and the autoconvolution is also discrete.
                    # But the scaling issue remains. Maybe there's an implicit constraint like sum(h) = 1 or max(h) = 1?
                    # Let me assume we should normalize so that sum(h) = 1 or something similar.
                    # Actually, looking at the seed program, it returns [1.0] * N, which has sum = N.
                    # Let's just work with the raw values and see what happens.
                    s_new = _score(h_new)
                    if s_new > current_score + 1e-10:
                        h = h_new
                        current_score = s_new
                        improved = True
                        break
            if not improved:
                break
        
        return h, current_score
    
    # Run local optimization from the best initial point
    h_opt, score_opt = _local_opt(best_h, max_iters=200)
    if score_opt > best_score:
        best_score = score_opt
        best_h = h_opt
    
    # Try more random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        h_rand = _random_positive(rng, N)
        h_rand = h_rand / np.sum(h_rand)
        h_opt, score_opt = _local_opt(h_rand, max_iters=100)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final polish with smaller steps
    def _polish(h, max_iters=200):
        h = h.copy().astype(float)
        current_score = _score(h)
        for it in range(max_iters):
            improved = False
            for i in range(N):
                for delta in [0.001, 0.0001, 0.00001, -0.001, -0.0001, -0.00001]:
                    h_new = h.copy()
                    h_new[i] = max(0, h_new[i] + delta)
                    s_new = _score(h_new)
                    if s_new > current_score + 1e-12:
                        h = h_new
                        current_score = s_new
                        improved = True
                        break
            if not improved:
                break
        return h, current_score
    
    h_polished, score_polished = _polish(best_h, max_iters=100)
    if score_polished > best_score:
        best_score = score_polished
        best_h = h_polished
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
