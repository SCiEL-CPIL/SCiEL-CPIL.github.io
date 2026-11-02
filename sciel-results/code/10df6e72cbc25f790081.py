import numpy as np
import time
from scipy.optimize import minimize

N = 600

def _compute_max_autoconv(h):
    """
    Compute the maximum autoconvolution value excluding lag 0.
    h is a vector of length N.
    Returns the max value of the autoconvolution at non-zero lags.
    """
    if np.sum(h) < 1e-12:
        return 0.0
    # Efficiently compute autoconvolution using FFT
    fft_size = 1 << (2 * N - 1).bit_length()
    h_fft = np.fft.fft(h, n=fft_size)
    conv_fft = h_fft * h_fft
    conv = np.fft.ifft(conv_fft).real
    
    # Ensure numerical stability
    conv = np.maximum(conv, 0)
    
    # Exclude center (lag 0)
    if N == 1:
        return 0.0
        
    max_val = np.max(np.concatenate([conv[:N-1], conv[N:]]))
    return max_val

def _compute_score(h):
    """
    Normalized score: max_autoconv / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return np.inf
    return _compute_max_autoconv(h) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Strategy: Convex Optimization via Spectral Constraints
    # We want to minimize max_{k != 0} (autoconv_k) / (sum h)^2.
    # Let H = sum h. The problem is scale invariant. We can fix H = 1.
    # Minimize f(h) = max_{k != 0} (autoconv_k) subject to sum h = 1, h >= 0.
    
    # The current seed uses Frank-Wolfe on the non-convex max of quadratics.
    # A different structural approach:
    # 1. Use the fact that autoconvolution is related to the power spectrum.
    #    Let H(omega) = FFT(h). Then autoconv_k = IDFT(|H(omega)|^2)_k.
    #    Minimizing the max of IDFT(|H|^2) excluding lag 0 is equivalent to minimizing
    #    the peak sidelobe level of the power spectrum in the time domain.
    
    # 2. Alternatively, we can use a convex formulation if we relax the "max" constraint.
    #    However, the max of convex functions is convex.
    #    autoconv_k(h) is quadratic. Is it convex?
    #    For k != 0, the matrix A_k has eigenvalues that can be negative (as noted in seed comments).
    #    So it's not convex.
    
    # Let's try a different design family: Parametric Families.
    # Instead of optimizing all 600 variables, we restrict h to a low-complexity family.
    # Inspiration 1 (Two-Block) scored 1.125.
    # Let's try a "Raised Cosine" or "Gaussian-like" family, or a "Windowed" family.
    
    # A very effective family for minimizing sidelobes is the "Optimal Window" or specific polynomial windows.
    # But we can also just run a more robust optimizer on the full space if we use a better objective.
    
    # Let's try a convex optimization approach by minimizing the L2 norm of h
    # subject to the constraint that the autoconvolution at all non-zero lags is <= T.
    # This is a Semidefinite Programming (SDP) or Second Order Cone (SOC) problem if linearized.
    # But autoconv_k is quadratic.
    
    # Let's stick to the parametric family idea but make it more flexible than 2 blocks.
    # We will use a "Piecewise Linear" or "Spline" basis with a small number of knots.
    # Or, we can use a "Fourier" basis with a few harmonics, constrained to be positive.
    
    # Actually, the "Flat function" insight says 1/N is the theoretical minimum for the normalized max?
    # "Flat function provably attains the theoretical minimum 1/N... since lag-0 term sum(h^2) >= 1/N... and is the convolution's maximum"
    # Wait. The task says "minimize the normalized maximum of its autoconvolution".
    # The autoconvolution at lag 0 is sum(h_i^2).
    # The autoconvolution at lag k is sum h_i h_{i+k}.
    # If h is uniform (flat), h_i = 1/N.
    # Lag 0: sum (1/N)^2 = N * (1/N^2) = 1/N.
    # Lag k != 0: sum (1/N) * (1/N) * (N-|k|) = (N-|k|)/N^2.
    # The max of these is at k=0, which is 1/N.
    # The "normalized" score in the seed code is _compute_max_autoconv(h) / (sum h)^2.
    # _compute_max_autoconv excludes lag 0!
    # So for flat h, max_{k!=0} conv_k = (N-1)/N^2.
    # (sum h)^2 = 1.
    # Score = (N-1)/N^2 = 1/N - 1/N^2.
    # For N=600, this is approx 1/600 = 0.001666.
    
    # The seed program returns a score of 1.0000.
    # The seed program's _compute_score returns _compute_max_autoconv(h) / (s*s).
    # If the seed returns 1.0, it means the normalized max is 1.0?
    # That implies the seed is NOT using the flat function effectively or the scoring is different.
    # Let's check the seed's initial state: h = np.ones(N) / N.
    # _compute_score(h) for flat h:
    # max_autoconv (k!=0) is (N-1)/N^2.
    # (sum h)^2 = 1.
    # Score = (N-1)/N^2.
    # For N=600, (599)/360000 = 0.0016638...
    # Why does the seed score 1.0?
    # Ah, the "score" in the task description says "score is the upper bound on C1".
    # And "search score is seed/value".
    # If the seed program's output yields a value V, and the seed baseline is B, then score = B/V?
    # Or is the "score" printed in the prompt (1.0000) the *search score*?
    # "score is the upper bound on C1" -> This is the raw value.
    # "search score is seed/value" -> This is the metric for improvement.
    # If the current program scores 1.0000, it means it is performing exactly as the seed baseline.
    # The seed baseline is likely the flat function or a trivial function.
    # If the flat function gives ~0.00166, and the seed score is 1.0, then the "seed" value in the denominator must be ~0.00166.
    # So the raw value (upper bound on C1) for the flat function is ~0.00166.
    
    # My goal: Minimize the raw value (max_{k!=0} conv_k / (sum h)^2).
    # The flat function gives (N-1)/N^2.
    # Can we do better?
    # We want to minimize max_{k!=0} (sum_i h_i h_{i+k}) / (sum h)^2.
    # Let's try to make the autoconvolution more "spread out" or lower peak.
    # If we use a Gaussian-like h, the autoconvolution is also Gaussian-like.
    # The peak of the autoconvolution at lag 0 is sum h^2.
    # But we EXCLUDE lag 0.
    # So we want the sidelobes (k!=0) to be as low as possible relative to the total mass.
    
    # A known result: The "Optimal" sequence for minimizing peak sidelobe is related to Barker codes or Zadoff-Chu, but those are complex. For real non-negative, we are limited.
    # However, if we allow h to be non-negative, we can't use sign changes.
    # The flat function has sidelobes decaying linearly.
    # A Gaussian h has sidelobes decaying faster (exponentially).
    # Let's test a Gaussian h.
    
    # h_i = exp(-alpha * (i - N/2)^2).
    # We can optimize alpha.
    
    # Let's implement a search over a parametric family of "Bump" functions.
    # Family: h(x) = exp(-alpha * |x|^beta) or similar, defined on [-1/4, 1/4].
    # Since we have discrete intervals, we can define h_i based on the center of the interval.
    
    centers = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    
    def _eval_param(params):
        alpha, beta = params
        if alpha < 0 or beta <= 0:
            return 1e9
        h = np.exp(-alpha * np.abs(centers)**beta)
        # Ensure non-negative (exp is always positive)
        # Normalize? No, score is scale invariant.
        return _compute_score(h)
    
    best_score = _compute_score(np.ones(N) / N)
    best_h = np.ones(N) / N
    
    # Optimize alpha, beta
    # Use scipy.optimize.minimize
    # Initial guess: alpha=10, beta=2 (Gaussian)
    
    # We need to be careful with time.
    # _compute_score is O(N log N).
    # minimize might call _eval_param many times.
    
    try:
        # Use L-BFGS-B or Nelder-Mead
        # Nelder-Mead is robust for low-dimensional.
        result = minimize(_eval_param, [10, 2], method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if result.fun < best_score:
            best_score = result.fun
            alpha, beta = result.x
            best_h = np.exp(-alpha * np.abs(centers)**beta)
            
    except Exception as e:
        pass
    
    # Try a second family: Polynomial window (e.g., Blackman-like)
    # h_i = 1 - a1*cos(2*pi*t) + a2*cos(4*pi*t) ...
    # Simpler: h_i = (1 - (2*t)^2)^2 where t in [-0.5, 0.5]
    # Let's just try a few fixed shapes and optimize scaling/width.
    
    # Shape 1: Triangle
    t = np.linspace(-1, 1, N)
    h_tri = np.maximum(0, 1 - np.abs(t))
    if _compute_score(h_tri) < best_score:
        best_score = _compute_score(h_tri)
        best_h = h_tri
        
    # Shape 2: Sinc-like (but must be non-negative, so use |sinc| or sinc^2)
    h_sinc = np.sinc(t * 2) ** 2
    h_sinc = np.maximum(h_sinc, 0)
    if _compute_score(h_sinc) < best_score:
        best_score = _compute_score(h_sinc)
        best_h = h_sinc

    # Shape 3: Raised Cosine
    h_rc = 0.5 * (1 + np.cos(np.pi * t))
    if _compute_score(h_rc) < best_score:
        best_score = _compute_score(h_rc)
        best_h = h_rc

    # If time permits, run a local search on the full vector starting from the best parametric solution.
    # But the parametric solution is already a good candidate.
    # The seed program uses Frank-Wolfe. We are using Parametric Families.
    # This is a structurally different approach.
    
    # One more check: Is the flat function actually the best?
    # For flat h, max_{k!=0} conv_k = (N-1)/N^2.
    # For Gaussian h, the sidelobes decay much faster.
    # The max sidelobe will be much smaller than (N-1)/N^2.
    # So Gaussian should be better.
    
    # Let's refine the Gaussian search.
    # We might have missed the optimum if Nelder-Mead failed.
    # Let's do a grid search on alpha for beta=2.
    
    if time.time() < deadline - 5:
        for alpha in np.linspace(1, 100, 50):
            h_g = np.exp(-alpha * centers**2)
            sc = _compute_score(h_g)
            if sc < best_score:
                best_score = sc
                best_h = h_g

    return list(best_h)
