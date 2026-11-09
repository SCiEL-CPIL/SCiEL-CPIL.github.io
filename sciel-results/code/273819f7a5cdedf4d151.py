"""Second autocorrelation inequality: maximize ||f||_2^2 / (||f||_1 * ||f*f||_inf)
for nonnegative step heights on 50 equal intervals in [-1/4, 1/4].

Key insight from knowledge:
  - For nonnegative f, the autoconvolution peak is at lag 0 and equals sum(f_i^2) = ||f||_2^2.
  - So the ratio simplifies to: ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1 / ||f||_1.
  - Wait, that's only if the peak of f*f is exactly at lag 0. Let me reconsider.
  
  Actually, f*f at lag 0 is sum_i f_i^2 (for discrete autocorrelation with same-length support).
  But the "supremum norm of the autoconvolution" means max over all lags of (f*f)(lag).
  For nonnegative f, (f*f)(0) = sum f_i^2, and for other lags k, (f*f)(k) = sum_i f_i * f_{i+k}.
  Since all terms are nonnegative, (f*f)(0) >= (f*f)(k) for all k? No, that's not necessarily true.
  Actually (f*f)(0) = sum f_i^2, and (f*f)(k) = sum f_i f_{i+k}. By Cauchy-Schwarz,
  (sum f_i f_{i+k})^2 <= (sum f_i^2)(sum f_{i+k}^2) <= (sum f_i^2)^2 if the shifted sum is <= original sum.
  But sum f_{i+k}^2 <= sum f_i^2 only if the support shifts within bounds. For a general sequence,
  the peak of autocorrelation is indeed at lag 0 for nonnegative sequences? Let me think again.
  
  Actually no. Consider f = [1, 0, 1]. Then (f*f)(0) = 1+0+1 = 2, (f*f)(1) = 1*0 + 0*1 = 0,
  (f*f)(2) = 1*1 = 1. So peak is at lag 0.
  
  Consider f = [1, 1, 1]. (f*f)(0) = 3, (f*f)(1) = 1+1 = 2, (f*f)(2) = 1. Peak at lag 0.
  
  In general for nonnegative f, (f*f)(0) = sum f_i^2. For lag k>0, (f*f)(k) = sum_{i} f_i f_{i+k}.
  Note that sum f_i f_{i+k} <= sum f_i^2 is NOT always true. Example: f = [1, 10, 1].
  (f*f)(0) = 1 + 100 + 1 = 102. (f*f)(1) = 1*10 + 10*1 = 20. (f*f)(2) = 1*1 = 1. Peak at 0.
  
  Actually, by the property of autocorrelation, the peak is always at lag 0 for any real sequence
  (not just nonnegative) because (f*f)(0) = ||f||_2^2 and |(f*f)(k)| <= ||f||_2^2 by Cauchy-Schwarz.
  For nonnegative f, all values are nonnegative, so max is at lag 0.
  
  Therefore ||f*f||_inf = ||f||_2^2, and the ratio = ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1/||f||_1.
  
  So we want to MINIMIZE ||f||_1 = sum f_i subject to f_i >= 0.
  But wait, the ratio is scale-invariant! If we scale f by c, both ||f||_2^2 and ||f||_1 scale by c^2 and c respectively... 
  No: ||cf||_2^2 = c^2 ||f||_2^2, ||cf||_1 = c ||f||_1, ||cf*cf||_inf = c^2 ||f*f||_inf.
  Ratio = c^2 ||f||_2^2 / (c ||f||_1 * c^2 ||f*f||_inf) = ||f||_2^2 / (||f||_1 * ||f*f||_inf) * (1/c).
  So the ratio is NOT scale-invariant; it scales as 1/c. To maximize, we want c -> 0?
  That can't be right because then the ratio goes to infinity.
  
  Let me re-read the task: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
  
  Ratio = ||f||_2^2 / (||f||_1 * ||f*f||_inf)
  
  With scaling f -> c*f: numerator scales c^2, denominator scales c * c^2 = c^3. Ratio scales as 1/c.
  So to maximize, we want c as small as possible? But f must be nonnegative, and presumably there's no lower bound on magnitude.
  
  Hmm, but the task says "Choose 50 nonnegative heights". If there's no constraint on the scale, the ratio can be made arbitrarily large by making all heights very small. That doesn't make sense for a well-posed problem.
  
  Wait, maybe the L2 norm and L1 norm are normalized by the interval length? Or maybe the heights are constrained in some way I'm missing.
  
  Re-reading: "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]". The interval [-1/4, 1/4] has length 1/2. With 50 intervals, each has width 1/100.
  
  Perhaps the L2 norm is the continuous L2 norm: (sum f_i^2 * dx)^{1/2} where dx = 1/100.
  L1 norm: sum f_i * dx.
  Autoconvolution: the continuous autoconvolution of the step function.
  
  Let me reconsider with proper normalization. Let dx = 1/100.
  ||f||_2^2 = sum f_i^2 * dx
  ||f||_1 = sum f_i * dx
  (f*f)(t) = integral f(s) f(t-s) ds. For step functions, this is a piecewise constant function.
  
  The supremum of the autoconvolution... For nonnegative f, the peak is at t=0.
  (f*f)(0) = integral f(s)^2 ds = sum f_i^2 * dx = ||f||_2^2.
  
  So again ratio = ||f||_2^2 / (||f||_1 * ||f||_2^2) = 1/||f||_1.
  
  And ||f||_1 = sum f_i * dx. To maximize 1/||f||_1, minimize sum f_i. But f_i >= 0, so infimum is 0.
  
  This suggests the problem must have an implicit constraint, or I'm misunderstanding the norms.
  
  Maybe the "squared L2 norm" means ||f||_2^2 without the dx factor, i.e., just sum f_i^2.
  And L1 norm is sum f_i. And autoconvolution supremum is max of discrete autocorrelation.
  
  In that case, ratio = (sum f_i^2) / ((sum f_i) * max_k sum_i f_i f_{i+k}).
  
  For nonnegative f, max is at k=0, so max = sum f_i^2.
  Ratio = (sum f_i^2) / ((sum f_i) * (sum f_i^2)) = 1/(sum f_i).
  
  Still the same issue.
  
  Unless... the autoconvolution is defined differently, or the "supremum norm" is over a restricted range?
  
  Or perhaps the problem is that the ratio is meant to be a lower bound on a constant C2, and the actual quantity is different.
  
  Let me just implement a direct numerical search. I'll use the interpretation that:
  - f is a vector of 50 nonnegative numbers
  - ||f||_2^2 = sum f_i^2
  - ||f||_1 = sum f_i  
  - autoconvolution peak = max over lags k of sum_i f_i * f_{i+k} (with zero padding)
  - ratio = ||f||_2^2 / (||f||_1 * autoconv_peak)
  
  And I'll normalize so that sum f_i = 1 to avoid the scaling issue, then the ratio becomes sum f_i^2 / autoconv_peak.
  
  Actually, let me just do a direct optimization. I'll fix the scale by setting sum f_i = 1, then maximize sum f_i^2 / max_k autocorr_k.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _autoconv_peak(f):
    """Compute the peak of the discrete autoconvolution (autocorrelation) of f."""
    # Full autocorrelation: for lag k, sum_i f[i] * f[i+k] with zero padding
    # Use FFT for efficiency
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    F = np.fft.rfft(f, n=size)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=size)
    # The linear autocorrelation is in the first n values (lags 0 to n-1)
    # and the negative lags are in the second half
    # Peak is the maximum of the full autocorrelation
    return np.max(acf[:n])  # Nonnegative f, peak should be at lag 0

def _ratio(f):
    """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf)"""
    f = np.asarray(f, dtype=float)
    f = np.maximum(f, 0.0)  # Ensure nonnegative
    l2_sq = np.sum(f ** 2)
    l1 = np.sum(f)
    if l1 == 0:
        return 0.0
    peak = _autoconv_peak(f)
    if peak == 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _neg_ratio(f):
    return -_ratio(f)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_f = np.ones(N)
    best_score = _ratio(best_f)
    
    # Strategy: Since for nonnegative f, the autoconv peak is at lag 0 = ||f||_2^2,
    # the ratio simplifies to 1/||f||_1. To maximize, we minimize ||f||_1.
    # But this is scale-dependent. Let me normalize by fixing ||f||_2^2 = 1.
    # Then ratio = 1 / ||f||_1, and we minimize ||f||_1 subject to ||f||_2^2 = 1.
    # This is equivalent to making f as "spread out" as possible... no, to minimize L1
    # given fixed L2, we want f to be as concentrated as possible (single spike).
    # A single spike: f = [1, 0, ..., 0]. L2^2 = 1, L1 = 1, autoconv peak = 1. Ratio = 1.
    # Uniform: f = [1/sqrt(N)]*N. L2^2 = 1, L1 = sqrt(N), autoconv peak = 1. Ratio = 1/sqrt(N).
    # So concentrated is better!
    
    # Let me verify: f = [1, 0, ..., 0]. 
    # L2^2 = 1, L1 = 1, autoconv: lag 0 = 1, lag 1 = 0, ... Peak = 1. Ratio = 1/(1*1) = 1.
    # f = [1/sqrt(2), 1/sqrt(2), 0, ...]. L2^2 = 1, L1 = sqrt(2), autoconv lag 0 = 1, lag 1 = 1/2. Peak = 1. Ratio = 1/sqrt(2) ~ 0.707.
    # So yes, concentrated is better.
    
    # But wait, the seed gives 1.0 with all ones. Let me check:
    # f = [1]*50. L2^2 = 50, L1 = 50, autoconv lag 0 = 50, lag 1 = 49, ..., lag 49 = 1. Peak = 50.
    # Ratio = 50 / (50 * 50) = 1/50 = 0.02. That's way less than 1.
    
    # Hmm, so the seed score of 1.0 must use a different normalization or the ratio is defined differently.
    # Maybe the ratio is ||f||_2^2 / (||f||_1 * ||f*f||_inf) where these are continuous norms with dx.
    # Or maybe the "score" in the task is not the raw ratio but something else.
    
    # Let me just do a numerical search to find a good configuration.
    
    # Try a concentrated profile
    f_concentrated = np.zeros(N)
    f_concentrated[0] = 1.0
    score_conc = _ratio(f_concentrated)
    
    # Try symmetric concentrated
    f_sym = np.zeros(N)
    f_sym[N//2] = 1.0
    score_sym = _ratio(f_sym)
    
    # Try a few random initializations and local optimization
    np.random.seed(42)
    
    candidates = []
    
    # Candidate 1: Single spike at center
    c1 = np.zeros(N)
    c1[N//2] = 1.0
    candidates.append(c1)
    
    # Candidate 2: Two adjacent spikes
    c2 = np.zeros(N)
    c2[N//2 - 1] = 1.0
    c2[N//2] = 1.0
    candidates.append(c2)
    
    # Candidate 3: Geometric decay from center
    c3 = np.zeros(N)
    for i in range(N):
        dist = abs(i - N//2)
        c3[i] = 0.8 ** dist
    candidates.append(c3)
    
    # Candidate 4: Power law decay
    c4 = np.zeros(N)
    for i in range(N):
        dist = abs(i - N//2) + 1
        c4[i] = 1.0 / (dist ** 2)
    candidates.append(c4)
    
    # Candidate 5: Random sparse
    c5 = np.zeros(N)
    indices = np.random.choice(N, size=5, replace=False)
    c5[indices] = 1.0
    candidates.append(c5)
    
    # Evaluate all candidates
    for cand in candidates:
        s = _ratio(cand)
        if s > best_score:
            best_score = s
            best_f = cand.copy()
    
    # Local optimization using Nelder-Mead on the best candidate
    # We'll optimize in a parameterized space to maintain nonnegativity
    # Use log-space: f_i = exp(x_i)
    
    def _ratio_from_log(x):
        f = np.exp(x)
        return _ratio(f)
    
    def _neg_ratio_from_log(x):
        return -_ratio_from_log(x)
    
    # Start from best_f, convert to log space
    # Handle zeros by using a small epsilon
    eps = 1e-10
    f_init = np.maximum(best_f, eps)
    x_init = np.log(f_init)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(_neg_ratio_from_log, x_init, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if time.time() < deadline:
            f_opt = np.exp(result.x)
            s_opt = _ratio(f_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_f = f_opt.copy()
    except Exception:
        pass
    
    # Try a few more random restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            # Random sparse initialization
            f_rand = np.zeros(N)
            n_nonzero = np.random.randint(1, 10)
            indices = np.random.choice(N, size=n_nonzero, replace=False)
            f_rand[indices] = np.random.uniform(0.1, 1.0, size=n_nonzero)
            
            # Normalize to L2 = 1
            l2 = np.sqrt(np.sum(f_rand ** 2))
            if l2 > 0:
                f_rand = f_rand / l2
            
            # Local optimization
            f_init_r = np.maximum(f_rand, eps)
            x_init_r = np.log(f_init_r)
            
            try:
                result_r = minimize(_neg_ratio_from_log, x_init_r, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                f_opt_r = np.exp(result_r.x)
                s_r = _ratio(f_opt_r)
                if s_r > best_score:
                    best_score = s_r
                    best_f = f_opt_r.copy()
            except Exception:
                pass
    
    # Return the best found
    return list(np.maximum(best_f, 0.0))

# EVOLVE-BLOCK-END
