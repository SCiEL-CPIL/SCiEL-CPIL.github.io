"""Rewritten program: targeted minimization of the third autocorrelation inequality value.

The objective is:
    value = abs(2 * N * max(conv) / (sum(h))**2)
where conv = np.convolve(h, h) (full, length 2N-1).

Key observations:
1. The max of conv is at index N-1 (zero-lag) and equals sum(h^2).
2. For a given sum S, minimizing value is equivalent to minimizing max(conv)/S^2.
3. The zero-lag term is sum(h^2) = ||h||^2. By Cauchy-Schwarz, ||h||^2 >= S^2/N,
   so value >= 2N * (S^2/N) / S^2 = 2.0. This is the seed value.
4. To beat 2.0, we need max(conv) < sum(h^2) * (some factor) -- but wait, max(conv)
   is at least sum(h^2) since that's the zero-lag term. So value >= 2.0 always?

Wait -- let me re-examine. The max is over ALL entries of conv, not just non-negative ones.
If some off-peak conv values are negative, the max could be less than sum(h^2) only if
sum(h^2) is not the maximum. But sum(h^2) = conv[N-1] is always positive (it's sum of squares).
So max(conv) >= conv[N-1] = sum(h^2) >= S^2/N.

Thus value >= 2N * (S^2/N) / S^2 = 2.0. The seed is optimal?

But the current program scores 1.3349, meaning value ~ 2.0/1.3349 ≈ 1.498. That's below 2.0!

Let me re-read: "The seed np.ones(400) gives 2.0." And "search score is seed/value".
So if value < 2.0, score > 1.0. My analysis above shows value >= 2.0, so something is wrong.

Re-reading the formula: value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)

Ah wait -- np.convolve(h, h) for h with negative entries: conv[N-1] = sum(h[i]*h[i]) = sum(h^2) > 0.
But other entries can be negative. The max picks the largest value, which is at least sum(h^2).

Unless... h can have negative entries making sum(h) small while sum(h^2) stays moderate?
No, that makes value larger.

Hmm, let me reconsider. If sum(h) is very large (positive), and we can make max(conv) relatively small...
But max(conv) >= sum(h^2) >= sum(h)^2/N by Cauchy-Schwarz.

So value = 2N * max(conv) / sum(h)^2 >= 2N * sum(h)^2/N / sum(h)^2 = 2.

This means value >= 2.0 is a hard lower bound! The seed achieves exactly 2.0.

But the score 1.3349 implies value ≈ 1.498 < 2.0. This contradicts my analysis.

Let me re-examine: maybe I'm wrong about which index is the peak.
np.convolve(h, h) for full mode: result[k] = sum_i h[i] * h[i + k - (N-1)] for valid i.
At k = N-1: sum_i h[i] * h[i] = sum(h^2). Yes.

Wait -- could np.max be picking a NEGATIVE value if all are negative? No, sum(h^2) > 0.

I'm confused. Let me just trust the empirical result and focus on optimization.
The current program achieves value ≈ 1.498. I'll try to improve further with:
- Better spectral methods
- More aggressive coordinate descent
- Randomized restarts with refinement
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 140.0

def _value(h):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=100.0):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Random perturbations of the seed with refinement
    # The seed is all ones. Small perturbations might help if the landscape is non-convex.
    for restart in range(30):
        if not deadline_ok():
            break
        # Create perturbed seed
        h = np.ones(N) + rng.normal(0, 0.1, N)
        # Ensure sum is not zero
        if abs(np.sum(h)) < 1.0:
            h += 1.0
        
        # Coordinate descent refinement
        for it in range(200):
            if not deadline_ok():
                break
            improved = False
            # Random subset of coordinates for speed
            idx = rng.choice(N, size=min(50, N), replace=False)
            for i in idx:
                for delta in [0.01, -0.01, 0.05, -0.05]:
                    h_trial = h.copy()
                    h_trial[i] += delta
                    v_trial = _value(h_trial)
                    if v_trial < best_v:
                        h = h_trial
                        best_v = v_trial
                        best_h = h.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Strategy 2: DCT-based optimization
    # Use low-frequency DCT modes to shape h
    K = 20
    n = np.arange(N)
    k_modes = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k_modes[None, :] / (2 * N)
    dct_basis = np.cos(angles)
    norms = np.sqrt(np.sum(dct_basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    dct_basis = dct_basis / norms
    
    def dct_h(coeffs):
        return np.ones(N) * 0.5 + dct_basis @ coeffs
    
    def dct_obj(coeffs):
        h = dct_h(coeffs)
        return _value(h)
    
    for restart in range(10):
        if not deadline_ok():
            break
        coeffs0 = rng.normal(0, 0.5, K)
        res = minimize(dct_obj, coeffs0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None:
            h = dct_h(res.x)
            update_best(h)
    
    # Strategy 3: Chirp-based parameterization
    def chirp_h(params):
        phase, rate, amp, base = params
        i = np.arange(N, dtype=float)
        ang = phase + rate * (i / N) ** 2
        h = base + amp * np.cos(ang)
        return h
    
    def chirp_obj(params):
        h = chirp_h(params)
        return _value(h)
    
    for restart in range(5):
        if not deadline_ok():
            break
        p0 = [rng.uniform(-np.pi, np.pi), rng.uniform(-100, 100), 
              rng.uniform(-1, 1), rng.uniform(0.5, 1.5)]
        res = minimize(chirp_obj, p0, method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500})
        if res.x is not None:
            h = chirp_h(res.x)
            update_best(h)
    
    # Strategy 4: Multi-chirp
    def multi_chirp_h(params):
        # params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
        i = np.arange(N, dtype=float)
        h = np.full(N, params[6])
        for j in range(2):
            phase = params[j * 3]
            rate = params[j * 3 + 1]
            amp = params[j * 3 + 2]
            ang = phase + rate * (i / N) ** 2
            h += amp * np.cos(ang)
        return h
    
    def multi_chirp_obj(params):
        h = multi_chirp_h(params)
        return _value(h)
    
    for restart in range(5):
        if not deadline_ok():
            break
        p0 = [rng.uniform(-np.pi, np.pi), rng.uniform(-50, 50), rng.uniform(-0.5, 0.5),
              rng.uniform(-np.pi, np.pi), rng.uniform(-50, 50), rng.uniform(-0.5, 0.5),
              rng.uniform(0.5, 1.5)]
        res = minimize(multi_chirp_obj, p0, method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 800})
        if res.x is not None:
            h = multi_chirp_h(res.x)
            update_best(h)
    
    # Strategy 5: Final coordinate descent polish on best_h
    if deadline_ok():
        h = best_h.copy()
        for it in range(300):
            if not deadline_ok():
                break
            improved = False
            idx = rng.choice(N, size=100, replace=False)
            for i in idx:
                for delta in [0.005, -0.005, 0.02, -0.02, 0.1, -0.1]:
                    h_trial = h.copy()
                    h_trial[i] += delta
                    v_trial = _value(h_trial)
                    if v_trial < best_v:
                        h = h_trial
                        best_v = v_trial
                        best_h = h.copy()
                        improved = True
                        break
            if not improved:
                break
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
