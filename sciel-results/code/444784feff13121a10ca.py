"""SDP-inspired approach: exploit the fact that h = D @ D^T (rank factor of PSD Toeplitz)
gives zero autocorrelation everywhere except the peak, yielding value = 2.0 exactly.
We parameterize h as a sum of structured PSD components + a large constant, then
optimize the structure to keep h realizable while making the autocorrelation as flat
as possible. Since we can't achieve perfect flatness in 400 dimensions with a rank-1
Toeplitz (that's the BPSK/LC bound), we use multiple rank components to approximate
a perfectly flat autocorrelation.

Key insight: if autocorr[l] = 0 for all l != 0, then value = 2.0 exactly (the seed).
We want value < 2.0, so we need negative autocorrelation values (since np.max picks
the maximum, and we want that maximum to be as close to 0 as possible while keeping
sum(h)^2 large).

Actually, re-reading: value = abs(2*N*max(conv)/sum(h)^2). For ones, conv max = 400, sum = 400,
so value = 2*400*400/160000 = 2.0. To get value < 2.0, we need 2*N*max(conv)/sum^2 < 2,
i.e., max(conv)*N < sum^2, i.e., max(conv)/N < mean^2.

The peak of autocorrelation is conv[0] = sum(h^2). So we need sum(h^2)/N < mean(h)^2,
which is impossible since by Cauchy-Schwarz sum(h^2) >= sum(h)^2/N.

So value >= 2.0 always? No wait: np.max(conv) takes the maximum over ALL entries of conv,
not just the peak. The peak conv[0] = sum(h^2) is always >= sum(h)^2/N. But other entries
could be larger? No, conv[0] = sum(h^2) >= (sum(h))^2/N by C-S. So max(conv) >= sum(h)^2/N,
meaning value >= 2.0 always.

So the theoretical minimum is 2.0 (achieved by constant sequences or any sequence where
all off-peak autocorrelations are <= 0 and the peak dominates). The seed achieves exactly 2.0.

Wait, but the seed score is 1.3349, and the seed value should be 2.0. Score = seed_value/value = 2.0/value.
So score 1.3349 means value ≈ 2.0/1.3349 ≈ 1.498. That's below 2.0!

Let me re-examine: for h = ones(400), conv = [1,2,...,400,...,2,1], max = 400.
value = 2*400*400/400^2 = 2.0. Score = 2.0/2.0 = 1.0. But the problem says "The seed np.ones(400) gives 2.0" for the value, and score is seed/value.

So current best score 1.3349 means best value = 2.0/1.3349 ≈ 1.498. That IS below 2.0!

How can max(conv) be less than sum(h)^2/N? That's impossible for conv[0]. Unless...
the autocorrelation is computed differently. Let me re-read: np.convolve(h,h) gives
full convolution, which for autocorrelation has the peak at index N-1 (not 0!).

np.convolve(h, h) for h of length N gives an array of length 2N-1. The peak (lag 0)
is at index N-1. So max over all entries includes all lags. For constant h=1:
conv = [1, 2, 3, ..., 400, 399, ..., 2, 1]. Max = 400 at index 399 (= N-1).
value = 2*400*400/160000 = 2.0.

For a random sequence, conv[399] = sum(h^2) > mean(h)^2 * N typically. But some entries
at large lags could be negative if h has negative values.

The key: we want max(conv) to be as small as possible relative to sum(h)^2. Since conv[399]
= sum(h^2) >= sum(h)^2/N, we have value >= 2.0 if max(conv) >= conv[399]. But what if some
off-peak entry exceeds the peak? That would make it worse. We want all entries to be
uniformly small.

Actually the theoretical lower bound: by Parseval, sum(conv[l]^2) = sum(h[k]^4) * ... no.
Let's think about it via DFT. Let H = DFT(h). Then conv = ifft(|H|^2) in the circular sense...
not exactly. Full linear convolution via FFT.

The constraint is: minimize max(conv[l]) / (sum(h))^2. Since conv[N-1] = sum(h^2),
and sum(h^2) >= (sum(h))^2/N, we get max(conv[l]) >= sum(h^2) >= (sum(h))^2/N,
so value = 2N * max(conv) / sum(h)^2 >= 2N * sum(h)^2/(N*sum(h)^2) = 2.

So value >= 2.0 ALWAYS. The best possible is exactly 2.0, achieved when sum(h^2) = sum(h)^2/N,
which means all h[i] are equal. So the optimal answer IS the seed?

But the current program scores 1.3349, implying value < 2.0. This contradicts my analysis.
Let me re-check: maybe I'm wrong about which index is the peak.

For h = [a, b] (N=2), np.convolve(h,h) = [a^2, ab+ba, b^2] = [a^2, 2ab, b^2].
Peak at lag 0 is at index N-1 = 1: value = 2ab. For h=[1,1]: [1,2,1], max=2, sum=2.
value = 2*2*2/4 = 2. For h=[1,-1]: [1,-2,1], max=1, sum=0. Undefined!

For h=[1,1]: value=2. For h=[1,1,1]: conv=[1,2,3,2,1], max=3, sum=3, value=2*3*3/9=2.

So constant gives exactly 2.0. To get below 2.0, we need max(conv) < sum(h)^2/N.
But conv[N-1] = sum(h^2) >= sum(h)^2/N. So max(conv) >= conv[N-1] >= sum(h)^2/N.
Thus value >= 2.0. Always.

Unless... sum(h) can be negative, and sum(h)^2 is the same. No difference.

I think the theoretical minimum IS 2.0, and the score can never exceed 1.0 for value,
meaning score = 2.0/value <= 1.0. But the problem states the current score is 1.3349,
which means value < 2.0. This is a contradiction unless I'm misunderstanding the formula.

Re-reading: "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"

And "The seed np.ones(400) gives 2.0" - this confirms the formula interpretation.
"search score is seed/value" where seed = 2.0. So score = 2.0 / value.

For score 1.3349, value = 2.0/1.3349 ≈ 1.498 < 2.0.

This is mathematically impossible by my C-S argument above. Let me re-examine...

OH WAIT. np.convolve(h, h) is the LINEAR convolution, and the "autocorrelation" in the
traditional sense has the peak at lag 0. But for linear convolution of h with itself,
the value at index i is sum_k h[k]*h[i-k]. The peak at lag 0 (i = N-1) is sum_k h[k]^2.
But other indices could be LARGER than the peak if h has special structure?

No! By Cauchy-Schwarz, |sum_k h[k] h[k+d]| <= sqrt(sum h[k]^2 * sum h[k+d]^2) <= sum h[k]^2.
So |conv[d]| <= conv[0] (the peak) for all d. Thus max(conv) = conv[N-1] = sum(h^2).

So value = 2*N*sum(h^2)/sum(h)^2 >= 2*N*sum(h)^2/(N*sum(h)^2) = 2.

The value is ALWAYS >= 2.0. The minimum is exactly 2.0. So the best possible score
is 2.0/2.0 = 1.0.

But the stated current score is 1.3349 > 1.0. This means either:
1. I'm misunderstanding the problem
2. The score metric is different
3. There's a numerical precision issue

Let me just focus on producing a valid program that returns a good result, keeping
the same interface. Given the mathematical constraint, the best achievable value is 2.0,
so the best score is 1.0. But since the problem claims 1.3349 is achievable, maybe I'm
wrong. Let me just optimize and see.

Given the ambiguity, I'll write a clean program that:
1. Tries to minimize max(conv)/sum(h)^2 using the structure of the problem
2. Uses PSD/Toeplitz structure awareness
3. Has a tight deadline guard
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0


def _value(h):
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))


def _build_toeplitz_rank_factorization(M, seed=0):
    """
    Generate a Toeplitz-like PSD matrix T of size N x N using M rank-1 components.
    T = sum_m u_m u_m^T where u_m are chirp-like vectors.
    The autocorrelation of h will relate to T.
    
    Actually, if we set h[k] = sum_m u_m[k] (real vector), then
    autocorr[l] = sum_k h[k] h[k+l] which relates to the structure of u_m.
    
    For a single chirp u[k] = exp(i*2*pi*f*k^2/N), the autocorrelation is approximately
    a delta at lag 0 (for N large). Multiple chirps with different frequencies give
    a sum of deltas, but they interfere.
    """
    rng = np.random.default_rng(seed)
    k = np.arange(N, dtype=float)
    t = k / N
    
    # Use quadratic phase (chirp) components
    frequencies = rng.uniform(1, N//2, size=M)
    phases = rng.uniform(0, 2*np.pi, size=M)
    amps = rng.uniform(0.5, 2.0, size=M)
    
    h = np.zeros(N)
    for m in range(M):
        angle = 2.0 * np.pi * frequencies[m] * t**2 + phases[m]
        h += amps[m] * np.cos(angle)
    
    return h


def _smooth_obj(coeffs, beta=60.0):
    """Smooth objective for optimization of h parameters."""
    h = np.asarray(coeffs, dtype=float)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = np.fft.irfft(np.fft.rfft(h, 2*N)**2, 2*N)[:2*N-1]
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -50, 50)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    val = abs(2.0 * N * lse / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _exact_obj(h):
    return _value(h)


def construct():
    t0 = time.time()
    
    def deadline_ok():
        return (time.time() - t0) < _DEADLINE
    
    # Start with seed
    best_h = np.ones(N)
    best_v = _value(best_h)  # Should be 2.0
    
    def update(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v and v > 0:
            best_v = v
            best_h = h.copy()
    
    # Strategy 1: Multi-chirp with optimization
    # Parameterize h as base + sum of chirps, optimize amplitudes/phases/rates
    
    M_CHIRPS = 8
    n_params = 1 + 3 * M_CHIRPS  # base + M*(phase, rate, amp)
    
    def chirp_obj(params):
        base = params[0]
        h = np.full(N, base)
        t = np.arange(N, dtype=float) / N
        for m in range(M_CHIRPS):
            phase = params[1 + 3*m]
            rate = params[2 + 3*m]
            amp = params[3 + 3*m]
            angle = 2.0 * np.pi * rate * t**2 + phase
            h += amp * np.cos(angle)
        return _smooth_obj(h)
    
    def chirp_h(params):
        base = params[0]
        h = np.full(N, base)
        t = np.arange(N, dtype=float) / N
        for m in range(M_CHIRPS):
            phase = params[1 + 3*m]
            rate = params[2 + 3*m]
            amp = params[3 + 3*m]
            angle = 2.0 * np.pi * rate * t**2 + phase
            h += amp * np.cos(angle)
        return h
    
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for trial in range(4):
                if not deadline_ok():
                    break
                x0 = np.zeros(n_params)
                x0[0] = 1.0  # base
                for m in range(M_CHIRPS):
                    x0[1 + 3*m] = rng.uniform(0, 2*np.pi)     # phase
                    x0[2 + 3*m] = rng.uniform(1, 20)           # rate
                    x0[3 + 3*m] = rng.uniform(-0.5, 0.5)       # amp
                
                res = minimize(chirp_obj, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = chirp_h(res.x)
                    update(h)
        except Exception:
            pass
    
    # Strategy 2: Direct optimization of h with normalization constraint
    # Use the fact that scaling h doesn't change value (both num and den scale)
    # So we can fix sum(h) = N and minimize max(conv)
    
    def _normalized_smooth_obj(h_raw, beta=60.0):
        h = np.asarray(h_raw, dtype=float)
        s = float(np.sum(h))
        if abs(s) < 1e-8:
            return 1e10
        h_norm = h * (N / s)  # Normalize so sum = N
        conv = np.fft.irfft(np.fft.rfft(h_norm, 2*N)**2, 2*N)[:2*N-1]
        m = np.max(conv)
        scaled = beta * (conv - m)
        scaled = np.clip(scaled, -50, 50)
        lse = m + np.log(np.sum(np.exp(scaled))) / beta
        return lse  # Since sum(h_norm)^2 = N^2, value = 2*N*lse/N^2 = 2*lse/N
    
    if deadline_ok():
        try:
            h_init = np.ones(N)
            h_init += 0.1 * np.random.default_rng(7).normal(size=N)
            
            res = minimize(_normalized_smooth_obj, h_init, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                update(res.x)
        except Exception:
            pass
    
    # Strategy 3: Coordinate descent on h directly (fast, per-element)
    # Minimize max(conv) / sum(h)^2 via coordinate-wise updates
    if deadline_ok():
        try:
            h = best_h.copy()
            beta = 50.0
            
            for _ in range(30):
                if not deadline_ok():
                    break
                improved = False
                s = np.sum(h)
                if abs(s) < 1e-8:
                    break
                
                # Try random elements with small perturbations
                rng = np.random.default_rng(123)
                for _ in range(200):
                    if not deadline_ok():
                        break
                    i = rng.integers(0, N)
                    for delta in [0.01, -0.01, 0.005, -0.005]:
                        h_new = h.copy()
                        h_new[i] += delta
                        s_new = np.sum(h_new)
                        if abs(s_new) < 1e-8:
                            continue
                        v = _value(h_new)
                        if v < best_v:
                            h = h_new
                            best_v = v
                            best_h = h_new.copy()
                            improved = True
                            break
                if not improved:
                    break
        except Exception:
            pass
    
    # Strategy 4: DFT-based parameterization - low frequency components
    # h = base + sum_k (a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N))
    # Optimize in this subspace
    
    K = 15
    n_dft_params = 1 + 2 * K
    
    def dft_to_h(params):
        base = params[0]
        h = np.full(N, base)
        n = np.arange(N)
        for k in range(1, K + 1):
            a = params[1 + 2*(k-1)]
            b = params[2 + 2*(k-1)]
            angle = 2.0 * np.pi * k * n / N
            h += a * np.cos(angle) + b * np.sin(angle)
        return h
    
    def dft_smooth_obj(params, beta=60.0):
        h = dft_to_h(params)
        s = float(np.sum(h))
        if abs(s) < 1e-8:
            return 1e10
        conv = np.fft.irfft(np.fft.rfft(h, 2*N)**2, 2*N)[:2*N-1]
        m = np.max(conv)
        scaled = beta * (conv - m)
        scaled = np.clip(scaled, -50, 50)
        lse = m + np.log(np.sum(np.exp(scaled))) / beta
        return abs(2.0 * N * lse / (s * s))
    
    if deadline_ok():
        try:
            # Initialize from best
            n_arr = np.arange(N)
            h_curr = best_h.copy()
            mean = float(np.mean(h_curr))
            x0 = np.zeros(n_dft_params)
            x0[0] = mean
            for k in range(1, K + 1):
                angle = 2.0 * np.pi * k * n_arr / N
                target = h_curr - mean
                cos_k = np.cos(angle)
                sin_k = np.sin(angle)
                A = np.column_stack([cos_k, sin_k])
                sol = np.linalg.lstsq(A, target, rcond=None)[0]
                x0[1 + 2*(k-1)] = sol[0]
                x0[2 + 2*(k-1)] = sol[1]
            
            res = minimize(dft_smooth_obj, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.x is not None:
                h = dft_to_h(res.x)
                update(h)
        except Exception:
            pass
    
    # Final polish: tight coordinate descent
    if deadline_ok():
        try:
            h = best_h.copy()
            s = np.sum(h)
            current_v = _value(h)
            rng = np.random.default_rng(999)
            
            for epoch in range(20):
                if not deadline_ok():
                    break
                step = 0.002 / (epoch + 1)
                for i in range(N):
                    if not deadline_ok():
                        break
                    best_local = current_v
                    best_val = h[i]
                    for delta in [step, -step, 2*step, -2*step]:
                        h_trial = h.copy()
                        h_trial[i] = best_val + delta
                        s_t = np.sum(h_trial)
                        if abs(s_t) < 1e-8:
                            continue
                        v_t = _value(h_trial)
                        if v_t < best_local:
                            best_local = v_t
                            best_val = h_trial[i]
                    h[i] = best_val
                s = np.sum(h)
                if abs(s) < 1e-8:
                    break
                current_v = _value(h)
                if current_v < best_v:
                    best_v = current_v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Validation
    s = float(np.sum(best_h))
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h


# Module-level init to ensure we're ready
import time as _time
_t0_module = _time.time()
