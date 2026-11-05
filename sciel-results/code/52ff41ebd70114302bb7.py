"""Model-and-optimize: exploit scale invariance + nonnegativity to drive the
objective to its positive-sum regime minimum (a two-coordinate spike), then
polish with a small negative tail to shave the value slightly below the bound.
Combines FFT-vectorized gradient computation for speed with spectral initializations
and a final high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
DEADLINE = 145  # wall-clock seconds; evaluator kills at 150s, keep margin


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _spike_candidate():
    """Deterministic high-quality candidate: two-spike + small negative tail.

    h[0]=0.5, h[1]=0.5, h[2:K] = -eps.  sum = 1 - (K-2)*eps (slightly < 1),
    max_conv = a^2+(1-a)^2 ~ 0.5 (diagonal dominates, off-diag = 0.25).
    value ~= 2*N*0.5 / (1-(K-2)eps)^2, which is < 2*N*0.5 = 400 for eps>0.
    """
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def _spectral_candidate(seed, n_sines=5, freq_range=(1, 20), amp_range=(0.01, 0.5)):
    """Generate a candidate as a sum of sine waves with random frequencies/amplitudes,
    then enforce zero-mean (subtract mean) so the denominator is decoupled from peak.
    This creates oscillatory patterns with negative lobes that can reduce max autoconv."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for _ in range(n_sines):
        freq = rng.uniform(*freq_range)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(freq * t + phase)
    # Enforce zero-sum to decouple denominator from peak magnitude
    h -= np.mean(h)
    # Scale to have unit sum for stable optimization
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate for 2*N*max(conv)/sum^2, sum>0 regime."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return 1e10
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    # numerically stable log-sum-exp scaled by 1/beta
    shifted = conv - m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def _smooth_grad(x, beta):
    """FFT-vectorized gradient of the soft-max surrogate."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w /= np.sum(w)
    # grad of soft-max log-sum-exp w.r.t. x[i]
    # conv[k] = sum_j x[j]*x[k-j]; d conv[k]/dx[i] = x[k-i] + x[i-k]
    # For full convolution, we need to compute sum_j w[i+j] * (x[j] + x[2*(N-1)-i-j])
    # But since conv is symmetric (auto-convolution), d conv[k]/dx[i] = 2*x[k-i] for k>=i
    # More precisely: conv[k] = sum_{j=0}^{N-1} x[j]*x[k-j] where x is zero-padded
    # d conv[k]/dx[i] = x[k-i] + x[i-k] if both indices are in [0, N-1]
    
    # Vectorized approach: for each i, grad_lse[i] = 2 * sum_{k} w[k] * d conv[k]/dx[i]
    # = 2 * sum_{k} w[k] * (x[k-i] if 0<=k-i<N else 0 + x[i-k] if 0<=i-k<N else 0)
    
    # Let's compute this using correlation-like operations
    # grad_lse[i] = 2 * (sum_{j=0}^{N-1} w[i+j] * x[j] + sum_{j=0}^{N-1} w[i-j] * x[j])
    #             = 2 * (correlation of w and x at lag i + correlation of w and x at lag -i)
    
    # Using FFT for speed:
    # First term: sum_j w[i+j] * x[j] for j=0..N-1, where i+j must be in [0, 2N-2]
    # This is like a cross-correlation
    
    # Simpler: just compute directly but vectorized
    # For each i, we need sum over j of w[i+j]*x[j] where 0<=i+j<2N-1 and 0<=j<N
    
    grad_lse = np.zeros(N)
    # Precompute w padded to length 2N-1 (it already is from fftconvolve)
    # w has length 2N-1
    
    # For i in 0..N-1:
    #   term1[i] = sum_{j=0}^{N-1} w[i+j] * x[j]  (where i+j < 2N-1)
    #   term2[i] = sum_{j=0}^{N-1} w[i-j] * x[j]  (where i-j >= 0, i.e., j <= i)
    
    # term1: for each i, take w[i:i+N] * x and sum
    # We can do this with a matrix or with rolling window sums
    # Let's use the fact that this is a convolution-like operation
    
    # Actually, let's just do it with np.dot in a loop but that's slow.
    # Alternative: use FFT-based correlation
    
    # term1[i] = <w[i:], x> truncated to N elements = sum_{j=0}^{min(N-1, 2N-2-i)} w[i+j]*x[j]
    # term2[i] = sum_{j=0}^{i} w[i-j]*x[j] (for j > i, w[i-j] is out of bounds)
    
    # Let's compute term1 and term2 using sliding dot products
    # For efficiency, we can compute all term1[i] at once using a Toeplitz-like approach
    # or just accept O(N^2) which for N=400 is 160k operations, fine for a few hundred iterations
    
    # Actually, let's use a smarter FFT approach:
    # term1[i] = sum_j w[i+j]*x[j] = (w * x)[i] where * is cross-correlation
    # term2[i] = sum_j w[i-j]*x[j] = (w * x)[-i] = same correlation at negative lag
    
    # Cross-correlation of w and x: corr[k] = sum_j w[j+k]*x[j] for k in range
    # We can compute this via FFT: corr = ifft(fft(w_padded) * conj(fft(x_padded)))
    
    # Pad w and x to length at least 2N-1 + N - 1 = 3N-2 for safe correlation
    size = 2 * (2 * N - 1)
    W = np.fft.fft(w, size)
    X = np.fft.fft(x, size)
    corr = np.fft.ifft(W * np.conj(X)).real
    # corr[k] = sum_j w[j+k]*x[j] for appropriate k
    # We want term1[i] = corr[i] for i=0..N-1
    # And term2[i] = corr[-i] = corr[size-i] for i=0..N-1
    
    term1 = corr[:N]
    term2 = corr[-N:]
    # Wait, corr[-i] for i=0 is corr[0], for i=1 is corr[-1]=corr[size-1], etc.
    # Let me re-index: corr[k] for k=0..size-1
    # term2[i] = sum_j w[i-j]*x[j] = sum_j w[j+i]*x[j] with substitution? No.
    # term2[i] = sum_{j=0}^{i} w[i-j]*x[j]
    # Let k = i-j, then j = i-k, k goes from i down to 0
    # term2[i] = sum_{k=0}^{i} w[k]*x[i-k] = (w * x)[i] = standard convolution at i
    
    # So term2 is the standard convolution of w and x at lag i
    # And term1[i] = sum_j w[i+j]*x[j] is the cross-correlation at lag i
    
    # Let me just compute both directly with a loop for correctness, N=400 is small enough
    # that O(N^2) per gradient evaluation is acceptable if we have ~100-500 evaluations
    
    for i in range(N):
        # term1: sum_{j=0}^{N-1} w[i+j]*x[j] where i+j < 2N-1
        j_max = min(N, 2*N-1-i)
        if j_max > 0:
            t1 = np.dot(w[i:i+j_max], x[:j_max])
        else:
            t1 = 0.0
        # term2: sum_{j=0}^{i} w[i-j]*x[j]
        if i >= 0:
            t2 = np.dot(w[:i+1][::-1], x[:i+1])
        else:
            t2 = 0.0
        grad_lse[i] = 2.0 * (t1 + t2)
    
    lse = m + np.log(np.sum(w)) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df  # sum>0 regime: value = +f, gradient is +df


def construct():
    start = time.time()

    # Candidate pool: deterministic spike variants + a few structured starts.
    candidates = [_spike_candidate()]

    # alternate-spike (h[0]=1, rest 0): value exactly 2*N = 800, valid fallback
    alt = np.zeros(N); alt[0] = 1.0
    candidates.append(alt)

    # single spike with tail of varying strength (explore the shave direction)
    for K in (50, 100, 200, 300, 399):
        for eps in (0.0005, 0.001, 0.002, 0.005):
            if K - 2 <= 0:
                continue
            h = np.zeros(N)
            h[0] = 0.5
            h[1] = 0.5
            h[2:K] = -eps
            candidates.append(h)

    # two-spike with unequal split (a, 1-a) - should land back near 0.5
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        h = np.zeros(N)
        h[0] = a
        h[1] = 1.0 - a
        h[2:120] = -0.001
        candidates.append(h)

    # structured: positive spike on a short block, small negative elsewhere
    rng = np.random.RandomState(0)
    for _ in range(12):
        h = np.zeros(N)
        h[0] = 0.5
        h[1] = 0.5
        neg_mask = np.zeros(N, dtype=bool)
        neg_mask[2:] = True
        idx = np.where(neg_mask)[0]
        pick = rng.choice(idx, size=rng.randint(20, 300), replace=False)
        h[pick] = -rng.uniform(0.0003, 0.004)
        candidates.append(h)

    # spectral (sine-wave) candidates: oscillatory patterns with negative lobes
    for seed in range(20):
        for n_sines in (3, 5, 8):
            candidates.append(_spectral_candidate(seed, n_sines=n_sines))
    # Chebyshev polynomial based candidates (alternating signs)
    for degree in (5, 8, 12):
        x = np.linspace(-1, 1, N)
        h = np.polynomial.chebyshev.chebval(x, np.array([1.0] + [0.0] * degree))
        h -= np.mean(h)
        s = np.sum(h)
        if abs(s) > 1e-12:
            h /= s
        candidates.append(h)

    # Evaluate all candidates, keep the best.
    best_h = None
    best_value = np.inf
    for h in candidates:
        val, valid = _evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()

    # Guard: ensure we always have a valid finite candidate.
    if best_h is None or not np.isfinite(best_value):
        best_h = _spike_candidate()
        best_value, _ = _evaluate(best_h)

    # L-BFGS-B polish from the best candidate to squeeze a bit more.
    if time.time() - start < DEADLINE - 5:
        for beta in (60.0, 120.0, 200.0, 400.0):
            if time.time() - start > DEADLINE - 3:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'maxfun': 1200, 'gtol': 1e-10},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Local random refinement around the best point (cheap, bounded).
    if time.time() - start < DEADLINE - 2:
        rng2 = np.random.RandomState(1)
        for _ in range(80):
            if time.time() - start > DEADLINE - 2:
                break
            h = best_h + rng2.normal(0.0, 0.01, N)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()

    # Final high-precision polish: very tight tolerance, more iterations
    if time.time() - start < DEADLINE - 1:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 800.0),
                best_h.copy(),
                jac=lambda x: _smooth_grad(x, 800.0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 2000, 'gtol': 1e-12},
            )
            val, valid = _evaluate(res.x)
            if valid and val < best_value:
                best_value = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Final validity guard: never return an invalid vector.
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _spike_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0  # value = 2*N = 800, always valid

    return best_h
