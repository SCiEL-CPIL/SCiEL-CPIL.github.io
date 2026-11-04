import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 110  # seconds, safety margin before 120s limit


def _compute_ratio(h):
    """Compute ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h*h is discrete autocorrelation."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Discrete autocorrelation: (h*h)[k] = sum_i h[i]*h[i+k] for k in [-n+1, n-1]
    # We need the max over all lags. Use FFT for efficiency.
    # Zero-pad to length >= 2n-1 to avoid circular convolution artifacts.
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(H * np.conj(H), n=size)
    # ac has length size; the linear autocorrelation lags are at indices 0..n-1 and size-n+1..size-1
    # But actually for real signal, ac[k] for k=0..n-1 gives sum_i h[i]*h[i+k] (with appropriate indexing)
    # Let's just take the max of the relevant portion.
    # Actually irfft gives the circular correlation of the zero-padded signal.
    # For zero-padded signal of length size >= 2n-1, the first n samples give lags 0..n-1,
    # and the last n-1 samples (indices size-n+1 to size-1) give negative lags.
    # The max of autocorrelation is at lag 0, which is always >= all others for nonnegative h.
    # Wait - for nonnegative h, (h*h)[0] = sum h[i]^2 which is the max? Not necessarily.
    # Actually (h*h)[0] = sum_i h[i]*h[i] = ||h||_2^2
    # (h*h)[k] for k>0 = sum_i h[i]*h[i+k]
    # For nonnegative h, (h*h)[0] = sum h[i]^2. By Cauchy-Schwarz, sum h[i]*h[i+k] <= sum h[i]^2.
    # So for nonnegative h, the max of autocorrelation is always at lag 0!
    # Therefore ||h*h||_inf = ||h||_2^2 and the ratio is always 1.
    # Wait, that can't be right. Let me reconsider.
    
    # Actually, the autoconvolution (h*h)(x) = integral h(t)h(x-t) dt.
    # For step functions on intervals, this is different from discrete autocorrelation.
    # The task says "autoconvolution" not "autocorrelation".
    # Autoconvolution of step heights: if h_k is the height on interval [x_k, x_{k+1}],
    # then (h*h)(x) = integral h(t)h(x-t) dt.
    # For piecewise constant h, this is a piecewise linear function.
    # The supremum of the autoconvolution can be less than ||h||_2^2.
    
    # Let me re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # So ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # where h*h is autoconvolution.
    
    # For a constant h=c on N intervals of width 1/N (total interval [-1/4, 1/4], width 1/2):
    # ||h||_2^2 = N * c^2 * (1/N) = c^2  (if intervals have width 1/N)
    # Wait, the intervals are on [-1/4, 1/4] so total width is 1/2, each interval width = 1/(2N).
    # ||h||_2^2 = sum h_k^2 * (1/(2N))
    # ||h||_1 = sum h_k * (1/(2N))
    # (h*h)(x) = integral h(t)h(x-t) dt
    
    # For constant h=c: ||h||_2^2 = N*c^2*(1/(2N)) = c^2/2
    # ||h||_1 = N*c*(1/(2N)) = c/2
    # (h*h)(x) for x in [0, 1/2] (support of autoconvolution):
    # (h*h)(x) = c^2 * (1/2 - x) for x in [0, 1/2]
    # So ||h*h||_inf = c^2/2
    # Ratio = (c^2/2) / ((c/2) * (c^2/2)) = (c^2/2) / (c^3/4) = 2/c
    # Hmm, that doesn't equal 1. Let me reconsider the normalization.
    
    # Actually I think the heights are just the step values, and the norms are computed as:
    # ||h||_2^2 = sum h_k^2 * dx where dx = 1/(2N)
    # But maybe the ratio is scale-invariant? Let's check:
    # If we scale h by alpha: ||alpha*h||_2^2 = alpha^2 ||h||_2^2
    # ||alpha*h||_1 = alpha ||h||_1
    # ||(alpha*h)*(alpha*h)||_inf = alpha^2 ||h*h||_inf
    # So ratio = alpha^2 ||h||_2^2 / (alpha ||h||_1 * alpha^2 ||h*h||_inf) = ||h||_2^2 / (||h||_1 * alpha * ||h*h||_inf)
    # That's NOT scale invariant. So the scale matters.
    
    # Wait, let me re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # This could be parsed as: (||h||_2^2) / (||h||_1 * ||h*h||_inf)
    # Or it could be: (||h||_2^2 / ||h||_1) * ||h*h||_inf
    
    # The task says "maximize the ratio of squared L2 norm to L1 norm times supremum norm"
    # I think it's: (||h||_2^2 / ||h||_1) * ||h*h||_inf ? No that would increase with scale.
    # Or: ||h||_2^2 / (||h||_1 * ||h*h||_inf) ? This has units of 1/(height).
    
    # For the seed program [1.0]*50, score is 1.0. Let me compute:
    # dx = 1/(2*50) = 1/100
    # h = [1]*50
    # ||h||_2^2 = 50 * 1^2 * (1/100) = 0.5
    # ||h||_1 = 50 * 1 * (1/100) = 0.5
    # Autoconvolution: h is 1 on [-0.25, 0.25], 0 elsewhere.
    # (h*h)(x) = integral h(t)h(x-t) dt = length of intersection of [t, t] with support...
    # = length of {t : t in [-0.25,0.25] and x-t in [-0.25,0.25]}
    # = length of {t : t in [-0.25,0.25] and t in [x-0.25, x+0.25]}
    # = length of intersection of [-0.25,0.25] and [x-0.25,x+0.25]
    # For x in [-0.5, 0.5]:
    #   If x >= 0: intersection is [x-0.25, 0.25], length = 0.5-x (for x in [0,0.5])
    #   If x < 0: intersection is [-0.25, x+0.25], length = 0.5+x (for x in [-0.5,0])
    # So ||h*h||_inf = 0.5 (at x=0)
    # Ratio = 0.5 / (0.5 * 0.5) = 2.0
    
    # But the score is 1.0 for the seed. So maybe the ratio is defined differently.
    # Perhaps the heights are not multiplied by dx? I.e., discrete norms:
    # ||h||_2^2 = sum h_k^2 = 50
    # ||h||_1 = sum h_k = 50
    # And autoconvolution is discrete: (h*h)[k] = sum_i h_i * h_{i+k}
    # For h=[1]*50: (h*h)[0] = 50, (h*h)[k] = 50-|k| for |k|<50
    # ||h*h||_inf = 50
    # Ratio = 50 / (50 * 50) = 0.02. Not 1.0 either.
    
    # Let me try: ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf) with discrete, no dx:
    # 50 / (50 * 50) = 0.02. No.
    
    # What if autoconvolution is normalized? Or what if it's:
    # ratio = (||h||_2^2 / ||h||_1) / ||h*h||_inf ?
    # = (50/50) / 50 = 0.02. No.
    
    # Hmm, what if the "autoconvolution" uses the continuous interpretation but with unit heights meaning unit area?
    # Or what if dx=1? I.e., the intervals are just indexed 0..49 and dx is irrelevant?
    # Then ||h||_2^2 = 50, ||h||_1 = 50, and if autoconvolution is (h*h)[k]=sum h_i h_{i+k}, max=50
    # ratio = 50/(50*50) = 0.02. Still not 1.
    
    # What if it's ||h||_2^2 / ||h*h||_inf ? (without the L1 norm)
    # = 50/50 = 1. Yes! That gives 1.0 for the seed.
    
    # But the task says "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # Maybe it's: (||h||_2^2 / ||h||_1) * (1/||h*h||_inf) ? No.
    # Or maybe: ||h||_2^2 / (||h||_1 * ||h*h||_inf) but with a specific normalization where for constant h this gives 1?
    
    # Actually wait - if we use continuous norms with dx:
    # ||h||_2^2 = sum h_k^2 dx, ||h||_1 = sum h_k dx, ||h*h||_inf = max of continuous autoconvolution
    # For h=[1]*50, dx=1/100:
    # ||h||_2^2 = 0.5, ||h||_1 = 0.5, ||h*h||_inf = 0.5
    # ratio = 0.5/(0.5*0.5) = 2.0. Not 1.
    
    # If dx=1: ||h||_2^2=50, ||h||_1=50, ||h*h||_inf = max of discrete autocorr = 50
    # ratio = 50/(50*50)=0.02. Not 1.
    
    # The only way to get 1 is: ||h||_2^2 / ||h*h||_inf (discrete, no dx) = 50/50 = 1.
    # Or: (||h||_2^2 / ||h||_1) / (||h*h||_inf / ||h||_1) = ||h||_2^2/||h*h||_inf.
    
    # I'll go with: score = ||h||_2^2 / ||h*h||_inf where both are discrete (no dx factor).
    # This is scale-invariant: if h -> alpha*h, ||alpha h||_2^2 = alpha^2||h||_2^2, 
    # ||(alpha h)*(alpha h)||_inf = alpha^2||h*h||_inf, ratio unchanged.
    
    l2_sq = np.dot(h, h)
    # Discrete autoconvolution (full): (h*h)[k] for k in [-(n-1), n-1]
    # Use FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(H * np.conj(H), n=size)
    # The linear convolution/correlation: ac[0:n] gives lags 0 to n-1
    # ac[n-size+1:] or similar for negative lags, but for nonnegative h, lag 0 is max
    # Actually let's just take the max of the entire ac array to be safe
    max_ac = np.max(ac)
    if max_ac < 1e-15:
        return 0.0
    return l2_sq / max_ac


def _compute_ratio_fast(h):
    """Faster computation using the fact that for nonnegative h, max autocorr is at lag 0."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l2_sq = np.dot(h, h)
    if l2_sq < 1e-15:
        return 0.0
    # For nonnegative h, the autocorrelation at lag 0 is sum h[i]^2 which is >= sum h[i]h[i+k] for any k
    # by rearrangement/Cauchy-Schwarz. So max is at lag 0.
    return 1.0  # l2_sq / l2_sq = 1 always!
    
    # Wait, this means the ratio is ALWAYS 1 for nonnegative h? That can't be right.
    # Let me reconsider. Maybe the autoconvolution is not the same as autocorrelation.
    # Autoconvolution: (f*g)(x) = integral f(t)g(x-t)dt with g=f.
    # For discrete: (h*h)[k] = sum_i h[i]*h[i+k] (this is actually correlation, not convolution)
    # Convolution would be: (h*h)[k] = sum_i h[i]*h[k-i]
    # For nonnegative h, convolution at k=0 is sum h[i]*h[-i] which requires h to be defined for negative indices.
    # If h is supported on indices 0..n-1, then (h*h)[k] = sum_i h[i]*h[k-i] is nonzero for k in [0, 2n-2].
    # (h*h)[0] = sum_i h[i]*h[-i] = 0 (since h[-i]=0 for i>0) unless h is defined differently.
    
    # I think the issue is the definition. Let me assume the evaluator computes:
    # score = ||h||_2^2 / (||h||_1 * ||h*h||_inf) where h*h is the continuous autoconvolution
    # and norms are continuous. For the seed to give 1.0, we need specific normalization.
    
    # Let me just try to maximize a reasonable objective and see.
    # Given the seed gives 1.0, and the task is to improve, let me try various strategies.


def construct():
    start_time = time.time()
    
    # Strategy: Try to find h that maximizes ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # where ||h*h||_inf is the max of the continuous autoconvolution.
    # 
    # For piecewise constant h on intervals of width dx=1/(2N):
    # ||h||_2^2 = dx * sum h_k^2
    # ||h||_1 = dx * sum h_k  
    # (h*h)(x) = integral h(t)h(x-t)dt
    # 
    # The ratio = dx * sum h_k^2 / (dx * sum h_k * ||h*h||_inf) = (sum h_k^2) / (sum h_k * ||h*h||_inf)
    # 
    # ||h*h||_inf depends on dx and the values h_k.
    # 
    # Let me just do a numerical optimization.
    
    def objective(x):
        """Negative of the ratio to be minimized."""
        h = np.maximum(x, 0.0)
        n = len(h)
        l2_sq = np.dot(h, h)
        l1 = np.sum(h)
        if l1 < 1e-15 or l2_sq < 1e-15:
            return 1e10
        # Compute autoconvolution via FFT
        dx = 1.0 / (2.0 * n)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        H = np.fft.rfft(h, n=size)
        ac = np.fft.irfft(H * np.conj(H), n=size)
        # ac is the circular correlation of the zero-padded h
        # For the continuous autoconvolution, we need to scale by dx
        # Actually, the discrete convolution sum h[i]*h[j] approximates the integral with factor dx
        # So ||h*h||_inf ≈ dx * max(ac)
        max_ac = np.max(ac)
        if max_ac < 1e-15:
            return 1e10
        ratio = l2_sq / (l1 * dx * max_ac)
        return -ratio
    
    best_h = None
    best_score = -1e10
    
    # Try several initial guesses
    initial_guesses = [
        np.ones(N),
        np.concatenate([np.ones(25), np.zeros(25)]),
        np.concatenate([np.ones(20), np.zeros(10), np.ones(20)]),
        np.concatenate([np.ones(10), np.zeros(30), np.ones(10)]),
        np.concatenate([np.ones(5), np.zeros(40), np.ones(5)]),
        np.exp(-0.5 * (np.linspace(-1, 1, N))**2),
        np.exp(-0.5 * (np.linspace(-2, 2, N))**2),
    ]
    
    for x0 in initial_guesses:
        if time.time() - start_time > DEADLINE - 5:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-6})
            if res.x is not None:
                h_candidate = np.maximum(res.x, 0.0)
                score = -objective(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate
        except Exception:
            pass
    
    # Also try differential evolution for global search
    if time.time() - start_time < DEADLINE - 10:
        try:
            bounds = [(0, 5)] * N
            de_result = differential_evolution(objective, bounds, maxiter=50, 
                                               popsize=10, seed=42, polish=True,
                                               tol=1e-5)
            h_candidate = np.maximum(de_result.x, 0.0)
            score = -objective(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate
        except Exception:
            pass
    
    # Return the best result
    if best_h is not None and best_score > 0:
        return list(best_h)
    else:
        return [1.0] * N
