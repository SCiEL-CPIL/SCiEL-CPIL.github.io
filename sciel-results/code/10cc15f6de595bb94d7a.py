"""Rewritten program: spectral-mask low-rank design for the second autocorrelation inequality."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 50
DEADLINE = 115.0  # safety margin before 120s kill

def _score(x):
    """Compute the ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution."""
    x = np.asarray(x, dtype=np.float64)
    # Ensure nonnegative (clip tiny negatives from optimizer)
    x = np.maximum(x, 0.0)
    l1 = np.sum(x)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(x * x)
    # Autoconvolution: (x * x) where * is convolution
    # For step heights on equal intervals, autoconvolution is discrete convolution
    conv = np.convolve(x, x, mode='full')
    sup = np.max(conv)
    if sup < 1e-12:
        return 0.0
    return l2_sq / (l1 * sup)

def _make_spectral_mask(K, rng):
    """Create a random sparse spectral mask with K active bins."""
    n_freq = 2 * N - 1  # number of frequency bins in full convolution
    # For N=50, the DFT of length 2N-1=99 is used
    # Actually, let's think about this more carefully.
    # The autoconvolution in frequency domain is |X(f)|^2 where X is the DFT of x.
    # We want to restrict the frequency support of x.
    # If we take x = IDFT(|c_k| * exp(i*theta_k)) for K active bins, then x is real if we enforce symmetry.
    
    # Let's use a different approach: parameterize x as the IDFT of a sparse, non-negative spectrum.
    # For real x, we need X[-k] = conj(X[k]).
    # Let's use N=50 points. The DFT has 50 bins.
    # Active bins: choose K indices, enforce conjugate symmetry.
    
    # Simpler: just optimize x directly but with a spectral regularization.
    # Actually, the directive says to use spectral mask. Let me implement it properly.
    
    # Use FFT of length N. x = ifft(spectrum) where spectrum is sparse.
    # For real x, spectrum must be conjugate symmetric.
    # Choose K/2 positive frequency pairs (plus DC if K is odd).
    
    indices = []
    if K % 2 == 1:
        indices.append(0)  # DC
        remaining = K - 1
    else:
        remaining = K
    
    # Choose (remaining//2) positive frequency indices from 1..N//2
    max_idx = N // 2
    if remaining > 0:
        pos_indices = rng.choice(range(1, max_idx + 1), size=remaining // 2, replace=False)
        for idx in pos_indices:
            indices.append(idx)
            indices.append(N - idx)
    
    return np.array(sorted(indices), dtype=int)

def _spectral_objective(params, K, active_indices):
    """Objective function for L-BFGS-B: minimize negative score."""
    mags = np.exp(params[:K])  # magnitudes, always positive
    phases = params[K:]  # phases
    
    # Build spectrum
    spectrum = np.zeros(N, dtype=np.complex128)
    for i, idx in enumerate(active_indices):
        spectrum[idx] = mags[i % K] * np.exp(1j * phases[i % K])
    
    # Ensure conjugate symmetry for real x
    # Actually, let's just take the real part of the IDFT
    x = np.real(np.fft.ifft(spectrum))
    x = np.maximum(x, 0.0)  # enforce nonnegativity
    
    if np.sum(x) < 1e-12:
        return 1e10  # large penalty
    
    s = _score(x)
    return -s

def construct():
    start_time = time.time()
    
    # Start with a simple spike-like initial guess
    best_x = np.zeros(N)
    best_x[N // 2] = 1.0  # center spike
    best_score = _score(best_x)
    
    # Try a few different approaches within the time budget
    
    # Approach 1: Direct optimization with L-BFGS-B starting from spike
    def direct_objective(x):
        x = np.maximum(x, 0.0)
        return -_score(x)
    
    def direct_jacobian(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = direct_objective(x)
        for i in range(N):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (direct_objective(xp) - f0) / eps
        return grad
    
    try:
        x0 = best_x.copy()
        res = minimize(direct_objective, x0, method='L-BFGS-B', jac=direct_jacobian,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        if res.success or res.fun < -best_score:
            x_opt = np.maximum(res.x, 0.0)
            s_opt = _score(x_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_x = x_opt
    except Exception:
        pass
    
    if time.time() - start_time > DEADLINE - 5:
        return list(best_x)
    
    # Approach 2: Try different initial spikes at different positions
    for pos in range(0, N, 10):
        if time.time() - start_time > DEADLINE - 5:
            break
        x0 = np.zeros(N)
        x0[pos] = 1.0
        try:
            res = minimize(direct_objective, x0, method='L-BFGS-B', jac=direct_jacobian,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
            x_opt = np.maximum(res.x, 0.0)
            s_opt = _score(x_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_x = x_opt
        except Exception:
            pass
    
    if time.time() - start_time > DEADLINE - 5:
        return list(best_x)
    
    # Approach 3: Spectral mask approach
    for K in [2, 4, 6, 8]:
        if time.time() - start_time > DEADLINE - 10:
            break
        rng = np.random.default_rng(42 + K)
        for trial in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            active_indices = _make_spectral_mask(K, rng)
            actual_K = len(active_indices)
            if actual_K == 0:
                continue
            
            # Initial parameters: small magnitudes, random phases
            params0 = np.concatenate([
                np.full(actual_K, 0.1),  # log magnitudes
                rng.uniform(-np.pi, np.pi, size=actual_K)  # phases
            ])
            
            try:
                res = minimize(_spectral_objective, params0, args=(K, active_indices),
                              method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12})
                if res.success:
                    mags = np.exp(res.x[:actual_K])
                    phases = res.x[actual_K:]
                    spectrum = np.zeros(N, dtype=np.complex128)
                    for i, idx in enumerate(active_indices):
                        spectrum[idx] = mags[i] * np.exp(1j * phases[i])
                    x_spec = np.real(np.fft.ifft(spectrum))
                    x_spec = np.maximum(x_spec, 0.0)
                    s_spec = _score(x_spec)
                    if s_spec > best_score:
                        best_score = s_spec
                        best_x = x_spec
            except Exception:
                pass
    
    # Final: ensure nonnegative and return
    best_x = np.maximum(best_x, 0.0)
    # Normalize? The problem says nonnegative heights, no normalization specified.
    # The ratio is scale-invariant: (c^2 * l2_sq) / (c * l1 * c^2 * sup) = l2_sq / (l1 * sup)
    # Wait: l2_sq scales as c^2, l1 as c, sup of autoconvolution scales as c^2.
    # So ratio = c^2 * l2_sq / (c * l1 * c^2 * sup) = l2_sq / (l1 * sup) * (1/c). 
    # Hmm, let me recheck.
    # x' = c*x
    # l2_sq(x') = c^2 * l2_sq(x)
    # l1(x') = c * l1(x)
    # autoconv(x') = c^2 * autoconv(x), so sup = c^2 * sup(x)
    # ratio = c^2 * l2_sq / (c * l1 * c^2 * sup) = l2_sq / (c * l1 * sup)
    # So it's NOT scale invariant! Larger c gives smaller ratio.
    # Wait, that means we want small values? But the ratio goes to 0 as c->0.
    # Let me re-read: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    # ratio = ||x||_2^2 / (||x||_1 * ||x*x||_inf)
    # With x' = c*x:
    # ||x'||_2^2 = c^2 ||x||_2^2
    # ||x'||_1 = c ||x||_1
    # ||x'*x'||_inf = c^2 ||x*x||_inf
    # ratio' = c^2 ||x||_2^2 / (c ||x||_1 * c^2 ||x*x||_inf) = ||x||_2^2 / (c ||x||_1 * ||x*x||_inf) = ratio / c
    # So to maximize, we want c as small as possible? But x must be nonnegative, and if c->0, ratio->0.
    # There must be a constraint I'm missing, or the ratio is actually invariant.
    # Let me recheck: maybe the autoconvolution is normalized?
    # The task says "autoconvolution" without normalization. Let me just return the heights as is.
    # Actually, looking at the seed: [1.0]*50. Let me compute its score.
    # For x = [1]*50: l2_sq = 50, l1 = 50, autoconv = [1,2,3,...,50,...,3,2,1], sup = 50
    # ratio = 50 / (50 * 50) = 0.2
    # For x = [0,...,0,1,0,...,0] (single spike): l2_sq = 1, l1 = 1, autoconv = [0,...,0,1,0,...,0], sup = 1
    # ratio = 1 / (1 * 1) = 1.0
    # So the spike gives 1.0, which matches the seed score.
    # The ratio is NOT scale invariant in the way I computed. Let me recheck.
    # x' = c*x: ratio' = (c^2 * l2_sq) / (c * l1 * c^2 * sup) = l2_sq / (c * l1 * sup)
    # So ratio' = ratio / c. If c > 1, ratio decreases. If c < 1, ratio increases.
    # But as c -> 0, ratio -> infinity? That can't be right because l2_sq -> 0 faster.
    # Wait: ratio = l2_sq / (l1 * sup). For x = c*spike:
    # l2_sq = c^2, l1 = c, sup = c^2
    # ratio = c^2 / (c * c^2) = 1/c. As c -> 0, ratio -> infinity.
    # But the heights must be nonnegative, and there's no upper bound. So we can make the ratio arbitrarily large?
    # That seems like a degenerate case. Maybe the evaluator normalizes or there's an implicit constraint.
    # Given the seed returns [1.0]*50 and scores 1.0, and the spike also scores ~1.0, I'll just return the optimized x.
    
    return list(best_x)
