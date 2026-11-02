"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize, LinearConstraint

N = 600
STEP = 1.0 / N  # step size for intervals on [-1/4, 1/4]

# EVOLVE-BLOCK-START
def _linear_autocorr(f):
    """Compute linear autocorrelation of f (length N) via FFT."""
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.fft.rfft(f, n=nfft)
    ac = np.fft.irfft(F * F.conj(), n=nfft)[:N]
    return ac

def _autocorr_grad(f, w):
    """
    Gradient of sum_k w[k]*ac[k] w.r.t. f, where ac = linear autocorrelation.
    ac[k] = sum_j f[j]*f[j+k]
    d/d f[i] sum_k w_k ac_k = sum_k w_k*(f[i+k]*1_{i+k<N} + f[i-k]*1_{i-k>=0})
    """
    nfft = 1
    while nfft < 2 * N:
        nfft *= 2
    F = np.fft.rfft(f, n=nfft)
    W = np.fft.rfft(w, n=nfft)
    # Term 1: sum_k w_k * f[i+k]  -> cross-correlation
    term1 = np.fft.irfft(np.conj(W) * F, n=nfft)[:N]
    # Term 2: sum_k w_k * f[i-k]  -> convolution
    term2 = np.fft.irfft(W * F, n=nfft)[:N]
    return term1 + term2

def _score(f):
    """Normalized max of linear autoconvolution."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 0.0)
    s = np.sum(f)
    if s < 1e-12:
        return np.inf
    ac = _linear_autocorr(f)
    return np.max(ac) / (s * s) * N

def construct():
    """
    Returns a sequence of 600 nonnegative step heights on equal intervals of [-1/4, 1/4]
    that minimizes the normalized maximum of the autoconvolution (upper bound on C1).

    Strategy:
    - The flat (constant) function achieves the Cauchy-Schwarz lower bound of 1.0.
    - We attempt to improve below 1.0 using a spectral shaping approach:
      parameterize f via its DFT, optimize a smooth surrogate of the max
      autoconvolution, then project back to nonnegativity.
    - Also try a direct projected gradient descent on the smooth max surrogate.
    - Return the best found; fall back to flat if no improvement.
    """
    t0 = time.time()
    deadline = 135.0  # safety margin before 150s kill

    flat = np.ones(N)
    best_f = flat.copy()
    best_val = _score(flat)  # should be ~1.0

    # --- Approach 1: Projected gradient descent on smooth max surrogate ---
    def smooth_obj_and_grad(f, beta):
        """
        Smooth surrogate: N * logsumexp(beta * ac) / (sum f)^2
        This approximates N * max(ac) / (sum f)^2 as beta -> inf.
        """
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10, np.zeros(N)
        ac = _linear_autocorr(f)
        ac_safe = np.maximum(ac, 1e-15)
        m = np.max(ac_safe)
        if m < 1e-12:
            return 1e10, np.zeros(N)
        exp_ac = np.exp(beta * (ac_safe - m))
        sum_exp = np.sum(exp_ac)
        weights = exp_ac / sum_exp
        lse = m + np.log(sum_exp / beta)
        grad_lse = _autocorr_grad(f, weights)
        s2 = s * s
        s3 = s2 * s
        obj = N * lse / s2
        obj_grad = N * (grad_lse * s - 2.0 * lse) / s3
        return obj, obj_grad

    f_cur = flat.copy()
    lr = 0.02
    beta = 30.0
    for it in range(2000):
        if time.time() - t0 > deadline:
            break
        obj, grad = smooth_obj_and_grad(f_cur, beta)
        f_new = f_cur - lr * grad
        f_new = np.maximum(f_new, 0.0)
        obj_new, _ = smooth_obj_and_grad(f_new, beta)
        if obj_new < obj:
            f_cur = f_new
            exact = _score(f_cur)
            if exact < best_val - 1e-10:
                best_val = exact
                best_f = f_cur.copy()
        else:
            lr *= 0.9
        if it % 100 == 0 and beta < 300:
            beta = min(beta * 1.3, 300.0)

    # --- Approach 2: Spectral shaping ---
    # Parameterize f via its power spectrum: f = |FFT^{-1}(sqrt(S))|^2 clamped to >= 0
    # Optimize S (log-scale) to flatten autocorrelation.
    # Simpler: optimize a small number of spectral coefficients.
    # Try: f[i] = 1 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N)
    # This keeps f real and symmetric, and we can optimize a_k.
    K = 20  # number of cosine modes
    # Precompute cosine basis
    idx = np.arange(N)
    cos_basis = np.zeros((K, N))
    for k in range(1, K + 1):
        cos_basis[k - 1] = np.cos(2.0 * np.pi * k * idx / N)

    def f_from_coeffs(a):
        """a has length K; f = 1 + sum a_k cos(...)"""
        f = 1.0 + cos_basis @ a
        return np.maximum(f, 0.0)

    def spectral_obj(a):
        f = f_from_coeffs(a)
        s = np.sum(f)
        if s < 1e-12:
            return 1e10
        ac = _linear_autocorr(f)
        return np.max(ac) / (s * s) * N

    # Optimize a using L-BFGS-B
    a0 = np.zeros(K)
    try:
        res = minimize(spectral_obj, a0, method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 2000,
                                'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - t0 < deadline:
            f_spec = f_from_coeffs(res.x)
            val_spec = _score(f_spec)
            if val_spec < best_val - 1e-10:
                best_val = val_spec
                best_f = f_spec.copy()
    except Exception:
        pass

    # --- Approach 3: Random restarts with small perturbations ---
    if time.time() - t0 < deadline:
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() - t0 > deadline:
                break
            # Random low-frequency perturbation
            a_rand = rng.normal(0, 0.05, size=K)
            f_trial = f_from_coeffs(a_rand)
            val_trial = _score(f_trial)
            if val_trial < best_val - 1e-10:
                best_val = val_trial
                best_f = f_trial.copy()
                # Local refinement
                try:
                    res2 = minimize(spectral_obj, a_rand, method='L-BFGS-B',
                                    options={'maxiter': 200, 'maxfun': 500,
                                             'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8})
                    if time.time() - t0 < deadline:
                        f_ref = f_from_coeffs(res2.x)
                        val_ref = _score(f_ref)
                        if val_ref < best_val - 1e-10:
                            best_val = val_ref
                            best_f = f_ref.copy()
                except Exception:
                    pass

    # Final check
    final_val = _score(best_f)
    if final_val < 1.0 - 1e-9:
        return list(best_f)
    return list(flat)
# EVOLVE-BLOCK-END
