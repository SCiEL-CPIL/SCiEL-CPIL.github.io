"""Structural rewrite: gradient-based optimization of a non-flat step function.

We minimize the normalized peak of the autoconvolution of a nonnegative step
function on 600 equal intervals over [-1/4, 1/4]. The "normalized" value we
optimize is scale-invariant: we fix the L1 norm of the heights to 1 and
minimize the maximum of the (discrete) autoconvolution.

Method:
- Parameterize heights h >= 0 (length N).
- Normalize h so that sum(h) == 1.
- Discrete autoconvolution c_k = sum_i h_i h_{i+k} (circular or linear? We use
  the linear convolution restricted to the support, matching the step function
  on [-1/4,1/4] whose autoconvolution lives on [-1/2,1/2]).
- Objective: log-sum-exp(c / T) * T  (smooth max), anneal T down.
- Use scipy.optimize.minimize with L-BFGS-B and analytical gradients.
- Deadline guard: track wall-clock and return best-so-far before the limit.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
_DEADLINE_SECONDS = 150.0
_SAFETY_MARGIN = 12.0  # stop searching this many seconds before the hard kill


def _autoconv_linear(h):
    """Linear autoconvolution of h (length N) -> length 2N-1."""
    return np.convolve(h, h, mode='full')


def _objective_and_grad(params, T):
    """params: unconstrained vector; we map to nonnegative h via softplus-ish
    exponential to keep h > 0, then normalize to L1 = 1.

    Returns (loss, grad) where loss = T * logsumexp(c / T).
    """
    # Map params -> positive h via exp (ensures positivity; exp is smooth).
    h = np.exp(params)
    # Normalize to L1 = 1
    s = h.sum()
    h = h / s
    c = _autoconv_linear(h)
    # Smooth max via log-sum-exp
    cm = c / T
    cm_max = cm.max()
    lse = cm_max + np.log(np.exp(cm - cm_max).sum())
    loss = T * lse

    # Gradient: d loss / d params
    # d loss / d c = softmax(c / T)  (since d/d c of T*logsumexp(c/T) = softmax)
    p = np.exp(cm - cm_max)
    p = p / p.sum()
    # c = conv(h, h); d c_k / d h_i = h_{k-i} + h_{k-i} (symmetric) -> use convolve
    # d c / d h = convolve(h, h) reversed? Actually d/d h_i of sum_j h_j h_{j+k}
    # For c_k = sum_j h_j h_{j+k}, d c_k/d h_i = h_{i+k} + h_{i-k} (with bounds).
    # This equals 2 * (h * h) cross-correlation. We compute grad_h via
    # np.convolve with the trick: d c / d h = convolve(h, h, 'full') reversed
    # Let's compute directly: grad_c = p (length 2N-1).
    # d loss / d h = (p * dc/dh) summed. dc/dh_i = sum_k p_k (h_{i+k} + h_{i-k})
    # = 2 * (p_reversed cross-correlate h). We use FFT or direct via convolve.
    # c = conv(h,h,'full'); d c / d h = [conv(h, h reversed? )]. Simpler:
    # grad_h = convolve(h, p, 'full') + convolve(h, p, 'full')[::-1]? Let's derive:
    # c_k = sum_j h_j h_{j+k}. d c_k / d h_i = h_{i+k} + h_{i-k}.
    # sum_k p_k h_{i+k} = (p * h)_i where (p*h) is cross-correlation.
    # In numpy: np.correlate(p, h, 'full') gives sum_k p_k h_{k+i}... let's just
    # compute via convolution: sum_k p_k h_{i+k} = convolve(p, h, 'full') at index
    # (len(p)-1) + i. And sum_k p_k h_{i-k} = convolve(p, h, 'full') at index
    # (len(p)-1) - i. So grad_h_i = C[(L-1)+i] + C[(L-1)-i] where C=conv(p,h,'full').
    C = np.convolve(p, h, mode='full')  # length (2N-1)+(N)-1 = 3N-2
    L = len(C)
    mid = L - 1
    # indices for i in 0..N-1:
    idx_plus = mid + np.arange(N)
    idx_minus = mid - np.arange(N)
    grad_h = C[idx_plus] + C[idx_minus]

    # Now chain through normalization h = exp(params)/s
    # Let u = exp(params), h = u / s, s = sum(u).
    # d h / d params_j: h_i = u_i / s.
    # d h_i / d u_j = (delta_ij * s - u_i) / s^2 = (delta_ij - h_i) / s.
    # d u_j / d params_j = u_j = s * h_j.
    # So d h_i / d params_j = (delta_ij - h_i) * h_j.
    # grad_params_j = sum_i grad_h_i * (delta_ij - h_i) * h_j
    #                = grad_h_j * h_j - h_j * sum_i grad_h_i * h_i
    g = grad_h
    hsum = np.dot(g, h)
    grad_params = h * (g - hsum)
    return loss, grad_params


def _score_from_heights(h):
    """Compute the normalized peak value (the metric) from heights h.

    We define the normalized value as max(autoconv) with L1(h) = 1.
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s <= 0:
        return float('inf')
    h = h / s
    c = _autoconv_linear(h)
    return float(c.max())


def _make_seeds():
    seeds = []
    # Flat
    seeds.append(np.ones(N))
    # Triangular (peak in middle)
    x = np.linspace(-1.0, 1.0, N)
    seeds.append(np.clip(1.0 - np.abs(x), 0.0, None))
    # Gaussian
    seeds.append(np.exp(-3.0 * x * x))
    # Two-lobe
    seeds.append(np.exp(-8.0 * (x - 0.5) ** 2) + np.exp(-8.0 * (x + 0.5) ** 2))
    # Cosine bump
    seeds.append(np.clip(np.cos(np.pi * x / 2.0), 0.0, None))
    return seeds


def _optimize_from_seed(seed, deadline):
    """Run annealed log-sum-exp gradient optimization from a seed.

    Returns best (score, heights) found.
    """
    h0 = np.asarray(seed, dtype=float)
    h0 = np.clip(h0, 1e-6, None)
    params0 = np.log(h0)

    best_score = _score_from_heights(h0)
    best_h = h0.copy()

    # Anneal temperature from high to low
    temperatures = [0.05, 0.02, 0.008, 0.003, 0.001, 0.0004]

    params = params0.copy()
    for T in temperatures:
        if time.time() > deadline:
            break
        # A few L-BFGS-B iterations per temperature
        try:
            res = minimize(
                _objective_and_grad,
                params,
                args=(T,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 40, 'gtol': 1e-5, 'maxls': 20},
            )
            if res.x is not None:
                h_new = np.exp(res.x)
                h_new = np.clip(h_new, 0.0, None)
                sc = _score_from_heights(h_new)
                if sc < best_score:
                    best_score = sc
                    best_h = h_new.copy()
                params = res.x
        except Exception:
            pass
    return best_score, best_h


def construct():
    start = time.time()
    deadline = start + _DEADLINE_SECONDS - _SAFETY_MARGIN

    seeds = _make_seeds()

    best_score = float('inf')
    best_h = None

    for seed in seeds:
        if time.time() > deadline:
            break
        sc, h = _optimize_from_seed(seed, deadline)
        if sc < best_score:
            best_score = sc
            best_h = h

    # Fallback: if nothing worked, return flat
    if best_h is None:
        return [1.0] * N

    # Ensure nonnegative and finite
    best_h = np.clip(np.asarray(best_h, dtype=float), 0.0, None)
    if not np.all(np.isfinite(best_h)) or best_h.sum() <= 0:
        return [1.0] * N

    return [float(v) for v in best_h]
