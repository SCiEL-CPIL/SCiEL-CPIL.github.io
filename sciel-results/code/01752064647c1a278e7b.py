"""Erdos minimum overlap: minimize the max overlap of h with shifted (1-h)."""
import time
import numpy as np
from scipy.optimize import minimize, SLSQP

N = 200
DEADLINE = 110.0  # safety margin under the 120 s / 150 s kill
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _all_overlaps(h):
    """Return array of overlaps of h with shifted (1-h) for all shifts k=0..N-1.
    Overlap for shift k = sum_i h_i * (1 - h_{(i+k) mod N})."""
    comp = 1.0 - h
    # For each shift k, overlap_k = sum_i h_i * comp_{(i+k) mod N}
    # Build the circulant via FFT for speed:
    # overlap_k = sum_i h_i * comp_{(i+k)} ; this is a circular cross-correlation.
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    # corr_k = sum_i h_i * comp_{(i+k) mod N} = ifft( H * conj(C) )_k
    corr = np.real(np.fft.ifft(H * np.conj(C)))
    return corr


def _max_overlap(h):
    return float(np.max(_all_overlaps(h)))


def _soft_max(x, lam):
    """log-sum-exp soft-max, stable."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)) / len(x)) / lam


def _objective_and_grad(h, lam):
    corr = _all_overlaps(h)
    val = _soft_max(corr, lam)
    # gradient: d/d h of soft-max
    p = np.exp(corr - np.max(corr))
    p = p / p.sum()
    # d corr_k / d h_i: corr_k = sum_i h_i comp_{(i+k)} = sum_i h_i (1 - h_{(i+k)})
    # d corr_k / d h_j = comp_{(j+k)} - h_j * d h_{(j+k)}/d h_j ... but comp depends on h.
    # corr_k = sum_i h_i - sum_i h_i h_{(i+k)} = 1 - sum_i h_i h_{(i+k)} (since sum h = N/2 fixed)
    # d corr_k / d h_j = - (h_{(j+k)} + h_{(j-k)})
    # (two terms: i=j gives h_{(j+k)}, and i such that i+k=j i.e. i=j-k gives h_j h_{(j-k)})
    dcorr = - (np.roll(h, -0) + np.roll(h, 0))  # placeholder
    # Build properly: for each k, dcorr_k_j = -h_{(j+k)} - h_{(j-k)}
    # grad_j = sum_k p_k * dcorr_k_j = -sum_k p_k (h_{(j+k)} + h_{(j-k)})
    # = -( (p * h) circulant )_j - ( (p * h) circulant )_j  (both same by symmetry)
    # Let q_j = sum_k p_k h_{(j+k)} = circular convolution of p and h at j.
    q = np.real(np.fft.ifft(np.fft.fft(p) * np.fft.fft(h)))
    grad = -2.0 * q
    return val, grad


def construct():
    # Start from a good non-uniform guess and refine with a smooth surrogate.
    # A sawtooth / two-level pattern reduces worst-case overlap.
    h = np.full(N, 0.5)
    # seed a perturbation to break symmetry
    rng = np.random.default_rng(0)
    h = h + 0.05 * rng.standard_normal(N)
    h = np.clip(h, 0.0, 1.0)
    h = h / h.sum() * (N / 2.0)

    best = h.copy()
    best_val = _max_overlap(h)
    lam = 50.0

    def obj(hv):
        v, g = _objective_and_grad(hv, lam)
        return v, g

    # An anneal on lam: start smooth, increase.
    for lam in [20.0, 40.0, 80.0, 160.0]:
        if _elapsed() > DEADLINE - 15.0:
            break
        try:
            res = minimize(
                obj, h, jac=True, method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 60, 'gtol': 1e-6, 'ftol': 1e-9},
            )
            h = res.x
            h = np.clip(h, 0.0, 1.0)
            h = h / h.sum() * (N / 2.0)
            v = _max_overlap(h)
            if v < best_val:
                best_val = v
                best = h.copy()
        except Exception:
            pass

    # Final exact polish with SLSQP on the true max objective (nonsmooth).
    if _elapsed() < DEADLINE - 8.0:
        def obj_exact(hv):
            return _max_overlap(hv)
        try:
            res = minimize(
                obj_exact, best, method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 40, 'ftol': 1e-10},
            )
            h = np.clip(res.x, 0.0, 1.0)
            h = h / h.sum() * (N / 2.0)
            v = _max_overlap(h)
            if v < best_val:
                best_val = v
                best = h.copy()
        except Exception:
            pass

    # Enforce exact sum constraint: sum == N/2.
    best = np.clip(best, 0.0, 1.0)
    best = best / best.sum() * (N / 2.0)
    return best
