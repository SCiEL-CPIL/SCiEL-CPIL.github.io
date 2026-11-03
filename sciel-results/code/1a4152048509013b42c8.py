"""Optimize nonnegative step heights to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin in seconds

def _autocorr_peak(h):
    # Full linear autoconvolution, take its maximum (supremum norm)
    # h is nonnegative, so max of autocorr is at lag 0? Not necessarily for linear conv.
    # Actually for nonnegative h, the autocorrelation (linear) at lag 0 is sum h_i^2,
    # and at other lags it's sum h_i h_{i+k}. Since h>=0, the max could be at lag 0
    # or elsewhere. Let's compute full and take max.
    ac = np.correlate(h, h, mode='full')
    return np.max(ac)

def _objective(params):
    # params: log-softplus transformed to ensure nonnegativity
    # Use x = exp(params) to keep positive, but that can blow up.
    # Better: h = softplus(params) = log(1+exp(params)), always positive and smooth.
    h = np.logaddexp(0.0, params)
    norm2_sq = np.sum(h * h)
    norm1 = np.sum(h)
    peak = _autocorr_peak(h)
    if norm1 == 0 or peak == 0:
        return 0.0
    ratio = norm2_sq / (norm1 * peak)
    return -ratio  # minimize negative

def _ratio_from_h(h):
    norm2_sq = np.sum(h * h)
    norm1 = np.sum(h)
    peak = _autocorr_peak(h)
    if norm1 == 0 or peak == 0:
        return 0.0
    return norm2_sq / (norm1 * peak)

def construct():
    global _START
    elapsed = time.time() - _START
    remaining = _DEADLINE - _MARGIN - elapsed
    if remaining <= 1.0:
        # Not enough time; return seed
        return [1.0] * N

    best_h = np.ones(N)
    best_ratio = _ratio_from_h(best_h)

    # Multiple random restarts with BFGS
    rng = np.random.default_rng(12345)
    max_restarts = 30
    max_iters = 500

    for r in range(max_restarts):
        if time.time() - _START > _DEADLINE - _MARGIN:
            break
        # Random initialization: mix of structures
        if r == 0:
            # Start near uniform but perturbed
            init = np.ones(N) + 0.1 * rng.standard_normal(N)
            init = np.clip(init, 0.01, None)
        elif r % 3 == 1:
            # Sparse-ish: mostly small, few large
            init = np.abs(rng.standard_normal(N)) * 0.5
            init[rng.choice(N, size=N//5, replace=False)] += 2.0
        elif r % 3 == 2:
            # Smooth bump
            x = np.linspace(0, 1, N)
            init = np.exp(-((x - 0.5) ** 2) / 0.05) + 0.01
        else:
            # Random positive
            init = np.abs(rng.standard_normal(N)) + 0.01

        # Transform to params for log-softplus: params = log(expm1(h)) for h>1, else small
        # Actually simpler: optimize directly on h with bounds using a transformation
        # Use params = log(h) so h = exp(params), but constrain via penalty or bounds.
        # Let's use the softplus parameterization: h = log(1+exp(params))
        # Invert: params = log(expm1(h)) for h > 1, for h <= 1 use log(expm1(h)) which is negative
        # For h close to 0, expm1(h) ~ h, so params ~ log(h). For h=0.01, params ~ log(0.01) = -4.6
        # For h=10, params ~ log(9.95) ~ 2.3
        with np.errstate(divide='ignore', invalid='ignore'):
            init_params = np.log(np.expm1(np.clip(init, 1e-6, None)))
        init_params = np.where(init_params > 10, 10, init_params)
        init_params = np.where(init_params < -10, -10, init_params)

        res = minimize(
            _objective,
            init_params,
            method='BFGS',
            options={'maxiter': max_iters, 'gtol': 1e-8, 'disp': False}
        )

        h_candidate = np.logaddexp(0.0, res.x)
        ratio_candidate = _ratio_from_h(h_candidate)

        if ratio_candidate > best_ratio:
            best_ratio = ratio_candidate
            best_h = h_candidate.copy()

    # Final polish: try a few local perturbations around best_h
    for _ in range(5):
        if time.time() - _START > _DEADLINE - _MARGIN:
            break
        perturbed = best_h * (1.0 + 0.05 * rng.standard_normal(N))
        perturbed = np.clip(perturbed, 1e-6, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            pert_params = np.log(np.expm1(np.clip(perturbed, 1e-6, None)))
        pert_params = np.where(pert_params > 10, 10, pert_params)
        pert_params = np.where(pert_params < -10, -10, pert_params)
        res = minimize(
            _objective,
            pert_params,
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-8, 'disp': False}
        )
        h_cand = np.logaddexp(0.0, res.x)
        ratio_cand = _ratio_from_h(h_cand)
        if ratio_cand > best_ratio:
            best_ratio = ratio_cand
            best_h = h_cand.copy()

    # Ensure nonnegative
    best_h = np.clip(best_h, 0.0, None)
    return [float(x) for x in best_h]
