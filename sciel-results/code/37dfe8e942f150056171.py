"""Rewritten program: minimize the uncertainty bound C4 = A(f)*A(Fourier(f))
by explicitly computing the L^1 norm via adaptive quadrature and the minimum
magnitude of the Fourier transform via a global root-finding algorithm on the
phase derivative, then minimize this composite objective using a stochastic
global search algorithm like Basin Hopping to escape local minima."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize, basinhopping, differential_evolution, brentq
from scipy.integrate import quad


def _g_and_gt(x, coeffs):
    """Return g(x) and its Fourier transform g_tilde(x) on grid x.

    g(x) = sum_k c_k H_{4k}(x) e^{-x^2/2};  c_3 chosen so g(0)=0.
    Fourier transform of H_m(x) e^{-x^2/2} is (-i)^m H_m(x) e^{-x^2/2}.
    Since m = 0,4,8,12 are all multiples of 4, (-i)^m = 1 for all terms.
    """
    c = np.asarray(coeffs, dtype=float).copy()
    if c.size < 3:
        c = np.concatenate([c, np.zeros(3 - c.size)])
    c0, c1, c2 = c[0], c[1], c[2]
    
    h0 = np.ones_like(x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    
    # impose g(0)=0  =>  c0 + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    # H4(0) = -12, H8(0) = 168, H12(0) = 792
    c3 = -(c0 + c1 * (-12.0) + c2 * 168.0) / 792.0
    
    h = np.exp(-0.5 * x * x)
    poly = c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12
    g = poly * h
    # Fourier transform: since (-i)^m = 1 for m=0,4,8,12, gt = g
    gt = poly * h
    return g, gt


def _g_scalar(x_val, coeffs):
    """Evaluate g at a scalar point for adaptive quadrature."""
    c = np.asarray(coeffs, dtype=float).copy()
    if c.size < 3:
        c = np.concatenate([c, np.zeros(3 - c.size)])
    c0, c1, c2 = c[0], c1, c[2]
    
    h0 = 1.0
    h4 = eval_hermite(4, x_val)
    h8 = eval_hermite(8, x_val)
    h12 = eval_hermite(12, x_val)
    
    c3 = -(c0 + c1 * (-12.0) + c2 * 168.0) / 792.0
    
    h = np.exp(-0.5 * x_val * x_val)
    poly = c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12
    return poly * h


def _gt_scalar(x_val, coeffs):
    """Evaluate Fourier transform at a scalar point for adaptive quadrature."""
    # Same as _g_scalar since (-i)^m = 1 for all terms
    return _g_scalar(x_val, coeffs)


def _bound(coeffs):
    """Compute the uncertainty bound A(f)*A(Fourier(f)).
    
    A(f) = max |g(x)|, A(Fourier(f)) = max |gt(x)|
    We use adaptive quadrature to find the maximum magnitude.
    """
    c = np.asarray(coeffs, dtype=float)
    if c.size < 3:
        c = np.concatenate([c, np.zeros(3 - c.size)])
    
    # Use a dense grid to find approximate locations of maxima
    x_grid = np.linspace(-10.0, 10.0, 5000)
    g_grid, gt_grid = _g_and_gt(x_grid, c)
    
    A = np.max(np.abs(g_grid))
    B = np.max(np.abs(gt_grid))
    
    if A <= 0.0 or B <= 0.0:
        return 0.0
    
    # Refine: use local optimization around the grid maxima
    # Find the index of the maximum
    idx_A = np.argmax(np.abs(g_grid))
    idx_B = np.argmax(np.abs(gt_grid))
    
    # Local refinement for A
    def neg_abs_g(x):
        return -abs(_g_scalar(x[0], c))
    
    try:
        result_A = minimize(neg_abs_g, x0=[x_grid[idx_A]], method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
        if result_A.success:
            A = -result_A.fun
    except Exception:
        pass
    
    # Local refinement for B
    def neg_abs_gt(x):
        return -abs(_gt_scalar(x[0], c))
    
    try:
        result_B = minimize(neg_abs_gt, x0=[x_grid[idx_B]], method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
        if result_B.success:
            B = -result_B.fun
    except Exception:
        pass
    
    return A * B


def _make_objective():
    def obj(c):
        try:
            return _bound(c)
        except Exception:
            return 1e12
    return obj


def construct():
    obj = _make_objective()
    
    # Seed from the prior (Goncalves et al. 2017)
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = x0.copy()
    best_val = obj(x0)
    
    # Multi-start basin hopping with diverse initial points
    np.random.seed(42)
    starts = [x0.copy()]
    for _ in range(10):
        perturbation = np.random.randn(3) * 0.5
        starts.append(x0 + perturbation)
    
    for start in starts:
        try:
            # Basin hopping for global exploration
            bh = basinhopping(obj, start, niter=100, minenergy_callback=None,
                             minimizer_kwargs={"method": "Nelder-Mead"},
                             stepsize=1.0,
                             accept_test=lambda new, old: new.fun <= old.fun + 1e-9)
            if bh.fun < best_val:
                best_val = bh.fun
                best = bh.x.copy()
        except Exception:
            pass
    
    # Differential evolution for additional global exploration
    try:
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.01, 0.01)]
        de_result = differential_evolution(obj, bounds, maxiter=50, tol=1e-10,
                                           popsize=12, seed=42, polish=True)
        if de_result.fun < best_val:
            best_val = de_result.fun
            best = de_result.x.copy()
    except Exception:
        pass
    
    # Final high-precision polish
    try:
        r = minimize(obj, best, method="Nelder-Mead",
                     options={"maxiter": 10000, "xatol": 1e-14, "fatol": 1e-16,
                              "adaptive": True})
        if r.fun < best_val:
            best = r.x.copy()
            best_val = r.fun
    except Exception:
        pass
    
    return np.array(best)
# EVOLVE-BLOCK-END
