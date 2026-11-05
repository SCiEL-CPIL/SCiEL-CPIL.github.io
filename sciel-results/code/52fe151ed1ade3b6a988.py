"""Compact dark-energy growth-index fit: Chebyshev spectral residual + min-max refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

# Fixed internal deadline with a safety margin (evaluator kills at 90 s).
_DEADLINE = 75.0

# Cosmology grid: a in (0, 1] mapped to x = ln a in [-3, 0].
_A_GRID = np.exp(np.linspace(-3.0, 0.0, 240))
_X_GRID = np.log(_A_GRID)


def _omega_m(a, om0, w0, wa):
    a2 = a * a
    a3 = a2 * a
    return om0 / (om0 + (1.0 - om0) * a3 * np.exp(3.0 * (w0 + 1.0 + wa) * (1.0 - a) / (1.0 + wa)))


def _gamma_of(c, a, om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    om = _omega_m(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)


def _build_gamma_basis(om0, w0, wa, a_grid):
    """Build the 6-column basis matrix for gamma as linear combination of coefficients."""
    om = _omega_m(a_grid, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a_grid)
    # Columns: 1, q, q^2, u, wa*a, u*q
    basis = np.column_stack([
        np.ones_like(a_grid),
        q,
        q * q,
        u,
        wa * a_grid,
        u * q
    ])
    return basis


def _f_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    g = _gamma_of(c, a, om0, w0, wa)
    return om ** g


def _norm_d_of(c, om0, w0, wa):
    # D(a)/D(1) = exp(-integral_{ln a}^{0} f(exp x) dx)
    f = _f_of(c, _A_GRID, om0, w0, wa)
    integ = np.zeros_like(_X_GRID)
    integ[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(_X_GRID))
    total = integ[-1]
    val = total - integ
    return np.exp(-val)


def _build_reference(om0, w0, wa):
    """Solve the second-order GR growth ODE for D(a) and f(a) = D'/(a D)."""
    N = 240
    x = np.linspace(-3.0, 0.0, N)
    a = np.exp(x)
    om = _omega_m(a, om0, w0, wa)
    ode = 1.0 - om
    # f(a) = D'/(aD); f(a) = (3/2) om + (3/2) om' - (3/2) om - ... use standard form:
    # f' + f^2 = (3/2) om + (3/2) om'  where derivative is d/da.
    # Work in x: d/dx = a d/da. Let g(x)=f. Then a f' = g - g^2.
    # a f' = (3/2) om + (3/2) a om' - (3/2) om? Standard: f' + f^2 = (3/2) om + (3/2) om' (d/d a).
    # a f' + f^2 = (3/2) om a + (3/2) a om'  -> not clean. Use d/dx form:
    # f' (d/da) = (1/a)(g - g^2). Equation: f' + f^2 = (3/2) om + (3/2) om'.
    # => (1/a)(g - g^2) + g^2 = (3/2) om + (3/2) om'
    # => g - g^2 + a g^2 = a*(3/2) om + a*(3/2) om'
    # => g*(1) + g^2*(a - 1) = (3/2) a (om + om')
    # Solve for g: (a-1) g^2 + g - (3/2) a (om+om') = 0.
    # Need om' = d om/d a.
    # om(a) = om0 / (om0 + (1-om0) a^3 E)  where E = exp(3 (w0+1+wa)(1-a)/(1+wa)).
    # Use numerical derivative for robustness.
    da = np.diff(a)
    om_prime = np.zeros_like(a)
    om_prime[1:] = (om[1:] - om[:-1]) / da
    om_prime[0] = om_prime[1]
    rhs = 1.5 * a * (om + om_prime)
    Acoef = (a - 1.0)
    # quadratic: Acoef g^2 + g - rhs = 0  => g = [-1 + sqrt(1 + 4 Acoef rhs)]/(2 Acoef)
    disc = 1.0 + 4.0 * Acoef * rhs
    disc = np.maximum(disc, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (-1.0 + np.sqrt(disc)) / (2.0 * Acoef)
    # fix a=1 limit: g(1) = 0.5 om(1) = 0.5 om0
    g[0] = 0.5 * om[0]
    # Now D(a)/D(1): d ln D / da = f => d ln D / dx = f * a = g
    lnD = np.zeros_like(x)
    lnD[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))
    D = np.exp(lnD)
    D = D / D[-1]
    return a, g, D


def _objective(c, models, ref_cache):
    """Returns max relative error over f and normalized D across models and a."""
    worst = 0.0
    for (om0, w0, wa) in models:
        g = _gamma_of(c, _A_GRID, om0, w0, wa)
        if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
            viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
            return 1.0 + float(np.max(viol))
        f = _f_of(c, _A_GRID, om0, w0, wa)
        D = _norm_d_of(c, om0, w0, wa)
        _, ref_f, ref_D = ref_cache[(om0, w0, wa)]
        er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
        er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
        worst = max(worst, er_f, er_D)
    return float(worst)


def _fit_ln_f_via_chebyshev(models, ref_cache, a_grid):
    """
    Fit ln(f) = ln(Om) * gamma using Chebyshev-inspired basis.
    We want to find c such that ln(f_approx) ≈ ln(f_ref) for all models.
    
    ln(f_approx) = ln(Om) * (c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q)
    
    This is linear in c! So we can do a direct least-squares fit.
    Then refine with min-max optimization.
    """
    # Build the linear system: for each (model, a) pair, we have:
    # ln(Om) * (c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q) = ln(f_ref)
    
    A_rows = []
    b_vals = []
    
    for (om0, w0, wa) in models:
        a = a_grid
        om = _omega_m(a, om0, w0, wa)
        _, ref_f, _ = ref_cache[(om0, w0, wa)]
        
        q = 1.0 - om
        u = 1.0 + w0 + wa * (1.0 - a)
        
        # Basis columns: 1, q, q^2, u, wa*a, u*q
        basis = np.column_stack([
            np.ones_like(a),
            q,
            q * q,
            u,
            wa * a,
            u * q
        ])
        
        # ln(f_approx) = ln(om) * gamma = ln(om) * basis @ c
        # So: (ln(om) * basis) @ c = ln(f_ref)
        ln_om = np.log(np.maximum(om, 1e-15))
        A_rows.append(ln_om[:, np.newaxis] * basis)
        b_vals.append(np.log(np.maximum(ref_f, 1e-15)))
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # Solve least squares
    c_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    c_ls = np.clip(c_ls, -10.0, 10.0)
    
    return c_ls


def _fit(models, seed=12345):
    """
    Two-stage fit:
    1. Linear least-squares on ln(f) to get a good initial guess.
    2. Refine with min-max (Nelder-Mead style) to reduce worst-case error.
    """
    # Build reference cache
    ref_cache = {}
    for m in models:
        ref_cache[m] = _build_reference(*m)
    
    # Stage 1: Linear fit on ln(f)
    c_ls = _fit_ln_f_via_chebyshev(models, ref_cache, _A_GRID)
    
    # Check feasibility of LS solution
    best_c = c_ls.copy()
    best_val = _objective(best_c, models, ref_cache)
    
    # If LS solution is infeasible, fall back to seed
    if best_val >= 1.0:
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_val = _objective(best_c, models, ref_cache)
    
    # Stage 2: Min-max refinement with multiple starts
    rng = np.random.default_rng(seed)
    starts = 0
    max_starts = 40
    deadline = _DEADLINE - 10.0
    
    # Nelder-Mead helper
    def _nelder_mead_local(c0, obj, max_iter=300, tol=1e-7):
        n = len(c0)
        simplex = [np.array(c0, dtype=float)]
        base = 0.03
        for i in range(n):
            v = np.array(c0, dtype=float)
            v[i] += base
            simplex.append(v)
        fvals = [obj(v) for v in simplex]
        for _ in range(max_iter):
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = obj(xr)
            if fr < fvals[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = obj(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = obj(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
                        fvals[i] = obj(simplex[i])
        best_i = int(np.argmin(fvals))
        return simplex[best_i], fvals[best_i]
    
    def obj(c):
        return _objective(c, models, ref_cache)
    
    # Start from LS solution
    c, val = _nelder_mead_local(best_c.copy(), obj, max_iter=400)
    c = np.clip(c, -10.0, 10.0)
    val = obj(c)
    if val < best_val:
        best_val = val
        best_c = c.copy()
    
    # Additional random starts around the best solution
    while time.time() < deadline and starts < max_starts:
        starts += 1
        c0 = best_c + rng.normal(0.0, 0.08, size=6)
        c0 = np.clip(c0, -10.0, 10.0)
        c, val = _nelder_mead_local(c0, obj, max_iter=250)
        c = np.clip(c, -10.0, 10.0)
        val = obj(c)
        if val < best_val:
            best_val = val
            best_c = c.copy()
    
    return best_c, best_val


def construct():
    t0 = time.time()
    # Public validation-style cosmologies (disjoint, smooth CPL):
    models = [
        (0.30, -0.9, 0.0),
        (0.28, -1.0, 0.0),
        (0.31, -1.1, 0.3),
        (0.27, -0.8, -0.3),
        (0.30, -1.0, 0.5),
        (0.33, -0.95, 0.2),
    ]
    
    best_c, best_val = _fit(models, seed=12345)
    
    # Final safety: ensure finite and in range.
    if not np.all(np.isfinite(best_c)):
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
