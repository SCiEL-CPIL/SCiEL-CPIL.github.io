"""Cosmology: universal luminosity-distance quadrature — Chebyshev-spectral design."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import time

# Deadline guard: finish well before 90 s wall-clock
_T0 = time.time()
_DEADLINE = 80.0  # safety margin

def _elapsed():
    return time.time() - _T0

def _time_left():
    return _DEADLINE - _elapsed()

def construct():
    """Return a universal 8-node positive quadrature for dL/(c/H0) = (1+z) z sum w_i / E(a_i)."""
    if _elapsed() > 5:
        # Fallback: composite midpoint
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    n = 8  # number of quadrature nodes

    # --- Step 1: Design a family of test integrands ---
    # The integrand in z-space is f(z) = z / E(a) where a = 1/(1+z).
    # For CPL dark energy: E(a) = a^{-3(1+w0+wa)} (1 - a)^{3 wa}
    # We sample a range of (w0, wa) to build a representative set.
    w0_vals = np.array([-2.5, -2.0, -1.5, -1.2, -1.0, -0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
    wa_vals = np.array([-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0])

    # z-grid for evaluating integrands
    z_grid = np.linspace(0, 1, 400)
    z_grid[0] = 1e-10  # avoid division by zero issues

    # Build matrix: rows = (w0, wa) pairs, cols = z_grid points
    # f(z; w0, wa) = z / E(a)
    integrands = []
    for w0 in w0_vals:
        for wa in wa_vals:
            a = 1.0 / (1.0 + z_grid)
            # E(a) = a^{-3(1+w0+wa)} * (1-a)^{3*wa}
            # Note: (1-a) = z/(1+z)
            log_E = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * np.log(1.0 - a)
            E = np.exp(log_E)
            f = z_grid / E
            integrands.append(f)

    F = np.array(integrands)  # shape (n_models, n_z)
    n_models = F.shape[0]

    # --- Step 2: Chebyshev spectral approximation ---
    # Map z in [0,1] to t in [-1,1]: t = 2z - 1
    # We want to find nodes z_i and weights w_i such that:
    #   sum_i w_i f(z_i) ≈ ∫_0^1 f(z) dz  for all test integrands
    #
    # Strategy: Use Chebyshev collocation to build a quadrature that is exact
    # for a Chebyshev polynomial basis up to some degree, then optimize.
    #
    # Alternative structural approach: 
    # 1. Find nodes via Chebyshev-Gauss-Lobatto points mapped to [0,1]
    # 2. Solve for weights via least-squares against the test integrands
    # 3. Optimize nodes to minimize worst-case error
    #
    # Let's use a two-stage approach:
    # Stage A: Start with Chebyshev-Gauss-Lobatto nodes, solve weights by LSQ
    # Stage B: Optimize both nodes and weights to minimize max relative error

    # Chebyshev-Gauss-Lobatto points on [-1, 1]
    j = np.arange(n)
    t_cgl = np.cos(np.pi * j / (n - 1))  # from 1 to -1
    t_cgl = t_cgl[::-1]  # from -1 to 1
    z_cgl = (t_cgl + 1.0) / 2.0  # map to [0, 1]

    # Solve for weights: minimize ||W @ F - 1||^2 where W is diagonal
    # Actually: for each model m, sum_i w_i F[m, i] ≈ 1 (exact integral of f over [0,1])
    # Wait: the exact integral is ∫_0^1 f(z) dz, not 1. Let me compute exact integrals.
    
    # Compute exact integrals using dense quadrature
    from scipy.integrate import quad
    exact_integrals = np.zeros(n_models)
    for m in range(n_models):
        f = F[m]
        # Use dense trapezoidal as reference (z_grid is dense enough)
        exact_integrals[m] = np.trapz(f, z_grid)

    # Build the system: A @ w = b where A[m, i] = F[m, i], b[m] = exact_integrals[m]
    A = F  # (n_models, n)
    b = exact_integrals

    # Solve least squares for weights given Chebyshev nodes
    w_init, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    # Ensure positivity: if any weight is negative, project
    if np.any(w_init < 0):
        # Use non-negative least squares via linprog
        # Minimize ||Aw - b||^2 subject to w >= 0, sum(w) = 1
        # Convert to QP... use scipy.optimize.minimize with SLSQP
        def obj(w):
            r = A @ w - b
            return np.sum(r**2)
        
        def grad(w):
            return 2.0 * A.T @ (A @ w - b)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(n)}]
        bounds = [(0, 1)] * n
        w0 = np.clip(w_init, 1e-6, None)
        w0 /= w0.sum()
        
        res = minimize(obj, w0, jac=grad, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-12})
        w_init = res.x
        w_init = np.maximum(w_init, 0)
        w_init /= w_init.sum()

    # --- Stage B: Optimize nodes to minimize worst-case relative error ---
    # Parameterize nodes in log-space to ensure they stay in (0,1) and increasing
    # nodes = sigmoid(u) ensures (0,1); enforce ordering via cumulative softmax or just optimize with bounds
    
    # Better: parameterize as z_i = 1/(1 + exp(-v_i)) with v sorted, or use a simpler approach:
    # z_i = (i + 1 + delta_i) / (n + 1) with small deltas, but that's limited.
    
    # Use: z_i = sigmoid(v_i), and enforce v_1 < v_2 < ... < v_n
    # Or simpler: z_i = (i-1 + eps_i) / (n-1) where eps_i in (0,1), but ordering is automatic if we use cumulative.
    
    # Simplest robust approach: parameterize by gaps
    # g_i > 0 for i=0..n-1, sum(g_i) = 1, z_i = cumsum(g)_i / (n-1) ... no
    # z_1 = g_1, z_2 = g_1 + g_2, ..., z_n = g_1 + ... + g_n = 1? No, z should be in (0,1)
    
    # Let's use: z_i = (cumsum(g))_i / sum(g), with g > 0. This gives 0 < z_1 < z_2 < ... < z_n = 1.
    # But we need z_n < 1 strictly. Use z_i = (cumsum(g))_i / (sum(g) + g_extra) where g_extra > 0.
    
    # Even simpler: just use the sigmoid parameterization with a sorting constraint handled by the optimizer.
    # Or: use the fact that we can optimize in the space of strictly increasing points via:
    # z_i = 1/(1 + exp(-v_i)), and require v_1 < v_2 < ... < v_n.
    
    # Let me use a direct approach: optimize z directly with bounds 0 < z_1 < ... < z_n < 1
    # using SLSQP with inequality constraints.
    
    def _compute_error(z, w):
        """Compute worst-case relative error across test models."""
        errors = np.zeros(n_models)
        for m in range(n_models):
            # Evaluate integrand at quadrature nodes
            a = 1.0 / (1.0 + z)
            w0 = w0_vals[m // len(wa_vals)]
            wa = wa_vals[m % len(wa_vals)]
            log_E = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * np.log(1.0 - a)
            E = np.exp(log_E)
            f_quad = z / E
            quad_val = np.dot(w, f_quad)
            exact_val = exact_integrals[m]
            if exact_val > 0:
                errors[m] = abs(quad_val - exact_val) / exact_val
        return np.max(errors)
    
    def _solve_weights(z):
        """Given nodes z, solve for optimal positive weights summing to 1."""
        # Build A matrix
        A = np.zeros((n_models, n))
        for m in range(n_models):
            a = 1.0 / (1.0 + z)
            w0 = w0_vals[m // len(wa_vals)]
            wa = wa_vals[m % len(wa_vals)]
            log_E = -3.0 * (1.0 + w0 + wa) * np.log(a) + 3.0 * wa * np.log(1.0 - a)
            E = np.exp(log_E)
            A[m, :] = z / E
        
        b = exact_integrals
        
        # Non-negative least squares with sum(w)=1
        def obj(w):
            r = A @ w - b
            return np.sum(r**2)
        
        def grad(w):
            return 2.0 * A.T @ (A @ w - b)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(n)}]
        bounds = [(1e-10, 1)] * n
        
        w0 = np.full(n, 1.0 / n)
        res = minimize(obj, w0, jac=grad, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 100, 'ftol': 1e-14})
        w_opt = res.x
        w_opt = np.maximum(w_opt, 1e-10)
        w_opt /= w_opt.sum()
        return w_opt
    
    # Initial nodes: Chebyshev-Gauss-Lobatto mapped to [0,1], slightly interior
    z_init = z_cgl.copy()
    z_init[0] = max(z_init[0], 1e-3)
    z_init[-1] = min(z_init[-1], 1 - 1e-3)
    
    # Optimize nodes
    def _node_objective(params):
        """params: (n-1) gap sizes, all > 0"""
        if _time_left() < 2:
            return 1e10
        gaps = np.exp(params)  # ensure positive
        z = np.cumsum(gaps)
        z = z / z[-1]  # normalize so z_n = 1
        # Shift interior: z_i = i/n + (z_i - i/n) * 0.5 to keep in (0,1)
        # Actually just use z as is, but ensure strict interior
        z[0] = max(z[0], 1e-4)
        z[-1] = min(z[-1], 1 - 1e-4)
        # Re-normalize to keep in (0,1)
        z = np.clip(z, 1e-4, 1 - 1e-4)
        # Ensure strictly increasing
        for i in range(1, n):
            if z[i] <= z[i-1]:
                z[i] = z[i-1] + 1e-6
        if z[-1] > 1 - 1e-4:
            z = z / z[-1] * (1 - 1e-4)
        
        w = _solve_weights(z)
        return _compute_error(z, w)
    
    # Initial gaps: uniform
    gaps_init = np.ones(n - 1)
    
    # Optimize
    best_z = z_init.copy()
    best_w = _solve_weights(best_z)
    best_err = _compute_error(best_z, best_w)
    
    # Try a few random restarts
    n_restarts = 5
    for r in range(n_restarts):
        if _time_left() < 5:
            break
        if r == 0:
            gaps = gaps_init.copy()
        else:
            gaps = np.ones(n - 1) * np.exp(np.random.normal(0, 0.3, n - 1))
        
        res = minimize(_node_objective, np.log(gaps), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            gaps_opt = np.exp(res.x)
            z_new = np.cumsum(gaps_opt)
            z_new = z_new / z_new[-1]
            z_new = np.clip(z_new, 1e-4, 1 - 1e-4)
            for i in range(1, n):
                if z_new[i] <= z_new[i-1]:
                    z_new[i] = z_new[i-1] + 1e-6
            if z_new[-1] > 1 - 1e-4:
                z_new = z_new / z_new[-1] * (1 - 1e-4)
            best_z = z_new
            best_w = _solve_weights(best_z)
    
    # Final check: ensure validity
    best_z = np.sort(best_z)
    best_z = np.clip(best_z, 1e-6, 1 - 1e-6)
    # Ensure strictly increasing
    for i in range(1, n):
        if best_z[i] <= best_z[i-1]:
            best_z[i] = best_z[i-1] + 1e-5
    best_z = np.clip(best_z, 1e-6, 1 - 1e-6)
    
    best_w = np.maximum(best_w, 1e-10)
    best_w /= best_w.sum()
    
    # Verify sum of weights
    if abs(best_w.sum() - 1.0) > 1e-12:
        best_w /= best_w.sum()
    
    return {"nodes": best_z, "weights": best_w}
# EVOLVE-BLOCK-END
