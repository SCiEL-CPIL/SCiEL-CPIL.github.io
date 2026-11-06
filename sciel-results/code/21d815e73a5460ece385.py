"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyt


def _cpl_g(x, w0, wa):
    """Model-dependent factor g(x)=1/E(1-x) for CPL. x = z/(1+z), a=1-x."""
    a = np.clip(1.0 - x, 1e-15, 1.0)
    wa_eff = wa / a
    exponent = 3.0 * (1.0 + w0 + wa_eff)
    g = np.power(a, exponent)
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def _project_positive(w, tol=1e-12):
    """Project weights onto the cone w_i>0, sum w = 1."""
    w = np.asarray(w, dtype=float).copy()
    n = w.size
    for _ in range(2000):
        if w.sum() <= 1.0:
            # Scale up
            w = w / max(w.sum(), 1e-30)
            break
        # Find the minimum weight
        idx_min = np.argmin(w)
        # Reduce all weights proportionally so the min hits tol
        excess = w.sum() - 1.0
        # Subtract from all, but clamp at tol
        w_new = w - excess / n
        w_new = np.maximum(w_new, tol)
        if np.abs(w_new.sum() - 1.0) < 1e-14:
            w = w_new
            break
        w = w_new
    w = np.maximum(w, tol)
    w = w / w.sum()
    return w


def _chebyshev_basis(x, n_terms):
    """Evaluate Chebyshev polynomials T_0..T_{n_terms-1} at x in [-1,1]."""
    n = len(x)
    basis = np.empty((n, n_terms))
    for k in range(n_terms):
        basis[:, k] = eval_chebyt(k, x)
    return basis


def _solve(w0_grid, wa_grid):
    """
    Fit the integrand f(x; w0, wa) = x * g(x; w0, wa) against Chebyshev basis
    spanning the CPL parameter space, then derive optimal 8-node positive quadrature.
    """
    n_models = len(w0_grid) * len(wa_grid)
    
    # Fine grid for integration and fitting
    M = 2000
    x = np.linspace(1e-6, 1.0 - 1e-6, M)
    
    # Map to [-1,1] for Chebyshev
    x_cheb = 2.0 * x - 1.0
    
    # Build model library: f(x; w0, wa) = x * g(x; w0, wa)
    f_lib = np.empty((n_models, M))
    f_true = np.empty(n_models)
    k = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            g = _cpl_g(x, w0, wa)
            f = x * g
            f_lib[k] = f
            f_true[k] = np.trapz(f, x)
            k += 1
    f_true = np.maximum(f_true, 1e-30)
    
    # Weighting: emphasize relative accuracy (small f -> larger relative error)
    wgt = 1.0 / np.sqrt(np.maximum(f_true, 1e-3))
    wgt = wgt / wgt.max()
    
    # Fit each model's integrand against Chebyshev basis
    n_terms = 12  # Number of Chebyshev terms
    basis = _chebyshev_basis(x_cheb, n_terms)  # (M, n_terms)
    
    # For each model, compute Chebyshev coefficients via weighted least squares
    # Coeffs[m, k] = <f_m * wgt_m, T_k> / <T_k, T_k> (weighted)
    coeffs = np.empty((n_models, n_terms))
    for m in range(n_models):
        f_m = f_lib[m] * wgt[m]
        # Solve: minimize ||B^T * c - f_m||^2 where B = basis
        # Using normal equations: (B B^T) c = B f_m
        B = basis  # (M, n_terms)
        # Weighted inner products
        BtB = B.T @ B  # (n_terms, n_terms)
        Btf = B.T @ f_m  # (n_terms,)
        # Add small regularization for stability
        BtB += 1e-10 * np.eye(n_terms)
        try:
            coeffs[m] = np.linalg.solve(BtB, Btf)
        except np.linalg.LinAlgError:
            coeffs[m] = np.linalg.lstsq(BtB, Btf, rcond=None)[0]
    
    # The integrand can be approximated as:
    # f(x; w0, wa) ≈ sum_k c_k(w0, wa) * T_k(2x-1)
    # where c_k varies with model parameters.
    
    # Key insight: The nodes and weights should be chosen to accurately integrate
    # any function in the span of these Chebyshev polynomials weighted by the
    # typical coefficient patterns.
    
    # Compute the "effective" integrand shape: average over models with weighting
    # to find where the function varies most.
    
    # Alternative approach: Use the Chebyshev coefficients to determine optimal nodes.
    # The nodes should be clustered where the Chebyshev basis has high leverage.
    
    # Let's use a different strategy: 
    # 1. Determine the range of exponents that appear
    # 2. Fit against a set of basis functions that capture this variation
    
    # The CPL exponent is: exponent = 3*(1 + w0 + wa/a)
    # As a goes from 0 to 1, wa/a goes from infinity to wa.
    # For a near 1, exponent ≈ 3*(1+w0+wa)
    # For a near 0, exponent → -infinity (if wa > 0) or +infinity (if wa < 0)
    
    # Let's parameterize by the "effective" behavior.
    # Instead, let's directly optimize nodes and weights to minimize worst-case
    # relative error over the model library, using the Chebyshev fit as initialization.
    
    # Initialize nodes: Gauss-Chebyshev nodes (optimal for Chebyshev-weighted integration)
    # Gauss-Chebyshev nodes in [-1,1]: x_k = cos((2k+1)π/(2N)), k=0..N-1
    N = 8
    k_vals = np.arange(N)
    x_cheb_nodes = np.cos((2.0 * k_vals + 1.0) * np.pi / (2.0 * N))
    # Map back to [0,1]
    nodes_init = 0.5 * (x_cheb_nodes + 1.0)
    nodes_init = np.sort(nodes_init)[::-1]  # cos is decreasing, so reverse for increasing
    nodes_init = np.clip(nodes_init, 1e-6, 1.0 - 1e-6)
    
    # Initialize weights: Gauss-Chebyshev weights
    weights_init = np.full(N, np.pi / N)
    weights_init = weights_init / weights_init.sum()
    
    nodes = nodes_init.copy()
    weights = weights_init.copy()
    
    # Now refine: alternating optimization of nodes and weights
    # to minimize worst relative error over the model library
    
    def _compute_samples(nd, w0_grid, wa_grid):
        """Compute integrand values at nodes for all models."""
        n_m = len(w0_grid) * len(wa_grid)
        S = np.empty((n_m, len(nd)))
        k = 0
        for w0 in w0_grid:
            for wa in wa_grid:
                g = _cpl_g(nd, w0, wa)
                S[k] = nd * g
                k += 1
        return S
    
    def _worst_rel_err(nd, wt):
        S = _compute_samples(nd, w0_grid, wa_grid)
        Q = S @ wt
        err = np.abs(Q - f_true) / f_true
        return err.max()
    
    # Alternating optimization
    for it in range(200):
        # --- Update weights: minimize max_m |S_m @ w - f_true_m| s.t. w>=0, sum w=1
        S = _compute_samples(nodes, w0_grid, wa_grid)  # (n_models, 8)
        
        # Linear programming: min t s.t. |S @ w - f_true| <= t, w >= 0, sum w = 1
        from scipy.optimize import linprog
        n_m = len(f_true)
        # Variables: w (8), t (1)
        c = np.zeros(9)
        c[8] = 1.0
        A_ub = np.zeros((2 * n_m, 9))
        b_ub = np.zeros(2 * n_m)
        A_ub[:n_m, :8] = S
        A_ub[:n_m, 8] = -1.0
        A_ub[n_m:, :8] = -S
        A_ub[n_m:, 8] = -1.0
        b_ub[:n_m] = f_true
        b_ub[n_m:] = f_true
        A_eq = np.zeros((1, 8))
        A_eq[0, :] = 1.0
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * 8 + [(0.0, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            weights = _project_positive(res.x[:8])
        
        # --- Update nodes: minimize worst relative error
        from scipy.optimize import minimize
        
        def obj_nodes(v):
            nd = np.sort(np.clip(v, 1e-6, 1.0 - 1e-6))
            # Ensure strict increase
            for i in range(1, len(nd)):
                if nd[i] <= nd[i-1] + 1e-8:
                    nd[i] = nd[i-1] + 1e-8
            S = _compute_samples(nd, w0_grid, wa_grid)
            Q = S @ weights
            err = np.abs(Q - f_true) / f_true
            return err.max()
        
        res2 = minimize(obj_nodes, nodes, method="L-BFGS-B",
                        bounds=[(1e-6, 1.0 - 1e-6)] * 8,
                        options={"maxiter": 100, "ftol": 1e-12})
        if res2.success:
            nodes = np.sort(res2.x)
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1] + 1e-8:
                    nodes[i] = nodes[i-1] + 1e-8
    
    # Final polish with Nelder-Mead on nodes
    from scipy.optimize import minimize
    
    def obj_nodes_final(v):
        nd = np.sort(np.clip(v, 1e-6, 1.0 - 1e-6))
        for i in range(1, len(nd)):
            if nd[i] <= nd[i-1] + 1e-8:
                nd[i] = nd[i-1] + 1e-8
        S = _compute_samples(nd, w0_grid, wa_grid)
        Q = S @ weights
        err = np.abs(Q - f_true) / f_true
        return err.max()
    
    res3 = minimize(obj_nodes_final, nodes, method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-12})
    if res3.success:
        nodes = np.sort(res3.x)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1] + 1e-8:
                nodes[i] = nodes[i-1] + 1e-8
    
    # Final weight optimization
    S = _compute_samples(nodes, w0_grid, wa_grid)
    from scipy.optimize import linprog
    n_m = len(f_true)
    c = np.zeros(9)
    c[8] = 1.0
    A_ub = np.zeros((2 * n_m, 9))
    b_ub = np.zeros(2 * n_m)
    A_ub[:n_m, :8] = S
    A_ub[:n_m, 8] = -1.0
    A_ub[n_m:, :8] = -S
    A_ub[n_m:, 8] = -1.0
    b_ub[:n_m] = f_true
    b_ub[n_m:] = f_true
    A_eq = np.zeros((1, 8))
    A_eq[0, :] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * 8 + [(0.0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.success:
        weights = _project_positive(res.x[:8])
    
    # Final sanity
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    nodes = np.clip(np.sort(nodes), 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-8:
            nodes[i] = nodes[i-1] + 1e-8
    
    return {"nodes": nodes, "weights": weights}


def construct():
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    try:
        rule = _solve(w0_grid, wa_grid)
    except Exception:
        # Fallback: Gauss-Chebyshev mapped to [0,1]
        N = 8
        k_vals = np.arange(N)
        x_cheb_nodes = np.cos((2.0 * k_vals + 1.0) * np.pi / (2.0 * N))
        nodes = np.sort(0.5 * (x_cheb_nodes + 1.0))[::-1]
        nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
        weights = np.full(N, 1.0 / N)
        rule = {"nodes": nodes, "weights": weights}

    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    # Enforce contract: strictly increasing in [0,1], positive weights summing to 1.
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
