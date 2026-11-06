"""
Strategy: Moment-matching quadrature with Chebyshev expansion.

Instead of optimizing nodes and weights jointly via a nonlinear minimax problem,
we use a different design family:
1. For a set of candidate node configurations (Gauss-Legendre, Chebyshev, Gauss-Jacobi,
   and perturbed variants), we solve for weights that minimize the maximum absolute
   error across a dense set of test integrals. This is a linear program (LP) in the
   weights for fixed nodes.
2. We use scipy.optimize.linprog for the LP, which is fast and reliable.
3. We use a fast fixed-order Gauss-Legendre quadrature (64-point) as the reference
   integral instead of adaptive quad, enabling many more node configurations to be
   evaluated within the time budget.
4. We then do a local refinement of the best node configuration using a coordinate
   descent or Nelder-Mead approach on the nodes, re-solving the LP at each step.

Key difference from current program:
- Current: joint nonlinear optimization of nodes and weights with least-squares + projection.
- New: alternating scheme where weights are solved via LP (minimax) for fixed nodes,
  and nodes are refined via local search. The LP gives a proper minimax solution for
  weights, not just a projected least-squares approximation.
"""

import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin


def _time_left():
    return _DEADLINE - time.time()


def _E_of_z_vec(z, w0, wa, Omega_m=0.3):
    """Vectorized E(z) for CPL dark energy."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)


def _ref_integral_fast(z, w0, wa, Omega_m=0.3, n_gauss=64):
    """Fast reference integral using 64-point Gauss-Legendre."""
    if z < 1e-10:
        return 1.0
    x, w = roots_legendre(n_gauss)
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    zt = z * t
    E = _E_of_z_vec(zt, w0, wa, Omega_m)
    return np.dot(wt, 1.0 / E)


def _quadrature_at_nodes(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z_vec(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)


def _build_test_set():
    """Build a dense test set of (z, w0, wa) with reference integrals."""
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])
    wa_vals = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    test_points = []
    refs = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_points.append((z, w0, wa))
                refs.append(_ref_integral_fast(z, w0, wa))
    
    return test_points, np.array(refs)


def _solve_lp_for_weights(nodes, test_points, refs, Omega_m=0.3):
    """
    Solve LP to find positive weights that minimize the maximum absolute error.
    
    Minimize:   epsilon
    Subject to: -epsilon <= sum_j A[i,j] w_j - refs[i] <= epsilon  for all i
                w_j >= 0
                sum_j w_j = 1
    
    Variables: [w_0, ..., w_7, epsilon]
    """
    n = len(nodes)
    N = len(test_points)
    
    # Build A matrix
    A = np.zeros((N, n))
    for idx, (z, w0, wa) in enumerate(test_points):
        if z < 1e-10:
            A[idx, :] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z_vec(zt, w0, wa, Omega_m)
            A[idx, :] = 1.0 / E
    
    # LP formulation:
    # Minimize [0, 0, ..., 0, 1] . [w, eps]
    # Constraints:
    #   A[i] @ w - eps <= refs[i]  =>  A[i] @ w - eps - refs[i] <= 0
    #   -A[i] @ w + eps <= -refs[i]  =>  -A[i] @ w + eps + refs[i] <= 0
    #   sum(w) = 1  (handled as equality)
    #   w >= 0, eps >= 0 (handled by bounds)
    
    c = np.zeros(n + 1)
    c[-1] = 1.0  # minimize epsilon
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each i: A[i] @ w - eps <= refs[i]  =>  [A[i], -1] @ [w, eps] <= refs[i]
    #            -A[i] @ w + eps <= -refs[i]  =>  [-A[i], 1] @ [w, eps] <= -refs[i]
    
    A_ub = np.zeros((2 * N, n + 1))
    b_ub = np.zeros(2 * N)
    
    for i in range(N):
        A_ub[2*i, :n] = A[i]
        A_ub[2*i, n] = -1.0
        b_ub[2*i] = refs[i]
        
        A_ub[2*i+1, :n] = -A[i]
        A_ub[2*i+1, n] = 1.0
        b_ub[2*i+1] = -refs[i]
    
    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_j >= 0, eps >= 0
    bounds = [(0, None)] * n + [(0, None)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        if result.success:
            w = result.x[:n]
            eps = result.x[n]
            # Ensure positivity and normalization
            w = np.maximum(w, 0.0)
            w_sum = np.sum(w)
            if w_sum > 1e-15:
                w = w / w_sum
            return w, eps
        else:
            return None, float('inf')
    except Exception:
        return None, float('inf')


def _eval_minimax(nodes, test_points, refs, Omega_m=0.3):
    """Evaluate the maximum relative error for a given node configuration."""
    w, eps = _solve_lp_for_weights(nodes, test_points, refs, Omega_m)
    if w is None:
        return None, float('inf')
    
    # Compute max relative error
    max_rel_err = 0.0
    for idx, (z, w0, wa) in enumerate(test_points):
        approx = _quadrature_at_nodes(nodes, w, z, w0, wa, Omega_m)
        ref = refs[idx]
        rel_err = abs(approx - ref) / max(ref, 1e-15)
        max_rel_err = max(max_rel_err, rel_err)
    
    return w, max_rel_err


def _gauss_legendre_nodes(n=8):
    nodes, _ = roots_legendre(n)
    return (nodes + 1.0) / 2.0


def _chebyshev_nodes(n=8):
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return (nodes + 1.0) / 2.0


def _generate_node_candidates():
    """Generate diverse node configurations."""
    candidates = []
    
    # Gauss-Legendre
    candidates.append(_gauss_legendre_nodes(8))
    
    # Chebyshev
    candidates.append(_chebyshev_nodes(8))
    
    # Gauss-Jacobi
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), 
                         (0.5, 1.0), (1.0, 0.5), (2.0, 0.0), (0.0, 2.0),
                         (1.5, 0.5), (0.5, 1.5)]:
        try:
            nodes, _ = roots_jacobi(8, alpha, beta)
            nodes = (nodes + 1.0) / 2.0
            if np.all(np.diff(nodes) > 1e-10):
                candidates.append(nodes)
        except:
            pass
    
    # Uniform
    candidates.append(np.linspace(1/17, 16/17, 8))
    
    # Log-spaced
    for base in [1.5, 2.0, 3.0, 5.0]:
        t = np.linspace(0, 1, 8)
        log_nodes = 1.0 - (1.0 - t) ** base
        log_nodes = np.clip(log_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(log_nodes) > 1e-6):
            candidates.append(log_nodes)
    
    # Power-law
    for base in [0.5, 0.3, 0.2, 0.7]:
        t = np.linspace(0, 1, 8)
        pow_nodes = t ** (1.0 / base)
        pow_nodes = np.clip(pow_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(pow_nodes) > 1e-6):
            candidates.append(pow_nodes)
    
    # Perturbed Gauss-Legendre
    gl = _gauss_legendre_nodes(8)
    rng = np.random.RandomState(42)
    for trial in range(30):
        perturb = rng.uniform(-0.05, 0.05, 8)
        perturbed = gl + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    # Perturbed Chebyshev
    ch = _chebyshev_nodes(8)
    for trial in range(15):
        perturb = rng.uniform(-0.05, 0.05, 8)
        perturbed = ch + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    return candidates


def construct():
    if time.time() > _DEADLINE - 2:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Build test set
    test_points, refs = _build_test_set()
    
    # Generate node candidates
    candidates = _generate_node_candidates()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    # Coarse search
    for cand in candidates:
        if time.time() > _DEADLINE - 10:
            break
        w, err = _eval_minimax(cand, test_points, refs)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Refine with local optimization on nodes
    n = 8
    
    def _objective_nodes(node_params):
        nodes_p = node_params.copy()
        if np.any(nodes_p <= 0) or np.any(nodes_p >= 1):
            return 1e10
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-8):
            return 1e10
        w, err = _eval_minimax(nodes_p, test_points, refs)
        if w is None:
            return 1e10
        return err
    
    x0 = best_nodes.copy()
    bounds = [(1e-4, 1 - 1e-4)] * n
    
    try:
        result = minimize(
            _objective_nodes, x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if result.fun < best_err:
            refined_nodes = result.x
            if np.all(refined_nodes > 0) and np.all(refined_nodes < 1):
                if np.all(np.diff(refined_nodes) > 1e-8):
                    w_ref, err_ref = _eval_minimax(refined_nodes, test_points, refs)
                    if w_ref is not None and err_ref < best_err:
                        best_nodes = refined_nodes
                        best_weights = w_ref
                        best_err = err_ref
    except Exception:
        pass
    
    # Second refinement with L-BFGS-B if time permits
    if time.time() < _DEADLINE - 15:
        try:
            result2 = minimize(
                _objective_nodes, best_nodes,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result2.fun < best_err:
                refined_nodes2 = result2.x
                if np.all(refined_nodes2 > 0) and np.all(refined_nodes2 < 1):
                    if np.all(np.diff(refined_nodes2) > 1e-8):
                        w_ref2, err_ref2 = _eval_minimax(refined_nodes2, test_points, refs)
                        if w_ref2 is not None and err_ref2 < best_err:
                            best_nodes = refined_nodes2
                            best_weights = w_ref2
                            best_err = err_ref2
        except Exception:
            pass
    
    # Final verification
    w_sum = np.sum(best_weights)
    if w_sum > 1e-12:
        best_weights = best_weights / w_sum
    
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    diffs = np.diff(best_nodes)
    if np.any(diffs <= 0):
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if np.any(best_weights <= 0):
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}


# EVOLVE-BLOCK-END
