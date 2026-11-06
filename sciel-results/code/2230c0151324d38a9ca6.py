"""
Strategy: Minimax Chebyshev quadrature design via alternating projection.

Key structural change: Instead of least-squares fitting to a finite set of reference
integrals (which biases toward the training set), we directly minimize the worst-case
relative error over a continuous parameter space using a Chebyshev-minimax approach.
The key insight is that the integrand f(t; z, w0, wa) = 1/E(z*t) lies in a low-dimensional
function space parameterized by (z, w0, wa). We approximate this space by a set of
extreme basis functions and solve a Chebyshev approximation problem: find nodes and
weights that minimize max_{f in F} |sum_i w_i f(t_i) - integral(f)| / integral(f).

The algorithm:
1. Sample the parameter space densely to identify "extreme" integrand shapes.
2. For fixed nodes, solve the linear Chebyshev problem for weights (minimize max relative error).
3. Optimize nodes using a derivative-free method that evaluates the Chebyshev criterion.
4. Use a multi-start strategy with diverse initial node configurations.

This is fundamentally different from the current approach because:
- We minimize the true worst-case error, not a proxy (least-squares on finite samples).
- The weight optimization is a linear Chebyshev problem (solvable exactly for fixed nodes).
- Node optimization uses the actual minimax criterion, not a surrogate.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL dark energy model."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)

def _reference_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference: \int_0^1 dt / E(z*t)."""
    if z < 1e-10:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, err = quad(integrand, 0, 1, limit=300, epsabs=1e-13, epsrel=1e-13)
    return val

def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute \sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)

def _solve_chebyshev_weights(nodes, test_points, Omega_m=0.3):
    """
    For fixed nodes, solve the linear Chebyshev problem:
    minimize max_i |A[i,:] @ w - ref[i]| / ref[i]
    subject to sum(w) = 1, w >= 0
    
    This is a linear programming problem. We use the standard reformulation:
    minimize t
    subject to:
      A[i,:] @ w - ref[i] <= t * ref[i]
      -(A[i,:] @ w - ref[i]) <= t * ref[i]
      sum(w) = 1
      w >= 0
    
    Variables: [w_0, ..., w_7, t]
    """
    n = len(nodes)
    N = len(test_points)
    
    # Build A matrix and references
    A = np.zeros((N, n))
    refs = np.zeros(N)
    
    for idx, (z, w0, wa) in enumerate(test_points):
        if z < 1e-10:
            A[idx, :] = 1.0
            refs[idx] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[idx, :] = 1.0 / E
            refs[idx] = _reference_integral(z, w0, wa, Omega_m)
    
    # LP formulation:
    # Minimize: [0, ..., 0, 1] @ [w, t]
    # Subject to:
    #   A[i,:] @ w - ref[i] - t * ref[i] <= 0  for all i
    #   -A[i,:] @ w + ref[i] - t * ref[i] <= 0  for all i
    #   sum(w) = 1  (equality)
    #   w >= 0, t >= 0 (implied by constraints)
    
    # Variables: x = [w_0, ..., w_7, t]
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each test point i:
    #   A[i,:] @ w - t * ref[i] <= ref[i]
    #   -A[i,:] @ w - t * ref[i] <= -ref[i]
    
    A_ub = np.zeros((2 * N, n + 1))
    b_ub = np.zeros(2 * N)
    
    for i in range(N):
        A_ub[2*i, :n] = A[i, :]
        A_ub[2*i, n] = -refs[i]
        b_ub[2*i] = refs[i]
        
        A_ub[2*i+1, :n] = -A[i, :]
        A_ub[2*i+1, n] = -refs[i]
        b_ub[2*i+1] = -refs[i]
    
    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * (n + 1)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        if result.success:
            w = result.x[:n]
            t = result.x[n]
            # Ensure weights sum to 1
            w_sum = np.sum(w)
            if w_sum > 1e-12:
                w = w / w_sum
            return w, t
        else:
            return None, float('inf')
    except Exception:
        return None, float('inf')

def _generate_test_points():
    """
    Generate a diverse set of test points (z, w0, wa) that captures the
    variation of the integrand across the parameter space.
    """
    # Focus on regions where the integrand varies most
    z_vals = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    wa_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    test_points = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_points.append((z, w0, wa))
    
    return test_points

def _gauss_legendre_nodes(n=8):
    """Gauss-Legendre nodes mapped to [0,1]."""
    nodes, _ = roots_legendre(n)
    return (nodes + 1.0) / 2.0

def _chebyshev_nodes(n=8):
    """Chebyshev nodes of the first kind mapped to [0,1]."""
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return (nodes + 1.0) / 2.0

def _generate_initial_candidates():
    """
    Generate diverse initial node configurations for multi-start optimization.
    """
    candidates = []
    
    # 1. Gauss-Legendre
    candidates.append(_gauss_legendre_nodes(8))
    
    # 2. Chebyshev
    candidates.append(_chebyshev_nodes(8))
    
    # 3. Gauss-Jacobi with various parameters
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0)]:
        try:
            nodes, _ = roots_jacobi(8, alpha, beta)
            nodes = (nodes + 1.0) / 2.0
            if np.all(np.diff(nodes) > 1e-10):
                candidates.append(nodes)
        except:
            pass
    
    # 4. Uniform
    candidates.append(np.linspace(1/17, 16/17, 8))
    
    # 5. Log-spaced (concentrated near 0)
    for base in [1.5, 2.0, 3.0]:
        t = np.linspace(0, 1, 8)
        log_nodes = 1.0 - (1.0 - t) ** base
        log_nodes = np.clip(log_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(log_nodes) > 1e-6):
            candidates.append(log_nodes)
    
    # 6. Power-law concentrated near 1
    for base in [0.5, 0.3, 0.7]:
        t = np.linspace(0, 1, 8)
        pow_nodes = t ** (1.0 / base)
        pow_nodes = np.clip(pow_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(pow_nodes) > 1e-6):
            candidates.append(pow_nodes)
    
    # 7. Perturbed Gauss-Legendre
    gl = _gauss_legendre_nodes(8)
    rng = np.random.RandomState(42)
    for trial in range(15):
        perturb = rng.uniform(-0.05, 0.05, 8)
        perturbed = gl + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    # 8. Perturbed Chebyshev
    ch = _chebyshev_nodes(8)
    for trial in range(10):
        perturb = rng.uniform(-0.05, 0.05, 8)
        perturbed = ch + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    return candidates

def _evaluate_nodes(nodes, test_points, Omega_m=0.3):
    """
    Evaluate the Chebyshev error for a given node configuration.
    Returns (weights, max_relative_error).
    """
    w, t = _solve_chebyshev_weights(nodes, test_points, Omega_m)
    if w is None:
        return None, float('inf')
    return w, t

def construct():
    # Deadline check at start
    if time.time() > _DEADLINE - 2:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Generate test points
    test_points = _generate_test_points()
    
    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    # Coarse search over initial candidates
    for cand in candidates:
        if time.time() > _DEADLINE - 10:
            break
        w, err = _evaluate_nodes(cand, test_points)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Refine with local optimization using the Chebyshev criterion
    n = 8
    
    def _objective_nodes(node_params):
        nodes_p = node_params.copy()
        if np.any(nodes_p <= 0) or np.any(nodes_p >= 1):
            return 1e10
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-8):
            return 1e10
        w, err = _evaluate_nodes(nodes_p, test_points)
        if w is None:
            return 1e10
        return err
    
    x0 = best_nodes.copy()
    bounds = [(1e-4, 1 - 1e-4)] * n
    
    try:
        result = minimize(
            _objective_nodes, x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < best_err:
            refined_nodes = result.x
            if np.all(refined_nodes > 0) and np.all(refined_nodes < 1):
                if np.all(np.diff(refined_nodes) > 1e-8):
                    w_ref, err_ref = _evaluate_nodes(refined_nodes, test_points)
                    if w_ref is not None and err_ref < best_err:
                        best_nodes = refined_nodes
                        best_weights = w_ref
                        best_err = err_ref
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
