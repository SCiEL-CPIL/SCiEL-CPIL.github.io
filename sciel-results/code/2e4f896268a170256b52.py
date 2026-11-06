"""
Strategy: Smooth L2 Least-Squares Quadrature Design.

Key structural change: The current program uses a non-smooth Chebyshev (minimax) criterion,
which is difficult to optimize efficiently with derivative-free methods like Nelder-Mead.
This new program reformulates the problem as a smooth least-squares (L2) approximation problem.

We define a dense grid of cosmological parameters (z, w0, wa) and minimize the sum of
squared relative errors between the quadrature approximation and the true integral.
This objective function is smooth and differentiable with respect to both nodes and weights.
We use scipy.optimize.minimize with a gradient-based method (L-BFGS-B or BFGS) to find
the optimal nodes and weights.

Why this may be better:
1. Smoothness: L2 error is differentiable, allowing gradient-based optimizers to take
   efficient steps. Nelder-Mead struggles with the kinks in the max-error function.
2. Globality: Minimizing the average error over a dense grid prevents overfitting to a
   few extreme points and provides a robust "universal" rule.
3. Efficiency: Gradient methods converge faster than simplex methods in high dimensions
   (16 dimensions: 8 nodes + 8 weights).

The constraint that weights must be positive and sum to 1 is handled by parameterizing
the weights using a softmax or by solving a constrained subproblem, but for simplicity
and robustness, we optimize unconstrained weights and nodes, then project/normalize
weights at the end. However, to ensure positivity during optimization, we can use
log-weights or just rely on the optimizer staying in the positive region if initialized
well. Given the physics, positive weights are expected. We will use bounds for nodes
[0,1] and optimize weights freely, then check positivity. If weights go negative, we
might need a penalty term or a different parameterization.

Actually, a cleaner approach for positive weights summing to 1 is to optimize 7 weights
and the 8th is 1 - sum(others), but this breaks symmetry. 
Better: Optimize 8 weights with bounds [0, 1] and 8 nodes with bounds [0, 1].
The sum-to-1 constraint can be handled by a penalty term in the objective function
or by using a constrained optimizer. L-BFGS-B supports bounds but not equality constraints.
We can use SLSQP which supports equality constraints, or simply add a large penalty
for deviation from sum=1.

Let's use a penalty method for simplicity and robustness:
Objective = Sum_Squared_Relative_Errors + Penalty * (Sum(Weights) - 1)^2

Nodes are bounded to [epsilon, 1-epsilon] to ensure strict increase isn't strictly enforced
by the optimizer, but we can sort them or penalize non-increasing nodes.
Actually, the formula assumes nodes are just values. The order doesn't matter for the sum,
but the problem statement says "strictly increasing". We can sort the nodes at the end.
During optimization, we don't need to enforce ordering, just that they are distinct.
We can add a penalty for nodes being too close or out of order, or just sort at the end
and hope the optimizer finds a good set. Sorting at the end is safe because the sum
is commutative.

Wait, if we sort nodes, the weights must be permuted accordingly.
If we optimize unordered nodes and weights, the pairing matters.
So we must optimize ordered nodes.
We can parameterize nodes as cumulative sums of positive increments.
Let nodes[0] = d0, nodes[1] = d0+d1, ..., nodes[7] = d0+...+d7.
Constraint: sum(d_i) < 1.
This ensures strictly increasing nodes.
Variables: d_0...d_7 (8 vars) and w_0...w_7 (8 vars).
Total 16 vars.

Let's stick to simpler bounds and penalty for now.
Nodes in [0, 1], Weights in [0, 1].
Penalty for (sum(w)-1)^2 and for (diff(nodes) < 0)^2.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL dark energy model."""
    if np.any(z < 0):
        # Handle z=0 carefully if vectorized, but here z is scalar or array
        pass
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)

def _reference_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference integral."""
    if z < 1e-10:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val

def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)

def _generate_grid():
    """
    Generate a dense grid of (z, w0, wa) to define the L2 error surface.
    We want to cover the relevant physical range.
    z: 0.1 to 10 (log spaced)
    w0: -1 to 1
    wa: -1 to 1
    """
    z_vals = np.logspace(-1, 1, 20) # 0.1 to 10
    w0_vals = np.linspace(-1.0, 1.0, 7)
    wa_vals = np.linspace(-1.0, 1.0, 7)
    
    grid = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                grid.append((z, w0, wa))
    return grid

def _objective(params, grid, Omega_m=0.3):
    """
    Objective function: Sum of squared relative errors + penalty terms.
    params: [8 nodes, 8 weights]
    """
    nodes = params[:8]
    weights = params[8:]
    
    # Penalty for sum(weights) != 1
    w_sum = np.sum(weights)
    penalty_sum = 100.0 * (w_sum - 1.0)**2
    
    # Penalty for non-increasing nodes or nodes out of bounds
    # Nodes should be in [0, 1]
    penalty_bounds = 0.0
    if np.any(nodes < 0) or np.any(nodes > 1):
        penalty_bounds += 100.0 * (np.sum(np.minimum(0, nodes)) + np.sum(np.maximum(0, nodes-1)))
    
    diffs = np.diff(nodes)
    if np.any(diffs <= 0):
        penalty_bounds += 100.0 * np.sum(np.maximum(0, -diffs))
        
    # Penalty for negative weights
    penalty_neg_w = 100.0 * np.sum(np.minimum(0, weights))
    
    total_penalty = penalty_sum + penalty_bounds + penalty_neg_w
    
    if total_penalty > 1.0:
        return total_penalty
    
    # Compute L2 error
    sq_err = 0.0
    for z, w0, wa in grid:
        ref = _reference_integral(z, w0, wa, Omega_m)
        if ref < 1e-10:
            continue
        approx = _quadrature_value(nodes, weights, z, w0, wa, Omega_m)
        rel_err = (approx - ref) / ref
        sq_err += rel_err**2
        
    return sq_err + total_penalty

def construct():
    if time.time() > _DEADLINE - 5:
        # Fallback: Gauss-Legendre
        from scipy.special import roots_legendre
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    grid = _generate_grid()
    
    # Initial guess: Gauss-Legendre
    from scipy.special import roots_legendre
    nodes0, weights0 = roots_legendre(8)
    nodes0 = (nodes0 + 1) / 2
    weights0 = weights0 / 2
    
    x0 = np.concatenate([nodes0, weights0])
    
    # Bounds: Nodes in [0,1], Weights in [0,1]
    bounds = [(1e-6, 1-1e-6)] * 8 + [(0, 1)] * 8
    
    best_result = None
    best_val = float('inf')
    
    # Try multiple starting points
    rng = np.random.RandomState(42)
    
    starts = [x0]
    # Add some random perturbations
    for i in range(5):
        pert_nodes = nodes0 + rng.uniform(-0.1, 0.1, 8)
        pert_weights = weights0 + rng.uniform(-0.05, 0.05, 8)
        pert_weights = np.clip(pert_weights, 1e-4, 1)
        pert_weights = pert_weights / np.sum(pert_weights)
        pert_nodes = np.sort(np.clip(pert_nodes, 1e-4, 1-1e-4))
        if np.all(np.diff(pert_nodes) > 1e-5):
            starts.append(np.concatenate([pert_nodes, pert_weights]))
            
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(
                _objective, start, args=(grid, 0.3),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        return {"nodes": nodes0, "weights": weights0}
        
    nodes_opt = best_result[:8]
    weights_opt = best_result[8:]
    
    # Final cleanup: sort nodes and corresponding weights
    sort_idx = np.argsort(nodes_opt)
    nodes_opt = nodes_opt[sort_idx]
    weights_opt = weights_opt[sort_idx]
    
    # Normalize weights
    w_sum = np.sum(weights_opt)
    if w_sum > 1e-12:
        weights_opt = weights_opt / w_sum
    else:
        weights_opt = weights0
        nodes_opt = nodes0
        
    # Ensure strict positivity and ordering
    if np.any(weights_opt <= 0) or np.any(np.diff(nodes_opt) <= 0):
        # Fallback to best valid from optimization or GL
        # Check if we can fix by clipping
        weights_opt = np.clip(weights_opt, 1e-10, None)
        weights_opt = weights_opt / np.sum(weights_opt)
        if np.any(np.diff(nodes_opt) <= 0):
             # Jitter nodes
             eps = 1e-5
             nodes_opt = np.linspace(eps, 1-eps, 8)
             
    return {"nodes": nodes_opt, "weights": weights_opt}

# EVOLVE-BLOCK-END
