"""Rewritten program with analytic-gradient L-BFGS optimization for QAOA angles."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _qaoa_forward_with_grads(n, cut_sizes, angles):
    """
    Forward pass computing state and exact analytic gradients of expected cut
    with respect to all gamma and beta parameters via backpropagation.
    
    Returns: (expected_cut, grad_array) where grad_array has shape (p, 2)
    with columns (d/dgamma, d/dbeta).
    """
    dim = 1 << n
    p = len(angles)
    
    # Initialize state: |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Store states after each layer for backprop
    states = [state.copy()]
    exp_factors = []
    
    # Forward pass
    for k in range(p):
        gamma_k = angles[k, 0]
        beta_k = angles[k, 1]
        
        exp_factor = np.exp(-1j * gamma_k * cut_sizes)
        exp_factors.append(exp_factor)
        
        # Cost layer
        state_after_cost = states[k] * exp_factor
        
        # X layer: apply exp(-i * beta_k * sum_j X_j)
        c = np.cos(beta_k)
        s = np.sin(beta_k)
        tensor = state_after_cost.reshape([2] * n)
        for i in range(n):
            moved = np.moveaxis(tensor, i, 0)
            flat = moved.reshape(2, -1)
            new_flat = np.empty_like(flat)
            new_flat[0] = c * flat[0] - 1j * s * flat[1]
            new_flat[1] = -1j * s * flat[0] + c * flat[1]
            tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
        state_after_x = tensor.flatten()
        
        states.append(state_after_x)
    
    # Final expected cut
    final_state = states[-1]
    probs = np.abs(final_state) ** 2
    expected_cut = float(np.sum(probs * cut_sizes))
    
    # Backward pass for gradients
    # grad_state[j] = dE/dpsi_j (the Wirtinger derivative w.r.t. psi_j)
    # For E = sum_j |psi_j|^2 * c_j:
    #   dE/dpsi_j* = psi_j * c_j
    #   dE/dpsi_j = conj(psi_j) * c_j
    grad_state = np.conj(final_state) * cut_sizes
    
    grads = np.zeros((p, 2), dtype=np.float64)
    
    for k in range(p - 1, -1, -1):
        gamma_k = angles[k, 0]
        beta_k = angles[k, 1]
        
        # Backprop through X layer
        # X matrix for each qubit: [[c, -is], [-is, c]]
        # X^dag = [[c, is], [is, c]]
        c = np.cos(beta_k)
        s = np.sin(beta_k)
        
        tensor_grad = grad_state.reshape([2] * n)
        for i in range(n):
            moved = np.moveaxis(tensor_grad, i, 0)
            flat = moved.reshape(2, -1)
            new_flat = np.empty_like(flat)
            new_flat[0] = c * flat[0] + 1j * s * flat[1]
            new_flat[1] = 1j * s * flat[0] + c * flat[1]
            tensor_grad = np.moveaxis(new_flat.reshape([2] * n), 0, i)
        grad_after_cost = tensor_grad.flatten()
        
        # Gradient w.r.t. beta_k:
        # X layer: state_after_x = X(beta) * state_after_cost
        # dX/dbeta = [[-s, -ic], [-ic, -s]]
        # dE/dbeta = sum_j grad_after_cost_j * d(state_after_x_j)/dbeta
        # d(state_after_x)/dbeta = dX/dbeta * state_after_cost
        # For each qubit: dX/dbeta * v = [[-s, -ic], [-ic, -s]] * [v0, v1]
        # = [-s*v0 - ic*v1, -ic*v0 - s*v1]
        state_after_cost = states[k] * exp_factors[k]
        
        tensor_v = state_after_cost.reshape([2] * n)
        tensor_g = grad_after_cost.reshape([2] * n)
        beta_grad_sum = 0.0
        for i in range(n):
            moved_v = np.moveaxis(tensor_v, i, 0)
            flat_v = moved_v.reshape(2, -1)
            moved_g = np.moveaxis(tensor_g, i, 0)
            flat_g = moved_g.reshape(2, -1)
            # dX/dbeta * v = [-s*v0 - ic*v1, -ic*v0 - s*v1]
            dv0 = -s * flat_v[0] - 1j * c * flat_v[1]
            dv1 = -1j * c * flat_v[0] - s * flat_v[1]
            beta_grad_sum += np.sum(flat_g[0] * dv0 + flat_g[1] * dv1)
        
        grads[k, 1] = float(np.real(beta_grad_sum))
        
        # Backprop through cost layer
        # state_after_cost = state_before * exp(-1j * gamma_k * cut_sizes)
        # grad_before = conj(exp(-1j * gamma_k * cut_sizes)) * grad_after_cost
        #             = exp(1j * gamma_k * cut_sizes) * grad_after_cost
        grad_before = np.exp(1j * gamma_k * cut_sizes) * grad_after_cost
        
        # Gradient w.r.t. gamma_k:
        # d(state_after_cost)/dgamma = -1j * cut_sizes * state_after_cost
        # dE/dgamma = sum_j grad_after_cost_j * (-1j * cut_sizes_j * state_after_cost_j)
        gamma_grad_sum = np.sum(grad_after_cost * (-1j * cut_sizes * state_after_cost))
        grads[k, 0] = float(np.real(gamma_grad_sum))
        
        grad_state = grad_before
    
    return expected_cut, grads


def _finite_diff_grad(n, cut_sizes, angles, current_value, eps=1e-7):
    """Compute gradient using central finite differences (fallback)."""
    p = len(angles)
    grad = np.zeros((p, 2), dtype=np.float64)
    
    for k in range(p):
        for j in range(2):
            angles_plus = angles.copy()
            angles_minus = angles.copy()
            angles_plus[k, j] += eps
            angles_minus[k, j] -= eps
            
            val_plus = _expected_cut(n, cut_sizes, angles_plus)
            val_minus = _expected_cut(n, cut_sizes, angles_minus)
            
            grad[k, j] = (val_plus - val_minus) / (2 * eps)
    
    return grad


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        c = np.cos(beta)
        s = np.sin(beta)
        tensor = state.reshape([2] * n)
        for i in range(n):
            moved = np.moveaxis(tensor, i, 0)
            flat = moved.reshape(2, -1)
            new_flat = np.empty_like(flat)
            new_flat[0] = c * flat[0] - 1j * s * flat[1]
            new_flat[1] = -1j * s * flat[0] + c * flat[1]
            tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
        state = tensor.flatten()
    
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective_and_grad(params, n, cut_sizes, p):
    """Objective function and gradient for L-BFGS."""
    angles = params.reshape(p, 2)
    value, grad = _qaoa_forward_with_grads(n, cut_sizes, angles)
    return -value, -grad.flatten()


def _lbfgs_optimize(x0, n, cut_sizes, p, maxiter=200, tol=1e-12):
    """Run L-BFGS-B optimization with analytic gradient."""
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': tol,
                'gtol': tol,
                'maxls': 50,
                'maxfun': maxiter * 10
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _graph_theoretic_init(n, edges, p):
    """Initialize angles based on graph properties."""
    m = len(edges)
    if m == 0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Average degree
    avg_degree = 2 * m / n
    
    # Heuristic initialization based on graph density
    gamma1 = min(0.5, 0.15 + 0.1 * avg_degree)
    beta1 = min(0.4, 0.1 + 0.05 * avg_degree)
    gamma2 = min(0.5, 0.1 + 0.08 * avg_degree)
    beta2 = min(0.35, 0.08 + 0.04 * avg_degree)
    
    return np.array([[gamma1, beta1], [gamma2, beta2]])


def _multi_restart_lbfgs(n, cut_sizes, p, num_restarts=10):
    """Multi-restart L-BFGS with diverse initial guesses and analytic gradients."""
    best_angles = None
    best_score = -np.inf
    
    np.random.seed(123)
    initial_guesses = []
    
    # Known good QAOA p=2 starting points for small graphs
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    initial_guesses.append(np.array([0.3, 0.3, 0.3, 0.3]))
    initial_guesses.append(np.array([0.45, 0.2, 0.45, 0.2]))
    initial_guesses.append(np.array([0.55, 0.25, 0.55, 0.25]))
    initial_guesses.append(np.array([0.25, 0.35, 0.25, 0.35]))
    
    # Random guesses
    for _ in range(num_restarts - len(initial_guesses)):
        gammas = np.random.uniform(0.1, np.pi * 0.8, p)
        betas = np.random.uniform(0.1, np.pi * 0.7, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    for x0 in initial_guesses:
        if _time_left() < 15.0:
            break
        
        x_opt, score = _lbfgs_optimize(x0, n, cut_sizes, p, maxiter=150, tol=1e-12)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # High-precision polish with tighter tolerance
    if best_angles is not None and _time_left() > 25.0:
        x_polished, score_polished = _lbfgs_optimize(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-15
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Graph-theoretic initialization
    init_angles = _graph_theoretic_init(n, edges, p)
    
    # Multi-restart L-BFGS optimization with analytic gradients
    best_angles, best_score = _multi_restart_lbfgs(n, cut_sizes, p, num_restarts=12)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
