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


def _apply_x_layer(state, beta, n):
    """Apply exp(-i*beta*sum_j X_j) to state vector."""
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
    return tensor.flatten()


def _apply_x_layer_dag(grad, beta, n):
    """Apply (exp(-i*beta*sum_j X_j))^dag to gradient vector."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = grad.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] + 1j * s * flat[1]
        new_flat[1] = 1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_forward_with_grads(n, cut_sizes, angles):
    """
    Forward pass computing state and analytic gradients of expected cut
    with respect to all gamma and beta parameters.
    
    Returns: (expected_cut, grad_array) where grad_array has shape (p, 2)
    with columns (d/dgamma, d/dbeta).
    """
    dim = 1 << n
    p = len(angles)
    
    # Initialize state: |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Store states after each cost layer for gradient computation
    # cost_states[k] = state after applying cost layer k (before X layer k)
    cost_states = []
    
    for k in range(p):
        gamma_k = angles[k, 0]
        beta_k = angles[k, 1]
        
        # Cost layer
        exp_factor = np.exp(-1j * gamma_k * cut_sizes)
        state_after_cost = state * exp_factor
        cost_states.append(state_after_cost)
        
        # X layer
        state = _apply_x_layer(state_after_cost, beta_k, n)
    
    # Final expected cut
    probs = np.abs(state) ** 2
    expected_cut = float(np.sum(probs * cut_sizes))
    
    # Backward pass for gradients
    # dE/dpsi_final = conj(psi_final) * cut_sizes  (this is dE/dpsi, not dE/dpsi*)
    grad_state = np.conj(state) * cut_sizes
    
    grads = np.zeros((p, 2), dtype=np.float64)
    
    for k in range(p - 1, -1, -1):
        gamma_k = angles[k, 0]
        beta_k = angles[k, 1]
        
        # Backprop through X layer: grad_after_cost = X^dag(grad_after_x)
        grad_after_cost = _apply_x_layer_dag(grad_state, beta_k, n)
        
        # Gradient with respect to beta_k:
        # dE/dbeta_k = 2 * Re[ sum_j conj(grad_after_cost_j) * dpsi_after_cost_j/dbeta_k ]
        # dpsi_after_cost_j/dbeta_k = d/dbeta [state_before_j * exp(-i*gamma*c_j)]
        #                           = 0 (beta doesn't appear in cost layer)
        # Wait, that's not right. The X layer is: psi_after_x = U_x(beta) * psi_after_cost
        # So psi_after_cost = U_x(beta)^dag * psi_after_x
        # dpsi_after_cost/dbeta = d(U_x^dag)/dbeta * psi_after_x
        # 
        # U_x(beta) for each qubit: [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
        # dU_x/dbeta = [[-sin(b), -i cos(b)], [-i cos(b), -sin(b)]]
        # U_x^dag = [[cos(b), i sin(b)], [i sin(b), cos(b)]]
        # d(U_x^dag)/dbeta = [[-sin(b), i cos(b)], [i cos(b), -sin(b)]]
        #
        # So dpsi_after_cost = d(U_x^dag)/dbeta * psi_after_x
        #
        # For each qubit i, the local operation is:
        # new[0] = -sin(b)*flat[0] + i*cos(b)*flat[1]
        # new[1] = i*cos(b)*flat[0] - sin(b)*flat[1]
        # where flat = psi_after_x reshaped for qubit i
        
        c = np.cos(beta_k)
        s = np.sin(beta_k)
        tensor_dx = grad_state.reshape([2] * n)
        for i in range(n):
            moved = np.moveaxis(tensor_dx, i, 0)
            flat = moved.reshape(2, -1)
            new_flat = np.empty_like(flat)
            new_flat[0] = -s * flat[0] + 1j * c * flat[1]
            new_flat[1] = 1j * c * flat[0] - s * flat[1]
            tensor_dx = np.moveaxis(new_flat.reshape([2] * n), 0, i)
        dpsi_dbeta = tensor_dx.flatten()
        
        grads[k, 1] = 2.0 * float(np.real(np.sum(np.conj(grad_after_cost) * dpsi_dbeta)))
        
        # Backprop through cost layer: grad_before = exp(1j*gamma*c) * grad_after_cost
        exp_factor_conj = np.exp(1j * gamma_k * cut_sizes)
        grad_before = exp_factor_conj * grad_after_cost
        
        # Gradient with respect to gamma_k:
        # psi_after_cost = psi_before * exp(-i*gamma*c)
        # dpsi_after_cost/dgamma = psi_before * exp(-i*gamma*c) * (-i*c) = psi_after_cost * (-i*c)
        # dE/dgamma = 2 * Re[ sum_j conj(grad_after_cost_j) * dpsi_after_cost_j/dgamma ]
        #           = 2 * Re[ sum_j conj(grad_after_cost_j) * (-i * c_j * psi_after_cost_j) ]
        
        dpsi_dgamma = -1j * cut_sizes * cost_states[k]
        grads[k, 0] = 2.0 * float(np.real(np.sum(np.conj(grad_after_cost) * dpsi_dgamma)))
        
        grad_state = grad_before
    
    return expected_cut, grads


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective_and_grad(params, n, cut_sizes, p):
    """Objective function and gradient for L-BFGS."""
    angles = params.reshape(p, 2)
    value, grad = _qaoa_forward_with_grads(n, cut_sizes, angles)
    return -value, -grad.flatten()


def _lbfgs_optimize(x0, n, cut_sizes, p, maxiter=200, tol=1e-12):
    """Run L-BFGS-B optimization with gradient."""
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
                'maxls': 50
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
    """Multi-restart L-BFGS with diverse initial guesses."""
    best_angles = None
    best_score = -np.inf
    
    np.random.seed(123)
    initial_guesses = []
    
    # Graph-theoretic initialization
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
        
        x_opt, score = _lbfgs_optimize(x0, n, cut_sizes, p, maxiter=100, tol=1e-10)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Final high-precision polish
    if best_angles is not None and _time_left() > 30.0:
        x_polished, score_polished = _lbfgs_optimize(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-14
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
    
    # Multi-restart L-BFGS optimization
    best_angles, best_score = _multi_restart_lbfgs(n, cut_sizes, p, num_restarts=12)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
