"""Rewritten program with analytic-gradient optimization for QAOA angle design."""
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


def _apply_x_layer_fast(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
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


def _qaoa_state_and_grads(n, cut_sizes, angles):
    """
    Simulate QAOA circuit and return final state and gradients of state w.r.t parameters.
    angles: (p, 2) array of [gamma, beta]
    Returns: state, grads where grads is a list of p*2 complex arrays.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Gradients for each parameter
    # We track d_state/d_gamma_i and d_state/d_beta_i
    # Initially all zero
    num_params = angles.shape[0] * 2
    grads = [np.zeros(dim, dtype=np.complex128) for _ in range(num_params)]
    
    for i in range(angles.shape[0]):
        gamma = angles[i, 0]
        beta = angles[i, 1]
        
        # Cost layer: U_C = exp(-i gamma C)
        # state <- U_C state
        # d_state/d_gamma <- -i C U_C state + U_C d_state/d_gamma
        # d_state/d_beta <- U_C d_state/d_beta
        
        # Apply cost layer to state
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Update gradients
        # Index for gamma_i is 2*i
        idx_g = 2 * i
        # Index for beta_i is 2*i + 1
        idx_b = 2 * i + 1
        
        # d_state/d_gamma gets contribution from derivative of U_C
        # d/dgamma exp(-i gamma C) = -i C exp(-i gamma C)
        # So new_grad_g = -i C * state (from the direct term) + U_C * old_grad_g
        # But old_grad_g is propagated through U_C
        # Actually, the chain rule:
        # |psi_new> = U_C |psi_old>
        # d|psi_new>/dtheta = U_C d|psi_old>/dtheta + dU_C/dtheta |psi_old>
        
        # For gamma:
        # dU_C/dgamma = -i C U_C
        # So d|psi_new>/dgamma = U_C d|psi_old>/dgamma - i C U_C |psi_old>
        # Note: U_C |psi_old> is the new state.
        grads[idx_g] = grads[idx_g] * np.exp(-1j * gamma * cut_sizes) - 1j * cut_sizes * state
        
        # For beta (from previous layer):
        # dU_C/dbeta = 0
        # So d|psi_new>/dbeta = U_C d|psi_old>/dbeta
        grads[idx_b] = grads[idx_b] * np.exp(-1j * gamma * cut_sizes)
        
        # X layer: U_X = exp(-i beta X)
        # state <- U_X state
        # d_state/d_beta <- U_X d_state/d_beta - i X U_X state
        # d_state/d_gamma <- U_X d_state/d_gamma
        
        # Apply X layer to state
        state = _apply_x_layer_fast(state, beta, n)
        
        # We need to apply U_X to the gradients as well
        # But _apply_x_layer_fast only works on one state.
        # We can apply it to each gradient vector.
        # This is expensive if done naively, but n is small.
        
        # Update grads for beta
        # dU_X/dbeta = -i X U_X
        # d|psi_new>/dbeta = U_X d|psi_old>/dbeta - i X U_X |psi_old>
        # Note: U_X |psi_old> is the new state.
        
        # We need to propagate all gradients through U_X
        # Let's do this for all gradients
        for k in range(num_params):
            grads[k] = _apply_x_layer_fast(grads[k], beta, n)
            
        # Add the direct term for beta
        # -i X state
        # We need to compute X |state>.
        # X is sum of X_i.
        # We can compute X |state> by applying X_i for each i and summing.
        # Or we can use the fact that X_i flips the i-th bit.
        
        x_state = np.zeros(dim, dtype=np.complex128)
        tensor_state = state.reshape([2] * n)
        for i in range(n):
            # Flip i-th bit
            # This is equivalent to moving axis i to 0, swapping, and moving back
            moved = np.moveaxis(tensor_state, i, 0)
            flat = moved.reshape(2, -1)
            # Swap 0 and 1
            swapped = np.empty_like(flat)
            swapped[0] = flat[1]
            swapped[1] = flat[0]
            # Move back
            new_tensor = np.moveaxis(swapped.reshape([2] * n), 0, i)
            x_state += new_tensor.flatten()
            
        grads[idx_b] -= 1j * x_state
        
    return state, grads


def _objective_and_grad(params, n, cut_sizes, p):
    """
    Compute objective (negative expected cut) and its gradient.
    params: flat array of length 2*p
    """
    angles = params.reshape(p, 2)
    state, grads = _qaoa_state_and_grads(n, cut_sizes, angles)
    
    # Expected cut = <state| C |state>
    # C is diagonal with entries cut_sizes
    # <state| C |state> = sum(state* * cut_sizes * state) = sum(|state|^2 * cut_sizes)
    # This is real.
    exp_cut = float(np.sum(np.abs(state) ** 2 * cut_sizes))
    
    # Gradient of exp_cut w.r.t params
    # d/dtheta <psi| C |psi> = 2 Re <dpsi/dtheta | C |psi>
    # Since C is diagonal, <dpsi| C |psi> = sum(dpsi* * cut_sizes * psi)
    
    grad_params = np.zeros_like(params)
    for k in range(2 * p):
        # dpsi/dtheta_k is grads[k]
        # We need 2 Re <grads[k]| C |state>
        # <grads[k]| C |state> = np.vdot(grads[k], cut_sizes * state)
        # np.vdot conjugates the first argument
        val = np.vdot(grads[k], cut_sizes * state)
        grad_params[k] = 2 * np.real(val)
        
    # We want to maximize exp_cut, so minimize -exp_cut
    # Gradient of -exp_cut is -grad_params
    return -exp_cut, -grad_params


def _optimize_with_gradient(x0, n, cut_sizes, p, maxiter=100):
    """
    Optimize using L-BFGS-B with analytic gradients.
    """
    try:
        res = minimize(
            lambda x: _objective_and_grad(x, n, cut_sizes, p)[0],
            x0,
            jac=lambda x: _objective_and_grad(x, n, cut_sizes, p)[1],
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-12,
                'gtol': 1e-8
            }
        )
        return res.x, -res.fun
    except Exception:
        # Fallback to finite difference if gradient fails
        try:
            res = minimize(
                lambda x: -_expected_cut(n, cut_sizes, x.reshape(p, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            return res.x, -res.fun
        except Exception:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer_fast(state, beta, n)
        
    return state


def _graph_theoretic_init(n, edges, p):
    """
    Generate initial guesses based on graph properties.
    """
    num_edges = len(edges)
    max_possible_edges = n * (n - 1) // 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.5
    
    # Heuristic: gamma proportional to density, beta inversely
    # Standard QAOA p=2 angles are often around gamma=pi/4, beta=pi/8
    # Let's create a few variations
    
    inits = []
    
    # 1. Density-based
    g1 = np.pi * density / 2
    b1 = np.pi * (1 - density) / 4
    inits.append(np.array([g1, b1, g1, b1]))
    
    # 2. Standard
    inits.append(np.array([0.4, 0.2, 0.4, 0.2]))
    
    # 3. Slightly different
    inits.append(np.array([0.5, 0.3, 0.5, 0.3]))
    
    # 4. Small angles
    inits.append(np.array([0.2, 0.1, 0.2, 0.1]))
    
    # 5. Large angles
    inits.append(np.array([0.7, 0.4, 0.7, 0.4]))
    
    # 6. Randomized around standard
    np.random.seed(42)
    for _ in range(5):
        g = np.random.uniform(0.2, 0.8, p)
        b = np.random.uniform(0.1, 0.5, p)
        inits.append(np.column_stack((g, b)).flatten())
        
    return inits


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Generate initial guesses
    initial_guesses = _graph_theoretic_init(n, edges, p)
    
    best_angles = None
    best_score = -np.inf
    
    for x0 in initial_guesses:
        if _time_left() < 10.0:
            break
            
        x_opt, score = _optimize_with_gradient(x0, n, cut_sizes, p, maxiter=100)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
            
    # Final polish with tighter tolerance
    if best_angles is not None and _time_left() > 15.0:
        x_polished, score_polished = _optimize_with_gradient(
            best_angles.flatten(), n, cut_sizes, p, maxiter=200
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
