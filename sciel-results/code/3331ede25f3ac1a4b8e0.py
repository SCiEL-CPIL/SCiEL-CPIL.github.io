"""Optimize 3-layer Ry/CZ circuit angles for transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, L_BFGS_B

# Global deadline guard
_DEADLINE = 145.0  # seconds, leave 5s safety margin
_START_TIME = None

# Cache for computed results
_result_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Pauli matrices
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    # Build Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Build operator for Z on qubit i and qubit i+1
        op = np.array([[1]], dtype=np.complex128)
        for q in range(n - 1, -1, -1):
            if q == i or q == i + 1:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H -= coupling * op
    
    # Build X_i terms
    for i in range(n):
        op = np.array([[1]], dtype=np.complex128)
        for q in range(n - 1, -1, -1):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H -= field * op
    
    return H


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of an n-qubit state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    
    # Reshape state to apply single-qubit gate
    # Qubit i is bit i (little-endian), so qubit 0 is the least significant bit
    # In the tensor representation, we reshape as (2,)*n where index order is (q_{n-1}, ..., q_0)
    # So qubit i corresponds to axis (n-1-i) in the tensor
    axis = n - 1 - qubit
    
    state_tensor = state.reshape([2] * n)
    
    # Move the target axis to the front
    state_tensor = np.moveaxis(state_tensor, axis, 0)
    new_shape = (2,) + state_tensor.shape[1:]
    state_tensor = state_tensor.reshape(new_shape)
    
    # Apply Ry
    result = np.zeros_like(state_tensor)
    result[0, :] = c * state_tensor[0, :] - s * state_tensor[1, :]
    result[1, :] = s * state_tensor[0, :] + c * state_tensor[1, :]
    
    # Move axis back
    result = result.reshape([2] * n)
    result = np.moveaxis(result, 0, axis)
    
    return result.reshape(-1)


def _apply_cz(state, q1, q2, n):
    """Apply CZ gate between qubits q1 and q2."""
    # CZ = diag(1, 1, 1, -1) in the basis |00>, |01>, |10>, |11>
    # For qubits q1, q2 (little-endian), the phase is -1 when both are 1
    
    mask = np.zeros(2 ** n, dtype=np.float64)
    for idx in range(2 ** n):
        bit1 = (idx >> q1) & 1
        bit2 = (idx >> q2) & 1
        if bit1 and bit2:
            mask[idx] = -1.0
        else:
            mask[idx] = 1.0
    
    return state * mask


def _apply_circuit(theta, n):
    """Apply the full 3-layer Ry/CZ circuit starting from |0>^n.
    theta has shape (layers, n).
    Between layers, apply CZ on adjacent pairs in ascending order.
    """
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    layers = theta.shape[0]
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to qubit i for all i
        for i in range(n):
            state = _apply_ry(state, theta[layer, i], i, n)
        
        # Apply CZ between adjacent pairs (i, i+1) in ascending order, except after last layer
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _energy(state, H):
    """Compute expectation value of H in state."""
    return np.real(np.vdot(state, H @ state))


def _objective(params, n, coupling, field, H, E0):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(3, n)
    state = _apply_circuit(theta, n)
    E = _energy(state, H)
    return (E - E0) / n


def _gradient(params, n, coupling, field, H, E0, eps=1e-7):
    """Numerical gradient via central differences."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus, n, coupling, field, H, E0) - 
                   _objective(params_minus, n, coupling, field, H, E0)) / (2 * eps)
    return grad


def angles(n, coupling, field, layers):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    key = (n, coupling, field, layers)
    if key in _result_cache:
        return _result_cache[key]
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 10:
        # Not enough time, return best guess
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _result_cache[key] = theta
        return theta
    
    # Build Hamiltonian and get exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    
    # Initial guess: try a few heuristics
    best_theta = None
    best_val = np.inf
    
    # Heuristic 1: uniform angles
    for init_scale in [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, np.pi/4, np.pi/3]:
        theta_init = np.zeros((layers, n))
        # First two layers: small rotation, last layer: larger
        theta_init[0] = init_scale * 0.3
        theta_init[1] = init_scale * 0.5
        theta_init[2] = init_scale
        params = theta_init.flatten()
        val = _objective(params, n, coupling, field, H, E0)
        if val < best_val:
            best_val = val
            best_theta = theta_init.copy()
    
    # Heuristic 2: symmetric angles
    for sym_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        theta_init = np.zeros((layers, n))
        theta_init[:, :] = sym_val
        params = theta_init.flatten()
        val = _objective(params, n, coupling, field, H, E0)
        if val < best_val:
            best_val = val
            best_theta = theta_init.copy()
    
    # Heuristic 3: based on field strength
    field_ratio = field / (coupling + field) if (coupling + field) > 0 else 0.5
    theta_init = np.zeros((layers, n))
    theta_init[0] = 0.2 * field_ratio
    theta_init[1] = 0.4 * field_ratio
    theta_init[2] = 0.8 * field_ratio
    params = theta_init.flatten()
    val = _objective(params, n, coupling, field, H, E0)
    if val < best_val:
        best_val = val
        best_theta = theta_init.copy()
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 20:
        _result_cache[key] = best_theta
        return best_theta
    
    # Local optimization using L-BFGS-B
    try:
        params_init = best_theta.flatten()
        res = minimize(
            _objective,
            params_init,
            args=(n, coupling, field, H, E0),
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        
        if res.fun < best_val:
            best_theta = res.x.reshape(3, n)
            best_val = res.fun
        
        # Check deadline
        if time.time() - _START_TIME > _DEADLINE - 15:
            _result_cache[key] = best_theta
            return best_theta
        
        # Second round: refine with tighter tolerance
        res2 = minimize(
            _objective,
            res.x,
            args=(n, coupling, field, H, E0),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        
        if res2.fun < best_val:
            best_theta = res2.x.reshape(3, n)
            best_val = res2.fun
            
    except Exception:
        pass
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE - 10:
        _result_cache[key] = best_theta
        return best_theta
    
    # Multi-start refinement: try a few more random-ish inits near the best
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() - _START_TIME > _DEADLINE - 15:
            break
        
        noise = rng.normal(0, 0.1, size=(3, n))
        theta_try = best_theta + noise
        params_try = theta_try.flatten()
        
        try:
            res_r = minimize(
                _objective,
                params_try,
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-13}
            )
            
            if res_r.fun < best_val:
                best_theta = res_r.x.reshape(3, n)
                best_val = res_r.fun
        except Exception:
            continue
    
    _result_cache[key] = best_theta
    return best_theta
# EVOLVE-BLOCK-END
