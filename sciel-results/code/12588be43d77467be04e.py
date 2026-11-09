"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    # Z_i Z_{i+1} terms (diagonal)
    for i in range(n - 1):
        for k in range(dim):
            z_i = 1.0 if ((k >> i) & 1) == 0 else -1.0
            z_j = 1.0 if ((k >> (i + 1)) & 1) == 0 else -1.0
            H[k, k] += -coupling * z_i * z_j
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for k in range(dim):
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field
            H[k_flip, k] += -field
    
    return H


def _apply_ry(theta, state, n):
    """Apply Ry(theta) to every qubit."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    for i in range(n):
        # Ry = [[c, -s], [s, c]] on qubit i
        new_state = np.zeros_like(state)
        for k in range(len(state)):
            if (k >> i) & 1 == 0:
                new_state[k] += c * state[k]
            else:
                new_state[k] -= s * state[k]
            # For the flipped bit
            k_flip = k ^ (1 << i)
            if (k >> i) & 1 == 0:
                new_state[k_flip] += s * state[k]
            else:
                new_state[k_flip] += c * state[k]
        state = new_state
    return state


def _apply_cz_adjacent(state, n):
    """Apply CZ on every adjacent pair in ascending order."""
    for i in range(n - 1):
        # CZ: |11> gets phase -1
        for k in range(len(state)):
            if ((k >> i) & 1) == 1 and ((k >> (i + 1)) & 1) == 1:
                state[k] *= -1.0
    return state


def _apply_circuit(params, n, layers):
    """Apply the full circuit to |0>^n and return final state."""
    state = np.zeros(1 << n)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        theta = params[layer]
        state = _apply_ry(theta, state, n)
        if layer < layers - 1:
            state = _apply_cz_adjacent(state, n)
    
    return state


def _circuit_energy(params, n, coupling, field, H):
    """Compute circuit energy."""
    state = _apply_circuit(params, n, 3)
    # Energy = state^T H state (H is real symmetric)
    return state @ H @ state


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate Ising ground state."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Build Hamiltonian and get exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E_ground = eigvals[0]
    
    dim = 1 << n
    
    def objective(params_flat):
        params = params_flat.reshape(layers, n)
        E = _circuit_energy(params, n, coupling, field, H)
        return (E - E_ground) / n
    
    best_params = np.zeros((layers, n))
    best_val = np.inf
    
    # Multiple random restarts
    n_restarts = 8
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start from seed-like initialization
            x0 = np.zeros(layers * n)
            x0[-n:] = np.pi / 2
        else:
            x0 = np.random.uniform(-np.pi, np.pi, size=layers * n)
        
        try:
            result = scipy.optimize.minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Additional local refinement if time permits
    if time.time() < deadline - 20:
        try:
            result = scipy.optimize.minimize(
                objective, best_params.flatten(), method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final check: ensure finite
    if not np.all(np.isfinite(best_params)):
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
    
    return best_params
# EVOLVE-BLOCK-END
