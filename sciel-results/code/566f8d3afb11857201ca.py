"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0


def _circuit_state(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j

    for layer in range(layers):
        # Apply Ry to all qubits in parallel
        # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
        # We need to apply this to every qubit.
        # For a state vector |psi>, the operation on qubit q is:
        # new_psi[|0...b...0>] += cos(t/2) * psi[|0...b...0>] - sin(t/2) * psi[|0...b...1>]
        # new_psi[|0...b...1>] += sin(t/2) * psi[|0...b...0>] + cos(t/2) * psi[|0...b...1>]
        
        # Vectorized approach:
        # Reshape state to (2,) * n dimensions? No, easier to iterate qubits but vectorize the index ops.
        # Or, use the fact that Ry on qubit q acts on the 2x2 blocks.
        
        # Let's do it efficiently by iterating qubits (n is small: 4 or 6)
        # but vectorizing the state update.
        
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            mask = 1 << q
            # Indices where qubit q is 0
            idx_0 = [i for i in range(dim) if (i & mask) == 0]
            idx_1 = [i + mask for i in idx_0]
            
            idx_0_arr = np.array(idx_0)
            idx_1_arr = np.array(idx_1)
            
            vals_0 = state[idx_0_arr]
            vals_1 = state[idx_1_arr]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            state[idx_0_arr] = new_vals_0
            state[idx_1_arr] = new_vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                mask1 = 1 << i
                mask2 = 1 << (i + 1)
                # Find indices where both bits are 1
                # (idx & mask1) and (idx & mask2)
                # We can vectorize this:
                idxs = np.arange(dim)
                both_set = ((idxs & mask1) != 0) & ((idxs & mask2) != 0)
                state[both_set] *= -1.0

    return state


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    
    # Try multiple random initializations
    # n=4: 12 params, n=6: 18 params
    # L-BFGS-B is efficient for smooth functions.
    
    num_restarts = 15 if n <= 4 else 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Use different seeds for reproducibility and diversity
        np.random.seed(42 + i)
        
        # Initialize angles in a reasonable range
        # Ry angles that are too large might be less optimal, but let's allow a wide range.
        # Since Ry(theta) has periodicity 2pi, and we want to approximate ground states,
        # angles near 0 or pi/2 are often good starting points.
        x_init = np.random.uniform(-1.5, 1.5, size=(layers * n))
        
        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerances if time remains
    if time.time() < deadline - 5.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0), best_x, 
                          method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
