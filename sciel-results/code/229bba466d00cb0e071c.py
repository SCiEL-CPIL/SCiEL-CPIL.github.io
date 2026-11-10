"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute exact ground energies and circuit energy function
_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim))
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            H[state, state] += -coupling * z_i * z_j
    # X terms: -field * sum X_i
    for i in range(n):
        for state in range(dim):
            bit_i = (state >> i) & 1
            flipped = state ^ (1 << i)
            H[flipped, state] += -field
    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    key = (n, coupling, field)
    if key not in _CACHE:
        H = _build_hamiltonian(n, coupling, field)
        eigvals = np.linalg.eigvalsh(H)
        _CACHE[key] = eigvals[0]
    return _CACHE[key]

def _ry_matrix(theta):
    """Return 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit in state vector."""
    dim = len(state)
    n = int(np.log2(dim))
    result = np.zeros_like(state)
    for state_idx in range(dim):
        bit = (state_idx >> qubit) & 1
        partner = state_idx ^ (1 << qubit)
        if bit == 0:
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            result[state_idx] = c * state[state_idx] + s * state[partner]
            result[partner] = -s * state[state_idx] + c * state[partner]
        else:
            # Already handled by partner when bit==0
            pass
    return result

def _apply_cz(state, qubit_i, qubit_j):
    """Apply CZ gate between qubit_i and qubit_j."""
    dim = len(state)
    result = state.copy()
    for state_idx in range(dim):
        bit_i = (state_idx >> qubit_i) & 1
        bit_j = (state_idx >> qubit_j) & 1
        if bit_i == 1 and bit_j == 1:
            result[state_idx] = -state[state_idx]
    return result

def _circuit_energy(params, n, coupling, field, layers):
    """Compute energy of circuit state for given parameters."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # Start |0>^n
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = params[layer, q]
            state = _apply_ry(state, theta, q)
        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)
    
    # Compute energy <psi|H|psi>
    H = _build_hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy

def _objective_flat(params_flat, n, coupling, field, layers):
    """Objective function for scipy minimize."""
    params = params_flat.reshape(layers, n)
    return _circuit_energy(params, n, coupling, field, layers)

def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate ground state."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    exact_e = _exact_ground_energy(n, coupling, field)
    
    # Initial guess: all zeros (|0> state)
    best_params = np.zeros((layers, n))
    best_energy = _circuit_energy(best_params, n, coupling, field, layers)
    
    # Try multiple initializations and optimizations
    inits = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.random.randn(layers, n) * 0.5,
        np.random.randn(layers, n) * 1.0,
    ]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_flat,
                init.flatten(),
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Try a second round with more iterations if time permits
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _objective_flat,
                best_params.flatten(),
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
