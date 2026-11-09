"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the transverse-field Ising model."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for k in range(dim):
            # z_i is 1 if bit i is 0, -1 if bit i is 1
            z_i = 1 if (k >> i) & 1 == 0 else -1
            z_j = 1 if (k >> (i + 1)) & 1 == 0 else -1
            H[k, k] += -coupling * z_i * z_j
            
    # X_i terms
    for i in range(n):
        for k in range(dim):
            # X_i flips bit i
            k_flip = k ^ (1 << i)
            H[k, k_flip] += -field / 2.0
            H[k_flip, k] += -field / 2.0
            
    return H

def _apply_ry(state, qubit, theta):
    """Apply Ry(theta) to the specified qubit in the state vector."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    mask = 1 << qubit
    dim = len(state)
    for k in range(dim):
        if (k & mask) == 0:
            k1 = k | mask
            c0 = state[k]
            c1 = state[k1]
            state[k] = c * c0 + s * c1
            state[k1] = -s * c0 + c * c1

def _apply_cz(state, i, j):
    """Apply CZ gate between qubits i and j."""
    mask_i = 1 << i
    mask_j = 1 << j
    mask = mask_i | mask_j
    dim = len(state)
    for k in range(dim):
        if (k & mask) == mask:
            state[k] *= -1.0

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s/150s limit
    
    dim = 1 << n
    H = _build_hamiltonian(n, coupling, field)
    
    # Initial state |0>^n
    init_state = np.zeros(dim, dtype=complex)
    init_state[0] = 1.0
    
    def objective(params):
        # params is a flat array of length layers * n
        # Reshape to (layers, n)
        thetas = params.reshape((layers, n))
        
        # Copy initial state
        state = init_state.copy()
        
        for l in range(layers):
            # Apply Ry to all qubits
            for i in range(n):
                _apply_ry(state, i, thetas[l, i])
            
            # Apply CZ if not the last layer
            if l < layers - 1:
                for i in range(n - 1):
                    _apply_cz(state, i, i + 1)
        
        # Compute energy <H>
        energy = np.real(np.vdot(state, H.dot(state)))
        return energy

    # Multi-restart optimization
    best_energy = np.inf
    best_params = np.zeros(layers * n)
    
    # Define a few initial guesses
    # 1. All zeros
    # 2. All pi/2 (maximally mixed-ish, often good for X terms)
    # 3. Random
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    initial_guesses = [np.zeros(layers * n)]
    initial_guesses.append(np.full(layers * n, np.pi / 2))
    for _ in range(num_restarts - 2):
        initial_guesses.append(rng.uniform(-np.pi, np.pi, size=layers * n))
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
