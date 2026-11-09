"""Optimized quantum circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal (each pair added once)
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:  # only add each pair once
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _apply_ry(state, n, qubit, theta):
    """Apply Ry(theta) to qubit in-place on state vector. Vectorized."""
    dim = 1 << n
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    mask = (1 << qubit)
    # Indices where qubit bit is 0
    idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
    idx1 = idx0 | mask
    new_state = state.copy()
    new_state[idx0] = c * state[idx0] - s * state[idx1]
    new_state[idx1] = s * state[idx0] + c * state[idx1]
    return new_state


def _apply_cz(state, n, qubit_a, qubit_b):
    """Apply CZ to qubits a and b in-place on state vector."""
    dim = 1 << n
    new_state = state.copy()
    for idx in range(dim):
        if ((idx >> qubit_a) & 1) and ((idx >> qubit_b) & 1):
            new_state[idx] = -state[idx]
    return new_state


def _circuit_state_vector(n, theta, layers):
    """
    Build the state vector from the circuit.
    theta: (layers, n) array of Ry angles
    Circuit: |0>^n, then for each layer l:
      - Apply Ry(theta[l, i]) to qubit i for all i
      - If not last layer, apply CZ on adjacent pairs (i, i+1) for i in ascending order
    """
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    for l in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            state = _apply_ry(state, n, i, theta[l, i])
        
        # Apply CZ on adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, n, i, i + 1)
    
    return state


def _circuit_energy(n, coupling, field, theta, layers, H=None):
    """Compute the energy of the circuit state."""
    if H is None:
        H = _build_hamiltonian(n, coupling, field)
    state = _circuit_state_vector(n, theta, layers)
    energy = np.vdot(state, H @ state).real
    return float(energy)


def _objective(params, n, coupling, field, layers, ground_energy, H):
    """Objective function for optimization."""
    theta = params.reshape((layers, n))
    energy = _circuit_energy(n, coupling, field, theta, layers, H)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate the ground state."""
    start_time = time.time()
    deadline = start_time + 140  # Leave safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    best_params = np.zeros(layers * n)
    best_score = float('inf')
    
    # Try multiple initializations with different strategies
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. All pi/4
    initializations.append(np.full(layers * n, np.pi / 4))
    
    # 3. Last layer pi/2
    init3 = np.zeros(layers * n)
    init3[-n:] = np.pi / 2
    initializations.append(init3)
    
    # 4. Physics-based interpolation
    init4 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init4[l * n + i] = np.pi / 2 * min(1.0, field)
    initializations.append(init4)
    
    # 5-8. Random initializations with different seeds
    for seed in [42, 123, 7, 99]:
        np.random.seed(seed)
        initializations.append(np.random.uniform(-np.pi/2, np.pi/2, layers * n))
    
    # 9. Symmetric initialization based on field strength
    init9 = np.zeros(layers * n)
    # For small field, ground state is close to |0>^n, so angles near 0
    # For large field, ground state is close to |+>^n, so angles near pi/2
    # Middle layers can have different angles
    base_angle = np.arctan(field / coupling) if coupling > 0 else np.pi/2
    for l in range(layers):
        scale = 1.0 - 0.3 * l  # Decrease angle in later layers
        for i in range(n):
            init9[l * n + i] = base_angle * scale
    initializations.append(init9)
    
    # 10. Another physics-inspired init
    init10 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            # Alternate pattern based on position
            pos_factor = 0.8 + 0.2 * np.sin(np.pi * i / max(1, n-1))
            init10[l * n + i] = (np.pi / 2) * min(1.0, field) * pos_factor * (1.0 - 0.2*l)
    initializations.append(init10)
    
    # 11. Uniform small angles
    initializations.append(np.full(layers * n, 0.1))
    
    # 12. Alternating signs
    init12 = np.zeros(layers * n)
    for l in range(layers):
        for i in range(n):
            init12[l * n + i] = 0.3 * (1 if i % 2 == 0 else -1)
    initializations.append(init12)
    
    # Run Nelder-Mead from each initialization
    for idx, init in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(n, coupling, field, layers, ground_energy, H),
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'maxfev': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Refine with L-BFGS-B from the best result
    if time.time() < deadline - 10:
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-10
                }
            )
            
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Try a few more local refinements from slightly perturbed best params
    if time.time() < deadline - 5:
        for pert in [0.01, 0.05, 0.1, 0.2]:
            if time.time() > deadline - 3:
                break
            try:
                np.random.seed(int(time.time() * 1000) % 10000)
                perturbed = best_params + np.random.uniform(-pert, pert, layers * n)
                result3 = minimize(
                    _objective,
                    perturbed,
                    args=(n, coupling, field, layers, ground_energy, H),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (layers * n),
                    options={
                        'maxiter': 300,
                        'maxfun': 800,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )
                
                if result3.fun < best_score:
                    best_score = result3.fun
                    best_params = result3.x.copy()
            except Exception:
                continue
    
    # Final polish with Nelder-Mead from the best result
    if time.time() < deadline - 2:
        try:
            result4 = minimize(
                _objective,
                best_params,
                args=(n, coupling, field, layers, ground_energy, H),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            if result4.fun < best_score:
                best_score = result4.fun
                best_params = result4.x.copy()
        except Exception:
            pass
    
    theta = best_params.reshape((layers, n))
    return theta
# EVOLVE-BLOCK-END
