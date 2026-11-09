"""Optimize shallow-circuit Ising ground states via fidelity maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Precompute Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I = np.eye(2, dtype=np.complex128)

# Cache for exact ground states
_ground_state_cache = {}


def _kron_all(ops):
    """Tensor product of a list of operators (left to right)."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _get_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    # ZZ terms
    for i in range(n - 1):
        terms = [I] * n
        terms[i] = Z
        terms[i + 1] = Z
        H -= coupling * _kron_all(terms)
    
    # X terms
    for i in range(n):
        terms = [I] * n
        terms[i] = X
        H -= field * _kron_all(terms)
    
    return H


def _get_ground_state(n, coupling, field):
    """Get exact ground state, cached."""
    key = (n, coupling, field)
    if key not in _ground_state_cache:
        H = _get_hamiltonian(n, coupling, field)
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_state = eigvecs[:, 0]
        # Fix global phase for consistency
        idx = np.argmax(np.abs(ground_state))
        phase = ground_state[idx] / np.abs(ground_state[idx])
        ground_state = ground_state / phase
        _ground_state_cache[key] = ground_state
    return _ground_state_cache[key]


def _ry_matrix(theta):
    """Single-qubit Ry rotation."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cz_matrix():
    """Two-qubit CZ gate."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)


def _simulate_circuit(params, n, layers):
    """
    Simulate the circuit and return the final state vector.
    params: (layers, n) array of angles.
    Start from |0>^n.
    Each layer: apply Ry(theta_i) to each qubit i.
    Between layers (not after last): apply CZ on adjacent pairs in ascending order.
    """
    dim = 2 ** n
    # Start from |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            theta = params[layer, i]
            ry = _ry_matrix(theta)
            # Apply Ry on qubit i
            ops = [I] * n
            ops[i] = ry
            gate = _kron_all(ops)
            state = gate @ state
        
        # Apply CZ between layers (not after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # Apply CZ on qubits i and i+1
                ops = [I] * n
                # Build CZ on qubits i and i+1
                cz = _cz_matrix()
                # Need to embed CZ into n-qubit space
                # Qubits are ordered 0, 1, ..., n-1
                # We need to apply CZ on positions i and i+1
                # This is tricky with kron; let's build it properly
                # CZ acts on qubits i and i+1, identity on others
                # We'll build it by iterating over basis states
                gate = np.zeros((dim, dim), dtype=np.complex128)
                for col in range(dim):
                    for row in range(dim):
                        # Check if bits i and i+1 are both 1
                        bit_i_row = (row >> (n - 1 - i)) & 1
                        bit_ip1_row = (row >> (n - 2 - i)) & 1
                        bit_i_col = (col >> (n - 1 - i)) & 1
                        bit_ip1_col = (col >> (n - 2 - i)) & 1
                        if row == col:
                            if bit_i_row and bit_ip1_row:
                                gate[row, col] = -1.0
                            else:
                                gate[row, col] = 1.0
                        # CZ is diagonal, so off-diagonal is 0
                state = gate @ state
    
    return state


def _fidelity_objective(params_flat, n, coupling, field, layers, ground_state):
    """Compute 1 - |<ground|circuit>|^2 to minimize."""
    params = params_flat.reshape(layers, n)
    state = _simulate_circuit(params, n, layers)
    overlap = np.vdot(ground_state, state)
    fidelity = np.abs(overlap) ** 2
    return 1.0 - fidelity


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    ground_state = _get_ground_state(n, coupling, field)
    
    best_params = None
    best_score = np.inf
    
    # Multi-start optimization
    n_params = layers * n
    
    # Generate multiple initial guesses
    rng = np.random.default_rng(42)
    
    # Initial guesses:
    # 1. Zeros
    # 2. Pi/2 on last layer
    # 3. Random small
    # 4. Random larger
    # 5. Uniform pi/4
    # 6. Based on field strength heuristics
    
    initial_guesses = []
    
    # Heuristic: for transverse field, angles related to field
    # Stronger field -> more rotation
    base_angle = np.arctan(field / coupling) if coupling != 0 else np.pi / 2
    base_angle = np.clip(base_angle, 0, np.pi)
    
    # Guess 1: zeros
    initial_guesses.append(np.zeros(n_params))
    
    # Guess 2: pi/2 on last layer
    g2 = np.zeros(n_params)
    g2[-n:] = np.pi / 2
    initial_guesses.append(g2)
    
    # Guess 3: uniform small
    initial_guesses.append(np.full(n_params, base_angle / 3))
    
    # Guess 4: uniform medium
    initial_guesses.append(np.full(n_params, base_angle))
    
    # Guess 5: random
    initial_guesses.append(rng.uniform(-np.pi, np.pi, n_params))
    
    # Guess 6: random small
    initial_guesses.append(rng.uniform(-0.5, 0.5, n_params))
    
    # Guess 7: alternating pattern
    g7 = np.zeros(n_params)
    for layer in range(layers):
        for i in range(n):
            g7[layer * n + i] = base_angle * (1 if i % 2 == 0 else 0.5)
    initial_guesses.append(g7)
    
    # Guess 8: based on exact ground state components (if feasible)
    # For small n, we can extract angles from ground state
    if n <= 6:
        # Try to match product state approximation
        gs = ground_state
        # For a product state, each qubit's reduced state gives angle
        # But ground state may be entangled, so this is approximate
        for i in range(n):
            # Extract single-qubit state for qubit i
            # This is complex; skip for now
            pass
    
    best_result = None
    best_obj = np.inf
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _fidelity_objective,
                guess,
                args=(n, coupling, field, layers, ground_state),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_obj:
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x.copy()
        except Exception:
            continue
    
    if best_result is not None:
        best_params = best_result.reshape(layers, n)
    else:
        # Fallback
        best_params = np.zeros((layers, n))
        best_params[-1] = np.pi / 2
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _fidelity_objective,
                best_params.ravel(),
                args=(n, coupling, field, layers, ground_state),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
