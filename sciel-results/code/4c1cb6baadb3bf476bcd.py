"""Optimize shallow Ry/CZ circuit angles to approximate Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute exact ground states and energies for the known test cases
# to avoid recomputation during optimization. We'll compute them on the fly
# but cache them.

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Z_i Z_{i+1} terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                partner = idx + (1 << i)
                H[idx, partner] += -field
                H[partner, idx] += -field
            # else: bit_i == 1, handled when partner has bit 0
    
    return H


def _apply_ry_vectorized(state, theta, qubit, n):
    """Apply Ry(theta) to the given qubit using vectorized operations."""
    dim = 2 ** n
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Create index arrays for all basis states
    indices = np.arange(dim)
    bit = (indices >> qubit) & 1
    
    # Separate indices where bit is 0 and 1
    mask0 = bit == 0
    mask1 = bit == 1
    
    idx0 = indices[mask0]
    idx1 = indices[mask1]
    partner_idx = idx0 + (1 << qubit)  # partner for each idx0
    
    new_state = state.copy()
    
    # Apply Ry to pairs (idx0, partner)
    v0 = state[idx0]
    v1 = state[partner_idx]
    
    new_state[idx0] = c * v0 - s * v1
    new_state[partner_idx] = s * v0 + c * v1
    
    return new_state


def _apply_cz_vectorized(state, qubit_a, qubit_b, n):
    """Apply CZ between qubit_a and qubit_b using vectorized operations."""
    dim = 2 ** n
    indices = np.arange(dim)
    bit_a = (indices >> qubit_a) & 1
    bit_b = (indices >> qubit_b) & 1
    
    mask = (bit_a == 1) & (bit_b == 1)
    state = state.copy()
    state[mask] *= -1.0
    
    return state


def _circuit_state_fast(theta, n, layers):
    """Apply the circuit to |0>^n and return the final state (vectorized)."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry(theta[layer, i]) to each qubit i
        for i in range(n):
            state = _apply_ry_vectorized(state, theta[layer, i], i, n)
        
        # Apply CZ on every adjacent pair in ascending order (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_vectorized(state, i, i + 1, n)
    
    return state


def _energy(state, H):
    """Compute the expectation value of H in the given state."""
    return np.real(np.vdot(state, H @ state))


# Cache for exact ground states and energies
_exact_cache = {}

def _get_exact_ground(n, coupling, field):
    """Get the exact ground state and energy, caching results."""
    key = (n, coupling, field)
    if key not in _exact_cache:
        H = _build_hamiltonian(n, coupling, field)
        eigvals, eigvecs = np.linalg.eigh(H)
        ground_energy = eigvals[0]
        ground_state = eigvecs[:, 0]
        _exact_cache[key] = (ground_state, ground_energy, H)
    return _exact_cache[key]


def _objective(params, n, coupling, field, layers):
    """Objective function: mean energy gap per spin."""
    theta = params.reshape(layers, n)
    state = _circuit_state_fast(theta, n, layers)
    ground_state, ground_energy, H = _get_exact_ground(n, coupling, field)
    e_circuit = _energy(state, H)
    return (e_circuit - ground_energy) / n


def _estimate_initial_from_ground(n, coupling, field, layers, ground_state):
    """
    Estimate initial angles by trying to match the ground state structure.
    For a product-like state, we can estimate single-qubit rotations.
    """
    dim = 2 ** n
    # For simplicity, use a heuristic based on the field strength
    # When field is large, ground state is close to |+>^n
    # When coupling dominates, ground state has more structure
    
    theta = np.zeros((layers, n))
    
    # Heuristic: the last layer should rotate towards the ground state
    # For transverse field, the "natural" angle is related to field/coupling ratio
    ratio = field / coupling if coupling != 0 else 1.0
    
    # Small field: angles close to pi/2 (maximally mixed in X basis)
    # Large field: angles closer to 0 or pi
    theta[-1, :] = np.arctan(ratio)  # Simple heuristic
    
    # Middle layers: small corrections
    theta[1, :] = 0.1 * np.sign(ratio)
    theta[0, :] = 0.05 * np.sign(ratio)
    
    return theta


def _optimize_for_case(n, coupling, field, layers, deadline):
    """Optimize angles for a single case using multi-start L-BFGS-B with polish."""
    ground_state, ground_energy, H = _get_exact_ground(n, coupling, field)
    
    best_theta = None
    best_score = np.inf
    
    # Multiple initial guesses
    initial_guesses = []
    
    # 1. Product |+> state (seed)
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    initial_guesses.append(theta0.flatten())
    
    # 2. All zeros
    initial_guesses.append(np.zeros((layers, n)).flatten())
    
    # 3. Heuristic from ground state structure
    theta_h = _estimate_initial_from_ground(n, coupling, field, layers, ground_state)
    initial_guesses.append(theta_h.flatten())
    
    # 4. Random initializations (fewer, since we're faster now)
    np.random.seed(42)
    for _ in range(6):
        initial_guesses.append(np.random.uniform(-np.pi, np.pi, (layers, n)).flatten())
    
    # 5. Small angle variations
    for scale in [0.1, 0.3, 0.5, 0.8]:
        theta_s = np.zeros((layers, n))
        theta_s[-1] = scale
        initial_guesses.append(theta_s.flatten())
    
    # Phase 1: Coarse optimization with moderate tolerances
    for x0 in initial_guesses:
        if time.time() > deadline - 30:  # Leave 30s for polish
            break
        
        try:
            result = minimize(
                _objective, x0, args=(n, coupling, field, layers),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            continue
    
    if best_theta is None:
        # Fallback to seed
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2
        best_score = _objective(best_theta.flatten(), n, coupling, field, layers)
    
    # Phase 2: High-precision polish from the best solution
    if time.time() < deadline - 5:
        try:
            result_polish = minimize(
                _objective, best_theta.flatten(), args=(n, coupling, field, layers),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result_polish.fun < best_score:
                best_score = result_polish.fun
                best_theta = result_polish.x.reshape(layers, n)
        except Exception:
            pass
    
    return best_theta


def angles(n, coupling, field, layers):
    """Optimize circuit angles for the given Ising chain parameters."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Try to optimize for this specific case
    try:
        theta = _optimize_for_case(n, coupling, field, layers, deadline)
    except Exception:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
# EVOLVE-BLOCK-END
