"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

# CZ gate
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=np.complex128)


def _kron_all(ops):
    """Kronecker product of a list of 2x2 matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the open transverse-field Ising chain."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    # ZZ terms: -coupling * sum Z_i Z_(i+1)
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * _kron_all(ops)
    
    # X terms: -field * sum X_i
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * _kron_all(ops)
    
    return H


def _ground_energy(n, coupling, field):
    """Compute the exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(np.real(eigvals[0]))


def _build_cz_layer_unitary(n):
    """Build the full unitary for the CZ layer: CZ on adjacent pairs in ascending order."""
    dim = 2**n
    U = np.eye(dim, dtype=np.complex128)
    
    for i in range(n - 1):
        U_cz = np.eye(dim, dtype=np.complex128)
        for state in range(dim):
            b_i = (state >> i) & 1
            b_ip1 = (state >> (i+1)) & 1
            if b_i == 1 and b_ip1 == 1:
                U_cz[state, state] = -1.0
        U = U_cz @ U
    
    return U


def _build_ry_unitary(theta_vec, n):
    """Build unitary for a layer of Ry gates given theta_vec of length n."""
    U = np.array([[1.0]], dtype=np.complex128)
    for q in range(n - 1, -1, -1):
        c = np.cos(theta_vec[q] / 2)
        s = np.sin(theta_vec[q] / 2)
        Ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
        U = np.kron(U, Ry)
    return U


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to a specific qubit in the state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    perm = list(range(n))
    perm[-1], perm[qubit] = perm[qubit], perm[-1]
    state_perm = np.transpose(state_tensor, perm)
    
    new_state_perm = np.einsum('ab,...b->...a', ry, state_perm)
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _apply_cz(state, i, j, n):
    """Apply CZ gate between qubits i and j (i < j)."""
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    perm = list(range(n))
    perm[-1], perm[j] = perm[j], perm[-1]
    perm[-2], perm[i] = perm[i], perm[-2]
    state_perm = np.transpose(state_tensor, perm)
    
    # CZ acts on the last two qubits in permuted order
    new_state_perm = np.copy(state_perm)
    # For states where both qubits are 1, flip sign
    # In permuted tensor, last two indices are i and j
    mask = (state_perm[..., 1, 1] != 0)
    new_state_perm[..., 1, 1] = -state_perm[..., 1, 1]
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _state_from_params_fast(theta, n, layers, U_cz):
    """Build the circuit state from angle parameters using precomputed unitaries."""
    dim = 2**n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry gates to each qubit
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q, n)
        
        if layer < layers - 1:
            # Apply CZ gates on adjacent pairs
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return float(np.real(np.vdot(state, H @ state)))


def _objective_and_grad(theta_flat, n, coupling, field, layers, ge, H, U_cz):
    """Compute objective and gradient using finite differences."""
    theta = theta_flat.reshape(layers, n)
    state = _state_from_params_fast(theta, n, layers, U_cz)
    e_circuit = _energy(state, H)
    gap = (e_circuit - ge) / n
    
    # Compute gradient via finite differences
    grad = np.zeros_like(theta_flat)
    h = 1e-5
    for k in range(len(theta_flat)):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[k] += h
        theta_minus[k] -= h
        
        tp = theta_plus.reshape(layers, n)
        tm = theta_minus.reshape(layers, n)
        
        sp = _state_from_params_fast(tp, n, layers, U_cz)
        sm = _state_from_params_fast(tm, n, layers, U_cz)
        
        ep = _energy(sp, H)
        em = _energy(sm, H)
        
        grad[k] = (ep - em) / (2 * h * n)
    
    return gap, grad


def _heuristic_init(n, field):
    """
    Physics-inspired initial guess.
    For small field, ground state is close to |+>^n (all pi/2 in last layer).
    For larger field, we need more rotation.
    Use arctan(1/field) as a starting point for the last layer.
    """
    theta = np.zeros((3, n))
    # Last layer: rotate based on field strength
    # For field=0, we want pi/2 (maximal superposition)
    # For large field, we want less rotation
    last_layer_angle = np.arctan(1.0 / max(field, 0.01))
    theta[2] = last_layer_angle
    return theta


def _optimize_one(n, coupling, field, layers, ge, H, U_cz, deadline):
    """Optimize angles for a single instance using L-BFGS-B with multiple restarts."""
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    best_theta = None
    best_val = np.inf
    
    # Try heuristic initial point first
    theta_init = _heuristic_init(n, field).flatten()
    
    # Also try the simple |+> state
    theta_plus = np.zeros(layers * n)
    theta_plus[-n:] = np.pi / 2
    
    # Try all-zeros
    theta_zero = np.zeros(layers * n)
    
    initial_points = [theta_init, theta_plus, theta_zero]
    
    # Add some random points
    rng = np.random.RandomState(42)
    for _ in range(10):
        initial_points.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())
    
    for idx, theta_start in enumerate(initial_points):
        if time.time() > deadline - 10:
            break
        
        try:
            # Use tighter tolerance for the first few (informed) starts
            if idx < 3:
                options = {'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8}
            else:
                options = {'maxiter': 100, 'maxfun': 300, 'gtol': 1e-7}
            
            res = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, U_cz)[0],
                theta_start,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, U_cz)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options=options
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass
    
    if best_theta is None:
        # Fallback to heuristic
        best_theta = theta_init
    
    # Final polish pass with very tight tolerance on the best result
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, U_cz)[0],
                best_theta,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, U_cz)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-10}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass
    
    return best_theta.reshape(layers, n)


# Cache for ground energies, Hamiltonians, and CZ unitaries
_ge_cache = {}
_H_cache = {}
_Ucz_cache = {}


def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ge_cache:
        _ge_cache[key] = _ground_energy(n, coupling, field)
    return _ge_cache[key]


def _get_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key not in _H_cache:
        _H_cache[key] = _build_hamiltonian(n, coupling, field)
    return _H_cache[key]


def _get_cz_unitary(n):
    key = n
    if key not in _Ucz_cache:
        _Ucz_cache[key] = _build_cz_layer_unitary(n)
    return _Ucz_cache[key]


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    U_cz = _get_cz_unitary(n)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, H, U_cz, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
# EVOLVE-BLOCK-END
