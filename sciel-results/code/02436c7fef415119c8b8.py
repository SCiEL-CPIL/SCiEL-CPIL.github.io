"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

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


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to a specific qubit in the state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    # Reshape state to n-dimensional tensor
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    # Apply Ry to the qubit dimension
    # We need to permute so the qubit is last, apply, then permute back
    perm = list(range(n))
    perm[-1], perm[qubit] = perm[qubit], perm[-1]
    state_perm = np.transpose(state_tensor, perm)
    
    # Apply Ry: state_perm[..., qubit] -> Ry @ state_perm[..., qubit]
    # state_perm has shape (2, 2, ..., 2) with qubit as last axis
    new_state_perm = np.einsum('ab,...b->...a', ry, state_perm)
    
    # Permute back
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _apply_cz(state, i, j, n):
    """Apply CZ gate between qubits i and j (i < j) in ascending order."""
    # Reshape to tensor
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    # Permute so i and j are the last two dimensions
    perm = list(range(n))
    # Move j to last, i to second-to-last
    perm[-1], perm[j] = perm[j], perm[-1]
    perm[-2], perm[i] = perm[i], perm[-2]
    state_perm = np.transpose(state_tensor, perm)
    
    # Apply CZ: reshape last two dims to 4, apply CZ
    flat_shape = [-1, 4]
    state_2d = state_perm.reshape(flat_shape)
    new_state_2d = np.einsum('ab,...b->...a', CZ, state_2d)
    new_state_perm = new_state_2d.reshape(state_perm.shape)
    
    # Permute back
    inv_perm = [0] * n
    for idx, p in enumerate(perm):
        inv_perm[p] = idx
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _state_from_params(theta, n, layers):
    """Build the circuit state from angle parameters."""
    # Start with |0>^n
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q, n)
        
        # Apply CZ between adjacent pairs in ascending order (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _energy(state, n, coupling, field):
    """Compute <state|H|state>."""
    H = _build_hamiltonian(n, coupling, field)
    return float(np.real(np.vdot(state, H @ state)))


def _objective(theta, n, coupling, field, layers, ge):
    """Compute the mean energy gap per spin."""
    state = _state_from_params(theta, n, layers)
    e_circuit = _energy(state, n, coupling, field)
    gap = (e_circuit - ge) / n
    return gap


def _optimize_one(n, coupling, field, layers, ge, deadline):
    """Optimize angles for a single instance using L-BFGS-B."""
    from scipy.optimize import minimize
    
    def obj_and_grad(theta_flat):
        theta = theta_flat.reshape(layers, n)
        # Compute energy
        state = _state_from_params(theta, n, layers)
        H = _build_hamiltonian(n, coupling, field)
        e_circuit = float(np.real(np.vdot(state, H @ state)))
        gap = (e_circuit - ge) / n
        
        # Compute gradient via finite differences (central)
        grad = np.zeros_like(theta_flat)
        h = 1e-5
        for k in range(len(theta_flat)):
            theta_plus = theta_flat.copy()
            theta_minus = theta_flat.copy()
            theta_plus[k] += h
            theta_minus[k] -= h
            
            tp = theta_plus.reshape(layers, n)
            tm = theta_minus.reshape(layers, n)
            
            sp = _state_from_params(tp, n, layers)
            sm = _state_from_params(tm, n, layers)
            
            ep = float(np.real(np.vdot(sp, H @ sp)))
            em = float(np.real(np.vdot(sm, H @ sm)))
            
            grad[k] = (ep - em) / (2 * h * n)
        
        return gap, grad
    
    # Initial guess: product |+> state (all pi/2 in last layer, 0 elsewhere)
    theta_init = np.zeros(layers * n)
    theta_init[-n:] = np.pi / 2
    
    # Bounds: angles in [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    best_theta = theta_init.copy()
    best_val = obj_and_grad(theta_init)[0]
    
    # Try L-BFGS-B
    try:
        res = minimize(
            obj_and_grad,
            theta_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8}
        )
        if res.success or res.fun < best_val:
            best_theta = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_theta.reshape(layers, n)
    
    # Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        theta_rand = rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten()
        try:
            res = minimize(
                obj_and_grad,
                theta_rand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 300, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_theta = res.x
                best_val = res.fun
        except Exception:
            pass
    
    return best_theta.reshape(layers, n)


# Cache for ground energies
_ge_cache = {}


def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ge_cache:
        _ge_cache[key] = _ground_energy(n, coupling, field)
    return _ge_cache[key]


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
# EVOLVE-BLOCK-END
