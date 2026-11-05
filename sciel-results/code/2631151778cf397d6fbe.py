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

# CZ gate matrix
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
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * _kron_all(ops)
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


def _apply_ry_to_state(state, theta, q, n, dim):
    """Apply Ry(theta) to qubit q of state vector (dim=2**n)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # Ry = [[c, -s], [s, c]]
    # For qubit q, we need to apply this to the q-th index
    shape = [2] * n
    st = state.reshape(shape)
    # Move qubit q to last axis
    perm = list(range(n))
    perm[-1], perm[q] = perm[q], perm[-1]
    st_perm = np.transpose(st, perm)
    # Apply Ry: new[..., 0] = c*old[...,0] - s*old[...,1]
    #           new[..., 1] = s*old[...,0] + c*old[...,1]
    old0 = st_perm[..., 0]
    old1 = st_perm[..., 1]
    new0 = c * old0 - s * old1
    new1 = s * old0 + c * old1
    st_new = np.empty_like(st_perm)
    st_new[..., 0] = new0
    st_new[..., 1] = new1
    # Inverse permute
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    st_final = np.transpose(st_new, inv_perm)
    return st_final.reshape(dim)


def _apply_cz_to_state(state, i, j, n, dim):
    """Apply CZ gate between qubits i and j (i < j)."""
    shape = [2] * n
    st = state.reshape(shape)
    # Move qubits i and j to last two axes
    perm = list(range(n))
    perm[-1], perm[j] = perm[j], perm[-1]
    perm[-2], perm[i] = perm[i], perm[-2]
    st_perm = np.transpose(st, perm)
    # Now last two axes are qubits i and j
    # Reshape: group last two indices into one of size 4
    flat_shape = list(st_perm.shape[:-2]) + [4]
    st_2d = st_perm.reshape(flat_shape)
    # CZ acts on the last index: CZ[0,0]=1, CZ[1,1]=1, CZ[2,2]=1, CZ[3,3]=-1
    # So just flip sign of index 3
    st_2d[..., 3] = -st_2d[..., 3]
    st_new = st_2d.reshape(st_perm.shape)
    # Inverse permute
    inv_perm = [0] * n
    for idx, p in enumerate(perm):
        inv_perm[p] = idx
    st_final = np.transpose(st_new, inv_perm)
    return st_final.reshape(dim)


def _state_from_params(theta, n, layers, dim):
    """Build the circuit state from angle parameters (layers, n)."""
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_to_state(state, theta[layer, q], q, n, dim)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_to_state(state, i, i + 1, n, dim)
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return float(np.real(np.vdot(state, H @ state)))


def _objective_and_grad(theta_flat, n, coupling, field, layers, ge, H, dim):
    """Compute objective and gradient using central finite differences."""
    theta = theta_flat.reshape(layers, n)
    state = _state_from_params(theta, n, layers, dim)
    e_circuit = _energy(state, H)
    gap = (e_circuit - ge) / n

    num_params = layers * n
    h = 1e-6
    grad = np.zeros(num_params)
    for k in range(num_params):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[k] += h
        theta_minus[k] -= h
        sp = _state_from_params(theta_plus.reshape(layers, n), n, layers, dim)
        sm = _state_from_params(theta_minus.reshape(layers, n), n, layers, dim)
        ep = _energy(sp, H)
        em = _energy(sm, H)
        grad[k] = (ep - em) / (2.0 * h * n)

    return gap, grad


def _heuristic_init(n, field):
    """Physics-inspired initial guess."""
    theta = np.zeros((3, n))
    # For small field, ground state is close to |+>^n
    # For larger field, less rotation
    last_layer_angle = np.arctan(1.0 / max(field, 0.01))
    theta[2] = last_layer_angle
    return theta


def _optimize_one(n, coupling, field, layers, ge, H, dim, deadline):
    """Optimize angles for a single instance using L-BFGS-B with multiple restarts."""
    bounds = [(-np.pi, np.pi)] * (layers * n)

    best_theta = None
    best_val = np.inf

    # Initial points
    theta_init = _heuristic_init(n, field).flatten()
    theta_plus = np.zeros(layers * n)
    theta_plus[-n:] = np.pi / 2
    theta_zero = np.zeros(layers * n)

    initial_points = [theta_init, theta_plus, theta_zero]

    # Add random points
    rng = np.random.RandomState(42)
    num_random = 10
    for _ in range(num_random):
        initial_points.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())

    # Also try symmetric patterns
    for ang in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        theta_sym = np.full((layers, n), ang)
        initial_points.append(theta_sym.flatten())

    for idx, theta_start in enumerate(initial_points):
        if time.time() > deadline - 20:
            break

        try:
            if idx < 3:
                options = {'maxiter': 300, 'maxfun': 800, 'gtol': 1e-10}
            else:
                options = {'maxiter': 200, 'maxfun': 500, 'gtol': 1e-9}

            res = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, dim)[0],
                theta_start,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, dim)[1],
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
        best_theta = theta_init

    # Final polish pass with very tight tolerance
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, dim)[0],
                best_theta,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, dim)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 800, 'gtol': 1e-12}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass

    return best_theta.reshape(layers, n)


# Cache for ground energies and Hamiltonians
_ge_cache = {}
_H_cache = {}


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


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    dim = 2 ** n

    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    theta = _optimize_one(n, coupling, field, layers, ge, H, dim, deadline)

    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)

    return theta
# EVOLVE-BLOCK-END
