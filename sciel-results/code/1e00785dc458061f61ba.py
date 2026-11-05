"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Pauli matrices (real for Ising model)
X = np.array([[0, 1], [1, 0]], dtype=float)
Z = np.array([[1, 0], [0, -1]], dtype=float)
I2 = np.eye(2, dtype=float)


def _kron_all(ops):
    """Kronecker product of a list of 2x2 matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the open transverse-field Ising chain."""
    H = np.zeros((2**n, 2**n), dtype=float)
    
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
    return float(eigvals[0])


# Cache index arrays for vectorized Ry application
_IDX_CACHE = {}

def _get_idx_arrays(n):
    """Precompute index arrays for Ry gates on each qubit."""
    if n in _IDX_CACHE:
        return _IDX_CACHE[n]
    dim = 1 << n
    arrays = []
    for i in range(n):
        mask = 1 << i
        all_idx = np.arange(dim)
        idx0 = all_idx[(all_idx & mask) == 0]
        idx1 = idx0 | mask
        arrays.append((idx0, idx1))
    _IDX_CACHE[n] = arrays
    return arrays


def _apply_cz_vectorized(state, n):
    """Apply CZ gates on adjacent pairs in ascending order (in-place)."""
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        all_idx = np.arange(1 << n)
        both_set = ((all_idx & mask_i) != 0) & ((all_idx & mask_ip1) != 0)
        state[both_set] *= -1.0
    return state


def _circuit_state(theta, n, layers, idx_arrays):
    """Compute the statevector of the circuit using vectorized bit-index arithmetic."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = idx_arrays[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            _apply_cz_vectorized(state, n)
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return float(state @ (H @ state))


def _objective_and_grad(theta_flat, n, coupling, field, layers, ge, H, idx_arrays):
    """Compute objective and gradient using parameter-shift rule (exact for Ry gates)."""
    theta = theta_flat.reshape(layers, n)
    state = _circuit_state(theta, n, layers, idx_arrays)
    e_circuit = _energy(state, H)
    gap = (e_circuit - ge) / n
    
    # Parameter-shift rule: for Ry(theta), d/dtheta <Ry(theta)|H|Ry(theta)> = 
    # (E(theta + pi/2) - E(theta - pi/2)) / 2
    grad = np.zeros_like(theta_flat)
    shift = np.pi / 2
    
    for k in range(len(theta_flat)):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[k] += shift
        theta_minus[k] -= shift
        
        tp = theta_plus.reshape(layers, n)
        tm = theta_minus.reshape(layers, n)
        
        sp = _circuit_state(tp, n, layers, idx_arrays)
        sm = _circuit_state(tm, n, layers, idx_arrays)
        
        ep = _energy(sp, H)
        em = _energy(sm, H)
        
        grad[k] = (ep - em) / (2.0 * n)
    
    return gap, grad


def _heuristic_init(n, field):
    """Physics-inspired initial guess."""
    theta = np.zeros((3, n))
    
    if field > 0.01:
        last_layer_angle = np.arctan(1.0 / field)
    else:
        last_layer_angle = np.pi / 2
    
    theta[2] = last_layer_angle
    theta[1] = 0.1 * last_layer_angle
    theta[0] = 0.05 * last_layer_angle
    
    return theta


def _optimize_one(n, coupling, field, layers, ge, H, idx_arrays, deadline):
    """Optimize angles for a single instance using L-BFGS-B with parameter-shift gradients."""
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    best_theta = None
    best_val = np.inf
    
    # Generate diverse initial points
    initial_points = []
    
    # Heuristic init
    theta_init = _heuristic_init(n, field).flatten()
    initial_points.append(theta_init)
    
    # |+> state
    theta_plus = np.zeros(layers * n)
    theta_plus[-n:] = np.pi / 2
    initial_points.append(theta_plus)
    
    # All zeros
    initial_points.append(np.zeros(layers * n))
    
    # Physics: structured inits
    for frac in [0.3, 0.5, 0.7, 1.0]:
        tp = np.zeros(layers * n)
        tp[-n:] = frac * np.pi / 2
        initial_points.append(tp)
    
    # Uniform
    initial_points.append(np.full(layers * n, np.pi / 4))
    
    # Random starts with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(20):
        initial_points.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())
    
    # Deduplicate similar points
    unique_points = [initial_points[0]]
    for pt in initial_points[1:]:
        if all(np.linalg.norm(pt - up) > 0.15 for up in unique_points):
            unique_points.append(pt)
    
    initial_points = unique_points
    
    # Phase 1: L-BFGS-B from each initial point
    for idx, theta_start in enumerate(initial_points):
        if time.time() > deadline - 20:
            break
        
        try:
            if idx < 5:
                options = {'maxiter': 400, 'maxfun': 1000, 'gtol': 1e-10}
            else:
                options = {'maxiter': 200, 'maxfun': 500, 'gtol': 1e-9}
            
            res = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[0],
                theta_start,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[1],
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
    
    # Phase 2: Powell polish from best L-BFGS-B result
    if time.time() < deadline - 15:
        try:
            res_powell = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[0],
                best_theta,
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if res_powell.fun < best_val:
                best_val = res_powell.fun
                best_theta = res_powell.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional random restarts with L-BFGS-B (enabled by vectorized speedup)
    restart_budget = 10
    for r in range(restart_budget):
        if time.time() > deadline - 10:
            break
        tc = best_theta + rng.normal(0, 0.5, size=(layers, n)).flatten()
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[0],
                tc,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final tight L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[0],
                best_theta,
                jac=lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-13}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final Powell polish
    if time.time() < deadline - 3:
        try:
            res_final = minimize(
                lambda x: _objective_and_grad(x, n, coupling, field, layers, ge, H, idx_arrays)[0],
                best_theta,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-15, 'ftol': 1e-17}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_theta = res_final.x.copy()
        except Exception:
            pass
    
    return best_theta.reshape(layers, n)


# Cache for ground energies and Hamiltonians
_ge_cache = {}
_H_cache = {}
_angle_cache = {}


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
    key = (n, coupling, field, layers)
    if key in _angle_cache:
        return _angle_cache[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    idx_arrays = _get_idx_arrays(n)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _angle_cache[key] = theta
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, H, idx_arrays, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    _angle_cache[key] = theta
    return theta
# EVOLVE-BLOCK-END
