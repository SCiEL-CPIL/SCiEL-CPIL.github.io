"""Rewritten program: Nelder-Mead multi-start optimization for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill
_cache = {}

def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _apply_ry(state, theta, qubit, n):
    if abs(theta) < 1e-15:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    gate = _ry_matrix(theta)
    state_moved = np.moveaxis(state_r, qubit, 0)
    state_out = np.tensordot(gate, state_moved, axes=([1], [0]))
    state_moved_back = np.moveaxis(state_out, 0, qubit)
    return state_moved_back.reshape(-1)

def _apply_cz(state, q1, q2, n):
    if q1 == q2:
        return state
    shape = [2] * n
    state_r = state.reshape(shape)
    state_moved = np.moveaxis(state_r, [q1, q2], [-2, -1])
    factor = np.ones(state_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    state_moved = state_moved * factor
    state_moved_back = np.moveaxis(state_moved, [-2, -1], [q1, q2])
    return state_moved_back.reshape(-1)

def _circuit_state(params, n, layers):
    thetas = params.reshape(layers, n)
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, thetas[layer, q], q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state

def _build_hamiltonian(n, coupling, field):
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_i + 1 == 0 else -1.0 # Wait, bit_ip1 is i+1
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H

def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])

def _objective(params, n, coupling, field, layers, H, e0):
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    return (energy - e0) / n

def _optimize_nm(n, coupling, field, layers):
    total_params = layers * n
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Generate diverse initial guesses
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(total_params))
    
    # 2. Pi/2
    inits.append(np.full(total_params, np.pi / 2))
    
    # 3. Heuristic based on field/coupling
    # If field is strong, rotate more. If coupling is strong, rotate less?
    # Actually, for pure Z, ground state is |0..0> (theta=0).
    # For pure X, ground state is |+..+> (theta=pi/2).
    # So theta ~ arctan(field/coupling) * something?
    ratio = field / coupling
    base_theta = np.arctan(ratio)
    inits.append(np.full(total_params, base_theta))
    
    # 4. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Scale random noise by pi
        inits.append(rng.uniform(-np.pi, np.pi, size=total_params))

    best_val = np.inf
    best_params = None
    
    # Nelder-Mead options
    # maxiter is per restart
    nm_options = {
        'maxiter': 500,
        'fatol': 1e-10,
        'xatol': 1e-8,
        'adaptive': True
    }

    for x0 in inits:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        try:
            # Use scipy.optimize.minimize with Nelder-Mead
            from scipy.optimize import minimize
            res = minimize(
                _objective, 
                x0, 
                args=(n, coupling, field, layers, H, e0),
                method='Nelder-Mead',
                options=nm_options
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception as e:
            pass

    if best_params is None:
        best_params = np.zeros(total_params)
        
    return best_params.reshape(layers, n)

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    try:
        theta = _optimize_nm(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
