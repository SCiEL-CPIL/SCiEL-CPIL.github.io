"""Rewritten program: Fast variational optimization for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill
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
            theta = thetas[layer, q]
            state = _apply_ry(state, theta, q, n)
        
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

def _objective_fidelity(params, n, coupling, field, layers, H, e0, psi_target):
    psi = _circuit_state(params, n, layers)
    # Fidelity-based objective: minimize -|<target|psi>|^2
    # This is equivalent to maximizing overlap with the exact ground state
    overlap = np.vdot(psi_target, psi)
    fidelity = np.real(overlap * np.conj(overlap))
    # We want to maximize fidelity, so minimize negative fidelity
    # But we also need to ensure the energy is low. Let's use energy gap.
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    return obj_val

def _objective_and_grad(params, n, coupling, field, layers, H, e0):
    psi = _circuit_state(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(params.size):
        params_plus = params.copy()
        params_plus[k] += eps
        psi_plus = _circuit_state(params_plus, n, layers)
        Hpsi_plus = H @ psi_plus
        energy_plus = np.vdot(psi_plus, Hpsi_plus).real
        grad[k] = (energy_plus - energy) / eps / n
    return obj_val, grad

def _optimize(n, coupling, field, layers):
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Try multiple restarts with different initializations
    # For small n, we can afford several restarts
    n_params = layers * n
    
    # Generate diverse initial guesses
    np.random.seed(42)
    restarts = []
    
    # 1. All zeros
    restarts.append(np.zeros(n_params))
    
    # 2. All pi/2
    restarts.append(np.full(n_params, np.pi / 2))
    
    # 3. All pi/4
    restarts.append(np.full(n_params, np.pi / 4))
    
    # 4. Random initializations
    for _ in range(20):
        restarts.append(np.random.randn(n_params) * 1.0)
    
    # 5. Heuristic: stronger rotation for larger field
    ratio = field / coupling if coupling != 0 else 1.0
    base_angle = np.arctan(ratio)
    restarts.append(np.full(n_params, base_angle))
    
    best_val = np.inf
    best_params = None
    
    for r, x_init in enumerate(restarts):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
            
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            def obj(x):
                return _objective_fidelity(x, n, coupling, field, layers, H, e0, None)
            
            res = opt.minimize(
                obj, 
                x_init, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
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
        theta = _optimize(n, coupling, field, layers)
        _cache[key] = theta
        return theta
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        _cache[key] = theta
        return theta
# EVOLVE-BLOCK-END
