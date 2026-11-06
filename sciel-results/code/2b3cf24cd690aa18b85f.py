"""Direct approach: exploit circuit structure for fast gradient computation."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
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


def _exact_ground_energy(H):
    """Compute exact ground state energy."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _get_state_from_full_params(params, n, layers):
    """Compute state from flattened full theta vector."""
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            shape = [2] * n
            psi_r = psi.reshape(shape)
            gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
            psi_moved = np.moveaxis(psi_r, q, 0)
            psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
            psi = np.moveaxis(psi_out, 0, q).reshape(-1)
        
        if layer < layers - 1:
            for q in range(n - 1):
                shape = [2] * n
                psi_r = psi.reshape(shape)
                psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
                factor = np.ones(psi_moved.shape, dtype=np.complex128)
                idx11 = (slice(None),) * (n - 2) + (1, 1)
                factor[idx11] = -1.0
                psi_moved = psi_moved * factor
                psi = np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)
    
    return psi

def _expand_symmetric_params(sym_params, n, layers):
    """
    Expand symmetric parameters to full (layers, n) array.
    Symmetry: theta[l, i] == theta[l, n-1-i]
    """
    full_params = np.zeros((layers, n))
    half_n = (n + 1) // 2
    # sym_params shape: (layers, half_n)
    # We assume sym_params is flattened (layers * half_n)
    sym_params = sym_params.reshape(layers, half_n)
    
    for l in range(layers):
        for i in range(half_n):
            val = sym_params[l, i]
            full_params[l, i] = val
            full_params[l, n - 1 - i] = val
            
    return full_params

def _get_state_from_sym_params(sym_params, n, layers):
    """Compute state from symmetric parameter vector."""
    full_params = _expand_symmetric_params(sym_params, n, layers)
    return _get_state_from_full_params(full_params, n, layers)


def _compute_energy_and_gradient_sym(sym_params, n, coupling, field, layers, H, e0):
    """
    Compute energy and gradient with respect to symmetric parameters.
    Uses parameter-shift rule.
    """
    half_n = (n + 1) // 2
    num_sym_params = layers * half_n
    
    # Compute main state
    psi = _get_state_from_sym_params(sym_params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Compute gradient using parameter shift rule
    # dE/dtheta_k = (E(theta_k + pi/2) - E(theta_k - pi/2)) / 2
    shift = np.pi / 2
    grad = np.zeros(num_sym_params)
    
    for k in range(num_sym_params):
        params_plus = sym_params.copy()
        params_minus = sym_params.copy()
        params_plus[k] += shift
        params_minus[k] -= shift
        
        psi_plus = _get_state_from_sym_params(params_plus, n, layers)
        psi_minus = _get_state_from_sym_params(params_minus, n, layers)
        
        e_plus = np.vdot(psi_plus, H @ psi_plus).real
        e_minus = np.vdot(psi_minus, H @ psi_minus).real
        
        grad[k] = (e_plus - e_minus) / (2 * n)
    
    return obj_val, grad


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    
    half_n = (n + 1) // 2
    num_sym_params = layers * half_n
    
    def objective_only(params):
        val, _ = _compute_energy_and_gradient_sym(params, n, coupling, field, layers, H, e0)
        return val
    
    def gradient_only(params):
        _, grad = _compute_energy_and_gradient_sym(params, n, coupling, field, layers, H, e0)
        return grad
    
    # Heuristic initial guesses
    ratio = field / coupling if coupling > 0 else 1.0
    base_angle = 0.5 * np.pi * np.tanh(ratio * 0.5)
    
    np.random.seed(42)
    
    restarts = [
        np.full(num_sym_params, base_angle),
        np.zeros(num_sym_params),
        np.full(num_sym_params, np.pi / 2),
        np.full(num_sym_params, np.pi / 4),
        np.random.randn(num_sym_params) * 0.5 + base_angle,
        np.random.randn(num_sym_params) * 0.3 + base_angle,
    ]
    
    if n <= 4:
        restarts.append(np.random.randn(num_sym_params) * 0.8)
        restarts.append(np.random.randn(num_sym_params) * 0.6)
    else:
        restarts.append(np.random.randn(num_sym_params) * 0.5)
    
    best_result = None
    best_val = np.inf
    
    time_budget_per_restart = 12.0  # seconds per restart
    
    for r, x_init in enumerate(restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
        
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(
                objective_only,
                x_init,
                jac=gradient_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-10,
                    'gtol': 1e-6,
                    'maxls': 20,
                    'maxfun': 200
                }
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: use the best heuristic
        best_result = np.full(num_sym_params, base_angle)
    
    # Expand to full params
    theta = _expand_symmetric_params(best_result, n, layers)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: return |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
