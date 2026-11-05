"""Shallow-circuit Ising ground states - combined analytical approach with precomputation."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_PRECOMP_CACHE = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _precompute_masks(n):
    """Precompute bit masks and index arrays for fast state evolution."""
    dim = 1 << n
    masks = [1 << i for i in range(n)]
    idx0_cache = {}
    idx1_cache = {}
    for i in range(n):
        m = masks[i]
        idx0 = np.array([b for b in range(dim) if (b & m) == 0], dtype=int)
        idx1 = idx0 | m
        idx0_cache[i] = idx0
        idx1_cache[i] = idx1
    cz_indices = []
    for i in range(n - 1):
        mi = masks[i]
        mip1 = masks[i + 1]
        both_set = np.array([b for b in range(dim) if (b & mi) and (b & mip1)], dtype=int)
        cz_indices.append(both_set)
    return masks, idx0_cache, idx1_cache, cz_indices

def _apply_cz_vectorized(state_flat, n, cz_indices):
    """Apply CZ gates on adjacent pairs in ascending order."""
    for both_set in cz_indices:
        state_flat[both_set] *= -1.0
    return state_flat

def _circuit_state(theta, n, layers, precomp=None):
    """Compute the statevector of the Ry/CZ circuit using precomputed indices."""
    if precomp is None:
        _, idx0_cache, idx1_cache, cz_indices = _precompute_masks(n)
    else:
        _, idx0_cache, idx1_cache, cz_indices = precomp
    
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0 = idx0_cache[i]
            idx1 = idx1_cache[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            state = _apply_cz_vectorized(state, n, cz_indices)
            
    return state

def _energy_from_state(state, H):
    return float(np.dot(state, H @ state))

def _exact_ground_state(n, coupling, field):
    """Get exact ground state and energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return psi0, E0, H

def _mean_field_init(n, coupling, field):
    """
    Mean-field / variational product state initialization.
    """
    theta = np.zeros((3, n))
    
    ratio = field / coupling
    if ratio < 2.0:
        t_opt = np.arcsin(min(ratio / 2.0, 1.0))
    else:
        t_opt = np.pi / 2
    
    theta[0] = t_opt * 0.5
    theta[1] = t_opt * 0.3
    theta[2] = t_opt * 0.2
    
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute masks and indices
    if n not in _PRECOMP_CACHE:
        _PRECOMP_CACHE[n] = _precompute_masks(n)
    precomp = _PRECOMP_CACHE[n]
    _, idx0_cache, idx1_cache, cz_indices = precomp
    
    # Build Hamiltonian and get exact ground state
    psi0, E0, H = _exact_ground_state(n, coupling, field)
    
    # Mean-field initialization
    theta_mf = _mean_field_init(n, coupling, field)
    
    # Also try simple heuristics
    theta_simple = np.zeros((3, n))
    theta_simple[-1] = np.pi / 2  # Put all rotation in last layer
    
    theta_uniform = np.full((3, n), np.pi / 4)
    
    # Evaluate initial candidates
    candidates = [theta_mf, theta_simple, theta_uniform]
    
    best_theta = None
    best_energy = float('inf')
    
    for tc in candidates:
        if time.time() > deadline:
            break
        state = _circuit_state(tc, n, 3, precomp)
        e = _energy_from_state(state, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Also add random starts
    rng = np.random.default_rng(12345)
    for _ in range(12):
        if time.time() > deadline - 5:
            break
        tc = rng.uniform(-np.pi, np.pi, size=(3, n))
        state = _circuit_state(tc, n, 3, precomp)
        e = _energy_from_state(state, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Add perturbations around best
    for _ in range(8):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        state = _circuit_state(tc, n, 3, precomp)
        e = _energy_from_state(state, H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Now do numerical optimization from the best starting point
    def objective(x):
        theta = x.reshape((3, n))
        state = _circuit_state(theta, n, 3, precomp)
        return _energy_from_state(state, H)
    
    # Strategy 1: L-BFGS-B with numerical gradients
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead for polish
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Strategy 3: Another L-BFGS-B pass with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final: try a few more random restarts if time permits
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.5, size=(3, n))
        try:
            res = opt.minimize(
                objective,
                tc.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Final high-precision polish: Powell with very tight tolerances
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((3, n))
        except Exception:
            pass
    
    # Ensure output is finite
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = np.zeros((3, n))
        best_theta[-1] = np.pi / 2
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
