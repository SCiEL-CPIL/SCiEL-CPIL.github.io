"""Shallow-circuit Ising ground states - direct analytical approach."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}

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

def _circuit_state(theta, n, layers):
    """Compute the statevector of the 3-layer Ry/CZ circuit."""
    dim = 1 << n
    state = np.zeros((2,) * n, dtype=float)
    state[(0,) * n] = 1.0
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            
            state = np.moveaxis(state, i, 0)
            new_shape = (2,) + state.shape[1:]
            state = state.reshape(new_shape)
            
            old0 = state[0].copy()
            old1 = state[1].copy()
            state[0] = c * old0 - s * old1
            state[1] = s * old0 + c * old1
            
            state = state.reshape((2,) * n)
            state = np.moveaxis(state, 0, i)
        
        if l < layers - 1:
            state_flat = state.flatten()
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                for b in range(dim):
                    if (b & mask_i) and (b & mask_ip1):
                        state_flat[b] *= -1.0
            state = state_flat.reshape((2,) * n)
            
    return state.flatten()

def _energy_from_state(state, H):
    return float(np.dot(state, H @ state))

def _energy(theta_flat, n, coupling, field, H, E0):
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def _exact_ground_state(n, coupling, field):
    """Get exact ground state and energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return psi0, E0

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

def _circuit_energy(theta_flat, n, H):
    """Energy of the circuit state."""
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    return _energy_from_state(state, H)

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    psi0, E0 = _exact_ground_state(n, coupling, field)
    
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
        e = _energy_from_state(_circuit_state(tc, n, 3), H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Also add random starts
    rng = np.random.default_rng(12345)
    for _ in range(12):
        if time.time() > deadline - 5:
            break
        tc = rng.uniform(-np.pi, np.pi, size=(3, n))
        e = _energy_from_state(_circuit_state(tc, n, 3), H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Add perturbations around best
    for _ in range(8):
        if time.time() > deadline - 5:
            break
        tc = best_theta + rng.normal(0, 0.3, size=(3, n))
        e = _energy_from_state(_circuit_state(tc, n, 3), H)
        if e < best_energy:
            best_energy = e
            best_theta = tc.copy()
    
    # Now do numerical optimization from the best starting point
    def objective(x):
        theta = x.reshape((3, n))
        state = _circuit_state(theta, n, 3)
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
    
    # Strategy 3: Another L-BFGS-B pass
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                objective,
                best_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
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
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
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
