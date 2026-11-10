import numpy as np
import scipy.optimize as sco
import time

# Precomputed CZ gate
_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)

def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i for open chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n - 1):
        for state_idx in range(dim):
            bi = (state_idx >> i) & 1
            bp = (state_idx >> (i + 1)) & 1
            H[state_idx, state_idx] += -coupling * (1 - 2*bi) * (1 - 2*bp)
    for i in range(n):
        for state_idx in range(dim):
            flip = state_idx ^ (1 << i)
            H[flip, state_idx] += -field
    return H

def _ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _kron_full(gate, q, n):
    """Build full n-qubit gate matrix for single-qubit gate on qubit q (little-endian)."""
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        if i == q:
            full = np.kron(gate, full)
        else:
            full = np.kron(np.eye(2, dtype=np.complex128), full)
    return full

def _kron_cz(q, n):
    """Build full n-qubit CZ gate matrix for CZ on qubits q, q+1 (little-endian)."""
    full = np.array([[1.0]], dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        if i == q or i == q + 1:
            full = np.kron(_CZ, full)
        else:
            full = np.kron(np.eye(2, dtype=np.complex128), full)
    return full

def _forward(theta, n, layers):
    """Return (final_state, list_of_states_after_each_Ry).
    theta shape: (layers, n)"""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    # states_after_ry[l] = state after Ry in layer l (before CZ)
    states_after_ry = []
    for l in range(layers):
        for i in range(n):
            g = _ry(theta[l, i])
            full = _kron_full(g, i, n)
            state = full @ state
        states_after_ry.append(state.copy())
        if l < layers - 1:
            for i in range(n - 1):
                full_cz = _kron_cz(i, n)
                state = full_cz @ state
    return state, states_after_ry

def _objective_and_grad(params, n, coupling, field, layers, H, ground_energy):
    """Return (objective_value, gradient_vector)."""
    theta = params.reshape(layers, n)
    state, states_after_ry = _forward(theta, n, layers)
    energy = np.real(np.vdot(state, H @ state))
    obj_val = (energy - ground_energy) / n
    
    # Gradient: for each Ry at layer l, qubit i:
    # d/dtheta * E = 2*Re(vdot(state_after_ry[l], dRy/dtheta @ H @ state_after_ry[l]))
    # where dRy/dtheta = -0.5 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
    #                 = 0.5 * [[sin(t/2), cos(t/2)], [-cos(t/2), sin(t/2)]]
    grad = np.zeros_like(params)
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2)
            s = np.sin(t / 2)
            # dRy/dtheta = 0.5 * [[s, c], [-c, s]]
            dRy = 0.5 * np.array([[s, c], [-c, s]], dtype=np.complex128)
            full_d = _kron_full(dRy, i, n)
            Hs = H @ states_after_ry[l]
            grad[l * n + i] = 2.0 * np.real(np.vdot(states_after_ry[l], full_d @ Hs))
    
    return obj_val, grad

def _mean_field_init(n, field, layers):
    """Mean-field initialization: theta_i ≈ arctan(field/(coupling_eff)) per layer."""
    coupling = 1.0
    # Simple mean-field: each spin sees effective transverse field
    # theta ≈ arctan(field / (coupling * neighbors))
    theta = np.zeros((layers, n))
    for l in range(layers):
        for i in range(n):
            # neighbors
            left = 1 if i > 0 else 0
            right = 1 if i < n - 1 else 0
            eff_z = coupling * (left + right)
            if eff_z > 0:
                theta[l, i] = np.arctan(field / eff_z)
            else:
                theta[l, i] = np.pi / 2
    return theta.flatten()

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140
    
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = np.linalg.eigvalsh(H)[0]
    n_params = layers * n
    
    best_params = np.zeros(n_params)
    best_val = np.inf
    
    # Initializations
    np.random.seed(42)
    mf_init = _mean_field_init(n, field, layers)
    initializations = [
        mf_init,
        np.zeros(n_params),
        np.full(n_params, np.pi / 4),
        np.random.uniform(-np.pi / 2, np.pi / 2, n_params),
        np.random.uniform(0, np.pi, n_params),
    ]
    
    # First pass: L-BFGS-B with analytic gradient from each init
    for init in initializations:
        if time.time() > deadline - 5:
            break
        try:
            result = sco.minimize(
                _objective_and_grad, init,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Second pass: Nelder-Mead polish from best
    if time.time() < deadline - 10:
        try:
            def _obj_only(params):
                return _objective_and_grad(params, n, coupling, field, layers, H, ground_energy)[0]
            
            result = sco.minimize(
                _obj_only, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Third pass: L-BFGS-B polish from best
    if time.time() < deadline - 5:
        try:
            result = sco.minimize(
                _objective_and_grad, best_params,
                args=(n, coupling, field, layers, H, ground_energy),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(layers, n)
