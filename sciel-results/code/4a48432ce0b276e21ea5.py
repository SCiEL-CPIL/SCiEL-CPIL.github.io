import numpy as np
import time
from scipy.optimize import minimize

def _precompute_indices(n):
    """Precompute index arrays for Ry and CZ gates."""
    dim = 1 << n
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_indices.append((idx0, idx1))
    
    cz_indices = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx_both = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_indices.append(idx_both)
    
    return ry_indices, cz_indices

def _circuit_state(n, layers, params, ry_indices, cz_indices):
    """Compute state from circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if layer < layers - 1:
            for q in range(n - 1):
                state[cz_indices[q]] = -state[cz_indices[q]]
    
    return state

def _hamiltonian(n, coupling, field):
    """Build Hamiltonian."""
    dim = 1 << n
    I = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for i in range(n - 1):
        term = np.array([1.0])
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    return H

def _closed_form_angle(coupling, field):
    """
    Closed-form optimal Ry angle for Ising model.
    
    For H = -J*ZZ - h*X on a single qubit (effective), the optimal rotation
    to minimize energy satisfies tan(theta/2) = h/J, so theta = 2*arctan(h/J).
    This is equivalent to arccos(J/sqrt(J^2+h^2)).
    """
    if coupling == 0:
        return np.pi / 2
    ratio = field / coupling
    # theta = 2 * arctan(ratio)
    # Clamp ratio to avoid numerical issues
    ratio = np.clip(ratio, -1e6, 1e6)
    return 2.0 * np.arctan(ratio)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute indices for fast circuit evaluation
    ry_indices, cz_indices = _precompute_indices(n)
    
    # Compute closed-form optimal angle
    base_angle = _closed_form_angle(coupling, field)
    
    # Initialize all layers with the closed-form angle
    params = np.full((layers, n), base_angle)
    
    # Build Hamiltonian and compute ground state energy
    H = _hamiltonian(n, coupling, field)
    e_ground = np.linalg.eigvalsh(H)[0]
    
    def obj(xf):
        p = xf.reshape((layers, n))
        st = _circuit_state(n, layers, p, ry_indices, cz_indices)
        energy = np.vdot(st, H @ st).real
        return (energy - e_ground) / n
    
    # Evaluate the closed-form solution
    best_val = obj(params.flatten())
    best_params = params.copy()
    
    # Try diverse variations to find better starting points
    variations = []
    
    # Uniform variations
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        v = np.full((layers, n), base_angle * scale)
        variations.append(v)
    
    # Layer-decaying patterns
    for decay in [0.5, 0.7, 0.85, 1.0]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (decay ** l)
        variations.append(v)
    
    # Layer-increasing patterns
    for growth in [1.2, 1.5]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (growth ** l)
        variations.append(v)
    
    # Alternating layer patterns
    for sign in [1.0, -1.0]:
        v = np.zeros((layers, n))
        for l in range(layers):
            v[l, :] = base_angle * (sign ** l)
        variations.append(v)
    
    # Evaluate all variations
    for v in variations:
        if time.time() > deadline - 10:
            break
        val = obj(v.flatten())
        if val < best_val:
            best_val = val
            best_params = v.copy()
    
    # Use L-BFGS-B from the best initial point
    if time.time() < deadline - 30:
        try:
            res = minimize(obj, best_params.flatten(), method='L-BFGS-B',
                         options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Use Nelder-Mead to polish
    if time.time() < deadline - 20:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # Final polish with BFGS
    if time.time() < deadline - 10:
        try:
            res = minimize(obj, best_params.flatten(), method='BFGS',
                         options={'maxiter': 300, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    # One more Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_params.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except:
            pass
    
    return best_params
