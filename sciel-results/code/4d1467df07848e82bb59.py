"""Tensor-network-informed optimizer for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_gs_cache = {}
_circuit_indices_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        mask = 1 << i
        for state in range(dim):
            flipped = state ^ mask
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _ground_state_vector(n, coupling, field):
    key = (n, coupling, field)
    if key in _gs_cache:
        return _gs_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    gs = eigvecs[:, idx]
    # Ensure real if possible
    if np.abs(np.imag(gs)).max() < 1e-10:
        gs = np.real(gs)
    else:
        # Phase align
        phase = np.angle(gs[np.argmax(np.abs(gs))])
        gs = gs * np.exp(-1j * phase)
        if np.abs(np.imag(gs)).max() < 1e-10:
            gs = np.real(gs)
    _gs_cache[key] = gs
    return gs

def _get_circuit_indices(n, layers):
    """Pre-compute index pairs for fast state evolution."""
    key = (n, layers)
    if key in _circuit_indices_cache:
        return _circuit_indices_cache[key]
    
    dim = 1 << n
    idx = np.arange(dim)
    
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_indices.append((idx_0, idx_1))
    
    cz_indices = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_indices.append(both_set)
    
    _circuit_indices_cache[key] = (ry_indices, cz_indices)
    return _circuit_indices_cache[key]

def _circuit_state_fast(n, thetas, layers):
    """Fast circuit state computation using pre-computed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    ry_indices, cz_indices = _get_circuit_indices(n, layers)
    
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = ry_indices[q]
            v0 = state[idx_0].copy()
            v1 = state[idx_1].copy()
            state[idx_0] = c * v0 - s * v1
            state[idx_1] = s * v0 + c * v1
        
        if layer < layers - 1:
            for both_set in cz_indices:
                state[both_set] *= -1.0
    
    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.vdot(psi, H @ psi))

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _mps_ground_state(n, coupling, field):
    """
    Compute ground state as MPS using DMRG-like approach on the small system.
    Returns the ground state vector reshaped as MPS tensors.
    """
    gs = _ground_state_vector(n, coupling, field)
    dim = 1 << n
    
    # Reshape into MPS with bond dimension 2
    # For n qubits, each site has local dim 2
    # MPS: psi = sum_{i1...in} T1[i1] T2[i2] ... Tn[in]
    # For exact state, we can just reshape the vector
    gs_reshaped = gs.reshape([2] * n)
    
    # SVD to get MPS tensors
    tensors = []
    remaining = gs_reshaped.copy()
    
    for i in range(n):
        if i == n - 1:
            # Last tensor is just the remaining vector
            tensors.append(remaining.copy())
        else:
            # Move current site to front
            remaining = np.moveaxis(remaining, 0, 0)
            # Reshape to (2, -1)
            shape = remaining.shape
            new_shape = (shape[0], -1)
            remaining = remaining.reshape(new_shape)
            # SVD
            U, S, Vh = np.linalg.svd(remaining, full_matrices=False)
            # Keep only largest singular value for exact MPS (bond dim 1 would lose info)
            # Actually for exact representation we need to keep all singular values
            # But for our purpose, we just need the angles, so let's use a different approach
            
            tensors.append(U)
            remaining = S[:, :1] * Vh[:1, :]  # This loses information
    
    # This approach is flawed. Let's use a simpler initialization strategy.
    return None

def _physics_informed_init(n, coupling, field, layers):
    """Generate physics-informed initial guesses."""
    inits = []
    
    # Paramagnetic limit
    inits.append(np.full((layers, n), np.pi / 2))
    
    # Ferromagnetic limit
    inits.append(np.zeros((layers, n)))
    
    # Mean-field estimate
    ratio = field / coupling if coupling > 0 else 1.0
    theta_mf = 2.0 * np.arctan(ratio)
    inits.append(np.full((layers, n), theta_mf))
    
    # Alternating
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                alt[l, q] = theta_mf
            else:
                alt[l, q] = -theta_mf
    inits.append(alt)
    
    # Layer-dependent
    layer_comp = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            layer_comp[l, q] = theta_mf * (1.0 + 0.1 * (l - 1))
    inits.append(layer_comp)
    
    # Perturbations
    for base in [0.0, np.pi / 2, theta_mf]:
        for delta in [-0.3, -0.1, 0.1, 0.3]:
            inits.append(np.full((layers, n), base + delta))
    
    return inits

def _fit_angles_from_gs(n, coupling, field, layers):
    """
    Try to fit circuit angles by matching the circuit state to the ground state.
    Uses the structure of the circuit to derive good initial angles.
    """
    gs = _ground_state_vector(n, coupling, field)
    dim = 1 << n
    
    # For a 3-layer circuit: Ry, CZ, Ry, CZ, Ry
    # The CZ gates create correlations between adjacent qubits
    # We can try to estimate the effective rotation angles
    
    # Approach: For each layer, try to find angles that best match the marginal
    # distributions of the ground state
    
    best_inits = []
    
    # Try different strategies for each layer
    for strat in range(5):
        thetas = np.zeros((layers, n))
        
        if strat == 0:
            # All same angle based on mean-field
            ratio = field / coupling if coupling > 0 else 1.0
            theta_mf = 2.0 * np.arctan(ratio)
            thetas[:, :] = theta_mf
        elif strat == 1:
            # Layer-dependent angles
            ratio = field / coupling if coupling > 0 else 1.0
            theta_mf = 2.0 * np.arctan(ratio)
            for l in range(layers):
                thetas[l, :] = theta_mf * (1.0 + 0.15 * (l - 1))
        elif strat == 2:
            # Alternating with layer dependence
            ratio = field / coupling if coupling > 0 else 1.0
            theta_mf = 2.0 * np.arctan(ratio)
            for l in range(layers):
                for q in range(n):
                    sign = 1.0 if q % 2 == 0 else -1.0
                    thetas[l, q] = sign * theta_mf * (1.0 + 0.1 * (l - 1))
        elif strat == 3:
            # Based on ground state marginals
            gs_real = np.real(gs) if np.abs(np.imag(gs)).max() < 1e-10 else gs
            for l in range(layers):
                for q in range(n):
                    # Estimate angle from probability of qubit being 1
                    mask = 1 << q
                    p1 = np.sum(np.abs(gs_real[(np.arange(dim) & mask) != 0])**2)
                    # For Ry(theta)|0>, P(1) = sin^2(theta/2)
                    # So theta = 2*arcsin(sqrt(p1))
                    p1 = np.clip(p1, 0, 1)
                    theta_q = 2.0 * np.arcsin(np.sqrt(p1))
                    thetas[l, q] = theta_q
        elif strat == 4:
            # Hybrid: use marginal for layer 0, adjust for later layers
            gs_real = np.real(gs) if np.abs(np.imag(gs)).max() < 1e-10 else gs
            for q in range(n):
                mask = 1 << q
                p1 = np.sum(np.abs(gs_real[(np.arange(dim) & mask) != 0])**2)
                p1 = np.clip(p1, 0, 1)
                theta_q = 2.0 * np.arcsin(np.sqrt(p1))
                thetas[0, q] = theta_q
                thetas[1, q] = theta_q * 0.8
                thetas[2, q] = theta_q * 0.6
        
        best_inits.append(thetas.flatten())
    
    return best_inits

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)
    
    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # Generate diverse initial points
    inits = _physics_informed_init(n, coupling, field, layers)
    inits += _fit_angles_from_gs(n, coupling, field, layers)
    
    # Add random inits
    num_random = 25 if n <= 4 else 15
    rng = np.random.default_rng(42)
    for i in range(num_random):
        if time.time() > deadline - 15:
            break
        if i % 3 == 0:
            x_init = rng.uniform(-np.pi, np.pi, size=(layers * n))
        elif i % 3 == 1:
            x_init = np.pi / 2 + rng.normal(0, 0.8, size=(layers * n))
        else:
            x_init = rng.normal(0, 1.0, size=(layers * n))
        inits.append(x_init)
    
    # Sort inits by objective value
    init_objs = []
    for x_init in inits:
        if time.time() > deadline - 10:
            break
        try:
            v = obj(x_init)
            init_objs.append((v, x_init))
        except Exception:
            continue
    
    init_objs.sort(key=lambda x: x[0])
    
    # Run L-BFGS-B from top starts
    max_starts = 12
    for v_init, x_init in init_objs[:max_starts]:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 50})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # Nelder-Mead polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(obj, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15,
                                   'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # One more polish if time remains
    if time.time() < deadline - 1.0 and best_val > 0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
