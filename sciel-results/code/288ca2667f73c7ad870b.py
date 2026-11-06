"""Improved optimizer for shallow-circuit Ising ground states with MPS-inspired init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_psi0_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = eigvals[0]
    psi0 = eigvecs[:, 0]
    _e0_cache[key] = (e0, psi0)
    return e0, psi0

def _get_indices(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    ry_idx = []
    for q in range(n):
        mask = 1 << q
        idx = np.arange(dim)
        mask_bit = (idx & mask) == 0
        idx_0 = idx[mask_bit]
        idx_1 = idx_0 + mask
        ry_idx.append((idx_0, idx_1))
    cz_idx = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        idx = np.arange(dim)
        both_set = ((idx & mask1) != 0) & ((idx & mask2) != 0)
        cz_idx.append(both_set)
    _idx_cache[key] = (ry_idx, cz_idx)
    return _idx_cache[key]

def _circuit_state_fast(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    ry_idx, cz_idx = _get_indices(n)

    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = ry_idx[q]
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1

        if layer < layers - 1:
            for i in range(n - 1):
                both_set = cz_idx[i]
                state[both_set] *= -1.0

    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _mps_informed_init(n, coupling, field, layers, psi0):
    """Fit Ry angles to match single-qubit reduced density matrices of ground state."""
    dim = 1 << n
    thetas = np.zeros((layers, n))
    
    # Compute single-qubit reduced density matrices
    for q in range(n):
        rho_q = np.zeros((2, 2), dtype=np.complex128)
        for state in range(dim):
            bit = (state >> q) & 1
            for state2 in range(dim):
                bit2 = (state2 >> q) & 1
                # Trace over all other qubits
                # Simplified: use expectation values
                pass
        
        # Use expectation values directly
        # <Z_q> and <X_q> from psi0
        Zq = np.zeros((dim, dim))
        Xq = np.zeros((dim, dim))
        for state in range(dim):
            bit = (state >> q) & 1
            Zq[state, state] = 1.0 if bit == 0 else -1.0
        for state in range(dim):
            flipped = state ^ (1 << q)
            Xq[state, flipped] = 1.0
            Xq[flipped, state] = 1.0
        
        ez = np.real(np.conj(psi0) @ Zq @ psi0)
        ex = np.real(np.conj(psi0) @ Xq @ psi0)
        
        # Match to Ry(theta)|0> state: <Z> = cos(theta), <X> = sin(theta)
        # Find theta that best matches
        theta = np.arctan2(ex, ez)
        
        # Distribute across layers (equal split for simplicity)
        per_layer = theta / layers
        for layer in range(layers):
            thetas[layer, q] = per_layer
    
    return thetas

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0
    
    e0, psi0 = _exact_ground(n, coupling, field)
    
    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    best_x = np.zeros(layers * n)
    best_val = obj(best_x)
    
    # MPS-inspired initialization
    try:
        mps_thetas = _mps_informed_init(n, coupling, field, layers, psi0)
        mps_x = mps_thetas.flatten()
        mps_val = obj(mps_x)
        if mps_val < best_val:
            best_val = mps_val
            best_x = mps_x
    except Exception:
        pass
    
    # Determine number of restarts
    if n <= 4:
        num_restarts = 25
    else:
        num_restarts = 12

    # Generate initial points
    inits = []
    
    # 1. Zero
    inits.append(np.zeros(layers * n))
    
    # 2. Pi/2
    inits.append(np.full(layers * n, np.pi/2))
    
    # 3. Small angles
    inits.append(np.full(layers * n, 0.1))
    inits.append(np.full(layers * n, -0.1))
    
    # 4. MPS-inspired if not already best
    try:
        mps_thetas2 = _mps_informed_init(n, coupling, field, layers, psi0)
        inits.append(mps_thetas2.flatten())
    except Exception:
        pass
    
    # 5. Physics-informed random
    if field > 1.0:
        center = np.pi/2
        spread = 0.8
    else:
        center = 0.0
        spread = 0.8

    for i in range(num_restarts):
        if i % 3 == 0:
            x_init = np.random.uniform(-np.pi, np.pi, size=(layers * n))
        elif i % 3 == 1:
            x_init = center + np.random.normal(0, spread, size=(layers * n))
        else:
            # Layer-wise variation
            x_init = np.array([center + np.random.normal(0, spread) 
                             for _ in range(layers * n)])
        
        inits.append(x_init)
        
        if time.time() > deadline - 15:
            break

    # Run optimization from each start
    for x_init in inits:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(obj, x_init, method='L-BFGS-B',
                           options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11})
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue

    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # One more ultra-precise pass if time remains
    if time.time() < deadline - 2.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
