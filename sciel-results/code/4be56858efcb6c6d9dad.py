import numpy as np
import scipy.optimize as opt
import time

# Cache for optimized angles
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_ip1 = (k >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[k, k] += -coupling * z_i * z_ip1
            
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        for k in range(dim):
            k_flipped = k ^ mask
            H[k, k_flipped] += -field
            H[k_flipped, k] += -field
            
    return H

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state and energy.
    """
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]
    return ground_state, ground_energy

def _compute_local_magnetizations(state, n):
    """
    Compute <Z_i> for each qubit i in the given state.
    """
    dim = 2 ** n
    probs = np.abs(state) ** 2
    idx = np.arange(dim)
    
    mags = np.zeros(n)
    for i in range(n):
        mask = 1 << i
        bit_vals = (idx >> i) & 1
        z_vals = 1 - 2 * bit_vals
        mags[i] = np.sum(probs * z_vals)
    return mags

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    """
    thetas = params.reshape(layers, n)
    
    state = np.zeros([2] * n, dtype=complex)
    state[tuple([0] * n)] = 1.0
    
    for l in range(layers):
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            axis = n - 1 - i
            state = np.moveaxis(state, axis, -1)
            
            old_0 = state[..., 0]
            old_1 = state[..., 1]
            
            new_0 = c * old_0 - s * old_1
            new_1 = s * old_0 + c * old_1
            
            new_state_slice = np.empty_like(state)
            new_state_slice[..., 0] = new_0
            new_state_slice[..., 1] = new_1
            
            state = np.moveaxis(new_state_slice, -1, axis)
        
        if l < layers - 1:
            flat_state = state.reshape(-1)
            dim = 2 ** n
            idx = np.arange(dim)
            
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                flat_state[cond] *= -1
            
            state = flat_state.reshape([2] * n)
            
    flat_state = state.reshape(-1)
    dim = 2 ** n
    idx = np.arange(dim)
    probs = np.abs(flat_state) ** 2
    
    E_zz = 0.0
    for i in range(n - 1):
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        E_zz += np.sum(probs * zz_val)
        
    E_1 = -coupling * E_zz
    
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        term = np.sum(np.conj(flat_state) * flat_state[idx_flipped])
        E_x += np.real(term)
        
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    return total_energy

def _optimize(n, coupling, field, layers, timeout=115.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        mags = _compute_local_magnetizations(ground_state, n)
        theta_last = np.arccos(np.clip(mags, -1, 1))
        
        init1 = np.zeros(n_params)
        init1[-n:] = theta_last
        
        init2 = np.tile(theta_last, layers)
        
        init3 = np.zeros(n_params)
        
        init4 = np.random.uniform(0, 2*np.pi, n_params)
        
        inits = [init1, init2, init3, init4]
        
    except Exception:
        inits = [np.random.uniform(0, 2*np.pi, n_params)]
        
    best_params = inits[0]
    best_energy = _circuit_energy(best_params, n, coupling, field, layers)
    
    def objective(params):
        return _circuit_energy(params, n, coupling, field, layers)
        
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = opt.minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    current_time = time.time()
    if current_time - start_time < timeout - 5:
        try:
            res_lbfgs = opt.minimize(
                objective, 
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_energy:
                best_energy = res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass

    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
