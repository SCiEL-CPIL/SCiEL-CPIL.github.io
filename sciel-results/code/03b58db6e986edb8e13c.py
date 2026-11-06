import numpy as np
import scipy.optimize as opt
import time

# Cache for optimized angles
_cache = {}

def _apply_ry(state, i, theta, n):
    """
    Apply Ry(theta) to qubit i of the state vector.
    Uses vectorized operations for speed.
    """
    dim = 2 ** n
    # State is indexed by binary number b_{n-1}...b_0.
    # Qubit i corresponds to bit i.
    # We can reshape the state into a tensor of shape (2, 2, ..., 2)
    # where the axes correspond to qubits n-1, n-2, ..., 0.
    # So qubit i corresponds to axis (n-1-i).
    
    # Reshape to (2,)*n
    state_tensor = state.reshape([2]*n)
    
    # Move the axis corresponding to qubit i to the front
    # Axis index in tensor: n-1-i
    target_axis = n - 1 - i
    state_tensor = np.moveaxis(state_tensor, target_axis, 0)
    
    # Apply Ry rotation
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    # Ry = [[c, -s], [s, c]]
    # new_0 = c * old_0 - s * old_1
    # new_1 = s * old_0 + c * old_1
    
    new_state_tensor = np.zeros_like(state_tensor)
    new_state_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
    new_state_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
    
    # Move axis back
    new_state_tensor = np.moveaxis(new_state_tensor, 0, target_axis)
    
    return new_state_tensor.reshape(-1)

def _apply_cz(state, i, j, n):
    """
    Apply CZ gate between qubit i and j.
    CZ flips the sign of the |11> component.
    """
    dim = 2 ** n
    mask_i = 1 << i
    mask_j = 1 << j
    
    # Create a mask for indices where both bits i and j are 1
    # We can do this efficiently by iterating or using boolean indexing
    # For small n, a simple loop or vectorized bit check is fine.
    indices = np.arange(dim)
    # Check if bit i is set AND bit j is set
    mask = (indices & mask_i) != 0
    mask &= (indices & mask_j) != 0
    
    state[mask] *= -1
    return state

def _circuit_energy(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value for the given circuit parameters.
    params: 1D array of angles (layers * n)
    """
    thetas = params.reshape(layers, n)
    
    # Initial state |0>^n
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    # Apply circuit
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = _apply_ry(state, i, thetas[l, i], n)
        
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    # Calculate Energy E = <psi| H |psi>
    # H = -coupling * sum(Z_i Z_{i+1}) - field * sum(X_i)
    
    # Term 1: Z Z interactions (Diagonal)
    E_zz = 0.0
    probs = np.abs(state)**2
    for idx in range(2**n):
        p = probs[idx]
        if p == 0: continue
        zz_val = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            zz_val += z_i * z_ip1
        E_zz += p * zz_val
    
    E_1 = -coupling * E_zz
    
    # Term 2: X interactions (Off-diagonal)
    # <psi| X_i |psi> = sum_k psi^*_k psi_{k^flip}
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        # Vectorized calculation of <psi| X_i |psi>
        # We need sum_{k} psi^*_k psi_{k^mask}
        # This can be done by flipping the bits of the indices
        indices = np.arange(2**n)
        flipped_indices = indices ^ mask
        E_x += np.sum(np.conj(state) * state[flipped_indices]).real
    
    E_2 = -field * E_x
    
    total_energy = E_1 + E_2
    return total_energy

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state of the Hamiltonian.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    
    # Build Hamiltonian matrix
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for idx in range(dim):
        zz_val = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            zz_val += z_i * z_ip1
        H[idx, idx] += -coupling * zz_val
        
    # X terms
    for i in range(n):
        mask = 1 << i
        for idx in range(dim):
            flipped = idx ^ mask
            H[idx, flipped] += -field
            # Note: H is Hermitian, H[flipped, idx] will be set in another iteration or symmetrically
            # Actually, X is real and symmetric, so H[idx, flipped] and H[flipped, idx] both get -field
            # The loop covers all idx, so H[idx, flipped] is set.
            # When idx becomes 'flipped', it sets H[flipped, idx].
            # So this is correct.
            
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Ground state is the eigenvector with the smallest eigenvalue
    ground_state = eigvecs[:, 0]
    ground_energy = eigvals[0]
    
    return ground_state, ground_energy

def _get_initial_guess(n, coupling, field, layers):
    """
    Generate initial guess for angles based on mean-field / local magnetization.
    """
    # Get exact ground state
    try:
        gs, e0 = _get_exact_ground_state(n, coupling, field)
        
        # Calculate local magnetizations <Z_i>
        # Z_i eigenvalues: +1 for |0>, -1 for |1>
        # <Z_i> = sum_k |psi_k|^2 * z_i(k)
        mags = np.zeros(n)
        for i in range(n):
            mask = 1 << i
            for idx in range(2**n):
                p = np.abs(gs[idx])**2
                bit = (idx >> i) & 1
                z_val = 1 if bit == 0 else -1
                mags[i] += p * z_val
                
        # Initial angles for layer 0
        # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
        # <Z> = cos^2(theta/2) - sin^2(theta/2) = cos(theta)
        # So we want cos(theta_0) = mags[i]
        # theta_0 = arccos(mags[i])
        # Note: This is only for the first layer. Subsequent layers start from 0 or previous best.
        
        init_thetas = np.zeros((layers, n))
        for i in range(n):
            # Clip to [-1, 1] for numerical safety
            m = np.clip(mags[i], -1.0, 1.0)
            theta = np.arccos(m)
            init_thetas[0, i] = theta
            
        return init_thetas.flatten()
        
    except Exception:
        # Fallback to random if exact diag fails (unlikely for n<=6)
        return np.random.uniform(0, 2*np.pi, n*layers)

def _optimize(n, coupling, field, layers, timeout=140.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # Get physically motivated initial guess
    x0 = _get_initial_guess(n, coupling, field, layers)
    
    # Bounds: [0, 2*pi]
    bounds = [(0, 2 * np.pi)] * n_params
    
    def objective(params):
        if time.time() - start_time > timeout:
            raise TimeoutError("Optimization timeout")
        return _circuit_energy(params, n, coupling, field, layers)
    
    best_params = x0
    best_energy = objective(x0)
    
    # Use L-BFGS-B with tight tolerances
    # Since the initial guess is good, L-BFGS-B should converge quickly.
    try:
        res = opt.minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 2000,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        if res.fun < best_energy:
            best_params = res.x
            best_energy = res.fun
            
    except TimeoutError:
        pass
    except Exception:
        # If L-BFGS-B fails, try Nelder-Mead as fallback
        try:
            res_nm = opt.minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < best_energy:
                best_params = res_nm.x
                best_energy = res_nm.fun
        except TimeoutError:
            pass
        
    # Final check for time
    if time.time() - start_time > timeout - 5:
        pass # Just return current best
        
    # Reshape to (layers, n)
    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    # Ensure types are consistent
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
