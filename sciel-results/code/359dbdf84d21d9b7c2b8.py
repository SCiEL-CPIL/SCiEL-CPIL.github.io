import numpy as np
import scipy.optimize as opt
import time

# Global cache for results
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    idx = np.arange(dim)
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = -coupling * z_i * z_ip1
        H[idx, idx] += zz_val
        
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        H[idx, idx_flipped] += -field
        H[idx_flipped, idx] += -field
        
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

def _circuit_energy_and_grad(params, n, coupling, field, layers):
    """
    Calculate the energy expectation value and its gradient for the given circuit parameters.
    Returns (energy, gradient).
    """
    thetas = params.reshape(layers, n)
    
    # Start with |0...0>
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    idx = np.arange(2**n)
    
    states_before_ry = [None] * layers
    current_state = state.copy()
    
    for l in range(layers):
        states_before_ry[l] = current_state.copy()
        
        # Apply Ry(theta) to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            mask_i = 1 << i
            idx_0 = idx[(idx & mask_i) == 0]
            idx_1 = idx[(idx & mask_i) != 0]
            
            amp_0 = current_state[idx_0].copy()
            amp_1 = current_state[idx_1].copy()
            
            new_amp_0 = c * amp_0 - s * amp_1
            new_amp_1 = s * amp_0 + c * amp_1
            
            current_state[idx_0] = new_amp_0
            current_state[idx_1] = new_amp_1
            
        # Apply CZ between adjacent pairs if not last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                current_state[cond] *= -1
                
    final_state = current_state
    
    probs = np.abs(final_state) ** 2
    
    z_vals = np.zeros((n, 2**n))
    zz_vals = np.zeros((n-1, 2**n))
    for i in range(n):
        mask_i = 1 << i
        bit_i = (idx >> i) & 1
        z_vals[i] = 1 - 2 * bit_i
        
    for i in range(n - 1):
        zz_vals[i] = z_vals[i] * z_vals[i+1]
        
    E_zz = np.sum(probs * zz_vals)
    E_x = 0.0
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        term = np.sum(np.conj(final_state) * final_state[idx_flipped])
        E_x += np.real(term)
        
    total_energy = -coupling * E_zz - field * E_x
    
    # Gradient Calculation (Adjoint)
    H_psi = np.zeros(2**n, dtype=complex)
    zz_diag = np.zeros(2**n)
    for i in range(n-1):
        zz_diag += zz_vals[i]
    H_psi -= coupling * zz_diag * final_state
    
    for i in range(n):
        mask = 1 << i
        idx_flipped = idx ^ mask
        H_psi[idx] -= field * final_state[idx_flipped]
        
    phi = H_psi.copy()
    grads = np.zeros_like(thetas)
    
    for l in range(layers - 1, -1, -1):
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
                phi[cond] *= -1
                
        psi_before = states_before_ry[l]
        
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            mask_i = 1 << i
            idx_0 = idx[(idx & mask_i) == 0]
            idx_1 = idx[(idx & mask_i) != 0]
            
            phi_0 = phi[idx_0]
            phi_1 = phi[idx_1]
            
            psi_0 = psi_before[idx_0]
            psi_1 = psi_before[idx_1]
            
            dpsi_0 = 0.5 * (-s * psi_0 - c * psi_1)
            dpsi_1 = 0.5 * (c * psi_0 - s * psi_1)
            
            inner = np.sum(np.conj(phi_0) * dpsi_0) + np.sum(np.conj(phi_1) * dpsi_1)
            
            grads[l, i] = 2 * np.real(inner)
            
        new_phi_0 = c * phi_0 + s * phi_1
        new_phi_1 = -s * phi_0 + c * phi_1
        
        phi[idx_0] = new_phi_0
        phi[idx_1] = new_phi_1
        
    return total_energy, grads.reshape(-1)

def _physics_informed_init(n, coupling, field, layers):
    """
    Generate initial parameters based on Mean Field / Ising approximation.
    We assume the ground state is close to a product state rotated by a common angle,
    or we fit the local magnetization to the exact ground state.
    """
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        dim = 2 ** n
        idx = np.arange(dim)
        probs = np.abs(ground_state) ** 2
        
        # Calculate local magnetizations <Z_i> from exact ground state
        mags = np.zeros(n)
        for i in range(n):
            mask_i = 1 << i
            bit_vals = (idx >> i) & 1
            z_vals = 1 - 2 * bit_vals
            mags[i] = np.sum(probs * z_vals)
            
        # Mean field approximation:
        # In the strong coupling limit, <Z_i> is large.
        # In the transverse field limit, <Z_i> is 0.
        # The circuit starts at |0> (Z=1).
        # Ry(theta) rotates Z towards X.
        # If we assume the state is roughly a product state |psi_i> = cos(theta/2)|0> + sin(theta/2)|1>,
        # then <Z_i> = cos(theta).
        # So theta_i = arccos(<Z_i>).
        
        # However, CZ gates introduce correlations.
        # A simple heuristic: distribute the rotation across layers.
        # If layers=3, maybe theta_0 = 0, theta_1 = theta_mag/2, theta_2 = theta_mag/2?
        # Or theta_0 = theta_mag/3, theta_1 = theta_mag/3, theta_2 = theta_mag/3?
        
        # Let's try a few variations and pick the best energy quickly.
        theta_mag = np.arccos(np.clip(mags, -1, 1))
        
        candidates = []
        
        # Candidate 1: All rotation in last layer
        p1 = np.zeros((layers, n))
        p1[-1, :] = theta_mag
        candidates.append(p1)
        
        # Candidate 2: Distributed evenly
        p2 = np.zeros((layers, n))
        for l in range(layers):
            p2[l, :] = theta_mag / layers
        candidates.append(p2)
        
        # Candidate 3: Last two layers
        p3 = np.zeros((layers, n))
        p3[-1, :] = theta_mag / 2
        p3[-2, :] = theta_mag / 2
        candidates.append(p3)
        
        # Candidate 4: Based on "effective" field
        # If field is small, theta is small. If field is large, theta is large.
        # This is already captured by mags.
        
        # Evaluate candidates
        best_p = candidates[0]
        best_e = np.inf
        for p in candidates:
            e, _ = _circuit_energy_and_grad(p.flatten(), n, coupling, field, layers)
            if e < best_e:
                best_e = e
                best_p = p
                
        return best_p.flatten()
        
    except Exception:
        return np.random.uniform(0, 2*np.pi, n * layers)

def _optimize(n, coupling, field, layers, timeout=145.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    # 1. Physics-informed initialization
    init = _physics_informed_init(n, coupling, field, layers)
    
    best_params = init
    best_energy, _ = _circuit_energy_and_grad(best_params, n, coupling, field, layers)
    
    def objective_and_grad(params):
        return _circuit_energy_and_grad(params, n, coupling, field, layers)
        
    # 2. Local Refinement with L-BFGS-B
    # We trust the physics init to be in the basin of attraction, so we can be aggressive with iterations.
    try:
        res = opt.minimize(objective_and_grad, init, jac=True, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # 3. Multi-start refinement if time permits
    # Generate a few perturbations of the best solution
    current_time = time.time()
    if current_time - start_time < timeout - 20:
        for _ in range(5):
            if time.time() - start_time > timeout - 10:
                break
            # Small random perturbation
            perturb = best_params + np.random.normal(0, 0.1, n_params)
            perturb = np.clip(perturb, 0, 2*np.pi)
            try:
                res_p = opt.minimize(objective_and_grad, perturb, jac=True, method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                if res_p.fun < best_energy:
                    best_energy = res_p.fun
                    best_params = res_p.x
            except Exception:
                continue
                
    # 4. Final Polish
    current_time = time.time()
    if current_time - start_time < timeout - 2:
        try:
            res_lbfgs = opt.minimize(objective_and_grad, best_params, jac=True, method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
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
