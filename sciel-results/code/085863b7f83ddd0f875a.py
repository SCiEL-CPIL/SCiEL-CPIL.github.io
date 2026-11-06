"""Tensor-network-inspired initialization for shallow Ising circuit."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain.
    
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    # Build Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        term = I.copy()
        for j in range(n):
            if j == i or j == i + 1:
                term = np.kron(term, Z)
            else:
                term = np.kron(term, I)
        H += -coupling * term
    
    # X terms
    for i in range(n):
        term = I.copy()
        for j in range(n):
            if j == i:
                term = np.kron(term, X)
            else:
                term = np.kron(term, I)
        H += -field * term
    
    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    ground_state = eigvecs[:, ground_idx]
    ground_energy = eigvals[ground_idx]
    
    return ground_state, ground_energy

def _simulate_circuit(angles, n, layers):
    """Simulate the Ry/CZ circuit starting from |0>^n."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    def apply_cz(state, i, j):
        new_state = state.copy()
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_j = (idx >> j) & 1
            if bit_i == 1 and bit_j == 1:
                new_state[idx] = -state[idx]
        return new_state

    def apply_ry(state, i, theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        new_state = np.zeros_like(state)
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                partner_idx = idx | (1 << i)
                amp_0 = state[idx]
                amp_1 = state[partner_idx]
                
                new_state[idx] = c * amp_0 + s * amp_1
                new_state[partner_idx] = -s * amp_0 + c * amp_1
        return new_state

    for layer in range(layers):
        for i in range(n):
            state = apply_ry(state, i, angles[layer, i])
        
        if layer < layers - 1:
            for i in range(n - 1):
                state = apply_cz(state, i, i + 1)
                
    return state

def _energy(state, n, coupling, field):
    """Compute expectation value of H for a given state."""
    dim = 2 ** n
    energy = 0.0
    
    for i in range(n - 1):
        exp_zz = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_j = (idx >> i + 1) & 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            prob = np.abs(state[idx])**2
            exp_zz += prob * z_i * z_j
        energy += -coupling * exp_zz
        
    for i in range(n):
        exp_x = 0.0
        for idx in range(dim):
            partner_idx = idx ^ (1 << i)
            exp_x += state[partner_idx] * np.conj(state[idx])
        exp_x = np.real(exp_x)
        energy += -field * exp_x
        
    return energy

def _fit_angles_via_mps(n, coupling, field, layers):
    """Fit circuit angles using MPS-inspired initialization and L-BFGS-B."""
    ground_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    def objective(angles_flat):
        angles_reshaped = angles_flat.reshape(layers, n)
        state = _simulate_circuit(angles_reshaped, n, layers)
        return _energy(state, n, coupling, field)
    
    theta_inits = []
    theta_inits.append(np.zeros((layers, n)))
    
    if coupling != 0:
        base_angle = np.arctan2(field, coupling)
    else:
        base_angle = np.pi / 2
    theta_inits.append(np.full((layers, n), base_angle / layers))
    
    local_theta = np.zeros(n)
    dim = 2 ** n
    for i in range(n):
        exp_z = 0.0
        exp_x = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            z_val = 1 if bit_i == 0 else -1
            prob = np.abs(ground_state[idx])**2
            exp_z += prob * z_val
            
            partner_idx = idx ^ (1 << i)
            exp_x += ground_state[partner_idx] * np.conj(ground_state[idx])
        exp_x = np.real(exp_x)
        local_theta[i] = np.arctan2(exp_x, exp_z)
        
    theta_mps_uniform = np.zeros((layers, n))
    for l in range(layers):
        theta_mps_uniform[l, :] = local_theta / layers
    theta_inits.append(theta_mps_uniform)
    
    theta_mps_first = np.zeros((layers, n))
    theta_mps_first[0, :] = local_theta
    theta_inits.append(theta_mps_first)
    
    theta_mps_last = np.zeros((layers, n))
    theta_mps_last[-1, :] = local_theta
    theta_inits.append(theta_mps_last)
    
    np.random.seed(42)
    for _ in range(2):
        theta_inits.append(np.random.uniform(-np.pi/2, np.pi/2, (layers, n)))
    
    best_energy = np.inf
    best_angles = None
    
    for init in theta_inits:
        x0 = init.flatten()
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12})
            
            x_refined = result.x
            res_nm = minimize(objective, x_refined, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            e = objective(res_nm.x.flatten())
            if e < best_energy:
                best_energy = e
                best_angles = res_nm.x.reshape(layers, n)
        except Exception:
            continue
            
    if best_angles is None:
        best_angles = np.zeros((layers, n))
        
    return best_angles

def angles(n, coupling, field, layers):
    """Return circuit parameters (layers, n) in radians."""
    try:
        best_angles = _fit_angles_via_mps(n, coupling, field, layers)
    except Exception:
        if coupling != 0:
            base_angle = np.arctan2(field, coupling)
        else:
            base_angle = np.pi / 2
        best_angles = np.full((layers, n), base_angle / layers)
    
    return best_angles
# EVOLVE-BLOCK-END
