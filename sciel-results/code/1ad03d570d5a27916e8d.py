import numpy as np
from scipy.optimize import minimize

# Cache for optimized parameters
_optimization_cache = {}

def _get_hamiltonian(n, coupling, field):
    """
    Construct the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    for state in range(dim):
        z_energy = 0
        for i in range(n - 1):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            z_energy += z_i * z_ip1
        H[state, state] += -coupling * z_energy
        
        for i in range(n):
            flipped_state = state ^ (1 << i)
            H[state, flipped_state] += -field

    return H

def _apply_circuit_vec(thetas, n, cz_masks):
    """
    Apply the circuit to the initial state |0...0> using vectorized operations.
    thetas: array of shape (layers, n)
    cz_masks: list of masks for CZ operations
    """
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    
    layers = thetas.shape[0]
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            # Reshape to isolate qubit i
            # Shape: (2**i, 2, 2**(n-i-1))
            shape = [2**i, 2, 2**(n-i-1)]
            psi_reshaped = psi.reshape(shape)
            
            p0 = psi_reshaped[:, 0, :]
            p1 = psi_reshaped[:, 1, :]
            
            new_p0 = c * p0 - s * p1
            new_p1 = s * p0 + c * p1
            
            new_psi_reshaped = np.stack([new_p0, new_p1], axis=1)
            psi = new_psi_reshaped.reshape(-1)
            
        # Apply CZ if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                psi *= cz_masks[i]
                
    return psi

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _optimization_cache:
        return _optimization_cache[key]
        
    H = _get_hamiltonian(n, coupling, field)
    
    # Get exact ground state
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0]
    
    # Precompute CZ masks
    cz_masks = []
    for i in range(n - 1):
        mask = np.ones(2**n, dtype=complex)
        for state in range(2**n):
            if ((state >> i) & 1) and ((state >> (i+1)) & 1):
                mask[state] = -1
        cz_masks.append(mask)
        
    def objective(x):
        thetas = x.reshape((layers, n))
        psi = _apply_circuit_vec(thetas, n, cz_masks)
        # Fidelity = |<psi | ground_state>|^2
        # We want to minimize 1 - Fidelity
        overlap = np.vdot(psi, ground_state)
        fidelity = np.abs(overlap)**2
        return 1.0 - fidelity
        
    # Multi-start optimization
    best_result = None
    best_val = np.inf
    
    # Define some initial guesses
    starts = []
    starts.append(np.zeros((layers, n)))
    
    init_theta = np.zeros((layers, n))
    if field < 1.0:
        init_theta[-1] = np.pi / 2
    else:
        init_theta[-1] = 0.0
        init_theta[0] = np.pi / 4
    starts.append(init_theta)
    
    # Add random starts
    np.random.seed(42)
    for _ in range(5):
        starts.append(np.random.uniform(0, np.pi, (layers, n)))
        
    for x0 in starts:
        res = minimize(objective, x0.flatten(), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x.reshape((layers, n))
            
    optimal_thetas = best_result
    
    # Store in cache
    _optimization_cache[key] = optimal_thetas
    
    return optimal_thetas
