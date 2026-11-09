"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import time

def _ry(theta):
    """2x2 Ry rotation matrix."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

def _cz():
    """2x2 CZ gate."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def _build_state(theta, n, layers):
    """Build the circuit state from angles."""
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0...0>
    
    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            c = np.cos(theta[layer, i] / 2)
            s = np.sin(theta[layer, i] / 2)
            new_state = np.zeros_like(state)
            for basis in range(dim):
                in_bit = (basis >> (n - 1 - i)) & 1
                # Contribution to basis where bit i = 0
                basis_out0 = basis & ~(1 << (n - 1 - i))
                if in_bit == 0:
                    new_state[basis_out0] += c * state[basis]
                    new_state[basis_out0 | (1 << (n - 1 - i))] += s * state[basis]
                else:
                    new_state[basis_out0] += -s * state[basis]
                    new_state[basis_out0 | (1 << (n - 1 - i))] += c * state[basis]
            state = new_state
        
        # Apply CZ gates between adjacent pairs (if not last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                for basis in range(dim):
                    bit_i = (basis >> (n - 1 - i)) & 1
                    bit_j = (basis >> (n - 1 - (i + 1))) & 1
                    if bit_i == 1 and bit_j == 1:
                        state[basis] *= -1
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute <H> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    dim = 1 << n
    energy = 0.0
    
    # ZZ terms: diagonal in computational basis
    for i in range(n - 1):
        for basis in range(dim):
            bit_i = (basis >> (n - 1 - i)) & 1
            bit_j = (basis >> (n - 1 - (i + 1))) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_j = 1.0 if bit_j == 0 else -1.0
            prob = np.abs(state[basis]) ** 2
            energy += -coupling * z_i * z_j * prob
    
    # X terms: off-diagonal
    for i in range(n):
        for basis in range(dim):
            basis_flipped = basis ^ (1 << (n - 1 - i))
            if basis < basis_flipped:  # Count each pair once
                amp = state[basis]
                amp_flip = state[basis_flipped]
                # <state|X_i|state> = state[basis_flipped]*conj(state[basis]) + state[basis]*conj(state[basis_flipped])
                # = 2*Re(state[basis_flipped]*conj(state[basis]))
                cross = state[basis_flipped] * np.conj(state[basis])
                energy += -field * 2.0 * cross.real
    
    return energy

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for basis in range(dim):
            bit_i = (basis >> (n - 1 - i)) & 1
            bit_j = (basis >> (n - 1 - (i + 1))) & 1
            z_i = 1 if bit_i == 0 else -1
            z_j = 1 if bit_j == 0 else -1
            H[basis, basis] += -coupling * z_i * z_j
    
    # X terms
    for i in range(n):
        for basis in range(dim):
            basis_flipped = basis ^ (1 << (n - 1 - i))
            H[basis, basis_flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, coupling, field, layers):
    """Objective function for optimization."""
    theta = params.reshape(layers, n)
    state = _build_state(theta, n, layers)
    energy = _compute_energy(state, n, coupling, field)
    return energy

def _physics_init(n, field, layers):
    """Physics-based initialization."""
    theta = np.zeros((layers, n))
    # For small field, want angles near 0 (ferromagnetic)
    # For large field, want angles near pi/2 (paramagnetic)
    # Interpolate
    frac = min(field / 2.0, 1.0)
    base_angle = frac * np.pi / 2
    theta[:, :] = base_angle
    return theta

def angles(n, coupling, field, layers):
    """Optimize circuit angles to approximate Ising ground state."""
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Physics-based initialization
    theta0 = _physics_init(n, field, layers)
    params0 = theta0.flatten()
    
    # Compute initial energy
    state0 = _build_state(theta0, n, layers)
    energy0 = _compute_energy(state0, n, coupling, field)
    
    best_params = params0.copy()
    best_energy = energy0
    
    # Try optimization
    try:
        from scipy.optimize import minimize
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(_objective, params0, args=(n, coupling, field, layers),
                         method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            theta_best = best_params.reshape(layers, n)
            return theta_best
        
        # Try a few restarts if time permits
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > deadline - 10:
                break
            
            # Perturb initial guess
            np.random.seed(r * 100 + n * 10 + int(field * 100))
            perturbation = np.random.normal(0, 0.3, size=params0.shape)
            params_r = params0 + perturbation
            
            result_r = minimize(_objective, params_r, args=(n, coupling, field, layers),
                               method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result_r.fun < best_energy:
                best_energy = result_r.fun
                best_params = result_r.x.copy()
        
    except Exception as e:
        pass
    
    theta_best = best_params.reshape(layers, n)
    return theta_best
# EVOLVE-BLOCK-END
