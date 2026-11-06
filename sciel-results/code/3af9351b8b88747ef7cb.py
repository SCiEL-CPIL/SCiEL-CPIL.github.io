"""
Optimized shallow-circuit Ising ground state approximation.
Strategy: Analytical mean-field initialization followed by local BFGS refinement.
This avoids the global search overhead and convergence issues of differential evolution
by starting the optimizer at a physically motivated point (product state approximation)
and using a fast, gradient-based local optimizer to polish the result.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Cache for exact ground energies and Hamiltonians
_HAM_CACHE = {}
_EE_CACHE = {}

def _get_hamiltonian(n, coupling):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    key = (n, coupling)
    if key in _HAM_CACHE:
        return _HAM_CACHE[key]
    
    dim = 1 << n
    
    # H = -coupling * sum Z_i Z_(i+1) - field * sum X_i
    # We store the parts separately to allow efficient energy calculation for different fields.
    # H_IZ = sum Z_i Z_(i+1)
    H_IZ = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H_IZ[state, state] += 1.0
            else:
                H_IZ[state, state] -= 1.0
                
    # H_IX = sum X_i
    H_IX = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_IX[state, flipped] = 1.0
            
    _HAM_CACHE[key] = (H_IZ, H_IX)
    return _HAM_CACHE[key]

def _compute_exact_ground_energy(n, coupling, field):
    """Compute the exact ground state energy."""
    key = (n, coupling, field)
    if key in _EE_CACHE:
        return _EE_CACHE[key]
    
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    # Exact diagonalization
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0].real
    _EE_CACHE[key] = e0
    return e0

def _apply_circuit(theta, n):
    """
    Apply the circuit to |0...0> and return the state vector.
    theta: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    layers = theta.shape[0]
    
    for l in range(layers):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            c = np.cos(theta[l, i] / 2)
            s = np.sin(theta[l, i] / 2)
            
            new_state = np.zeros_like(state)
            for idx in range(dim):
                if ((idx >> i) & 1) == 0:
                    partner = idx ^ (1 << i)
                    new_state[idx] = c * state[idx] - s * state[partner]
                    new_state[partner] = s * state[idx] + c * state[partner]
            state = new_state
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask = (1 << i) | (1 << (i + 1))
                for idx in range(dim):
                    if (idx & mask) == mask:
                        state[idx] *= -1.0
                        
    return state

def _compute_energy(theta, n, coupling, field):
    """Compute the energy of the state produced by the circuit."""
    state = _apply_circuit(theta, n)
    H_IZ, H_IX = _get_hamiltonian(n, coupling)
    H = -coupling * H_IZ - field * H_IX
    
    Hpsi = H.dot(state)
    energy = np.vdot(state, Hpsi).real
    return energy

def _mean_field_initial_guess(n, coupling, field, layers):
    """
    Generate an initial guess based on a product state approximation.
    For a product state |theta/2, theta/2, ...>, the energy is:
    E(theta) = -coupling * sum_{i=0}^{n-2} cos^2(theta/2) - field * n * sin(theta)
    
    We minimize this single-parameter function to find a good starting angle,
    then set all angles to this value.
    """
    # Minimize product state energy
    def prod_energy(theta):
        # theta is the Ry angle for all qubits in a product state
        # <Z> = cos(theta), <X> = -sin(theta) ... wait.
        # Ry(theta) |0> = cos(theta/2)|0> + sin(theta/2)|1>
        # <Z> = cos^2(theta/2) - sin^2(theta/2) = cos(theta)
        # <X> = 2 cos(theta/2) sin(theta/2) = sin(theta)
        # H = -J sum Z_i Z_{i+1} - h sum X_i
        # <H> = -J sum <Z_i><Z_{i+1}> - h sum <X_i>
        # For product state: <Z_i> = cos(theta), <X_i> = sin(theta)
        # <H> = -J (n-1) cos^2(theta) - h n sin(theta)
        
        return -coupling * (n - 1) * np.cos(theta)**2 - field * n * np.sin(theta)
    
    # Use a simple grid search or scipy minimize_scalar
    # Grid search is robust and fast for 1D
    thetas = np.linspace(-np.pi, np.pi, 1000)
    energies = np.array([prod_energy(t) for t in thetas])
    idx_min = np.argmin(energies)
    theta_opt = thetas[idx_min]
    
    # Refine with BFGS on the 1D function
    res = minimize(prod_energy, theta_opt, method='BFGS')
    if res.success:
        theta_opt = res.x[0]
        
    # Create the full parameter array
    # The circuit has 3 layers.
    # A product state is achieved if the entangling gates (CZ) don't change the state,
    # or if we can "undo" their effect, or if we just set all angles to the optimal product angle.
    # Setting all angles to the optimal product angle is a reasonable heuristic.
    # However, the CZ gates might mess up the product structure.
    # A better heuristic: 
    # Layer 3 (last): Set to optimal product angle to maximize local terms?
    # Layers 1 & 2: Set to 0? Or also optimal?
    # If we set all to optimal, we get a state close to the product state but with some entanglement from CZ.
    # Let's try setting all layers to the optimal product angle.
    
    theta = np.full((layers, n), theta_opt)
    
    # Alternative: Maybe the last layer should be different?
    # Let's stick with uniform for now, BFGS will fix it.
    
    return theta

def _objective(params, n, coupling, field, e_exact):
    """Objective function for the optimizer."""
    theta = params.reshape(3, n)
    e_circ = _compute_energy(theta, n, coupling, field)
    return e_circ - e_exact

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate ground states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Compute exact ground energy
    e_exact = _compute_exact_ground_energy(n, coupling, field)
    
    # 2. Get initial guess from mean-field
    theta_init = _mean_field_initial_guess(n, coupling, field, layers)
    x0 = theta_init.flatten()
    
    # 3. Optimization
    # Use BFGS or L-BFGS-B for local optimization.
    # This is much faster than global methods and likely to converge to the local minimum near the heuristic.
    
    best_x = x0
    best_val = _objective(x0, n, coupling, field, e_exact)
    
    try:
        res = minimize(
            _objective, 
            x0, 
            args=(n, coupling, field, e_exact),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if time.time() < deadline:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
            else:
                # If it didn't improve, keep the initial guess
                pass
        else:
            # If timeout, we might have a partial result, but minimize usually returns best found
            # Let's check if res.x is valid
            if hasattr(res, 'x') and res.x is not None:
                val_res = _objective(res.x, n, coupling, field, e_exact)
                if val_res < best_val:
                    best_val = val_res
                    best_x = res.x
                    
    except Exception:
        # Fallback to initial guess if anything goes wrong
        pass
    
    # 4. Final Polish with Nelder-Mead if time permits and if we want to be extra sure
    # L-BFGS-B is usually sufficient. Let's skip to save time and complexity.
    
    # Convert back to (layers, n) array
    theta_opt = best_x.reshape(layers, n)
    
    return theta_opt

# EVOLVE-BLOCK-END
