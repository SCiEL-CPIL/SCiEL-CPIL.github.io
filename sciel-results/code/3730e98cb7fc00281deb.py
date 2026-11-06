import numpy as np
from scipy.optimize import minimize

def _simulate_energy(state, n, coupling, field):
    """Compute <H> for a given state vector."""
    probs = np.abs(state)**2
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        z_i = np.where((np.arange(len(state)) & mask_i) == 0, 1.0, -1.0)
        z_j = np.where((np.arange(len(state)) & mask_j) == 0, 1.0, -1.0)
        exp_zz = np.sum(probs * z_i * z_j)
        energy += -coupling * exp_zz
        
    # X terms
    for i in range(n):
        mask = 1 << i
        indices_0 = np.where((np.arange(len(state)) & mask) == 0)[0]
        indices_1 = indices_0 | mask
        exp_x = 2.0 * np.real(np.sum(np.conj(state[indices_0]) * state[indices_1]))
        energy += -field * exp_x
        
    return energy

def _run_circuit(params, n, coupling, field, layers):
    """
    Run the circuit and return the final state vector.
    params: 1D array of length layers*n
    """
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    
    # Reshape params to (layers, n)
    angles = params.reshape((layers, n))
    
    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = angles[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            mask = 1 << i
            
            indices_0 = np.where((np.arange(len(state)) & mask) == 0)[0]
            indices_1 = indices_0 | mask
            
            v0 = state[indices_0].copy()
            v1 = state[indices_1].copy()
            
            new_state = state.copy()
            new_state[indices_0] = c * v0 - s * v1
            new_state[indices_1] = s * v0 + c * v1
            state = new_state
            
        # Apply CZ between adjacent pairs if not the last layer
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_j = 1 << (i + 1)
                indices = np.where(((np.arange(len(state)) & mask_i) != 0) & ((np.arange(len(state)) & mask_j) != 0))[0]
                state = state.copy()
                state[indices] *= -1.0
                
    return state

def _energy_and_gradient(params, n, coupling, field, layers):
    """
    Compute energy and analytic gradient using Parameter Shift Rule.
    """
    # Base energy
    state_base = _run_circuit(params, n, coupling, field, layers)
    energy = _simulate_energy(state_base, n, coupling, field)
    
    # Gradient
    grad = np.zeros_like(params)
    n_params = len(params)
    
    for k in range(n_params):
        # Shift + pi/2
        params_plus = params.copy()
        params_plus[k] += np.pi / 2
        state_plus = _run_circuit(params_plus, n, coupling, field, layers)
        energy_plus = _simulate_energy(state_plus, n, coupling, field)
        
        # Shift - pi/2
        params_minus = params.copy()
        params_minus[k] -= np.pi / 2
        state_minus = _run_circuit(params_minus, n, coupling, field, layers)
        energy_minus = _simulate_energy(state_minus, n, coupling, field)
        
        grad[k] = (energy_plus - energy_minus) / 2.0
        
    return energy, grad

def angles(n, coupling, field, layers):
    """
    Optimize parameters using L-BFGS-B with analytic gradients (Parameter Shift Rule).
    """
    n_params = layers * n
    
    def objective(params):
        e, _ = _energy_and_gradient(params, n, coupling, field, layers)
        return e

    def gradient(params):
        _, g = _energy_and_gradient(params, n, coupling, field, layers)
        return g

    # Multi-start
    best_energy = np.inf
    best_params = None
    
    # Deterministic seeds based on field and n to ensure reproducibility
    np.random.seed(int(field * 1000) + n)
    
    # Initial guesses
    guesses = [
        np.zeros(n_params),
        np.full(n_params, np.pi / 2),
        np.full(n_params, np.pi / 4),
        np.random.uniform(-np.pi, np.pi, n_params),
        np.random.uniform(-np.pi, np.pi, n_params)
    ]
    
    for guess in guesses:
        try:
            res = minimize(objective, guess, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
        
    return best_params.reshape((layers, n))
