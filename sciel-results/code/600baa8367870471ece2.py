import numpy as np
from scipy.optimize import minimize

def _simulate_circuit_vec(n, angles, layers):
    """
    Simulate the circuit using vectorized NumPy operations.
    angles: (layers, n)
    """
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    state_tensor = state.reshape([2] * n)
    indices = np.arange(2 ** n)
    
    for layer in range(layers):
        for q in range(n):
            theta = angles[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            perm = [q] + [i for i in range(n) if i != q]
            state_tensor = np.transpose(state_tensor, perm)
            
            old_state = state_tensor
            new_state = np.empty_like(old_state)
            new_state[0] = c * old_state[0] - s * old_state[1]
            new_state[1] = s * old_state[0] + c * old_state[1]
            state_tensor = new_state
            
            inv_perm = [0] * n
            for new_idx, old_idx in enumerate(perm):
                inv_perm[old_idx] = new_idx
            state_tensor = np.transpose(state_tensor, inv_perm)
        
        if layer < layers - 1:
            for q in range(n - 1):
                flat_state = state_tensor.flatten()
                mask = (((indices >> q) & 1) == 1) & (((indices >> (q + 1)) & 1) == 1)
                flat_state[mask] *= -1
                state_tensor = flat_state.reshape([2] * n)

    return state_tensor.flatten()

def _compute_energy_vec(state, n, coupling, field):
    """
    Compute energy expectation value.
    """
    probs = np.abs(state) ** 2
    energy = 0.0
    indices = np.arange(2 ** n)
    
    for i in range(n - 1):
        bit_i = (indices >> i) & 1
        bit_ip1 = (indices >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = z_i * z_ip1
        energy += -coupling * np.sum(probs * zz_val)
        
    for i in range(n):
        mask = 1 << i
        flipped = indices ^ mask
        x_exp = np.sum(np.conj(state) * state[flipped])
        energy += -field * 2 * np.real(x_exp)
        
    return energy

def _energy_objective(angles_flat, n, coupling, field, layers):
    angles = angles_flat.reshape(layers, n)
    state = _simulate_circuit_vec(n, angles, layers)
    return _compute_energy_vec(state, n, coupling, field)

def angles(n, coupling, field, layers):
    bounds = [(-np.pi, np.pi)] * (layers * n)
    n_params = layers * n
    
    # Basin Hopping with L-BFGS-B local optimization
    rng = np.random.default_rng(42)
    
    def local_minimize(x):
        try:
            res = minimize(
                _energy_objective,
                x,
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            return res.x, res.fun
        except Exception:
            return x, float('inf')

    # Initial guess
    x = rng.uniform(-np.pi, np.pi, size=n_params)
    x, e = local_minimize(x)
    best_x, best_e = x, e
    
    # Hopping steps
    step_size = 0.5
    for i in range(30):
        # Perturb
        x_perturbed = x + rng.normal(0, step_size, size=n_params)
        # Wrap angles to [-pi, pi]
        x_perturbed = (x_perturbed + np.pi) % (2 * np.pi) - np.pi
        
        # Local minimize from perturbed
        x_new, e_new = local_minimize(x_perturbed)
        
        if e_new < best_e:
            x = x_new
            best_e = e_new
            best_x = x_new
        else:
            # Greedy acceptance: keep best found so far
            pass
            
    best_angles = best_x.reshape(layers, n)
    if not np.all(np.isfinite(best_angles)):
        return np.zeros((layers, n))
    return best_angles
