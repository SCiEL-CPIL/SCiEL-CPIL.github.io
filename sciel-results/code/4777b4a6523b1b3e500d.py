import numpy as np
from scipy.optimize import minimize

def _simulate_circuit_vec(n, angles, layers):
    """
    Simulate the circuit using vectorized NumPy operations.
    angles: (layers, n)
    """
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    indices = np.arange(2 ** n)
    
    for layer in range(layers):
        for q in range(n):
            theta = angles[layer, q]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            
            perm = [q] + [i for i in range(n) if i != q]
            state = state.reshape([2] * n).transpose(perm)
            
            new_state = np.empty_like(state)
            new_state[0] = c * state[0] - s * state[1]
            new_state[1] = s * state[0] + c * state[1]
            state = new_state
            
            inv_perm = [0] * n
            for new_idx, old_idx in enumerate(perm):
                inv_perm[old_idx] = new_idx
            state = state.transpose(inv_perm).flatten()

        if layer < layers - 1:
            for q in range(n - 1):
                mask = (((indices >> q) & 1) == 1) & (((indices >> (q + 1)) & 1) == 1)
                state[mask] *= -1

    return state

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

def _gradient_param_shift(angles_flat, n, coupling, field, layers):
    """
    Compute gradient using Parameter Shift Rule.
    """
    grad = np.zeros_like(angles_flat)
    angles = angles_flat.reshape(layers, n)
    
    for layer in range(layers):
        for q in range(n):
            idx = layer * n + q
            
            angles_plus = angles.copy()
            angles_plus[layer, q] += np.pi / 2
            e_plus = _energy_objective(angles_plus.flatten(), n, coupling, field, layers)
            
            angles_minus = angles.copy()
            angles_minus[layer, q] -= np.pi / 2
            e_minus = _energy_objective(angles_minus.flatten(), n, coupling, field, layers)
            
            grad[idx] = 0.5 * (e_plus - e_minus)
            
    return grad

def angles(n, coupling, field, layers):
    bounds = [(-np.pi, np.pi)] * (layers * n)
    n_params = layers * n
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    n_restarts = 10
    
    for r in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)
        
        try:
            res = minimize(
                _energy_objective,
                x0,
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                jac=_gradient_param_shift,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < best_e:
                best_e = res.fun
                best_x = res.x
                
        except Exception:
            pass
            
    if best_x is None:
        return np.zeros((layers, n))
        
    return best_x.reshape(layers, n)
