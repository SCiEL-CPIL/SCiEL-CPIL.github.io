import numpy as np
import scipy.optimize as opt

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    """
    dim = 2 ** n
    
    # Construct Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[b, b] += -coupling * z_i * z_ip1
    
    # X terms
    for i in range(n):
        for b in range(dim):
            b_prime = b ^ (1 << i)
            H[b, b_prime] += -field
    
    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E_exact = eigvals[0].real
    
    # Precompute Ry matrices for each angle (will be updated in objective)
    def apply_circuit(theta, state):
        """Apply the full circuit to state vector."""
        for l in range(layers):
            # Apply Ry(theta[l, i]) to each qubit i
            for i in range(n):
                th = theta[l, i]
                c = np.cos(th / 2)
                s = np.sin(th / 2)
                # Apply Ry to qubit i
                # Reshape state to tensor
                state_tensor = state.reshape((2,) * n)
                axis = n - 1 - i
                state_tensor = np.swapaxes(state_tensor, axis, 0)
                # Ry = [[c, -s], [s, c]]
                # new_state[i, j, ...] = c*old[i, j, ...] - s*old[i^1, j, ...]
                # More precisely: Ry @ v where v is 2-vector
                # We can do this with slicing
                new_tensor = np.empty_like(state_tensor)
                new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
                new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
                state_tensor = new_tensor
                state_tensor = np.swapaxes(state_tensor, 0, axis)
                state = state_tensor.reshape(dim)
            
            # Apply CZ between layers (not after final layer)
            if l < layers - 1:
                for i in range(n - 1):
                    # CZ: phase -1 when both qubits i and i+1 are 1
                    mask = np.zeros(dim, dtype=bool)
                    for b in range(dim):
                        if ((b >> i) & 1) == 1 and ((b >> (i + 1)) & 1) == 1:
                            mask[b] = True
                    state[mask] *= -1
        
        return state
    
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        state = apply_circuit(theta, state)
        energy = np.vdot(state, H @ state).real
        return energy
    
    # Better initial guess based on physics
    # For transverse field Ising, the ground state in the intermediate regime
    # has a product-state approximation where each qubit is rotated by an angle
    # that balances the ZZ and X terms.
    # Simple heuristic: start with angles that produce a state close to 
    # the mean-field solution.
    
    theta0 = np.zeros((layers, n))
    # The final layer often needs a significant rotation
    theta0[-1] = np.pi / 4  # Start with a moderate angle
    
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    # Multi-start optimization
    best_theta = None
    best_energy = np.inf
    
    # Try several initial guesses
    initial_guesses = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 2),
        np.full((layers, n), np.pi / 6),
        np.array([[np.pi/4]*n, [np.pi/6]*n, [np.pi/3]*n]),
    ]
    
    for guess in initial_guesses:
        try:
            res = opt.minimize(objective, guess.ravel(), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_theta is None:
        best_theta = theta0
    
    return best_theta
