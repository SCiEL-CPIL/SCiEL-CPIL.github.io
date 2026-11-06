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
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            if bit_i == bit_ip1:
                H[b, b] += coupling
            else:
                H[b, b] += -coupling

    # X terms (Off-diagonal)
    for i in range(n):
        for b in range(dim):
            b_prime = b ^ (1 << i)
            H[b, b_prime] += -field

    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E_exact = eigvals[0].real

    # Precompute CZ phase vector for one layer of CZ gates
    # Phase is -1 for each adjacent pair of 1s.
    # Total phase is (-1)^(count of adjacent 1s).
    cz_phases = np.ones(dim, dtype=complex)
    for i in range(n - 1):
        for b in range(dim):
            if ((b >> i) & 1) and ((b >> (i + 1)) & 1):
                cz_phases[b] *= -1

    def apply_circuit(theta, state):
        """Apply the full circuit to state vector."""
        for l in range(layers):
            # Apply Ry(theta[l, i]) to each qubit i
            for i in range(n):
                th = theta[l, i]
                c = np.cos(th / 2)
                s = np.sin(th / 2)
                # Apply Ry to qubit i
                # Bit i corresponds to axis n-1-i in the tensor representation
                # where axis 0 is MSB (bit n-1) and axis n-1 is LSB (bit 0).
                
                ax = n - 1 - i
                state_tensor = state.reshape((2,) * n)
                state_tensor = np.swapaxes(state_tensor, ax, 0)
                
                new_tensor = np.empty_like(state_tensor)
                new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
                new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
                state_tensor = new_tensor
                state_tensor = np.swapaxes(state_tensor, 0, ax)
                state = state_tensor.reshape(dim)
            
            # Apply CZ between layers (not after final layer)
            if l < layers - 1:
                state *= cz_phases
        
        return state

    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        state = apply_circuit(theta, state)
        energy = np.vdot(state, H @ state).real
        return energy

    # Global Search: Differential Evolution
    # Bounds
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    best_theta = None
    best_energy = np.inf
    
    # Run DE with a few different seeds to increase robustness
    for seed in [42, 123, 999]:
        try:
            de_res = opt.differential_evolution(
                objective, 
                bounds, 
                maxiter=30, 
                popsize=8, 
                tol=1e-6,
                seed=seed
            )
            if de_res.fun < best_energy:
                best_energy = de_res.fun
                best_theta = de_res.x.reshape(layers, n)
        except Exception:
            pass

    if best_theta is None:
        # Fallback
        best_theta = np.zeros((layers, n))
        best_energy = objective(best_theta.ravel())

    # Local Polish: L-BFGS-B
    try:
        res = opt.minimize(
            objective, 
            best_theta.ravel(), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.fun < best_energy:
            best_theta = res.x.reshape(layers, n)
            best_energy = res.fun
    except Exception:
        pass

    return best_theta
