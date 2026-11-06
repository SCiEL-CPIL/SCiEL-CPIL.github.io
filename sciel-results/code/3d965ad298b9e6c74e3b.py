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
    
    # Precompute CZ phase vectors for each adjacent pair
    # CZ on pair (i, i+1) flips sign of states where both bits are 1
    cz_phase_vectors = []
    for i in range(n - 1):
        mask = np.zeros(dim, dtype=float)
        for b in range(dim):
            if ((b >> i) & 1) == 1 and ((b >> (i + 1)) & 1) == 1:
                mask[b] = -1.0
            else:
                mask[b] = 1.0
        cz_phase_vectors.append(mask)

    def apply_circuit(theta, state):
        """Apply the full circuit to state vector. Optimized for speed."""
        state = state.copy()
        for l in range(layers):
            # Apply Ry(theta[l, i]) to each qubit i
            for i in range(n):
                th = theta[l, i]
                c = np.cos(th / 2)
                s = np.sin(th / 2)
                # Reshape state to tensor for efficient slicing
                # Qubit i corresponds to axis n-1-i in little-endian tensor layout
                # But we can also use bitwise logic on the flat array if we are careful.
                # Tensor reshaping is usually fast enough for dim=64.
                state_tensor = state.reshape((2,) * n)
                axis = n - 1 - i
                # Move axis to front
                state_tensor = np.swapaxes(state_tensor, axis, 0)
                
                # Apply Ry
                new_tensor = np.empty_like(state_tensor)
                new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
                new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
                state_tensor = new_tensor
                
                # Move axis back
                state_tensor = np.swapaxes(state_tensor, 0, axis)
                state = state_tensor.reshape(dim)
            
            # Apply CZ between layers (not after final layer)
            if l < layers - 1:
                for i in range(n - 1):
                    state *= cz_phase_vectors[i]
        
        return state

    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        state = apply_circuit(theta, state)
        energy = np.vdot(state, H @ state).real
        return energy

    # Physics-based initialization
    # Heuristic: The transverse field h favors |+> (theta ~ pi/2 in Ry convention? 
    # Ry(pi/2) |0> = (|0> + |1>)/sqrt(2) = |+>.
    # The ZZ term favors aligned spins.
    # Let's estimate an effective local field.
    # For a chain, the effective field on a qubit is roughly J * (m_{left} + m_{right}) + h.
    # If we assume a uniform magnetization m, the effective field is ~ 2*J*m + h.
    # The optimal angle for a single qubit in field B_z and B_x is theta = arctan(B_x / B_z) ?
    # Actually, ground state of -B_z Z - B_x X is rotated by angle alpha where tan(2alpha) = B_x/B_z.
    # Here, the "effective B_z" comes from neighbors.
    # Let's try a simple interpolation.
    # If h is small, theta should be small (stay near |0>).
    # If h is large, theta should be near pi/2 (go to |+>).
    # Let's set the final layer to carry the main rotation, and earlier layers to be small or zero.
    # Or distribute evenly.
    
    # A common heuristic for such circuits is that the "effective" angle needed is roughly
    # proportional to the transverse field.
    # Let's guess theta_final ~ arctan(field / coupling) or similar.
    # Let's try: theta_i = arctan(field / coupling) for all i, distributed across layers.
    # Since we have 3 layers, maybe split the angle?
    # Or just put it in the last layer?
    # Let's try putting the estimated angle in the last layer, and 0 in others.
    # Or, better, use a mean-field estimate.
    
    # Simple heuristic:
    # theta_est = np.arctan(field / coupling)
    # This gives ~0.43 for h=0.45, J=1.
    # ~0.78 for h=1.0, J=1.
    # ~1.02 for h=1.65, J=1.
    
    # Let's initialize all layers with this angle / layers?
    # Or just the last layer?
    # Let's try a few variations in the initialization logic.
    
    theta_init = np.zeros((layers, n))
    # Estimate the "required" rotation angle
    theta_req = np.arctan(field / coupling)
    
    # Distribute the rotation. Often, the final layer is most effective for aligning with the final field direction.
    # But CZs between layers can create correlations.
    # Let's try: theta[0] = 0, theta[1] = theta_req/2, theta[2] = theta_req/2
    # Or theta[0]=theta_req/3, theta[1]=theta_req/3, theta[2]=theta_req/3
    
    # Let's use a heuristic that works well for Ising:
    # The ground state is often close to a product state of |+> if h > J, and |0> if h < J.
    # The crossover is around h=J.
    # Let's set the initial guess to be the "mean-field" product state angle.
    # And assume the circuit can prepare it.
    
    # Let's try initializing with the estimated angle in the last layer only.
    theta_init[-1, :] = theta_req
    
    # Alternative: distribute evenly
    # theta_init[:, :] = theta_req / layers
    
    # Let's use the evenly distributed one as it might be more robust to the CZ entanglers.
    theta_init = np.full((layers, n), theta_req / layers)
    
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    # Optimization
    # Use L-BFGS-B with a good start
    try:
        res = opt.minimize(objective, theta_init.ravel(), method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        best_theta = res.x.reshape(layers, n)
    except Exception:
        # Fallback to a simpler guess if optimization fails
        best_theta = theta_init
    
    return best_theta
