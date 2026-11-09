# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_indices(n):
    """Precompute index arrays for efficient gate application."""
    dim = 1 << n
    ry_indices = {}
    for qubit in range(n):
        mask = 1 << qubit
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int32)
        idx1 = idx0 | mask
        ry_indices[qubit] = (idx0, idx1)
    
    cz_indices = {}
    for q1 in range(n - 1):
        q2 = q1 + 1
        mask1 = 1 << q1
        mask2 = 1 << q2
        both_set = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int32)
        cz_indices[(q1, q2)] = both_set
    
    return ry_indices, cz_indices

def _apply_ry(state, n, qubit, theta, ry_indices):
    """Apply Ry(theta) to qubit on state of n qubits. Vectorized."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    idx0, idx1 = ry_indices[qubit]
    new_state = state.copy()
    new_state[idx0] = c * state[idx0] - s * state[idx1]
    new_state[idx1] = s * state[idx0] + c * state[idx1]
    return new_state

def _apply_cz(state, n, q1, q2, cz_indices):
    """Apply CZ between qubits q1 and q2 (q1 < q2) on state of n qubits."""
    both_set = cz_indices[(q1, q2)]
    new_state = state.copy()
    new_state[both_set] = -state[both_set]
    return new_state

def _circuit_state(n, layers, params, ry_indices, cz_indices):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, n, q, params[layer, q], ry_indices)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, n, q, q + 1, cz_indices)
    
    return state

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _circuit_energy(params, n, coupling, field, layers, H, e_ground, ry_indices, cz_indices):
    """Compute (circuit_energy - ground_energy) / n"""
    state = _circuit_state(n, layers, params, ry_indices, cz_indices)
    e_circuit = _energy(state, H)
    return (e_circuit - e_ground) / n

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    H = _hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    ry_indices, cz_indices = _precompute_indices(n)
    
    def obj(params):
        state = _circuit_state(n, layers, params, ry_indices, cz_indices)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    # Better initial guesses based on physics
    ratio = field / coupling
    
    # For single qubit in transverse field, optimal angle is arctan(field/coupling)/2
    # But here we have interactions, so use a heuristic
    if ratio < 0.5:
        base_angle = 0.3 * np.arctan(ratio)
    elif ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 8 + 0.25 * np.arctan(ratio - 1)
    else:
        base_angle = np.pi / 4 - 0.25 * np.arctan(1.0 / (ratio - 1))
    
    np.random.seed(42)
    
    initial_guesses = []
    
    # Guess 1: All angles equal to base_angle
    initial_guesses.append(np.full((layers, n), base_angle))
    
    # Guess 2: Alternating pattern
    guess2 = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            if q % 2 == 0:
                guess2[l, q] = base_angle
            else:
                guess2[l, q] = -base_angle
    initial_guesses.append(guess2)
    
    # Guess 3: All zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Guess 4: All pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # Guess 5: All pi
    initial_guesses.append(np.full((layers, n), np.pi))
    
    # Guess 6: All -pi/2
    initial_guesses.append(np.full((layers, n), -np.pi / 2))
    
    # Guess 7-12: Random perturbations
    for i in range(6):
        guess = np.full((layers, n), base_angle) + np.random.normal(0, 0.3, (layers, n))
        initial_guesses.append(guess)
    
    # Also try angles scaled by layer (deeper layers may need different angles)
    for scale in [0.5, 1.5, 2.0]:
        guess = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                guess[l, q] = base_angle * scale * (l + 1) / layers
        initial_guesses.append(guess)
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize, basinhopping
        
        # Use basinhopping for global search capability
        # This is more robust than simple multi-start for these landscapes
        
        def objective_flat(xf):
            return obj(xf.reshape((layers, n)))
        
        # Run basinhopping from a few starting points
        start_points = [initial_guesses[0].flatten(), initial_guesses[2].flatten(), initial_guesses[3].flatten()]
        
        for start in start_points:
            if time.time() > deadline - 10:
                break
            
            try:
                res_bh = basinhopping(objective_flat, start, 
                                      minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': 1e-12}},
                                      niter=20, 
                                      stepsize=0.5)
                if res_bh.fun < best_val:
                    best_val = res_bh.fun
                    best_params = res_bh.x.reshape((layers, n))
            except Exception:
                pass
        
        # Final polish with high precision L-BFGS-B
        if time.time() < deadline - 2:
            try:
                x_flat = best_params.flatten()
                res3 = minimize(objective_flat, x_flat, method='L-BFGS-B',
                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x.reshape((layers, n))
            except Exception:
                pass
                
    except ImportError:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
