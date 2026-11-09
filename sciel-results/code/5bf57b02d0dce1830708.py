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

def _apply_ry_inplace(state, n, qubit, theta, ry_indices):
    """Apply Ry(theta) to qubit in-place on state of n qubits."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    idx0, idx1 = ry_indices[qubit]
    s0 = state[idx0].copy()
    s1 = state[idx1].copy()
    state[idx0] = c * s0 - s * s1
    state[idx1] = s * s0 + c * s1

def _apply_cz_inplace(state, n, q1, q2, cz_indices):
    """Apply CZ between qubits q1 and q2 in-place."""
    both_set = cz_indices[(q1, q2)]
    state[both_set] = -state[both_set]

def _circuit_state(n, layers, params, ry_indices, cz_indices):
    """Compute the state produced by the circuit."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            _apply_ry_inplace(state, n, q, params[layer, q], ry_indices)
        if layer < layers - 1:
            for q in range(n - 1):
                _apply_cz_inplace(state, n, q, q + 1, cz_indices)
    
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

def _gradient(params, n, layers, H, e_ground, ry_indices, cz_indices):
    """Compute gradient of (circuit_energy - ground_energy)/n using finite differences."""
    dim = 1 << n
    grad = np.zeros((layers, n))
    eps = 1e-7
    base_energy = _circuit_energy(params, n, 1.0, 0.0, layers, H, e_ground, ry_indices, cz_indices)
    
    for l in range(layers):
        for q in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[l, q] += eps
            params_minus[l, q] -= eps
            e_plus = _circuit_energy(params_plus, n, 1.0, 0.0, layers, H, e_ground, ry_indices, cz_indices)
            e_minus = _circuit_energy(params_minus, n, 1.0, 0.0, layers, H, e_ground, ry_indices, cz_indices)
            grad[l, q] = (e_plus - e_minus) / (2 * eps)
    
    return grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    H = _hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    ry_indices, cz_indices = _precompute_indices(n)
    
    def obj(params):
        state = _circuit_state(n, layers, params, ry_indices, cz_indices)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    ratio = field / coupling
    
    if ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 4
    else:
        base_angle = 0.5 * (np.pi / 2 + np.arctan(1.0 / (ratio - 1)))
    
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
    
    # Also try angles scaled by layer
    for scale in [0.5, 1.5, 2.0]:
        guess = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                guess[l, q] = base_angle * scale * (l + 1) / layers
        initial_guesses.append(guess)
    
    # Additional physics-motivated guesses
    # For transverse field Ising, the ground state in the strong-field limit is |+>^n
    # In the strong-coupling limit it's |0>^n or |1>^n
    if ratio < 0.5:
        # Strong coupling: ground state is mostly |0>^n, small field correction
        g = np.zeros((layers, n))
        for l in range(layers):
            for q in range(n):
                g[l, q] = ratio * 0.1
        initial_guesses.append(g)
    elif ratio > 2.0:
        # Strong field: ground state is mostly |+>^n
        g = np.full((layers, n), np.pi / 4)
        initial_guesses.append(g)
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            try:
                res = minimize(obj_flat, x_flat, method='L-BFGS-B',
                              options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 60})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            if time.time() > deadline - 5:
                break
            
            try:
                res2 = minimize(obj_flat, x_flat, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-13, 'fatol': 1e-15})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        pass
    
    # Final polish: high-precision refinement from the best solution found
    if time.time() < deadline - 3:
        try:
            x_flat = best_params.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            res3 = minimize(obj_flat, x_flat, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 80})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
