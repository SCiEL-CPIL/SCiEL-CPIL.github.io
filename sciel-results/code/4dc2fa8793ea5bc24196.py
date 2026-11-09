# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_indices(n):
    """Precompute index masks for Ry and CZ gates."""
    dim = 1 << n
    ry_idx = []
    for q in range(n):
        mask = 1 << q
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        ry_idx.append((idx0, idx1))
    cz_idx = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        idx = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.int64)
        cz_idx.append(idx)
    return ry_idx, cz_idx

def _apply_ry_inplace(state, idx0, idx1, theta):
    """Apply Ry(theta) in-place using precomputed indices."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    v0 = state[idx0].copy()
    v1 = state[idx1].copy()
    state[idx0] = c * v0 - s * v1
    state[idx1] = s * v0 + c * v1

def _apply_cz_inplace(state, idx):
    """Apply CZ in-place using precomputed indices."""
    state[idx] *= -1.0

def _circuit_state(n, layers, params, ry_idx, cz_idx):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        for q in range(n):
            _apply_ry_inplace(state, ry_idx[q][0], ry_idx[q][1], params[layer, q])
        if layer < layers - 1:
            for q in range(n - 1):
                _apply_cz_inplace(state, cz_idx[q])
    
    return state

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal (each pair added once)
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:  # only add each pair once
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

def _objective(params, n, coupling, field, layers, ry_idx, cz_idx, H, e_ground):
    """Compute the objective: (circuit_energy - ground_energy) / n"""
    state = _circuit_state(n, layers, params, ry_idx, cz_idx)
    e_circuit = _energy(state, H)
    return (e_circuit - e_ground) / n

def _objective_and_grad(params, n, layers, ry_idx, cz_idx, H, e_ground):
    """Compute objective and exact gradient via parameter-shift rule for Ry gates.
    For U = Ry(theta) = exp(-i theta Y/2), dU/dtheta = -i/2 Y U = -i/2 Ry(theta+pi/2) Ry(-theta) U.
    Using the shift rule: d<psi|H|psi>/dtheta = (<psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)>) / 2
    """
    dim = 1 << n
    base_state = _circuit_state(n, layers, params, ry_idx, cz_idx)
    e_base = _energy(base_state, H)
    obj_val = (e_base - e_ground) / n
    
    grad = np.zeros_like(params)
    for layer in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[layer, q] = params[layer, q] + np.pi / 2
            p_minus[layer, q] = params[layer, q] - np.pi / 2
            s_plus = _circuit_state(n, layers, p_plus, ry_idx, cz_idx)
            s_minus = _circuit_state(n, layers, p_minus, ry_idx, cz_idx)
            e_plus = _energy(s_plus, H)
            e_minus = _energy(s_minus, H)
            grad[layer, q] = (e_plus - e_minus) / (2.0 * n)
    return obj_val, grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin.
    
    Uses exact parameter-shift gradients for Ry gates with precomputed
    index masks, enabling fast L-BFGS-B convergence from multiple
    physics-motivated initializations.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute index masks, Hamiltonian, and ground energy
    ry_idx, cz_idx = _precompute_indices(n)
    H = _hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    
    def obj(params):
        state = _circuit_state(n, layers, params, ry_idx, cz_idx)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    def obj_grad(params):
        return _objective_and_grad(params, n, layers, ry_idx, cz_idx, H, e_ground)
    
    # Physics-motivated base angle from field/coupling ratio
    ratio = field / coupling
    
    if ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 4
    else:
        base_angle = 0.5 * (np.pi / 2 + np.arctan(1.0 / (ratio - 1)))
    
    # Layer-dependent scaling: first layer rotates most, later layers refine
    # This matches the structure where early layers build the bulk of the state
    layer_scale = np.array([1.0, 0.6, 0.3])[:layers]
    
    # Create diverse structured initial guesses
    np.random.seed(42)
    
    initial_guesses = []
    
    # G1: Uniform base angle
    initial_guesses.append(np.full((layers, n), base_angle))
    
    # G2: Layer-scaled (first layer does most rotation)
    g2 = np.zeros((layers, n))
    for l in range(layers):
        g2[l, :] = base_angle * layer_scale[l]
    initial_guesses.append(g2)
    
    # G3: Alternating sign (symmetry breaking)
    g3 = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g3[l, q] = base_angle * layer_scale[l] * (1 if q % 2 == 0 else -1)
    initial_guesses.append(g3)
    
    # G4: All zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # G5: All pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # G6: Layer-scaled alternating
    g6 = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            g6[l, q] = base_angle * layer_scale[l] * (1 if q % 2 == 0 else -1)
            g6[l, q] *= 1.2  # slightly larger amplitude
    initial_guesses.append(g6)
    
    # G7-G12: Random perturbations around base_angle and layer-scaled
    for i in range(6):
        if i < 3:
            guess = np.full((layers, n), base_angle) + np.random.normal(0, 0.3, (layers, n))
        else:
            guess = np.zeros((layers, n))
            for l in range(layers):
                guess[l, :] = base_angle * layer_scale[l]
            guess += np.random.normal(0, 0.25, (layers, n))
        initial_guesses.append(guess)
    
    # Evaluate all initial guesses
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    for g in initial_guesses[1:]:
        v = obj(g)
        if v < best_val:
            best_val = v
            best_params = g.copy()
    
    # Optimize from each initial guess using exact gradients
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def obj_grad_flat(xf):
                v, g = obj_grad(xf.reshape((layers, n)))
                return v, g.flatten()
            
            try:
                res = minimize(obj_grad_flat, x_flat, method='L-BFGS-B',
                              jac=True,
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Tighter polish pass on the best result so far
            if time.time() > deadline - 10:
                break
            
            try:
                res2 = minimize(obj_grad_flat, best_params.flatten(), method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        pass
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            from scipy.optimize import minimize
            def obj_grad_flat_final(xf):
                v, g = obj_grad(xf.reshape((layers, n)))
                return v, g.flatten()
            res_final = minimize(obj_grad_flat_final, best_params.flatten(), method='L-BFGS-B',
                                jac=True,
                                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x.reshape((layers, n))
        except Exception:
            pass
    
    # Verify final result
    if time.time() < deadline - 1:
        try:
            final_val = obj(best_params)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
