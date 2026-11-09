# EVOLVE-BLOCK-START
import numpy as np
import time

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: diagonal
    diag = np.zeros(dim)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        bits = np.arange(dim)
        bi = (bits >> i) & 1
        bj = (bits >> (i + 1)) & 1
        zval = (1 - 2 * bi) * (1 - 2 * bj)
        diag += -coupling * zval
    H[np.arange(dim), np.arange(dim)] = diag

    # X terms: off-diagonal
    for i in range(n):
        mask = 1 << i
        bits = np.arange(dim)
        idx0 = bits[(bits & mask) == 0]
        idx1 = idx0 | mask
        H[idx0, idx1] += -field
        H[idx1, idx0] += -field
        
    return H

def _circuit_state(n, layers, params):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Precompute index pairs for Ry gates
    ry_indices = []
    for q in range(n):
        mask = 1 << q
        bits = np.arange(dim)
        idx0 = bits[(bits & mask) == 0]
        idx1 = idx0 | mask
        ry_indices.append((idx0, idx1))
        
    # Precompute indices for CZ gates
    cz_indices = []
    for q in range(n - 1):
        mask1 = 1 << q
        mask2 = 1 << (q + 1)
        bits = np.arange(dim)
        both_set = bits[(bits & mask1) != 0]
        both_set = both_set[(both_set & mask2) != 0]
        cz_indices.append(both_set)

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_indices[q]
            temp0 = state[idx0].copy()
            temp1 = state[idx1].copy()
            state[idx0] = c * temp0 - s * temp1
            state[idx1] = s * temp0 + c * temp1
            
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                idxs = cz_indices[q]
                state[idxs] = -state[idxs]
    
    return state

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Precompute Hamiltonian and ground energy
    H = _hamiltonian(n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    
    def obj(params):
        state = _circuit_state(n, layers, params)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    # Gradient computation using finite differences
    def gradient(params):
        grad = np.zeros_like(params)
        eps = 1e-5
        base = obj(params)
        for i in range(params.shape[0]):
            for j in range(params.shape[1]):
                params_plus = params.copy()
                params_plus[i, j] += eps
                f_plus = obj(params_plus)
                grad[i, j] = (f_plus - base) / eps
        return grad
    
    # Structured initial guesses
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
    
    # Guess 5-12: Random perturbations around base_angle
    for i in range(8):
        guess = np.full((layers, n), base_angle) + np.random.normal(0, 0.3, (layers, n))
        initial_guesses.append(guess)
    
    # Also try random starting points
    for i in range(4):
        guess = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))
        initial_guesses.append(guess)
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 15:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                g = gradient(p)
                return g.flatten()
            
            # Try L-BFGS-B with analytical gradient
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Polish with Nelder-Mead (derivative free, robust)
            if time.time() > deadline - 10:
                break
            
            try:
                res2 = minimize(obj_flat, x_flat, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        # Fallback: simple local optimization
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            
            params = guess.copy()
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                grad = gradient(params)
                lr = 0.01 / (1 + step * 0.1)
                params_new = params - lr * grad
                val_new = obj(params_new)
                if val_new < best_val:
                    best_val = val_new
                    best_params = params_new
                params = params_new
    
    # Final verification
    if time.time() < deadline - 1:
        try:
            final_val = obj(best_params)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
