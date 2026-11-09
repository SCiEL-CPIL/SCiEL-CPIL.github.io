"""Optimize Ry/CZ circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonian(n, coupling, field):
    """
    Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i.
    Uses vectorized operations for speed.
    """
    dim = 1 << n
    # Basis states as integers 0 to dim-1
    basis = np.arange(dim, dtype=np.int64)
    
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: Diagonal
    # Z_i Z_j for adjacent i, j
    # Z_i = 1 if bit i is 0, -1 if bit i is 1
    # Z_i Z_j = 1 if bits are same, -1 if different
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        bits_i = (basis & mask_i) != 0
        bits_j = (basis & mask_j) != 0
        # Product of Z values: (1 - 2*b_i) * (1 - 2*b_j)
        z_prod = (1 - 2 * bits_i.astype(np.int64)) * (1 - 2 * bits_j.astype(np.int64))
        H[basis, basis] += -coupling * z_prod

    # X terms: Off-diagonal
    # X_i flips bit i
    for i in range(n):
        mask = 1 << i
        # Pairs (b, b^mask)
        # We only add to H[b, b^mask] and H[b^mask, b]
        # To avoid double counting and ensure symmetry, we can just iterate all b
        # but since H is symmetric, we can fill both or just use the fact that 
        # X is its own inverse and real.
        flipped = basis ^ mask
        # H[b, flipped] += -field
        # H[flipped, b] += -field
        H[basis, flipped] += -field
        H[flipped, basis] += -field
        
    return H

def _circuit_state(params, n, layers, ry_masks, cz_masks):
    """
    Compute circuit state vector efficiently.
    params: (layers, n)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = ry_masks[q]
            
            # In-place update is tricky due to overlapping indices in general, 
            # but here idx0 and idx1 are disjoint.
            # However, state[idx0] and state[idx1] are read before write.
            # We must be careful not to overwrite state[idx0] before using it for state[idx1].
            # Using temporary arrays is safest and fast for small dim.
            val0 = state[idx0]
            val1 = state[idx1]
            state[idx0] = c * val0 - s * val1
            state[idx1] = s * val0 + c * val1
        
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                both = cz_masks[q]
                state[both] = -state[both]
    
    return state

def _build_masks(n):
    """Precompute index masks for Ry and CZ operations."""
    dim = 1 << n
    ry_masks = []
    for q in range(n):
        mask = 1 << q
        # idx0: indices where bit q is 0
        # idx1: indices where bit q is 1
        # Efficient construction:
        # idx0 are those i where i & mask == 0
        # idx1 are those i where i & mask != 0
        # We can construct idx0 and derive idx1
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx1 = idx0 | mask
        ry_masks.append((idx0, idx1))
    
    cz_masks = []
    for q in range(n - 1):
        m1 = 1 << q
        m2 = 1 << (q + 1)
        both = np.array([i for i in range(dim) if (i & m1) and (i & m2)], dtype=np.int64)
        cz_masks.append(both)
    
    return ry_masks, cz_masks

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns params array of shape (layers, n).
    """
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Build Hamiltonian and Ground Energy
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = eigvals[0]
    
    # 2. Precompute Masks
    ry_masks, cz_masks = _build_masks(n)
    
    # 3. Objective Function
    def obj(params):
        state = _circuit_state(params, n, layers, ry_masks, cz_masks)
        e_circuit = np.real(np.vdot(state, H @ state))
        return (e_circuit - e_ground) / n
    
    # 4. Gradient Function (Parameter Shift Rule)
    def grad(params):
        g = np.zeros((layers, n), dtype=np.float64)
        for layer in range(layers):
            for q in range(n):
                p_p = params.copy()
                p_p[layer, q] += np.pi / 2
                sp = _circuit_state(p_p, n, layers, ry_masks, cz_masks)
                ep = np.real(np.vdot(sp, H @ sp))
                
                p_m = params.copy()
                p_m[layer, q] -= np.pi / 2
                sm = _circuit_state(p_m, n, layers, ry_masks, cz_masks)
                em = np.real(np.vdot(sm, H @ sm))
                
                # Gradient of energy w.r.t theta is (E_plus - E_minus) / (2 * pi/2) ? 
                # No, parameter shift rule for Ry(theta):
                # d/dtheta <psi(theta)|H|psi(theta)> = (E(theta+pi/2) - E(theta-pi/2)) / 2
                # Wait, standard rule: f'(x) = 0.5 * (f(x+pi/2) - f(x-pi/2))
                # Yes.
                g[layer, q] = 0.5 * (ep - em)
        
        # The objective is (E - E_ground)/n. The gradient of the objective is grad_E / n.
        # But L-BFGS-B minimizes the objective. So we return grad_obj.
        return g / n
        
    # 5. Initial Guesses
    ratio = field / coupling
    # Physics-based initial guess
    if ratio < 1.0:
        base_angle = 0.5 * np.arctan(ratio)
    elif ratio < 2.0:
        base_angle = np.pi / 4
    else:
        base_angle = 0.5 * (np.pi / 2 + np.arctan(1.0 / (ratio - 1)))
        
    np.random.seed(42)
    initial_guesses = []
    
    # Symmetric uniform
    initial_guesses.append(np.full((layers, n), base_angle))
    
    # Alternating signs
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            alt[l, q] = base_angle if q % 2 == 0 else -base_angle
    initial_guesses.append(alt)
    
    # Zeros
    initial_guesses.append(np.zeros((layers, n)))
    
    # Pi/2
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    # Layer dependent scaling
    for scale in [0.5, 0.8, 1.2]:
        ld = np.zeros((layers, n))
        for l in range(layers):
            ld[l, :] = base_angle * scale * (1.0 + 0.1 * l)
        initial_guesses.append(ld)
        
    # Random perturbations
    for _ in range(10):
        g = np.full((layers, n), base_angle) + np.random.normal(0, 0.5, (layers, n))
        initial_guesses.append(g)
        
    # Random uniform
    for _ in range(5):
        g = np.random.uniform(-np.pi, np.pi, (layers, n))
        initial_guesses.append(g)

    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 20:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                return grad(p).flatten()
            
            # L-BFGS-B
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            # Nelder-Mead Polish
            if time.time() > deadline - 10:
                continue
            
            try:
                res2 = minimize(obj_flat, best_params.flatten(), method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
                
            # Final tight L-BFGS-B
            if time.time() > deadline - 5:
                continue
            
            try:
                res3 = minimize(obj_flat, best_params.flatten(), jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_params = res3.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        # Fallback: Simple gradient descent
        for guess in initial_guesses:
            if time.time() > deadline - 10:
                break
            params = guess.copy()
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                g = grad(params)
                lr = 0.05 / (1 + step * 0.1)
                params_new = params - lr * g
                val_new = obj(params_new)
                if val_new < best_val:
                    best_val = val_new
                    best_params = params_new
                params = params_new

    return best_params
# EVOLVE-BLOCK-END
