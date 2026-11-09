# EVOLVE-BLOCK-START
import numpy as np
import time

def _circuit_state(n, layers, params, precomp=None):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    Uses precomputed indices for speed.
    """
    if precomp is None:
        precomp = _precompute_indices(n)
    
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit (vectorized)
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx0, idx1 = precomp['ry_indices'][q]
            state[idx0] = c * state[idx0] - s * state[idx1]
            state[idx1] = s * state[idx0] + c * state[idx1]
        
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ flips sign when both qubits are 1
                mask = precomp['cz_masks'][q]
                state[mask] = -state[mask]
    
    return state

def _precompute_indices(n):
    """Precompute index arrays for fast circuit simulation."""
    dim = 1 << n
    precomp = {}
    
    # Ry indices for each qubit
    precomp['ry_indices'] = []
    for q in range(n):
        mask = (1 << q)
        idx0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.intp)
        idx1 = idx0 | mask
        precomp['ry_indices'].append((idx0, idx1))
    
    # CZ masks for adjacent pairs
    precomp['cz_masks'] = []
    for q in range(n - 1):
        mask1 = (1 << q)
        mask2 = (1 << (q + 1))
        # Indices where both bits are set
        mask = np.array([i for i in range(dim) if (i & mask1) and (i & mask2)], dtype=np.intp)
        precomp['cz_masks'].append(mask)
    
    return precomp

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

def _objective_and_grad(params, n, coupling, field, layers, H, e_ground, precomp):
    """Compute objective and analytic gradient."""
    state = _circuit_state(n, layers, params, precomp)
    e_circuit = _energy(state, H)
    obj_val = (e_circuit - e_ground) / n
    
    # Compute gradient via parameter-shift or finite differences
    # For efficiency, use central finite differences with small step
    grad = np.zeros_like(params)
    step = 1e-5
    for layer in range(layers):
        for q in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[layer, q] += step
            params_minus[layer, q] -= step
            
            e_plus = _energy(_circuit_state(n, layers, params_plus, precomp), H)
            e_minus = _energy(_circuit_state(n, layers, params_minus, precomp), H)
            
            grad[layer, q] = ((e_plus - e_minus) / (2 * step)) / n
    
    return obj_val, grad

def _nelder_mead(objective, x0, maxiter=200, tol=1e-8):
    """Simple Nelder-Mead implementation"""
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    for it in range(maxiter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-1]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Precompute ground energy, Hamiltonian, and circuit indices
    e_ground = _exact_ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)
    precomp = _precompute_indices(n)
    
    def obj(params):
        state = _circuit_state(n, layers, params, precomp)
        e_circuit = _energy(state, H)
        return (e_circuit - e_ground) / n
    
    def obj_grad(params):
        return _objective_and_grad(params, n, coupling, field, layers, H, e_ground, precomp)
    
    # Initial guess: all zeros
    x0 = np.zeros((layers, n))
    
    best_params = x0.copy()
    best_val = obj(x0)
    
    np.random.seed(42)
    
    # Multi-restart with L-BFGS-B using gradients
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 15:  # leave 15s for final polish
            break
        
        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            # Symmetric guess: all angles same
            x_init = np.full((layers, n), np.pi / 6)
        elif restart == 2:
            # Another symmetric guess
            x_init = np.full((layers, n), np.pi / 4)
        elif restart == 3:
            # Negative symmetric
            x_init = np.full((layers, n), -np.pi / 6)
        else:
            x_init = np.random.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        
        x_flat = x_init.flatten()
        
        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))
        
        def obj_grad_flat(xf):
            val, grad = obj_grad(xf.reshape((layers, n)))
            return val, grad.flatten()
        
        # Phase 1: L-BFGS-B with analytic gradients
        try:
            from scipy.optimize import minimize
            res = minimize(obj_grad_flat, x_flat, jac=True, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
            x_flat = res.x
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead polish for high precision
        if time.time() < deadline - 8:
            try:
                x_nm, f_nm = _nelder_mead(obj_flat, x_flat, maxiter=100, tol=1e-10)
                if f_nm < best_val:
                    best_val = f_nm
                    best_params = x_nm.reshape((layers, n))
            except Exception:
                pass
    
    # Final high-precision polish on best params
    if time.time() < deadline - 3:
        try:
            x_flat = best_params.flatten()
            x_final, f_final = _nelder_mead(obj_flat, x_flat, maxiter=200, tol=1e-11)
            if f_final < best_val:
                best_val = f_final
                best_params = x_final.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
