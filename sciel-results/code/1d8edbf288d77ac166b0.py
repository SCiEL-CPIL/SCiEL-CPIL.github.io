"""Optimized shallow-circuit Ising ground state approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _time_left(margin=3.0):
    return 147.0 - _elapsed() - margin

def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _apply_single_qubit_gate(state, n, q, U):
    """Apply a 2x2 unitary U to qubit q of an n-qubit state."""
    dim = 2 ** n
    new_state = np.zeros_like(state)
    mask = 1 << q
    for basis in range(dim):
        bit = (basis >> q) & 1
        if bit == 0:
            idx0 = basis
            idx1 = basis | mask
            new_state[idx0] += U[0, 0] * state[idx0] + U[0, 1] * state[idx1]
            new_state[idx1] += U[1, 0] * state[idx0] + U[1, 1] * state[idx1]
        else:
            idx1 = basis
            idx0 = basis & ~mask
            new_state[idx0] += U[0, 0] * state[idx0] + U[0, 1] * state[idx1]
            new_state[idx1] += U[1, 0] * state[idx0] + U[1, 1] * state[idx1]
    return new_state

def _apply_cz_layer(state, n):
    """Apply CZ gates on all adjacent pairs (i, i+1) in ascending order."""
    for i in range(n - 1):
        mask0 = 1 << i
        mask1 = 1 << (i + 1)
        for basis in range(2 ** n):
            if ((basis >> i) & 1) == 1 and ((basis >> (i + 1)) & 1) == 1:
                state[basis] *= -1.0
    return state

def _evolve_circuit(params, n, layers):
    """Evolve |0>^n through the circuit. Returns final state vector."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    for layer in range(layers):
        for q in range(n):
            U = _ry_matrix(params[layer, q])
            state = _apply_single_qubit_gate(state, n, q, U)
        if layer < layers - 1:
            state = _apply_cz_layer(state, n)
    return state

def _build_hamiltonian(n, coupling, field):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bip1 = (state >> (i + 1)) & 1
            zz = (1 - 2 * bi) * (1 - 2 * bip1)
            H[state, state] += -coupling * zz
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    return H

def _ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    return np.linalg.eigvalsh(H)[0]

def _objective(params_flat, n, coupling, field, layers, ground_e, H):
    params = params_flat.reshape(layers, n)
    state = _evolve_circuit(params, n, layers)
    energy = np.real(np.vdot(state, H @ state))
    return (energy - ground_e) / n

def _objective_grad(params_flat, n, coupling, field, layers, ground_e, H):
    """Compute objective and analytical gradient via parameter-shift rule."""
    params = params_flat.reshape(layers, n)
    state = _evolve_circuit(params, n, layers)
    energy = np.real(np.vdot(state, H @ state))
    val = (energy - ground_e) / n
    
    grad = np.zeros_like(params_flat)
    shift = np.pi / 4
    for layer in range(layers):
        for q in range(n):
            # Parameter-shift rule for Ry: d/dtheta <psi|Ry|psi> = 0.5*(f(+)-f(-))
            # Actually for Ry(theta), the gradient is: 0.5 * (E(theta+pi/2) - E(theta-pi/2))
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[layer, q] = params[layer, q] + shift
            p_minus[layer, q] = params[layer, q] - shift
            
            st_plus = _evolve_circuit(p_plus, n, layers)
            st_minus = _evolve_circuit(p_minus, n, layers)
            e_plus = np.real(np.vdot(st_plus, H @ st_plus))
            e_minus = np.real(np.vdot(st_minus, H @ st_minus))
            
            grad[layer * n + q] = 0.5 * (e_plus - e_minus) / n
    
    return val, grad.flatten()

def _init_meanfield(n, coupling, field, layers):
    """Mean-field inspired initialization."""
    theta = np.zeros((layers, n), dtype=np.float64)
    # For H = -J sum ZZ - h sum X, mean-field angle ~ arctan(h/J)
    if coupling > 0:
        base = np.arctan2(field, coupling)
    else:
        base = np.pi / 2
    # Distribute rotation across layers with increasing magnitude
    for l in range(layers):
        frac = (l + 1) / layers
        theta[l] = base * frac
    return theta

def _init_product_plus(n, coupling, field, layers):
    """Initialize as |+>^n (all angles 0 except last layer pi/2)."""
    theta = np.zeros((layers, n), dtype=np.float64)
    theta[-1] = np.pi / 2
    return theta

def _init_zero(n, coupling, field, layers):
    """All zeros (product |0>^n)."""
    return np.zeros((layers, n), dtype=np.float64)

def _init_staggered(n, coupling, field, layers):
    """Staggered initialization exploiting symmetry."""
    theta = np.zeros((layers, n), dtype=np.float64)
    base = np.arctan2(field, coupling) if coupling > 0 else np.pi / 2
    for l in range(layers):
        frac = (l + 1) / layers
        for q in range(n):
            sign = 1 if q % 2 == 0 else -1
            theta[l, q] = sign * base * frac
    return theta

def _init_random(seed, n, layers):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(layers, n))

def angles(n, coupling, field, layers):
    # Cache Hamiltonian and ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_e = _ground_energy(n, coupling, field)
    
    best_params = None
    best_val = np.inf
    
    def try_optimize(theta_init, method, maxiter, ftol, gtol, maxfun):
        nonlocal best_params, best_val
        if _time_left(5.0) < 1.0:
            return
        params_flat = theta_init.flatten()
        val = _objective(params_flat, n, coupling, field, layers, ground_e, H)
        if val < best_val:
            best_val = val
            best_params = params_flat.copy()
        
        if method == 'L-BFGS-B':
            try:
                res = minimize(
                    lambda x: _objective(x, n, coupling, field, layers, ground_e, H),
                    params_flat,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * (layers * n),
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxfun}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        elif method == 'Nelder-Mead':
            try:
                res = minimize(
                    lambda x: _objective(x, n, coupling, field, layers, ground_e, H),
                    params_flat,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': ftol, 'fatol': ftol}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        elif method == 'gradient_descent':
            try:
                x = params_flat.copy()
                lr = 0.5
                for it in range(maxiter):
                    if _time_left(5.0) < 1.0:
                        break
                    val, grad = _objective_grad(x, n, coupling, field, layers, ground_e, H)
                    if val < best_val:
                        best_val = val
                        best_params = x.copy()
                    x = x - lr * grad
                    x = np.clip(x, -np.pi, np.pi)
            except Exception:
                pass

    # Strategy 1: Mean-field init + L-BFGS-B
    theta_mf = _init_meanfield(n, coupling, field, layers)
    try_optimize(theta_mf, 'L-BFGS-B', 300, 1e-12, 1e-10, 1000)
    
    # Strategy 2: Product |+> init + L-BFGS-B
    if _time_left(5.0) > 2.0:
        theta_plus = _init_product_plus(n, coupling, field, layers)
        try_optimize(theta_plus, 'L-BFGS-B', 300, 1e-12, 1e-10, 1000)
    
    # Strategy 3: Staggered init + L-BFGS-B
    if _time_left(5.0) > 2.0:
        theta_stag = _init_staggered(n, coupling, field, layers)
        try_optimize(theta_stag, 'L-BFGS-B', 300, 1e-12, 1e-10, 1000)
    
    # Strategy 4: Zero init + Nelder-Mead (robust, no gradient needed)
    if _time_left(5.0) > 2.0:
        theta_zero = _init_zero(n, coupling, field, layers)
        try_optimize(theta_zero, 'Nelder-Mead', 200, 1e-10, 1e-10, 1000)
    
    # Strategy 5: Gradient descent from mean-field
    if _time_left(5.0) > 2.0:
        theta_mf2 = _init_meanfield(n, coupling, field, layers)
        try_optimize(theta_mf2, 'gradient_descent', 100, 1e-10, 1e-10, 1000)
    
    # Strategy 6: Random restarts
    if _time_left(5.0) > 2.0:
        for seed in [1, 2, 3]:
            if _time_left(5.0) < 2.0:
                break
            theta_rand = _init_random(seed, n, layers)
            try_optimize(theta_rand, 'L-BFGS-B', 200, 1e-12, 1e-10, 800)
    
    # Final polish: tighter tolerance on best solution
    if best_params is not None and _time_left(5.0) > 1.0:
        try:
            res = minimize(
                lambda x: _objective(x, n, coupling, field, layers, ground_e, H),
                best_params,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = _init_product_plus(n, coupling, field, layers).flatten()
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
