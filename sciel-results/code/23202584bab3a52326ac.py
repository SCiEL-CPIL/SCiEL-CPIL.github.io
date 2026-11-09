"""QAOA parameter design for MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_start_time = None

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_left():
    _init_time()
    return _DEADLINE - (time.time() - _start_time)

def _apply_single_qubit_gate(psi, n, j, U):
    """Apply 2x2 unitary U to qubit j (bit j, LSB=0)."""
    dim = 1 << n
    psi_tensor = psi.reshape((2,) * n)
    axis = n - 1 - j
    psi_moved = np.moveaxis(psi_tensor, axis, -1)
    psi_moved = np.einsum('ab,...b->...a', U, psi_moved, optimize=True)
    psi_moved = np.moveaxis(psi_moved, -1, axis)
    return psi_moved.reshape(dim)

def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut value for given QAOA parameters."""
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute C eigenvalues: number of cut edges for each state
    c_eigs = np.zeros(dim)
    for (i, j) in edges:
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                c_eigs[state] += 1.0
    
    for gamma, beta in zip(gammas, betas):
        # Apply U_C: diagonal phase
        psi = psi * np.exp(-1j * gamma * c_eigs)
        
        # Apply U_B = prod_j (cos(beta) I - i sin(beta) X_j)
        sin_b = np.sin(beta)
        cos_b = np.cos(beta)
        U = np.array([[cos_b, -1j * sin_b],
                     [-1j * sin_b, cos_b]], dtype=complex)
        
        for j in range(n):
            psi = _apply_single_qubit_gate(psi, n, j, U)
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    return np.real(np.sum(probs * c_eigs))

def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force."""
    best = 0
    for state in range(1 << n):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return best

def angles(n, edges, p):
    """Return QAOA angles (p, 2) array of (gamma, beta) pairs."""
    _init_time()
    
    if n == 0 or len(edges) == 0:
        return np.zeros((p, 2))
    
    num_edges = len(edges)
    exact_opt = _exact_maxcut(n, edges)
    
    if exact_opt == 0:
        return np.zeros((p, 2))
    
    # Initial guess
    gamma0 = 0.4
    beta0 = 0.3
    
    x0 = np.array([gamma0, beta0] * p)
    
    def objective(x):
        gammas = x[0::2]
        betas = x[1::2]
        val = _expected_cut(n, edges, gammas, betas)
        return -val
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    n_restarts = 15
    rng = np.random.RandomState(42)
    
    for r in range(n_restarts):
        if _time_left() < 12:
            break
        if r == 0:
            x_try = x0.copy()
        else:
            x_try = rng.uniform(0.05, np.pi - 0.05, size=2 * p)
        
        try:
            res = minimize(objective, x_try, method='L-BFGS-B', 
                          bounds=[(0.01, np.pi - 0.01)] * (2 * p),
                          options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-7})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Final refinement
    if _time_left() > 8:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                          bounds=[(0.01, np.pi - 0.01)] * (2 * p),
                          options={'maxiter': 100, 'ftol': 1e-11, 'gtol': 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    gammas = best_x[0::2]
    betas = best_x[1::2]
    result = np.column_stack([gammas, betas])
    
    return result
# EVOLVE-BLOCK-END
