import numpy as np
from scipy.optimize import minimize
import time

def _build_cost_diagonal(n, edges):
    """Build the diagonal of the MaxCut cost operator C.
    C = sum_{(i,j) in E} (I - Z_i Z_j)/2
    Eigenvalue for basis state x (as integer 0..2^n-1):
    c_x = (1/2) * sum_{(i,j) in E} (1 - z_i * z_j)
    where z_i = 1 if bit i is 0, -1 if bit i is 1 (or vice versa, sign convention)
    """
    # Actually, let's use: for bit position i (0-indexed from LSB), 
    # z_i = 1 if bit is 0, -1 if bit is 1
    # Then Z_i Z_j eigenvalue = z_i * z_j
    # c_x = 0.5 * sum (1 - z_i * z_j) over edges
    dim = 2 ** n
    cost_diag = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        c = 0.0
        for (i, j) in edges:
            # Get bits
            zi = 1 if (x >> i) & 1 == 0 else -1
            zj = 1 if (x >> j) & 1 == 0 else -1
            c += 0.5 * (1 - zi * zj)
        cost_diag[x] = c
    return cost_diag

def _exact_maxcut(n, edges):
    """Brute force exact MaxCut value."""
    best = 0
    for x in range(2 ** n):
        cut = 0
        for (i, j) in edges:
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return float(best)

def _simulate(n, edges, gammas, betas):
    """Simulate QAOA circuit, return final state vector."""
    dim = 2 ** n
    # Initial state |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    cost_diag = _build_cost_diagonal(n, edges)
    
    for k in range(len(gammas)):
        # Apply exp(-i * gamma_k * C)
        phase = np.exp(-1j * gammas[k] * cost_diag)
        psi = psi * phase
        
        # Apply exp(-i * beta_k * sum_j X_j)
        beta = betas[k]
        cb = np.cos(beta)
        sb = np.sin(beta)
        # For each qubit, apply [[cb, -i*sb], [-i*sb, cb]]
        # This is a tensor product, so we can do it qubit by qubit
        for j in range(n):
            # For each pair of states differing only in bit j
            # Apply the 2x2 rotation
            # Bit j: states where bit j = 0 and bit j = 1
            mask = 1 << j
            # We need to apply the rotation to all pairs (s, s^mask)
            # where s has bit j = 0
            for s in range(dim):
                if (s & mask) == 0:
                    s1 = s | mask
                    # psi[s] -> cb*psi[s] - i*sb*psi[s1]
                    # psi[s1] -> -i*sb*psi[s] + cb*psi[s1]
                    old_s = psi[s]
                    old_s1 = psi[s1]
                    psi[s] = cb * old_s - 1j * sb * old_s1
                    psi[s1] = -1j * sb * old_s + cb * old_s1
    return psi

def _expected_cut(n, edges, psi):
    """Compute expected cut value from state vector."""
    # For each edge (i,j), contribution = 0.5 - 0.5 * Re(<psi|Z_i Z_j|psi>)
    # Z_i Z_j is diagonal: eigenvalue = z_i * z_j for basis state x
    total = 0.0
    for (i, j) in edges:
        # Build diagonal of Z_i Z_j
        # For basis state x: z_i * z_j
        diag = np.zeros(2**n, dtype=np.float64)
        for x in range(2**n):
            zi = 1 if (x >> i) & 1 == 0 else -1
            zj = 1 if (x >> j) & 1 == 0 else -1
            diag[x] = zi * zj
        # <psi|Z_i Z_j|psi> = sum_x |psi[x]|^2 * diag[x]
        probs = np.abs(psi)**2
        exp_val = np.dot(probs, diag)
        total += 0.5 - 0.5 * exp_val.real
    return total

def _objective(params, n, edges, p):
    """Objective: negative expected cut (to minimize)."""
    gammas = params[:p]
    betas = params[p:]
    psi = _simulate(n, edges, gammas, betas)
    return -_expected_cut(n, edges, psi)

def angles(n, edges, p):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    # Compute exact MaxCut for reference
    exact_opt = _exact_maxcut(n, edges)
    
    # Multi-start optimization
    best_angles = np.array([0.35, 0.25] * p, dtype=np.float64)
    best_obj = -_expected_cut(n, edges, _simulate(n, edges, best_angles[:p], best_angles[p:]))
    
    np.random.seed(42)
    
    num_starts = 8
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        if start_idx == 0:
            x0 = np.array([0.35, 0.25] * p, dtype=np.float64)
        else:
            # Random initial guess in [0, pi]
            x0 = np.random.uniform(0, np.pi, size=2*p)
        
        try:
            result = minimize(
                _objective, x0, args=(n, edges, p),
                method='L-BFGS-B',
                bounds=[(0, np.pi)] * 2 * p,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_angles = result.x.copy()
        except Exception:
            continue
    
    return best_angles.reshape(p, 2)
