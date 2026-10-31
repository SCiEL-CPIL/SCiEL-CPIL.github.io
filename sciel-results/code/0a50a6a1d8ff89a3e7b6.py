import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return (N, N) matrix of squared pairwise distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, 1)
    d_sq_upper = d_sq[iu]
    min_d_sq = np.min(d_sq_upper)
    max_d_sq = np.max(d_sq_upper)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _get_hexagonal_guess():
    """
    Generate a good initial guess based on the hexagonal lattice.
    We pick the 16 points closest to the origin in the hexagonal lattice.
    """
    # Axial coordinates (q, r)
    # Convert to Cartesian: x = q + r/2, y = r * sqrt(3)/2
    
    # Generate points in a hexagonal region
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Hex distance check? Or just generate a large area and sort by distance
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            points.append((q, r, x, y, q*q + q*r + r*r))
            
    # Sort by distance squared from origin
    points.sort(key=lambda p: p[4])
    
    # Take the first 16
    selected = points[:16]
    
    P = np.zeros((N, 2))
    for i, (q, r, x, y, d2) in enumerate(selected):
        P[i] = [x, y]
        
    return P

def _objective_exact(params):
    """Objective: (max_d_sq / min_d_sq)."""
    P = params.reshape(-1, 2)
    return _score(P)

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # 1. Initial Guess
    P_init = _get_hexagonal_guess()
    
    # Normalize P_init so that min distance is 1?
    # The score is scale invariant, but for numerical stability in optimizers,
    # it's good to have a reasonable scale.
    # Let's just use the raw coordinates.
    
    best_score = _score(P_init)
    best_P = P_init.copy()
    
    # 2. Optimization with Nelder-Mead
    # Nelder-Mead works on flattened params
    params_init = P_init.flatten()
    
    try:
        res_nm = minimize(
            _objective_exact,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_P = res_nm.x.reshape(-1, 2)
    except Exception:
        pass
    
    # 3. Optimization with Powell
    if time.time() < deadline - 10:
        try:
            res_pw = minimize(
                _objective_exact,
                best_P.flatten(),
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if res_pw.fun < best_score:
                best_score = res_pw.fun
                best_P = res_pw.x.reshape(-1, 2)
        except Exception:
            pass
            
    # 4. Another Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res_nm2 = minimize(
                _objective_exact,
                best_P.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm2.fun < best_score:
                best_score = res_nm2.fun
                best_P = res_nm2.x.reshape(-1, 2)
        except Exception:
            pass
            
    # Final validation
    final_score = _score(best_P)
    if final_score > 1e6:
        # Fallback to grid if something went wrong
        return np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
        
    return best_P

# EVOLVE-BLOCK-END
