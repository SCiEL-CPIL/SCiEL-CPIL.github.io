"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 15.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Uses state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: exp(-i gamma1 * C)
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: exp(-i beta1 * sum X)
    if abs(beta1) > 1e-9:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)
            
    # Layer 2: exp(-i gamma2 * C)
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: exp(-i beta2 * sum X)
    if abs(beta2) > 1e-9:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using a multi-start Nelder-Mead strategy.
    This is a different strategy from the current program's Dual Annealing.
    We use a grid of initial points to cover the parameter space more uniformly,
    then run Nelder-Mead from each. This can be more effective for the specific
    landscape of QAOA MaxCut, which may have many local optima but also
    smooth basins of attraction.
    """
    
    # Define objective for minimization (we want to maximize expected cut)
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val

    # Bounds: Gamma in [0, pi], Beta in [0, pi/2] is typical for QAOA MaxCut.
    # However, to be safe, let's allow [0, pi] for both, but restrict Beta to [0, pi/2] 
    # as it's known that for MaxCut, beta > pi/2 is not optimal due to symmetry.
    # Actually, let's stick to [0, pi] for both to be fully general, but the 
    # initial points will be biased towards the expected region.
    
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi, np.pi, np.pi]
    
    # Generate a set of initial points.
    # We'll use a combination of random points and points based on known heuristics.
    # Heuristic: gamma1 ~ pi/4, beta1 ~ pi/8, gamma2 ~ pi/8, beta2 ~ pi/16 (roughly)
    # Or simply uniform random in the box.
    
    num_starts = 20
    rng = np.random.RandomState(42)
    
    initial_points = []
    # Add a few heuristic points
    initial_points.append([0.5, 0.25, 0.25, 0.125])
    initial_points.append([0.7, 0.35, 0.35, 0.17])
    initial_points.append([0.3, 0.15, 0.15, 0.07])
    initial_points.append([1.0, 0.5, 0.5, 0.25])
    
    # Add random points
    for _ in range(num_starts - len(initial_points)):
        pt = rng.uniform(lower, upper)
        initial_points.append(pt)
    
    best_x = None
    best_val = np.inf
    
    for x0 in initial_points:
        if _remaining() < 5.0:
            break
            
        try:
            res = minimize(
                obj, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        # Fallback
        best_x = np.array([0.5, 0.25, 0.25, 0.125])
        best_val = -_expected_cut(n, cuts, *best_x)
        
    # Final polish
    def obj_polish(x):
        if _remaining() < 1.0:
            return 1e10
        return -_expected_cut(n, cuts, *x)
        
    try:
        res_polish = minimize(
            obj_polish, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_polish.fun < best_val:
            best_x = res_polish.x
            best_val = res_polish.fun
    except Exception:
        pass
        
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
