import numpy as np
import time
from scipy.optimize import minimize

def _precompute(n, edges):
    """Precompute cut values and edge cut indicators."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int32)
    cut_vals = np.zeros(dim, dtype=np.float64)
    edge_cuts = []
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        ec = (bits_i ^ bits_j).astype(np.float64)
        edge_cuts.append(ec)
        cut_vals += ec
    return cut_vals, edge_cuts, states

def _eval(n, cut_vals, edge_cuts, states, params):
    """Evaluate expected cut for given parameters [gamma1, beta1, gamma2, beta2]."""
    dim = 1 << n
    gamma1, beta1, gamma2, beta2 = params
    
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # exp(-i*gamma1*C)
    psi *= np.exp(-1j * gamma1 * cut_vals)
    
    # exp(-i*beta1*sum X)
    c1 = np.cos(beta1 / 2)
    s1 = np.sin(beta1 / 2)
    for k in range(n):
        mask = 1 << k
        idx = np.where((states & mask) == 0)[0]
        partner = idx ^ mask
        oi = psi[idx].copy()
        op = psi[partner].copy()
        psi[idx] = c1 * oi - 1j * s1 * op
        psi[partner] = -1j * s1 * oi + c1 * op
    
    # exp(-i*gamma2*C)
    psi *= np.exp(-1j * gamma2 * cut_vals)
    
    # exp(-i*beta2*sum X)
    c2 = np.cos(beta2 / 2)
    s2 = np.sin(beta2 / 2)
    for k in range(n):
        mask = 1 << k
        idx = np.where((states & mask) == 0)[0]
        partner = idx ^ mask
        oi = psi[idx].copy()
        op = psi[partner].copy()
        psi[idx] = c2 * oi - 1j * s2 * op
        psi[partner] = -1j * s2 * oi + c2 * op
    
    probs = np.abs(psi) ** 2
    expected_cut = 0.0
    for ec in edge_cuts:
        expected_cut += np.dot(probs, ec)
    
    return expected_cut

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s kill
    
    # Precompute
    cut_vals, edge_cuts, states = _precompute(n, edges)
    
    # Multi-start Nelder-Mead optimization
    best_val = -1
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Initial guesses: spread out
    np.random.seed(42)
    num_starts = 200
    
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # Random initial point in [0, pi] x [0, pi] x [0, pi] x [0, pi]
        x0 = np.random.uniform(0, np.pi, size=4)
        
        def objective(x):
            return -_eval(n, cut_vals, edge_cuts, states, x)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        def objective(x):
            return -_eval(n, cut_vals, edge_cuts, states, x)
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except:
            pass
    
    # Return angles as (p, 2) array: rows are (gamma, beta)
    result = np.array([[best_params[0], best_params[1]],
                       [best_params[2], best_params[3]]])
    return result
