"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 10.0
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

def _apply_beta(psi, beta, n):
    """Apply exp(-i beta sum_k X_k) to state vector psi."""
    c = np.cos(beta)
    s = np.sin(beta)
    for k in range(n):
        shape = (2**(n-k-1), 2, 2**k)
        psi_reshaped = psi.reshape(shape)
        p0 = psi_reshaped[:, 0, :].copy()
        p1 = psi_reshaped[:, 1, :].copy()
        psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
        psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
        psi = psi_reshaped.reshape(-1)
    return psi

def _state_after_3_layers(n, cuts, gamma1, beta1, gamma2):
    """Compute state vector after gamma1, beta1, gamma2."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: Gamma
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    
    # Layer 1: Beta
    psi = _apply_beta(psi, beta1, n)
    
    # Layer 2: Gamma
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    
    return psi

def _optimal_beta2(psi, cuts):
    """
    Find the beta2 that maximizes expected cut for a given state psi 
    after the first 3 operations.
    The final state is exp(-i beta2 sum X) psi.
    We want to maximize <psi| exp(i beta2 sum X) C exp(-i beta2 sum X) |psi>.
    This is equivalent to maximizing the expectation of C in the rotated frame.
    However, a simpler heuristic for p=2 QAOA is that the optimal beta2 
    often aligns with the "max cut" direction. 
    Actually, we can just scan beta2 finely for the best local value, 
    or use the fact that for small p, the optimal beta2 is often near pi/2 or 0.
    Let's do a fine scan for beta2 in [0, pi] since the objective is periodic.
    """
    best_val = -np.inf
    best_beta = 0.0
    # The function is smooth. A coarse-to-fine scan is efficient.
    # First pass: 100 points
    for b in np.linspace(0, np.pi, 100):
        psi_final = _apply_beta(psi, b, psi.shape[0].bit_length() - 1) # n is log2(len)
        probs = np.abs(psi_final)**2
        val = np.sum(probs * cuts)
        if val > best_val:
            best_val = val
            best_beta = b
    # Second pass: refine around best_beta
    step = np.pi / 100
    for b in np.linspace(best_beta - step, best_beta + step, 50):
        if b < 0 or b > np.pi: continue
        psi_final = _apply_beta(psi, b, psi.shape[0].bit_length() - 1)
        probs = np.abs(psi_final)**2
        val = np.sum(probs * cuts)
        if val > best_val:
            best_val = val
            best_beta = b
    return best_beta, best_val

def _expected_cut_4params(n, cuts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut for 2-layer QAOA with 4 independent parameters."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    psi = _apply_beta(psi, beta1, n)
    
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    psi = _apply_beta(psi, beta2, n)
    
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using Grid Search on (g1, b1, g2) 
    with analytic/scan optimization of b2, followed by L-BFGS-B polish.
    """
    
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([np.pi, 2*np.pi, np.pi, 2*np.pi])
    
    best_x = np.array([0.5, 0.5, 0.5, 0.5])
    best_val = -_expected_cut_4params(n, cuts, *best_x)
    
    # Phase 1: Grid Search on g1, b1, g2
    # Rationale: The landscape for the first 3 params is often smoother 
    # and the global optimum is more likely to be found in a coarse grid 
    # than by stochastic DE which might get stuck in bad local minima 
    # due to the high dimensionality and non-convexity.
    
    # We use a moderate grid resolution to stay within time limits.
    # n is small (6-10), so state vector ops are fast.
    
    num_g = 15
    num_b = 15
    num_g2 = 15
    
    g_vals = np.linspace(0.1, np.pi - 0.1, num_g)
    b_vals = np.linspace(0.1, np.pi - 0.1, num_b)
    g2_vals = np.linspace(0.1, np.pi - 0.1, num_g2)
    
    grid_best_val = -np.inf
    grid_best_x = None
    
    total_checks = num_g * num_b * num_g2
    count = 0
    
    for g1 in g_vals:
        for b1 in b_vals:
            for g2 in g2_vals:
                if _remaining() < 20.0:
                    break
                count += 1
                # Compute state after 3 layers
                psi = _state_after_3_layers(n, cuts, g1, b1, g2)
                
                # Find optimal b2 for this prefix
                b2_opt, val_opt = _optimal_beta2(psi, cuts)
                
                if val_opt > grid_best_val:
                    grid_best_val = val_opt
                    grid_best_x = np.array([g1, b1, g2, b2_opt])
            if _remaining() < 20.0:
                break
        if _remaining() < 20.0:
            break
            
    if grid_best_x is not None:
        best_x = grid_best_x
        best_val = -grid_best_val # obj is negative
        
    # Phase 2: Local Polish with L-BFGS-B
    if _remaining() > 10.0:
        def obj(x):
            if _remaining() < 2.0:
                return 1e10
            g1, b1, g2, b2 = x
            return -_expected_cut_4params(n, cuts, g1, b1, g2, b2)
        
        def obj_grad(x):
            val = obj(x)
            if val >= 1e9:
                return val, np.zeros(4)
            eps = 1e-6
            grad = np.zeros(4)
            for i in range(4):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                grad[i] = (obj(xp) - obj(xm)) / (2 * eps)
            return val, grad
        
        try:
            res = minimize(
                obj_grad, 
                best_x, 
                method='L-BFGS-B', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if res.fun < 1e9 and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Nelder-Mead polish for any remaining local structure
    if _remaining() > 5.0:
        def obj(x):
            if _remaining() < 1.0:
                return 1e10
            g1, b1, g2, b2 = x
            return -_expected_cut_4params(n, cuts, g1, b1, g2, b2)
        
        try:
            res_nm = minimize(
                obj, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 20, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res_nm.fun < 1e9 and res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
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
