"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _expected_cut_4params(n, cuts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut for 2-layer QAOA with 4 independent parameters."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: Gamma then Beta
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    psi = _apply_beta(psi, beta1, n)
    
    # Layer 2: Gamma then Beta
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    psi = _apply_beta(psi, beta2, n)
    
    # Expected cut
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _fidelity_4params(n, cuts, optimal_states, gamma1, beta1, gamma2, beta2):
    """Compute fidelity with the uniform superposition of optimal states."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    psi = _apply_beta(psi, beta1, n)
    
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    psi = _apply_beta(psi, beta2, n)
    
    # Target state: uniform over optimal states
    target = np.zeros(2**n, dtype=np.complex128)
    target[optimal_states] = 1.0 / np.sqrt(len(optimal_states))
    
    overlap = np.vdot(target, psi)
    return np.abs(overlap)**2

def _optimize_4params(n, cuts):
    """Optimize 4 QAOA parameters using a hybrid strategy."""
    
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([np.pi, 2*np.pi, np.pi, 2*np.pi])
    
    max_cut = np.max(cuts)
    optimal_states = np.where(cuts == max_cut)[0]
    
    def obj_cut(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        return -_expected_cut_4params(n, cuts, g1, b1, g2, b2)
    
    def obj_fidelity(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        return -_fidelity_4params(n, cuts, optimal_states, g1, b1, g2, b2)
    
    best_x = None
    best_val = np.inf
    
    # Phase 1: Global search with Differential Evolution on the cut objective
    try:
        de_res = differential_evolution(
            obj_cut, 
            bounds=list(zip(lower, upper)), 
            maxiter=25, 
            popsize=12, 
            tol=1e-5, 
            seed=42,
            polish=False
        )
        if de_res.fun < 1e9:
            best_val = de_res.fun
            best_x = de_res.x.copy()
    except Exception:
        pass
    
    # Phase 2: Multi-start local optimization with Nelder-Mead on cut objective
    if _remaining() > 10.0:
        # Physics-informed initial guesses
        initial_guesses = [
            [0.5, 0.5, 0.5, 0.5],
            [0.7, 0.4, 0.7, 0.4],
            [0.3, 0.3, 0.3, 0.3],
            [0.2, 0.6, 0.2, 0.6],
            [1.0, 0.5, 1.0, 0.5],
            [0.4, 0.2, 0.4, 0.2],
            [0.8, 0.3, 0.8, 0.3],
            [0.1, 0.1, 0.1, 0.1],
            [0.6, 0.5, 0.6, 0.5],
            [0.9, 0.3, 0.9, 0.3],
        ]
        
        if best_x is not None:
            initial_guesses.append(best_x.copy())
        
        for x0 in initial_guesses:
            if _remaining() < 5.0:
                break
            try:
                res = minimize(
                    obj_cut, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < 1e9 and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # Phase 3: Fidelity-based optimization as a complementary signal
    if _remaining() > 15.0:
        # Use the best cut-optimal point as a starting point for fidelity
        start_points = [best_x] if best_x is not None else [[0.5, 0.5, 0.5, 0.5]]
        
        # Also try a few random points for fidelity
        rng = np.random.default_rng(123)
        for _ in range(5):
            start_points.append(rng.uniform(lower, upper))
        
        for x0 in start_points:
            if _remaining() < 5.0:
                break
            try:
                res = minimize(
                    obj_fidelity, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < 1e9:
                    # Evaluate this fidelity-optimal point on the actual cut objective
                    cut_val = obj_cut(res.x)
                    if cut_val < best_val:
                        best_val = cut_val
                        best_x = res.x.copy()
            except Exception:
                pass

    # Phase 4: High-precision polish with L-BFGS-B using numerical gradients
    if _remaining() > 5.0 and best_x is not None:
        def obj_grad(x):
            val = obj_cut(x)
            if val >= 1e9:
                return val, np.zeros(4)
            eps = 1e-7
            grad = np.zeros(4)
            for i in range(4):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                grad[i] = (obj_cut(xp) - obj_cut(xm)) / (2 * eps)
            return val, grad
        
        try:
            res = minimize(
                obj_grad, 
                best_x, 
                method='L-BFGS-B', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < 1e9 and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Phase 5: Final ultra-fine Nelder-Mead polish
    if _remaining() > 3.0 and best_x is not None:
        try:
            res_nm = minimize(
                obj_cut, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if res_nm.fun < 1e9 and res_nm.fun < best_val:
                best_val = res_nm.fun
                best_x = res_nm.x.copy()
        except Exception:
            pass

    if best_x is None:
        best_x = np.array([0.5, 0.5, 0.5, 0.5])

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
