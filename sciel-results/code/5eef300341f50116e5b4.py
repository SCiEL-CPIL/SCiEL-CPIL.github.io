# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

_IDX_D = (_XS - _YS) + 9
_N_D = 19
_IDX_H0 = _XS
_N_H0 = 10
_IDX_H1 = _XS + _YS
_N_H1 = 19
_IDX_H2 = _XS + 2 * _YS
_N_H2 = 28


def _shannon_entropy(p):
    p = p[p > 1e-30]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_entropies(P):
    p = P.ravel()
    
    d_counts = np.zeros(_N_D)
    np.add.at(d_counts, _IDX_D, p)
    d_counts /= d_counts.sum()
    H_D = _shannon_entropy(d_counts)
    
    h0_counts = np.zeros(_N_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0_counts /= h0_counts.sum()
    H_0 = _shannon_entropy(h0_counts)
    
    h1_counts = np.zeros(_N_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1_counts /= h1_counts.sum()
    H_1 = _shannon_entropy(h1_counts)
    
    h2_counts = np.zeros(_N_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2_counts /= h2_counts.sum()
    H_2 = _shannon_entropy(h2_counts)
    
    return H_D, H_0, H_1, H_2


def _objective(theta):
    # theta: 100 log-weights
    p_flat = np.exp(theta - np.max(theta))
    p_flat /= p_flat.sum()
    P = p_flat.reshape(_N, _N)
    
    H_D, H_0, H_1, H_2 = _compute_entropies(P)
    
    # Soft max of H_0, H_1, H_2
    beta = 50.0
    Hs = np.array([H_0, H_1, H_2])
    M_soft = (1.0 / beta) * np.log(np.sum(np.exp(beta * Hs)))
    
    if M_soft <= 1e-9:
        return 0.0
    
    return H_D / M_soft


def _neg_objective(theta):
    return -_objective(theta)


def _get_P_from_theta(theta):
    p_flat = np.exp(theta - np.max(theta))
    p_flat /= p_flat.sum()
    P = p_flat.reshape(_N, _N)
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    t_start = time.time()
    time_limit = 630.0
    
    best_val = -np.inf
    best_P = None
    
    # Strategy 1: Differential Evolution for global exploration
    bounds = [(-20, 20)] * (_N * _N)
    
    try:
        # Use a smaller popsize and maxiter to fit in time
        res_de = differential_evolution(_neg_objective, bounds, maxiter=50, popsize=10, tol=1e-6, seed=42, polish=False)
        
        if time.time() - t_start > time_limit - 30:
            raise TimeoutError("DE took too long")
            
        theta_best = res_de.x
        val_best = _objective(theta_best)
        
        if val_best > best_val:
            best_val = val_best
            best_P = _get_P_from_theta(theta_best)
            
    except Exception:
        pass
    
    # Strategy 2: L-BFGS-B Polish
    polish_starts = []
    if best_P is not None:
        # We need to recover theta from P? Or just store theta_best from DE.
        # Let's store theta_best separately.
        pass
    
    # Let's re-run DE logic to capture theta_best
    theta_de = None
    try:
        res_de = differential_evolution(_neg_objective, bounds, maxiter=50, popsize=10, tol=1e-6, seed=42, polish=False)
        if time.time() - t_start < time_limit - 30:
            theta_de = res_de.x
            val_de = _objective(theta_de)
            if val_de > best_val:
                best_val = val_de
                best_P = _get_P_from_theta(theta_de)
    except Exception:
        pass
    
    # Add uniform start
    theta_uniform = np.zeros(_N * _N)
    
    # Add theta_de if available
    if theta_de is not None:
        polish_starts.append(theta_de)
    
    polish_starts.append(theta_uniform)
    
    # Add a few random starts
    for _ in range(3):
        polish_starts.append(np.random.uniform(-5, 5, _N * _N))
    
    for theta_init in polish_starts:
        if time.time() - t_start > time_limit - 10:
            break
        
        try:
            res = minimize(_neg_objective, theta_init, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
            
            val_res = -res.fun
            if val_res > best_val:
                best_val = val_res
                best_P = _get_P_from_theta(res.x)
        except Exception:
            pass
    
    if best_P is None:
        best_P = np.ones((_N, _N)) / (_N * _N)
    
    # Final check
    P = np.clip(best_P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P /= s
        
    return P

# EVOLVE-BLOCK-END
