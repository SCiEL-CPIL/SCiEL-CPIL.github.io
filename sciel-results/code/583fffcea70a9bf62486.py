"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
EVAL_DEADLINE = 150.0


def _raw_value(h):
    """Compute the objective: max correlation / len * 2. Lower is better."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / len(h) * 2.0


def _is_valid(h):
    """Check constraints: finite, in [0,1], sum within tolerance."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return False
    if abs(np.sum(h) - len(h) / 2.0) > 1e-9 * len(h):
        return False
    return True


def _project(h):
    """Project h onto the feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to meet sum constraint
    target = len(h) / 2.0
    s = np.sum(h)
    if abs(s - target) < 1e-12:
        return h
    # Simple projection: scale non-bound variables
    # Use iterative water-filling style approach
    for _ in range(100):
        s = np.sum(h)
        diff = target - s
        if abs(diff) < 1e-14:
            break
        # Adjust only non-bound variables
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if not np.any(mask):
            break
        n_active = np.sum(mask)
        adj = diff / n_active
        h[mask] += adj
        h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max(corr, beta=100.0):
    """Soft-max of the correlation vector."""
    # Subtract max for numerical stability
    m = np.max(corr)
    return m + np.log(np.sum(np.exp(beta * (corr - m)))) / beta


def _objective_and_grad(h, beta=100.0):
    """Compute soft-max objective and its gradient."""
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    
    # Soft-max value
    m = np.max(corr)
    exp_terms = np.exp(beta * (corr - m))
    log_sum = np.log(np.sum(exp_terms))
    val = m + log_sum / beta
    
    # Gradient of soft-max w.r.t. corr
    probs = exp_terms / np.sum(exp_terms)
    
    # Now we need d(corr)/d(h)
    # corr[k] = sum_i h[i] * (1-h[i+k-99]) for appropriate ranges
    # Actually np.correlate(h, 1-h, mode="full") has length 2*N-1
    # corr[k] = sum_i h[i] * (1-h)[i + k - (N-1)] for valid i
    # Let's compute gradient via the structure
    # corr[k] = sum_i h[i] * (1 - h[i + k - N + 1])
    # d(corr[k])/d(h[j]) = (1-h[j+k-N+1]) - h[j] * (if j+k-N+1 in range... )
    # This is getting complex. Let's use finite differences for gradient or just use SLSQP with finite diff.
    
    return val


def _construct():
    """Construct 100 step heights minimizing the Erdos overlap constant."""
    start_time = time.time()
    
    # Start with a structured initial guess
    # Alternate high and low values to minimize overlap
    h0 = np.zeros(N)
    h0[::2] = 1.0
    h0[1::2] = 0.0
    # This gives sum = 50, which is N/2 = 50. Good.
    # But let's verify: 50 ones and 50 zeros, sum = 50. Yes.
    
    best_h = h0.copy()
    best_val = _raw_value(best_h)
    
    # Try several different initial patterns
    patterns = []
    
    # Pattern 1: alternating
    patterns.append(h0.copy())
    
    # Pattern 2: block structure
    h2 = np.zeros(N)
    h2[:50] = 1.0
    patterns.append(h2.copy())
    
    # Pattern 3: sinusoidal
    t = np.linspace(0, 4 * np.pi, N)
    h3 = 0.5 + 0.5 * np.sin(t)
    h3 = np.clip(h3, 0, 1)
    patterns.append(h3.copy())
    
    # Pattern 4: random
    np.random.seed(42)
    h4 = np.random.uniform(0, 1, N)
    patterns.append(h4.copy())
    
    # Pattern 5: more alternating with noise
    h5 = h0.copy()
    h5 += np.random.normal(0, 0.1, N)
    h5 = np.clip(h5, 0, 1)
    patterns.append(h5.copy())
    
    for i, h_init in enumerate(patterns):
        if time.time() - start_time > EVAL_DEADLINE - DEADLINE_MARGIN:
            break
        
        # Project to feasible set
        h_init = _project(h_init)
        
        if not _is_valid(h_init):
            continue
        
        val_init = _raw_value(h_init)
        if val_init < best_val:
            best_val = val_init
            best_h = h_init.copy()
        
        # Use SLSQP to minimize
        try:
            def obj(h_vec):
                return _raw_value(h_vec)
            
            result = minimize(
                obj,
                h_init,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_val:
                h_res = _project(result.x)
                if _is_valid(h_res):
                    val_res = _raw_value(h_res)
                    if val_res < best_val:
                        best_val = val_res
                        best_h = h_res.copy()
        except Exception:
            pass
        
        # Try COBYLA as well
        if time.time() - start_time > EVAL_DEADLINE - DEADLINE_MARGIN:
            break
        try:
            def obj2(h_vec):
                return _raw_value(h_vec)
            
            result2 = minimize(
                obj2,
                h_init,
                method='COBYLA',
                bounds=[(0, 1)] * N,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}
                ],
                options={'maxiter': 1000, 'rhobeg': 0.01}
            )
            
            if result2.fun < best_val:
                h_res2 = _project(result2.x)
                if _is_valid(h_res2):
                    val_res2 = _raw_value(h_res2)
                    if val_res2 < best_val:
                        best_val = val_res2
                        best_h = h_res2.copy()
        except Exception:
            pass
    
    # Final polish with more iterations
    if time.time() - start_time < EVAL_DEADLINE - DEADLINE_MARGIN:
        try:
            def obj3(h_vec):
                return _raw_value(h_vec)
            
            result3 = minimize(
                obj3,
                best_h,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            
            if result3.fun < best_val:
                h_res3 = _project(result3.x)
                if _is_valid(h_res3):
                    val_res3 = _raw_value(h_res3)
                    if val_res3 < best_val:
                        best_val = val_res3
                        best_h = h_res3.copy()
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project(best_h)
    if not _is_valid(best_h):
        # Fallback to seed
        best_h = np.full(N, 0.5)
    
    return best_h


def construct():
    return _construct()
# EVOLVE-BLOCK-END
