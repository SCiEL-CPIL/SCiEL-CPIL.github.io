"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import correlate

def _objective_and_grad(params):
    """
    params: (N-1,) vector of differences d where h[i] = 0.5 + cumsum(d)
    But we need sum(h) = 50. Let's use a different parametrization.
    
    Actually, let's use: h = 0.5 + A @ s, where A is a matrix that ensures sum(h) = 50.
    Simplest: let h = 0.5 + c, where sum(c) = 0.
    Use c = P @ s, where P is a projection onto the hyperplane sum=0.
    Or simpler: parametrize by N-1 free variables and set the last to enforce sum=0.
    
    Let h[0..N-2] be free, and h[N-1] = 50 - sum(h[0..N-2]).
    Then h[i] in [0,1] for all i.
    
    For the objective: value = max(correlate(h, 1-h, 'full')) / N * 2
    We want to minimize this.
    """
    h = np.empty(N)
    h[:-1] = params
    h[-1] = 50.0 - np.sum(params)
    
    # Check bounds
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return 1e6, None
    
    # Compute correlation
    # np.correlate(h, 1-h, mode='full')
    # This is O(N^2) but N=100 so it's fine
    corr = correlate(h, 1.0 - h, mode='full')
    val = np.max(corr) / N * 2.0
    return val, None


def _objective_with_grad(params):
    """Objective with gradient for SLSQP."""
    h = np.empty(N)
    h[:-1] = params
    h[-1] = 50.0 - np.sum(params)
    
    # Check bounds
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return 1e6
    
    # Compute correlation
    corr = correlate(h, 1.0 - h, mode='full')
    val = np.max(corr) / N * 2.0
    
    # Gradient: we need d(val)/d(params)
    # val = max_k corr[k] / N * 2
    # The max is at some index k*. Let's compute the gradient of corr[k*] w.r.t. h
    # corr[k] = sum_i h[i] * (1-h)[k-i] (depending on correlation definition)
    # Actually np.correlate(h, 1-h, mode='full') gives:
    # result[m] = sum_n h[n] * (1-h)[n + m - (N-1)] for m in 0..2N-2
    # Let's just use numerical gradient since analytical is complex
    # But for speed, let's use finite differences
    eps = 1e-7
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        h_plus = np.empty(N)
        h_plus[:-1] = params_plus
        h_plus[-1] = 50.0 - np.sum(params_plus)
        corr_plus = correlate(h_plus, 1.0 - h_plus, mode='full')
        val_plus = np.max(corr_plus) / N * 2.0
        
        params_minus = params.copy()
        params_minus[i] -= eps
        h_minus = np.empty(N)
        h_minus[:-1] = params_minus
        h_minus[-1] = 50.0 - np.sum(params_minus)
        corr_minus = correlate(h_minus, 1.0 - h_minus, mode='full')
        val_minus = np.max(corr_minus) / N * 2.0
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad


def _project_to_simplex(h):
    """Project h onto [0,1]^N with sum=50."""
    # First clip to [0,1]
    h_clipped = np.clip(h, 0.0, 1.0)
    # Adjust to make sum = 50
    # If sum is too high, reduce the largest values
    # If sum is too low, increase the smallest values
    s = np.sum(h_clipped)
    if abs(s - 50.0) < 1e-10:
        return h_clipped
    
    # Use iterative water-filling
    for _ in range(100):
        s = np.sum(h_clipped)
        diff = s - 50.0
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to reduce sum by diff
            # Reduce the largest values
            idx = np.argsort(h_clipped)[::-1]
            for i in idx:
                if diff <= 0:
                    break
                reduce = min(h_clipped[i], diff)
                h_clipped[i] -= reduce
                diff -= reduce
        else:
            # Need to increase sum by -diff
            # Increase the smallest values
            idx = np.argsort(h_clipped)
            for i in idx:
                if diff >= 0:
                    break
                increase = min(1.0 - h_clipped[i], -diff)
                h_clipped[i] += increase
                diff += increase
    
    return h_clipped


def _evaluate(h):
    """Compute the objective value for a valid h."""
    h = np.asarray(h)
    if h.shape != (N,) or not np.all(np.isfinite(h)):
        return np.inf
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return np.inf
    if abs(np.sum(h) - 50.0) > 1e-9 * N:
        return np.inf
    corr = correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial best: seed
    best_h = np.full(N, 0.5)
    best_val = _evaluate(best_h)  # Should be 0.5
    
    # Try a few random starts and SLSQP
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random start: h[i] in [0,1], sum=50
        # Generate random h and project
        h_random = np.random.uniform(0, 1, N)
        h_random = _project_to_simplex(h_random)
        
        # Parametrize: h[:-1] free, h[-1] = 50 - sum(h[:-1])
        params = h_random[:-1].copy()
        
        # Check if valid
        h_check = np.empty(N)
        h_check[:-1] = params
        h_check[-1] = 50.0 - np.sum(params)
        if np.any(h_check < -1e-9) or np.any(h_check > 1 + 1e-9):
            continue
        
        # SLSQP optimization
        try:
            res = minimize(
                _objective_with_grad,
                params,
                method='SLSQP',
                jac=True,
                bounds=[(0, 1)] * (N - 1),
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_val:
                h_new = np.empty(N)
                h_new[:-1] = res.x
                h_new[-1] = 50.0 - np.sum(res.x)
                val_new = _evaluate(h_new)
                if val_new < best_val and np.all(np.isfinite(h_new)):
                    best_h = h_new
                    best_val = val_new
        except Exception:
            pass
    
    # Multi-scale polish on the best solution
    if time.time() < deadline - 10.0:
        # Fine-tune with random coordinate search
        for phase in range(3):
            step_sizes = [0.05, 0.01, 0.001]
            steps = step_sizes[phase]
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                # Pick a random coordinate
                i = np.random.randint(N)
                # Try to adjust h[i] and h[j] to maintain sum
                j = np.random.randint(N)
                if i == j:
                    continue
                
                # Try moving mass from i to j
                delta = steps * np.random.choice([-1, 1])
                h_trial = best_h.copy()
                h_trial[i] += delta
                h_trial[j] -= delta
                
                if np.any(h_trial < 0) or np.any(h_trial > 1):
                    continue
                
                val_trial = _evaluate(h_trial)
                if val_trial < best_val:
                    best_h = h_trial
                    best_val = val_trial
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure sum is exactly 50
    s = np.sum(best_h)
    if abs(s - 50.0) > 1e-12:
        # Adjust the largest element
        idx = np.argmax(best_h)
        best_h[idx] += 50.0 - s
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
