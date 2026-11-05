"""
Minimizes the third autocorrelation inequality constant C3 upper bound.

Strategy:
1. Direct optimization of the 400-dim vector h using L-BFGS.
2. The objective function uses a log-sum-exp surrogate for the max operator 
   to provide smooth gradients, avoiding the non-differentiability of np.max.
3. Multiple random restarts with different initial conditions (Gaussian noise, 
   constant, sinusoidal).
4. A final phase of simulated annealing on the raw h vector to escape local 
   minima found by the gradient descent.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0  # safety margin before 150s kill

def _evaluate(h):
    """Compute the exact objective value as defined by the task."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_objective(h):
    """
    Smooth surrogate objective using log-sum-exp for max.
    We want to minimize 2*N*max(conv)/S^2.
    Let C = convolve(h, h).
    max(C) approx sum(exp(beta*C)) / exp(beta) ... actually logsumexp is (1/beta)log(sum(exp(beta*C))).
    To keep it simple and differentiable:
    f(h) = 2*N * logsumexp(beta * conv) / (sum(h)^2)
    Note: logsumexp is an upper bound on max. Minimizing it drives the max down.
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    
    # Log-sum-exp trick for numerical stability
    beta = 50.0  # Sharpness of the max approximation
    max_conv = np.max(conv)
    # If max_conv is large, exp(beta*conv) might overflow.
    # We can scale by max_conv: logsumexp(x) = max(x) + log(sum(exp(x-max(x))))
    shifted = beta * (conv - max_conv)
    # exp(shifted) is <= 1, so sum is <= len(conv)
    lse = max_conv + np.log(np.sum(np.exp(shifted))) / beta
    
    # The objective is 2*N * max_conv / S^2
    # Using lse as a proxy for max_conv
    return 2.0 * N * lse / (s * s)

def _surrogate_grad(h):
    """
    Numerical gradient is too slow for 400 dims in L-BFGS if done naively?
    Actually, L-BFGS in scipy uses numerical diff by default if jac is not provided.
    For 400 dims, numerical diff takes ~400 evaluations per step.
    It might be better to provide an analytical gradient or use a simpler method.
    
    However, calculating the gradient of logsumexp(conv(h)) is feasible.
    Let L = logsumexp(beta * c), where c_i = sum_k h_k h_{i-k}.
    dL/dh_j = sum_i (beta * c'_i) * (beta * h_{i-j} * exp(beta*c_i) / sum(exp(beta*c)))
    
    This is complex. Let's stick to numerical gradient but limit iterations.
    Or, use a simpler objective that is easier to differentiate?
    
    Alternative: Use Nelder-Mead on the 400-dim space? No, too slow.
    
    Let's use L-BFGS with numerical gradient. To speed it up, we can use a 
    coarser beta or fewer iterations.
    
    Actually, let's try a different surrogate that is easier:
    Minimize 2*N * (sum(exp(beta*conv)) / exp(beta*max_conv)) / S^2 ?
    
    Let's just use scipy.optimize.minimize with method='L-BFGS-B' and let it 
    compute gradients numerically. We will limit maxiter to keep time down.
    """
    pass # Not implementing analytical gradient to keep code simple, relying on scipy's numerical diff.

def _optimize_from_start(h_start, time_budget):
    """Run L-BFGS optimization from a starting point."""
    h_start = np.array(h_start, dtype=np.float64)
    
    # Normalize to avoid scaling issues
    # The objective is scale invariant: f(ah) = f(h).
    # We can fix sum(h) = N to remove this degree of freedom?
    # No, the constraint is just sum != 0.
    # However, L-BFGS might drift in scale. 
    # Let's add a penalty for deviation from sum=0 or sum=N?
    # Actually, just normalizing h to sum=1 at each step is not straightforward in minimize.
    # We can just let it optimize.
    
    def obj(h):
        return _surrogate_objective(h)
        
    # Use L-BFGS-B
    # maxiter is the number of function evaluations? No, iterations.
    # With numerical gradient, 1 iter ~ 400 evals.
    # Let's allow ~20-30 iterations.
    start_time = time.time()
    try:
        res = minimize(
            obj, h_start, 
            method='L-BFGS-B', 
            options={'maxiter': 30, 'ftol': 1e-9, 'gtol': 1e-6, 'disp': False}
        )
        if res.fun < 1e9:
            return res.x, _evaluate(res.x)
    except Exception:
        pass
    return h_start, _evaluate(h_start)

def _simulated_annealing(h_init, time_budget):
    """
    Simple simulated annealing on the raw h vector.
    Moves: Add Gaussian noise to a random subset of elements.
    """
    h = np.array(h_init, dtype=np.float64, copy=True)
    current_val = _evaluate(h)
    best_h = h.copy()
    best_val = current_val
    
    start_time = time.time()
    T = 0.5  # Initial temperature
    cooling_rate = 0.95
    step_size = 0.5
    
    np.random.seed(123)
    
    while (time.time() - start_time) < time_budget:
        if T < 1e-4:
            break
            
        # Generate a move: perturb a few random indices
        num_changes = np.random.randint(1, 10)
        indices = np.random.randint(0, N, num_changes)
        deltas = np.random.normal(0, step_size, num_changes)
        
        h_new = h.copy()
        h_new[indices] += deltas
        
        new_val = _evaluate(h_new)
        
        # Accept if better, or with probability exp(-(new-old)/T)
        if new_val < current_val:
            h = h_new
            current_val = new_val
            if new_val < best_val:
                best_h = h_new.copy()
                best_val = new_val
        else:
            delta_e = new_val - current_val
            if delta_e < 0: # Should not happen if new_val < current_val handled above
                pass
            else:
                prob = np.exp(-delta_e / T)
                if np.random.rand() < prob:
                    h = h_new
                    current_val = new_val
        
        T *= cooling_rate
        # Decay step size slightly
        step_size *= 0.999
        
    return best_h, best_val

def construct():
    """
    Returns the optimal step heights.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE - 5.0  # 5s safety margin

    np.random.seed(42)

    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # 2.0

    # Phase 1: Multiple L-BFGS restarts
    # Different initializations
    initializations = []
    
    # 1. Constant
    initializations.append(np.ones(N))
    
    # 2. Gaussian noise
    initializations.append(np.random.randn(N))
    
    # 3. Sine wave
    n = np.arange(N)
    initializations.append(np.sin(2 * np.pi * n / N))
    
    # 4. Cosine wave
    initializations.append(np.cos(2 * np.pi * n / N))
    
    # 5. Random sparse
    h_sparse = np.zeros(N)
    h_sparse[np.random.choice(N, 100, replace=False)] = 1.0
    initializations.append(h_sparse)
    
    # 6. Ramp
    initializations.append(np.linspace(1, -1, N))
    
    # 7. Triangle
    initializations.append(1 - np.abs(n - N/2) / (N/2))
    
    # 8. Random smooth (low freq noise)
    freqs = np.random.randint(1, 5, 3)
    phases = np.random.uniform(0, 2*np.pi, 3)
    h_smooth = np.zeros(N)
    for f, p in zip(freqs, phases):
        h_smooth += np.sin(2*np.pi*f*n/N + p)
    initializations.append(h_smooth)
    
    # Run L-BFGS on each
    for i, h_init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        # Budget for this restart
        remaining = deadline - time.time() - 5.0
        budget_per_restart = min(15.0, remaining / max(1, len(initializations) - i))
        
        if budget_per_restart < 2.0:
            break
            
        h_opt, val_opt = _optimize_from_start(h_init, budget_per_restart)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            print(f"Restart {i}: Improved to {best_val:.6f}")

    # Phase 2: Simulated Annealing on the best result
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        # Use a portion of the remaining time for SA
        sa_budget = min(30.0, remaining - 3.0)
        if sa_budget > 5.0:
            h_sa, val_sa = _simulated_annealing(best_h, sa_budget)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
                print(f"SA: Improved to {best_val:.6f}")

    return best_h

# Ensure the function signature matches
# def construct(): ...
