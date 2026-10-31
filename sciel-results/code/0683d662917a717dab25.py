"""Erdos minimum overlap - optimize step heights to minimize max overlap.

Strategy:
- Directly optimize N=100 step heights h in [0,1] with sum N/2.
- Objective: Minimize the maximum overlap O_k for shifts k=0..N-1.
- Use a smooth max (log-sum-exp) surrogate to make the objective differentiable.
- Use L-BFGS-B with analytical gradients for speed.
- Handle the sum constraint via a penalty term in the objective for L-BFGS-B,
  then refine with SLSQP using hard constraints for the best candidate.
- Multiple random restarts to escape local minima.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """
    Compute the overlap values O_k for all shifts k=0..N-1.
    O_k = (2/N) * (N/2 - sum_i h_i * h_{i+k})
    """
    # Circular autocorrelation
    # sum_i h_i h_{i+k} is the circular cross-correlation at lag k
    # Using FFT for efficiency, though N=100 is small, it's cleaner.
    # Actually, for N=100, a simple loop or np.roll is fast enough and avoids FFT overhead in tight loops if not careful.
    # Let's use np.roll for clarity and sufficient speed.
    
    overlaps = np.zeros(N)
    # Pre-compute sum(h) which is constant during optimization if constraint is enforced, 
    # but here we use penalty, so sum(h) varies.
    sum_h = np.sum(h)
    
    for k in range(N):
        # sum_i h_i * h_{i+k}
        # h_{i+k} is h shifted by -k (circular)
        shifted = np.roll(h, -k)
        dot = np.dot(h, shifted)
        overlaps[k] = (2.0 / N) * (sum_h - dot)
        
    return overlaps

def _soft_max_and_grad(h, beta):
    """
    Compute the soft max of the overlaps and its gradient.
    Returns: (soft_max_value, gradient_vector)
    """
    overlaps = _compute_overlaps(h)
    
    # Stable log-sum-exp
    max_o = np.max(overlaps)
    # Shift to prevent overflow
    exps = np.exp(beta * (overlaps - max_o))
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return max_o, np.zeros(N)
        
    soft_max_val = max_o + np.log(sum_exps) / beta
    
    # Weights for the soft max
    weights = exps / sum_exps
    
    # Gradient of the soft max
    # d(soft_max)/dh_j = sum_k weights[k] * d(O_k)/dh_j
    # d(O_k)/dh_j = -2/N * (h_{j+k} + h_{j-k})  [Note: derivative of sum_h - dot term]
    # Wait, O_k = 2/N * (sum_h - dot_k).
    # d(sum_h)/dh_j = 1.
    # d(dot_k)/dh_j = h_{j+k} + h_{j-k} (since dot_k = sum_i h_i h_{i+k}, derivative w.r.t h_j is h_{j+k} from first term and h_{j-k} from second term when i+k=j => i=j-k).
    # So d(O_k)/dh_j = 2/N * (1 - (h_{j+k} + h_{j-k})).
    
    grad = np.zeros(N)
    # We can compute this efficiently.
    # grad[j] = 2/N * sum_k weights[k] * (1 - h_{j+k} - h_{j-k})
    # grad[j] = 2/N * (sum_k weights[k] - sum_k weights[k]h_{j+k} - sum_k weights[k]h_{j-k})
    # sum_k weights[k] = 1.
    # So grad[j] = 2/N * (1 - (conv_w_h[j] + conv_w_h_rev[j]))
    # where conv_w_h is circular convolution of weights and h.
    
    # Let's just use the direct loop for N=100, it's O(N^2) = 10k ops, very fast.
    for k in range(N):
        if weights[k] < 1e-12:
            continue
        factor = (2.0 / N) * weights[k]
        # d(O_k)/dh_j = factor * (1 - h_{j+k} - h_{j-k})
        # h_{j+k} is np.roll(h, -k)
        # h_{j-k} is np.roll(h, k)
        # We add to grad
        grad += factor * (1.0 - np.roll(h, -k) - np.roll(h, k))
        
    return soft_max_val, grad

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Target sum
    target_sum = N / 2.0
    
    # Penalty coefficient for sum constraint in L-BFGS-B
    penalty_weight = 100.0
    
    def objective_and_grad(h):
        """
        Objective: Soft Max of Overlaps + Penalty for Sum Constraint.
        Gradient: Analytical gradient of Soft Max + Gradient of Penalty.
        """
        sm_val, sm_grad = _soft_max_and_grad(h, beta=20.0)
        
        # Penalty term
        sum_h = np.sum(h)
        penalty = penalty_weight * (sum_h - target_sum) ** 2
        penalty_grad = 2.0 * penalty_weight * (sum_h - target_sum) * np.ones(N)
        
        # Clip h to [0,1] for the objective evaluation to keep values sane,
        # but L-BFGS-B handles bounds. The gradient should ideally respect bounds,
        # but we rely on L-BFGS-B's box constraints.
        # However, if h goes out of bounds, the overlap formula might be weird.
        # L-BFGS-B keeps h in bounds.
        
        val = sm_val + penalty
        grad = sm_grad + penalty_grad
        
        return val, grad

    def objective_only(h):
        """For SLSQP, which doesn't take a jac tuple easily or we provide separate."""
        sm_val, _ = _soft_max_and_grad(h, beta=50.0) # Higher beta for sharper max in final polish
        return sm_val

    def grad_only(h):
        """Gradient for SLSQP."""
        _, grad = _soft_max_and_grad(h, beta=50.0)
        return grad

    def constraint_sum(h):
        return np.sum(h) - target_sum

    def constraint_sum_jac(h):
        return np.ones(N)

    # Initializations
    inits = []
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Sinusoidal patterns (exploiting symmetry or breaking it)
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.2, 0.4, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            # Project to sum constraint and bounds roughly
            h = np.clip(h, 0, 1)
            diff = target_sum - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            inits.append(h)
            
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        h = rng.uniform(0, 1, N)
        # Project to sum
        diff = target_sum - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        inits.append(h)

    # Optimization Loop
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        h_init = h0.copy()
        
        try:
            # Phase 1: L-BFGS-B with penalty
            # We use a lower beta for smoother landscape
            res1 = minimize(
                objective_and_grad, 
                h_init, 
                method='L-BFGS-B',
                jac=True, # Returns (f, g)
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            h_opt1 = res1.x
            # Fix sum constraint roughly
            diff = target_sum - np.sum(h_opt1)
            h_opt1 += diff / N
            h_opt1 = np.clip(h_opt1, 0, 1)
            
            # Check score
            overlaps1 = _compute_overlaps(h_opt1)
            score1 = np.max(overlaps1)
            
            if score1 < best_score:
                best_score = score1
                best_h = h_opt1.copy()
                
            # Phase 2: SLSQP Refinement with Hard Constraint
            # Only if we have time and the initial result is good
            if time.time() < deadline - 5.0:
                res2 = minimize(
                    objective_only,
                    h_opt1,
                    method='SLSQP',
                    jac=grad_only,
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_jac}],
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if res2.success:
                    h_opt2 = res2.x
                    # Enforce exact sum
                    diff = target_sum - np.sum(h_opt2)
                    # Distribute error to the variable with the most "room" (closest to 0.5 or away from bounds?)
                    # Simple average distribution
                    h_opt2 += diff / N
                    h_opt2 = np.clip(h_opt2, 0, 1)
                    
                    overlaps2 = _compute_overlaps(h_opt2)
                    score2 = np.max(overlaps2)
                    
                    if score2 < best_score:
                        best_score = score2
                        best_h = h_opt2.copy()

        except Exception:
            continue

    # Final sanity check and exact constraint enforcement
    if np.sum(best_h) != target_sum:
        diff = target_sum - np.sum(best_h)
        # Find indices where we can adjust without hitting bounds
        # Simple approach: add to all, clip, then fix residual on one element
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
        # If still not exact due to clipping, adjust one element
        diff = target_sum - np.sum(best_h)
        if abs(diff) > 1e-9:
            # Pick an element not at 0 or 1
            mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
            if np.any(mask):
                idx = np.where(mask)[0][0]
                best_h[idx] += diff
            else:
                # Fallback: uniform
                best_h = np.full(N, 0.5)
                
    return best_h
