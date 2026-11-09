# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import itertools

def _compute_lp_coefficients(support):
    """
    For a fixed support grid, compute the linear coefficients c such that
    P = c @ w, where w are the weights (unnormalized, but LP handles sum(w)=1).
    
    P = P[X1 + X2 + X3 < 2*X4]
    Let S = X1 + X2. P[S + X3 < 2*X4] = sum_{k,l,i,j} w_k w_l w_i w_j * 1[s_i + s_j < 2*s_l - s_k]
    
    We can rearrange this sum. Note that the indicator depends on (i, j, k, l).
    Let I_{i,j,k,l} = 1 if s_i + s_j < 2*s_l - s_k else 0.
    P = sum_{k,l,i,j} w_k w_l w_i w_j I_{i,j,k,l}
    
    This is not linear in w. It is a quartic form.
    However, if we fix the support, can we linearize?
    No, P is homogeneous of degree 4 in w.
    Wait, the task says: "Formulate the problem as a linear program where the objective function coefficients are the exact probabilities..."
    
    Let's re-read carefully.
    "P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)"
    
    If we define Q_{kl} = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j, then P = sum_{k,l} w_k w_l Q_{kl}.
    This is still quadratic in w.
    
    Is there a way to make it linear?
    Maybe the inspiration implies fixing some weights? Or maybe it's a relaxation?
    Or maybe I should use a different structure.
    
    Actually, let's look at the structure again.
    If we assume a specific distribution shape, or if we can reduce the dimensionality?
    
    Alternative interpretation:
    Maybe the "linear program" refers to optimizing the weights given a *fixed* joint distribution structure?
    No, the weights are the variables.
    
    Let's reconsider the quartic nature.
    Maximize w^T A w w^T B w? No.
    
    Let's look at the inspiration text again: "Formulate the problem as a linear program where the objective function coefficients are the exact probabilities of the four independent draws summing to specific inequality outcomes, using scipy.optimize.linprog to find the optimal weight vector for a fixed, sparse support grid..."
    
    This implies the objective IS linear in the decision variables.
    How can P be linear in w?
    It can't, unless some w are fixed or the problem is structured differently.
    
    UNLESS: The "decision variables" are not the weights w_i, but something else?
    Or maybe the inspiration is slightly misleading and it's actually a *quadratic* program solved via other means, or it fixes the support and uses a method that works?
    
    Wait, if I fix the support, and I want to maximize P.
    P is a polynomial of degree 4 in w.
    Maximizing a non-convex polynomial is hard.
    
    However, if we restrict the support to be very small, or if we use a specific trick?
    
    Let's look at the metric again.
    P[X1+X2+X3 < 2X4].
    
    Is it possible that the inspiration meant "Linear Program" in the sense of Linear Programming being used to bound or approximate? Or maybe it's a typo in my understanding of the inspiration?
    
    Let's try a different angle. What if we use **Nelder-Mead** on the weights only, for a fixed, well-chosen support grid?
    The current program optimizes both support and weights.
    Optimizing support is non-smooth (due to the indicator function discontinuities when support points cross thresholds).
    Optimizing weights for a FIXED support is smooth (polynomial).
    
    So, a strategy materially different from "stochastic hill climbing on both" is:
    1. Define a fixed, high-quality support grid (e.g., based on quantiles of a known good distribution, or a uniform grid).
    2. Use a smooth optimization method (like Nelder-Mead or even Gradient Descent if we derive gradients) to optimize ONLY the weights.
    
    This avoids the non-smoothness of the support optimization.
    
    Let's refine the support grid.
    The inequality is $S_3 < 2 X_4$.
    If we choose support points that are "symmetric" or have specific spacing, we might get better results.
    
    Let's try:
    1. Generate a candidate support grid.
    2. Use `scipy.optimize.minimize` (Nelder-Mead) to maximize P by adjusting weights (in log-space to keep them positive).
    3. Try multiple different support grids.
    
    This is different from the current program which perturbs support and weights simultaneously in a random walk.
    
    Let's implement this.
    
    Grid ideas:
    - Uniform on [0, 1]
    - Geometric progression
    - Points based on the 2-point optimum (0, 1) expanded?
    
    Let's try a set of structured grids and optimize weights for each.
    """
    pass

def _evaluate_measure_fast(support, weights):
    """
    Fast evaluation for fixed support.
    """
    n = len(support)
    w = weights / np.sum(weights)
    
    # Precompute pairwise sums and their probabilities
    # S_ij = s_i + s_j
    # P(S_ij) = w_i w_j
    
    # We need P[X1+X2+X3 < 2X4]
    # Let Y = X1+X2. We need P[Y + X3 < 2X4].
    # This is sum_{k,l} w_k w_l P(Y < 2s_l - s_k).
    # Let T_kl = 2s_l - s_k.
    # P(Y < T_kl) = sum_{i,j: s_i+s_j < T_kl} w_i w_j.
    
    # Precompute all pair sums S and their weights W_pair
    S = support[:, None] + support[None, :]
    W_pair = w[:, None] * w[None, :]
    
    # Flatten
    S_flat = S.flatten()
    W_pair_flat = W_pair.flatten()
    
    # Sort S_flat to use searchsorted?
    # But W_pair_flat is tied to specific (i,j).
    # If we sort S_flat, we must permute W_pair_flat accordingly.
    # Then P(Y < T) = sum of W_pair_flat where S_flat < T.
    # This is a cumulative sum query.
    
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_pair_flat[sort_idx]
    
    # Cumulative sum of weights
    CDF = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            target = 2.0 * support[l] - support[k]
            # Find index of first element >= target
            # We want sum of weights where S < target
            # searchsorted returns index where target would be inserted to maintain order
            # So indices < idx have S < target (if we use 'left')?
            # np.searchsorted(S_sorted, target, side='left') returns idx such that S_sorted[idx-1] < target <= S_sorted[idx]
            # So we want sum(W_sorted[:idx])
            idx = np.searchsorted(S_sorted, target, side='left')
            prob_y = CDF[idx-1] if idx > 0 else 0.0
            P += w[k] * w[l] * prob_y
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590
    
    best_score = -np.inf
    best_support = None
    best_weights = None
    
    # Define a set of candidate support grids
    # We will optimize weights for each grid.
    
    grids = []
    
    # 1. Uniform grids of varying sizes
    for n in [4, 5, 6, 8, 10, 12]:
        grids.append(np.linspace(0, 1, n))
        grids.append(np.linspace(0, 2, n))
        grids.append(np.linspace(0, 5, n))
        
    # 2. Geometric grids
    for base in [1.5, 2.0, 3.0]:
        for n in [4, 5, 6]:
            grids.append(np.array([base**i for i in range(n)]))
            
    # 3. Specific structured grids
    # Points related to the inequality structure
    grids.append(np.array([0, 0.5, 1, 2, 4]))
    grids.append(np.array([0, 1, 2, 4, 8]))
    grids.append(np.array([0, 0.1, 0.5, 1, 2, 5]))
    
    # 4. Random grids (sorted)
    np.random.seed(42)
    for _ in range(20):
        n = np.random.randint(4, 9)
        pts = np.sort(np.random.uniform(0, 5, n))
        grids.append(pts)
        
    # Remove duplicate grids (approximate)
    unique_grids = []
    for g in grids:
        is_dup = False
        for ug in unique_grids:
            if len(g) == len(ug) and np.allclose(g, ug):
                is_dup = True
                break
        if not is_dup:
            unique_grids.append(g)
            
    grids = unique_grids
    
    from scipy.optimize import minimize
    
    for support in grids:
        if time.time() > deadline:
            break
            
        n = len(support)
        
        # Initial weights: Uniform
        w_init = np.ones(n) / n
        
        # We optimize log-weights to ensure positivity
        # Let log_w be the variable.
        # We want to maximize P(w) where w = softmax(log_w).
        # Note: softmax is invariant to adding a constant, so we can fix one log_w to 0 or just let it vary.
        # Nelder-Mead works on unconstrained variables.
        
        def objective(log_w):
            # Convert to weights
            # Subtract max for numerical stability
            max_lw = np.max(log_w)
            exp_lw = np.exp(log_w - max_lw)
            w = exp_lw / np.sum(exp_lw)
            
            # Evaluate P
            p = _evaluate_measure_fast(support, w)
            return -p # Minimize negative P
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, np.zeros(n), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < 0: # res.fun is -P, so lower is better
                # Extract best weights
                max_lw = np.max(res.x)
                exp_lw = np.exp(res.x - max_lw)
                w_opt = exp_lw / np.sum(exp_lw)
                
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_support = support.copy()
                    best_weights = w_opt.copy()
                    
        except Exception:
            continue
            
    # If no grid worked, fallback
    if best_support is None:
        best_support = np.arange(4, dtype=np.float64)
        best_weights = np.ones(4, dtype=np.float64) / 4
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            best_support = np.arange(4, dtype=np.float64)
            best_weights = np.ones(4, dtype=np.float64) / 4
            
    return best_support, best_weights
# EVOLVE-BLOCK-END
