import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in an ACO for MCKP.
    Strategy: 
    1. Solve LP relaxation to get dual prices (shadow prices) for constraints.
    2. Use these to define an initial 'efficiency' score.
    3. Perform fast vectorized hill-climbing over heuristic parameters 
       (Lagrange multipliers + prize scaling) to maximize simulated ACO performance.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n_items = P.shape[0]
    n_constraints = w.shape[0]
    capacity = 25.0
    
    t0 = time.time()
    deadline = t0 + 290.0  # 40s buffer for safety

    # --- Step 1: LP Relaxation for Initial Duals ---
    # Maximize P^T x subject to w^T x <= capacity, 0 <= x <= 1
    # c = -P (for minimization)
    c = -P
    A_ub = w.T  # (n_items, n_constraints)
    b_ub = np.full(n_constraints, capacity)
    bounds = [(0, 1)] * n_items
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            # Duals correspond to A_ub constraints (the capacity constraints)
            # In scipy linprog, res.ineqlin.marginals gives the duals for A_ub
            # Note: sign convention: duals are for min c^T x s.t. A_ub x <= b_ub
            # The marginal value of increasing b_ub is -dual (for max problem)
            # Let's use the absolute value or just the raw duals as penalties.
            # If dual > 0, it means constraint is tight and valuable.
            duals = np.abs(res.ineqlin.marginals)
            # Ensure non-negative
            duals = np.maximum(duals, 0)
        else:
            # Fallback if LP fails
            duals = np.ones(n_constraints) / n_constraints
    except:
        duals = np.ones(n_constraints) / n_constraints

    # --- Step 2: Define Heuristic Model ---
    # h[i] = exp( alpha * P[i] - sum_j lambda_j * w[j, i] )
    # This is equivalent to exp( P[i] * alpha - w[i]^T * lambda )
    # We optimize alpha (scalar) and lambda (vector of n_constraints).
    
    # Initial guess:
    # alpha: scale prize
    # lambda: scaled duals
    # Normalize duals to have similar magnitude to prize/weight ratios
    # Prize ~ U[0,1], Weight ~ U[0,1]. 
    # If we set lambda = duals * scale, we need scale to balance.
    
    # Let's define the heuristic calculation function
    def compute_h(alpha, lambdas):
        # h = exp(alpha * P - w.T @ lambdas)
        # w is (n_constraints, n_items)
        # w.T @ lambdas is (n_items,)
        scores = alpha * P - (w.T @ lambdas)
        # Shift to avoid underflow/overflow, center around 0?
        # Or just exp and clip.
        h = np.exp(scores)
        # Clip to valid range
        h = np.clip(h, 1e-9, 1e6)
        return h

    # --- Step 3: Fast Vectorized ACO Simulation ---
    # We need a fast way to simulate the colony.
    # The bottleneck in the current program is the Python loop over items/ants.
    # We can vectorize the "fits" check and the probability selection somewhat,
    # but the sequential nature of item selection makes full vectorization hard.
    # However, we can precompute things and use numpy arrays for state.
    
    def fast_colony_sim(h, seed=42):
        """
        Simulates ACO. Returns best total prize.
        h: heuristic vector (n_items,)
        """
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n_items)
        best_total = 0.0
        
        # Pre-compute weights as (n_items, n_constraints) for easier slicing?
        # w is (n_constraints, n_items). w.T is (n_items, n_constraints).
        W_T = w.T 
        
        for _ in range(30):
            max_ant_total = 0.0
            for _ in range(20):
                remaining_cap = np.full(n_constraints, capacity)
                used = np.zeros(n_items, dtype=bool)
                total_prize = 0.0
                
                # We will loop until no item fits
                # To speed up, we can check fits in batches? No, selection is sequential.
                # But we can make the inner loop tight.
                
                while True:
                    # Check fits: W_T[i] <= remaining_cap
                    # This is (n_items, n_constraints) <= (n_constraints,)
                    fits_mask = np.all(W_T <= remaining_cap[None, :], axis=1) & ~used
                    
                    if not np.any(fits_mask):
                        break
                    
                    fitting_indices = np.where(fits_mask)[0]
                    
                    probs = pheromone[fitting_indices] * h[fitting_indices]
                    sum_probs = probs.sum()
                    if sum_probs < 1e-9:
                        probs = np.ones(len(fitting_indices))
                        sum_probs = len(fitting_indices)
                    probs = probs / sum_probs
                    
                    idx = rng.choice(fitting_indices, p=probs)
                    used[idx] = True
                    remaining_cap -= W_T[idx]
                    total_prize += P[idx]
                
            if total_prize > max_ant_total:
                max_ant_total = total_prize
                
            if max_ant_total > best_total:
                best_total = max_ant_total
                
            # Update pheromone
            pheromone = pheromone * 0.9 + max_ant_total / 10.0
            
        return best_total

    # --- Step 4: Optimization ---
    # Parameters: alpha (1), lambdas (n_constraints)
    # Total params: 1 + 5 = 6
    
    # Initial parameters
    # Scale duals: if duals are large, lambdas should be small? 
    # h = exp(alpha*P - w^T lambda). 
    # If lambda is large, high weight items get penalized.
    # Let's start with alpha=1.0, lambdas = duals * 0.5
    initial_lambdas = duals * 0.5
    initial_alpha = 1.0
    
    best_alpha = initial_alpha
    best_lambdas = initial_lambdas.copy()
    best_h = compute_h(best_alpha, best_lambdas)
    best_score = fast_colony_sim(best_h)
    
    current_alpha = initial_alpha
    current_lambdas = initial_lambdas.copy()
    current_h = compute_h(current_alpha, current_lambdas)
    current_score = best_score
    
    # Hill Climbing / Random Restart
    # We will use a simple coordinate ascent or random perturbations.
    # Given the small dimension (6), random perturbations are fine.
    
    step_alpha = 0.5
    step_lambda = 0.1
    
    # We will run for a fixed number of iterations or until time limit.
    # Each evaluation takes ~0.1-0.5s? Let's measure one.
    t_eval_start = time.time()
    _ = fast_colony_sim(current_h)
    t_eval_end = time.time()
    avg_eval_time = t_eval_end - t_eval_start
    
    # Estimate max iterations
    time_left = deadline - time.time()
    max_iters = int(time_left / (avg_eval_time * 1.2)) # 20% overhead
    
    # If avg_eval_time is too high, we might only get a few iters.
    # If it's low, we get many.
    
    # Use a mix of small and large steps
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturb
        # Occasionally do a large jump (restart)
        if np.random.rand() < 0.05:
            new_alpha = current_alpha + np.random.randn() * 2.0
            new_lambdas = current_lambdas + np.random.randn(n_constraints) * 0.5
        else:
            new_alpha = current_alpha + np.random.randn() * step_alpha
            new_lambdas = current_lambdas + np.random.randn(n_constraints) * step_lambda
            
        # Ensure lambdas are non-negative? 
        # The model allows negative lambdas (rewarding weight?), but physically 
        # weight is a cost. However, in MCKP, sometimes a "heavy" item in one 
        # constraint might be light in others. But the term is -w^T lambda.
        # If lambda < 0, it rewards weight. This might be useful if a constraint
        # is loose. Let's allow it, but clip to avoid extreme values.
        new_lambdas = np.clip(new_lambdas, -5.0, 5.0)
        
        new_h = compute_h(new_alpha, new_lambdas)
        
        # Evaluate
        new_score = fast_colony_sim(new_h)
        
        # Accept if better
        if new_score > current_score:
            current_alpha = new_alpha
            current_lambdas = new_lambdas.copy()
            current_h = new_h
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_alpha = current_alpha
                best_lambdas = current_lambdas.copy()
                best_h = new_h.copy()
                
        # Decay step sizes slightly?
        # step_alpha *= 0.99
        # step_lambda *= 0.99

    # Final result
    final_h = best_h
    final_h = np.where(final_h > 0, final_h, 1e-9)
    final_h = np.clip(final_h, 1e-9, 1e6)
    
    return final_h
