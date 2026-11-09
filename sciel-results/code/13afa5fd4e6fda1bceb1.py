"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Lagrangian Dual Approximation (Analytical Efficiency Ratio).
    
    1. Problem Framing:
       Instead of simulating random packings (Monte Carlo), we approximate the 
       solution to the Lagrangian relaxation of the Multiple-Constraint Knapsack.
       The Lagrangian dual problem seeks non-negative multipliers (lambda) for each 
       constraint such that the sum of lambda_c * capacity_c is maximized, subject 
       to the condition that no item has a "reduced cost" (prize - sum(lambda * weight)) 
       greater than 0 if it were to be selected in the relaxed knapsack.
       
       In practice, for a knapsack, the optimal Lagrangian multipliers correspond to 
       the "shadow prices" of the constraints. An item's attractiveness in the ACO 
       heuristic should reflect its value relative to these shadow prices.
       
    2. Method:
       - We use a subgradient method to find the optimal dual variables (lambda).
       - For a given lambda, the "value" of an item is V_i = prize_i - sum(lambda_c * weight_ic).
       - Items with V_i > 0 are "profitable" in the relaxed problem.
       - The subgradient of the dual function is (total_weight_used_by_profitable_items - capacity).
       - We iterate lambda updates until convergence or time limit.
       
    3. Heuristic Construction:
       - Once we have converged lambda, we compute the final reduced cost for each item:
         R_i = prize_i - sum(lambda_c * weight_ic).
       - To ensure the heuristic guides the ACO towards items that are "efficiency leaders" 
         in the context of the tight constraints, we use a transformed score.
       - A common effective heuristic for ACO in knapsack variants is the ratio of prize 
         to the "weighted cost" determined by the duals.
       - However, simply using R_i can be negative. 
       - A robust alternative is to use the "dual-adjusted prize": 
         H_i = prize_i / (1 + sum(lambda_c * weight_ic)).
         This rewards high prize and penalizes items that consume capacity in expensive (tight) constraints.
       
       - We will compute the duals via a few hundred subgradient steps, which is 
         deterministic, fast, and structurally different from random simulation.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 10.0  # Subgradient is very fast, 10s is generous
    
    # Initialize dual variables (lambda)
    # Start with small positive values to avoid zero division or symmetry issues
    lambdas = np.ones(n_constraints) * 0.1
    
    # Subgradient Method Parameters
    # The step size should decay to ensure convergence.
    # Standard form: t_k = alpha / k or alpha / sqrt(k).
    # We use a simple decaying step size.
    
    max_iter = 500
    best_dual_obj = -np.inf
    best_lambdas = lambdas.copy()
    
    # We want to maximize the dual function D(lambda) = sum(lambda * capacity)
    # subject to the constraint that the "knapsack" of items with positive reduced cost 
    # doesn't violate capacities too badly? 
    # Actually, the dual function is defined as:
    # D(lambda) = sum_{i: V_i > 0} (prize_i - sum(lambda_c * weight_ic)) + sum(lambda_c * capacity_c)
    # Wait, the standard Lagrangian relaxation for max z s.t. Ax <= b is:
    # L(x, lambda) = z - lambda^T (Ax - b) = (z - lambda^T Ax) + lambda^T b
    # Max over x in {0,1}^n of (z - lambda^T Ax) is just sum of max(0, z_i - lambda^T A_i)
    # So D(lambda) = sum_i max(0, prize_i - sum_c lambda_c weight_ic) + sum_c lambda_c capacity_c
    
    # We want to MINIMIZE D(lambda) over lambda >= 0 to get the upper bound.
    # The subgradient of D with respect to lambda_c is:
    # g_c = capacity_c - sum_{i: V_i > 0} weight_ic
    # (Note: derivative of max(0, v_i) is 1 if v_i > 0, else 0. 
    #  Derivative of -sum lambda w is -w. So term is -w_i if V_i>0.
    #  Plus derivative of lambda*C is C.
    #  So g_c = C_c - sum_{i: V_i > 0} w_ic)
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute reduced costs V_i = prize_i - sum_c lambda_c * weight_ic
        # weights shape (C, N), lambdas shape (C,)
        # dot product: (C, N) @ (C,) -> (N,)
        weighted_cost = weights.T @ lambdas  # shape (N,)
        reduced_costs = prizes - weighted_cost
        
        # Identify items with positive reduced cost
        mask = reduced_costs > 0
        
        # Compute subgradient
        # g_c = capacity - sum_{i in mask} weight_ic
        if np.any(mask):
            used_weights = weights[:, mask].sum(axis=1)  # shape (C,)
        else:
            used_weights = np.zeros(n_constraints)
            
        subgrad = capacity - used_weights
        
        # Compute current dual objective value to track best
        dual_obj = np.sum(lambdas * capacity) + np.sum(np.maximum(0, reduced_costs))
        
        if dual_obj < best_dual_obj:
            best_dual_obj = dual_obj
            best_lambdas = lambdas.copy()
            
        # Update lambdas
        # Step size: We want to move in direction of -subgrad (to minimize D)
        # But lambdas must be >= 0.
        # Step size decay: 1/sqrt(k+1) or similar.
        # Let's use a fixed step size scaled by 1/k for stability, or adaptive.
        # A common choice is step = alpha / k. Let's use alpha=1.0.
        step_size = 1.0 / (k + 1.0)
        
        # Update: lambda_new = lambda_old - step * subgrad
        # Since we are minimizing D, and g is subgradient of D, we do lambda -= step * g.
        # If g > 0 (capacity > used), it means we are "underusing" capacity in the relaxed problem,
        # so we should increase lambda? 
        # Let's check:
        # D(lambda) = sum max(0, v_i) + lambda^T b.
        # If we increase lambda, v_i decreases (if w_i > 0), so sum max(0, v_i) decreases.
        # But lambda^T b increases.
        # The optimal lambda balances these.
        # Subgradient g_c = b_c - sum_{i: v_i>0} w_ic.
        # If g_c > 0, it means b_c > used. The term lambda^T b is growing faster than the reduction in the item sum?
        # Actually, the subgradient points in the direction of steepest ascent for convex functions.
        # We are MINIMIZING a convex function D(lambda).
        # So we should move in the direction of -g.
        # lambda_new = lambda_old - step * g.
        
        new_lambdas = lambdas - step_size * subgrad
        
        # Project onto non-negative orthant
        new_lambdas = np.maximum(0, new_lambdas)
        
        # Optional: Normalize or cap to prevent explosion?
        # With decaying step size, it should converge.
        
        lambdas = new_lambdas

    # Use the best lambdas found (minimizing dual objective)
    lambdas = best_lambdas
    
    # Compute final heuristic scores
    # We want a score that reflects the "efficiency" under these dual prices.
    # Option A: Reduced Cost R_i = prize_i - sum(lambda * weight_i). 
    #   Problem: Can be negative. ACO needs positive.
    # Option B: Prize / (1 + sum(lambda * weight_i)).
    #   This is always positive. It rewards high prize and penalizes items that are "expensive" 
    #   under the dual prices.
    # Option C: (prize_i / (sum(lambda * weight_i) + epsilon))
    #   This is the ratio.
    
    # Let's use Option B: H_i = prize_i / (1 + weighted_cost_i)
    # This is smooth and positive.
    
    weighted_cost = weights.T @ lambdas  # shape (N,)
    
    # Avoid division by zero or very small numbers
    denom = 1.0 + weighted_cost
    scores = prizes / denom
    
    # Ensure positive
    # prizes are in [0, 1], so scores are in [0, 1].
    # If prize is 0, score is 0.
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So we map 0 to 1e-9.
    
    # To sharpen the distribution and help ACO distinguish between good and great items,
    # we can apply a power. 
    # If we use scores^2, we emphasize higher scores more.
    # Given the ACO probability is P * H, and P starts at 1, 
    # a sharper H helps the ants find the best items faster in early iterations.
    
    # Let's try squaring the normalized scores.
    # First, normalize to [0, 1] range to make power application consistent.
    max_score = np.max(scores)
    if max_score > 0:
        scores_norm = scores / max_score
    else:
        scores_norm = np.ones(n_items)
        
    # Apply power 2.0 to sharpen
    scores_final = scores_norm ** 2.0
    
    # Map non-positive to 1e-9
    scores_final = np.where(scores_final > 0, scores_final, 1e-9)
    
    # Ensure finite
    scores_final = np.where(np.isfinite(scores_final), scores_final, 1e-9)
    
    return scores_final
# EVOLVE-BLOCK-END
