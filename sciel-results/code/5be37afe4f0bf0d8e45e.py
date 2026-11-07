"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a greedy Lagrangian dual relaxation.
    
    Instead of a static geometric mean of efficiencies, this method treats the
    heuristic as a dynamic ranking of items based on their marginal contribution
    to the optimal solution of a specific instance. It performs a subgradient
    optimization on the Lagrangian dual to find multipliers that best balance
    the constraints, then ranks items by their adjusted profit (prize - sum(lambda*weight)).
    
    This adapts to the specific "shape" of the instance's constraint bottlenecks.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape (5, 100)
    # prizes: (5, 100) -> we assume the first dimension is constraints? 
    # Wait, the prompt says "prizes and weights, of shape (5, 100)".
    # Usually prizes are per item, so shape should be (1, 100) or (100,).
    # But the prompt says "prizes ... shape (5, 100)".
    # Let's re-read carefully: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # This is unusual for prizes. Prizes are usually scalar per item.
    # If prizes is (5, 100), maybe it means 5 different prize vectors? Or is it a typo in my understanding?
    # "Item scores ... for 3 fixed instances with 100 items and 5 constraints."
    # "prizes and weights, of shape (5, 100)"
    # If weights is (5, 100), that's 5 constraints x 100 items. That makes sense.
    # If prizes is (5, 100), that's weird. 
    # Let's look at the seed program:
    # `ratios = prizes / (weights + epsilon)`
    # If prizes is (5, 100) and weights is (5, 100), this works element-wise.
    # Then `np.mean(log_ratios, axis=0)` reduces the 5 constraints to 100 items.
    # This implies the seed program treats prizes as having 5 rows, just like weights.
    # Perhaps "prizes" in this context is actually "value densities" or something similar?
    # Or maybe the problem definition implies that each constraint has a "prize" component?
    # No, "highest total prize found by any ant".
    # If prizes is (5, 100), what is the "total prize"?
    # If I sum prizes over items, and prizes is (5, 100), I get (5,).
    # The scoring says "mean over instances of the highest total prize".
    # This is ambiguous. However, the seed program works with (5, 100) for both.
    # And the seed program computes a ratio `prizes / weights`.
    # If I assume the standard MCKP: Max sum(p_i x_i) s.t. sum(w_ij x_i) <= C_j.
    # Here p_i is scalar.
    # If the input `prizes` is (5, 100), maybe the "prize" for an item is the sum of its row? Or the mean?
    # Or maybe the task description implies that `prizes` is actually the same shape as weights for some reason?
    # Let's look at the seed code again.
    # `geometric_mean = np.exp(np.mean(log_ratios, axis=0))`
    # This produces a (100,) array.
    # The function returns `array of one score per item`.
    # So the output is (100,).
    # The input `prizes` is (5, 100).
    # If I use a Lagrangian approach, I need a scalar prize per item to define the objective.
    # Let's assume the "prize" of an item is the mean or sum of its `prizes` row.
    # Given "uniform in [0, 1]", and 5 rows, the sum would be in [0, 5], mean in [0, 1].
    # Let's assume the objective is to maximize the sum of `prizes[i, j]` over selected items?
    # If so, the total prize of a set S is sum_{j in S} sum_{k=0}^4 prizes[k, j].
    # Let's define `item_prizes = np.sum(prizes, axis=0)` or `np.mean(prizes, axis=0)`.
    # Since the seed uses `prizes / weights` element-wise and then aggregates, it effectively weights the prize by the inverse weight.
    # Let's try to infer the "true" prize from the seed's logic.
    # The seed effectively maximizes the geometric mean of `p_kj / w_kj`.
    # This is not standard.
    # However, for the Lagrangian, I need a clear objective.
    # Let's assume the objective is `sum(item_prizes)`.
    # I will use `item_prizes = np.sum(prizes, axis=0)` as the value of an item.
    
    item_prizes = np.sum(prizes, axis=0)
    C = 25.0
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # If time is already up, return uniform
    if time.time() > deadline:
        return np.ones(n_items)

    # Lagrangian Dual Minimization
    # Minimize g(lambda) = sum_i max(0, p_i - sum_j lambda_j w_ij) + sum_j lambda_j C_j
    # We use subgradient method to find lambda.
    # lambda_j >= 0
    
    # Initialize lambda to 0
    lambdas = np.zeros(n_constraints)
    
    # Number of iterations for dual optimization
    # Keep it small to ensure speed, but enough to converge somewhat.
    # 50-100 iterations is usually fast for N=100, K=5.
    max_dual_iters = 100
    
    # Step size for subgradient
    # A common choice is t_k = c / k or adaptive.
    # Let's use a simple decreasing step size.
    step_sizes = []
    for k in range(1, max_dual_iters + 1):
        step_sizes.append(1.0 / k)
        
    best_lambda = lambdas.copy()
    best_dual_value = -np.inf
    
    for k in range(max_dual_iters):
        if time.time() > deadline - 5:
            break
            
        # Compute reduced costs: r_i = p_i - sum_j lambda_j w_ij
        # r shape: (100,)
        # weights shape: (5, 100)
        # lambdas shape: (5,)
        reduced_costs = item_prizes - np.dot(lambdas, weights)
        
        # Items with positive reduced cost are "selected" in the dual problem
        selected = reduced_costs > 0
        
        # Compute dual objective value:
        # g(lambda) = sum_{i: r_i > 0} r_i + sum_j lambda_j * C_j
        dual_value = np.sum(reduced_costs[selected]) + np.dot(lambdas, C)
        
        if dual_value > best_dual_value:
            best_dual_value = dual_value
            best_lambda = lambdas.copy()
            
        # Subgradient of g(lambda) w.r.t lambda_j:
        # s_j = C_j - sum_{i: r_i > 0} w_ij
        # s shape: (5,)
        s = C - weights[:, selected].sum(axis=1)
        
        # Update lambda: lambda_new = max(0, lambda + step * s)
        step = step_sizes[k]
        lambdas = np.maximum(0, lambdas + step * s)
        
        # Optional: Normalize or limit step size to prevent divergence
        # The step size 1/k is standard for subgradient.
        
    # After finding good lambdas, compute the final heuristic scores.
    # The heuristic should reflect the "attractiveness" of items.
    # A good heuristic for ACO is often the reduced cost itself, or a transformed version.
    # However, reduced costs can be negative.
    # The task says: "Heuristic entries that are not positive become 1e-9".
    # So we can just use the reduced costs calculated with the best lambda.
    # But wait, if an item has negative reduced cost, it's unattractive.
    # If it has positive, it's attractive.
    # This perfectly aligns with the ACO probability proportional to pheromone * heuristic.
    
    # Calculate final reduced costs with best_lambda
    final_reduced_costs = item_prizes - np.dot(best_lambda, weights)
    
    # Ensure positive values, replace non-positive with 1e-9
    scores = np.where(final_reduced_costs > 0, final_reduced_costs, 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores

# EVOLVE-BLOCK-END
