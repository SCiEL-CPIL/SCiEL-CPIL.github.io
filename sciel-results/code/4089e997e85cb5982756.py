import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Greedy Lagrangian Relaxation approach.
    
    The idea is that the 'value' of an item is not just its prize, but its prize 
    minus the 'cost' of the resources it consumes. Since resources are scarce, 
    we estimate the 'price' (dual variable) of each resource by observing how 
    much a greedy knapsack solver values them.
    
    We iterate:
    1. Start with uniform prices (or based on total weight).
    2. Solve a relaxed knapsack: maximize sum(prize_i - sum(lambda_c * weight_ic)) 
       subject to no capacity constraints (just pick all positive net value items).
       Actually, to mimic the ACO's constraint handling, we should simulate a 
       greedy fill. But a simpler proxy for duals in MCKP is to use the 
       'opportunity cost' derived from a single-constraint greedy sort.
       
    A more robust and fast heuristic for MCKP in ACO contexts is often:
    h_i = prize_i / (1 + sum_c (w_ic / cap_c) * lambda_c)
    
    Let's estimate lambda_c by checking how 'tight' each constraint is.
    If a constraint is tight, its lambda should be high.
    
    Algorithm:
    1. Calculate total prize and total weight per constraint.
    2. Estimate scarcity s_c = 1 / (1 + total_weight_c / capacity_c). 
       High scarcity -> low s_c? No, high scarcity means high price.
       Let's use price_c = 1 / (capacity_c - total_weight_c + epsilon) ? 
       No, total_weight is sum of ALL items, which is ~50 for cap 25. 
       This is not the weight of the *selected* set.
       
    Better approach: 
    The ACO selects items until constraints are full. 
    The "critical" items are those that fill the constraints.
    
    Let's use a "Greedy Simulation" to find a good solution, then use the 
    marginal values from that solution to price the resources.
    
    1. Run a fast greedy algorithm:
       - Sort items by prize/weight ratio (using a composite weight).
       - Fill knapsack.
       - Identify which constraints are binding.
    
    Actually, the most effective simple heuristic that outperforms static ratios 
    in MCKP-ACO is often derived from the **Lagrangian Dual** of the problem.
    
    Let's implement a few iterations of a subgradient method to find lambda.
    Objective: max sum(prize_i * x_i) s.t. sum(w_ic * x_i) <= cap_c
    Lagrangian: L(lambda) = sum_i max(0, prize_i - sum_c lambda_c * w_ic) + sum_c lambda_c * cap_c
    
    We want lambda such that the "shadow price" reflects scarcity.
    If we pick all items with positive reduced cost, the total weight will likely 
    exceed capacity. We want to adjust lambda so that the "desire" to pick items 
    balances the capacity.
    
    However, doing subgradient optimization might be slow or unstable.
    
    Alternative: **Geometric Mean Ratio** blended with **Sum Ratio**.
    The inspiration mentions "geo-ratio" and "sum-ratio" blend scored 1.006.
    The current program uses a normalized sum cost.
    
    Let's try the **Geometric Mean** of the prize-to-weight ratios for each constraint.
    h_i = (prod_c (prize_i / (w_ic + eps)))^(1/5)
    This penalizes items that are heavy in ANY constraint heavily.
    
    But we need to account for the fact that not all constraints are equally tight.
    
    Let's refine the "Greedy Simulation" idea.
    1. Compute a static ratio: r_i = sum(prizes) / sum(weights).
    2. Sort by r_i.
    3. Simulate adding items until constraints are violated.
    4. The items that *don't* fit are the "expensive" ones in terms of opportunity cost.
    5. This doesn't directly give a score for all items, but we can use the 
       "bottleneck" constraints.
    
    Let's go with a **Adaptive Lagrangian** heuristic computed via a few 
    subgradient steps. It is robust and theoretically grounded.
    
    Steps:
    1. Initialize lambda = ones(5) * 0.1
    2. For iter in range(20):
       a. Compute reduced cost: rc_i = prize_sum_i - dot(lambda, weights[:, i])
       b. Select items with rc_i > 0.
       c. Calculate total weight of selected items: W_c = sum(w_ic for i in selected)
       d. If W_c > cap_c, increase lambda_c. If W_c < cap_c, decrease lambda_c.
       e. Step size: step = 0.1 / (sum(lambda^2))
       f. lambda_c += step * (W_c - cap_c)
       g. Clip lambda to [0, 10]
    3. Final Heuristic: h_i = prize_sum_i / (1 + dot(lambda, weights[:, i]))
       Or h_i = exp(prize_sum_i - dot(lambda, weights[:, i])) ?
       The ACO multiplies pheromone * heuristic. 
       If heuristic is too large, it dominates. 
       Standard form is h_i = (prize_i) / (1 + sum(lambda_c * w_ic)).
       
    Let's verify the scale.
    prizes ~ U[0,1], sum ~ 2.5.
    weights ~ U[0,1], sum ~ 2.5.
    lambda ~ 0.1 to 1.
    denominator ~ 1 + 0.25 = 1.25.
    h_i ~ 2.
    
    This seems reasonable.
    
    To improve, we can normalize the heuristic so that the mean is 1 or max is 1?
    The task says "larger is more attractive". It doesn't specify scale.
    However, pheromone starts at 1. If h is 2, probability is 2*1.
    If h is 100, probability is 100*1.
    Relative ratios matter most.
    
    Let's implement the Subgradient Lagrangian.
    """
    start_time = time.time()
    deadline = 300.0  # Hard limit, but we should be much faster
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    capacities = np.full(n_constraints, 25.0)
    
    total_prizes = np.sum(prizes, axis=0)
    
    # Initial lambda estimation: uniform small positive
    lambdas = np.ones(n_constraints) * 0.1
    
    # Subgradient method to find dual variables
    # We want to find lambdas such that the "unconstrained" solution 
    # (pick all items with positive reduced cost) roughly respects capacities.
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
            
        # Reduced cost for each item
        # rc_i = total_prize_i - sum_c(lambda_c * w_ic)
        # weights is (5, 100), lambdas is (5,)
        # dot product: lambdas @ weights -> (100,)
        cost = lambdas @ weights
        reduced_costs = total_prizes - cost
        
        # Select items with positive reduced cost
        mask = reduced_costs > 0
        selected = np.where(mask)[0]
        
        if len(selected) == 0:
            # If no items are selected, lambdas are too high.
            # Decrease lambdas.
            lambdas = np.maximum(0, lambdas * 0.5)
            continue
            
        # Calculate total weight of selected items per constraint
        # weights[:, selected] is (5, len(selected))
        total_weight_selected = np.sum(weights[:, selected], axis=1)
        
        # Subgradient: (total_weight_selected - capacities)
        grad = total_weight_selected - capacities
        
        # Step size
        # Standard step size: step = alpha / ||grad||^2 or similar
        # Let's use a simple adaptive step
        norm_grad_sq = np.sum(grad ** 2)
        if norm_grad_sq < 1e-6:
            step = 0.01
        else:
            step = 1.0 / np.sqrt(norm_grad_sq)
            
        # Update lambdas
        lambdas = lambdas + step * grad
        
        # Ensure non-negative
        lambdas = np.maximum(0, lambdas)
        
        # Optional: Cap lambdas to prevent explosion
        lambdas = np.minimum(10.0, lambdas)
        
        # Check convergence (optional, but subgradient is slow to converge)
        # We'll just run fixed iterations.
        
    # Final heuristic calculation
    # h_i = prize_i / (1 + sum_c(lambda_c * w_ic))
    cost = lambdas @ weights
    heuristic = total_prizes / (1.0 + cost)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Optional: Normalize? 
    # The ACO uses prob proportional to pheromone * heuristic.
    # If we scale all heuristics by a constant, the probabilities don't change.
    # So normalization is not strictly necessary for the *distribution*, 
    # but it keeps numbers stable.
    
    return heuristic

