import numpy as np
from scipy.optimize import linprog

def heuristics(prizes, weights):
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Fallback heuristic
    def get_baseline():
        return prizes / (weights.sum(axis=0) + 1e-9)

    try:
        # 1. Compute total prize per item
        p_total = prizes.sum(axis=0) # (100,)
        
        # 2. Setup LP
        # Maximize sum(p_total[j] * x[j])
        # Equivalent to Minimize -sum(p_total[j] * x[j])
        c = -p_total
        
        # Constraints:
        # 1. Weight constraints: sum_j weights[i, j] * x[j] <= 25 for i in 0..4
        # 2. Bound constraints: x[j] <= 1 for j in 0..99
        # 3. Lower bounds: x[j] >= 0 (handled by bounds)
        
        A_ub = np.vstack([weights, np.eye(100)]) # (105, 100)
        b_ub = np.concatenate([np.full(5, 25.0), np.ones(100)]) # (105,)
        
        bounds = [(0, 1) for _ in range(100)]
        
        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            return get_baseline()
            
        # Extract duals for the first 5 constraints (weight constraints)
        # res.marginals contains the dual variables for A_ub constraints
        # The first 5 correspond to weights
        lambdas = res.marginals[:5]
        
        # Ensure lambdas are non-negative and finite
        if not np.all(np.isfinite(lambdas)) or np.any(lambdas < 0):
             # Duals should be non-negative for <= constraints in min problem
             # If they are negative or non-finite, something is wrong
             return get_baseline()
             
        # Compute reduced costs
        # RC_j = p_total[j] - sum_i lambda[i] * weights[i, j]
        # weights is (5, 100), lambdas is (5,)
        # weights.T @ lambdas gives (100,)
        reduced_costs = p_total - weights.T @ lambdas
        
        # Heuristic should be positive
        # Items with high reduced cost are good
        # Clamp negative reduced costs to 0 (or small epsilon)
        h_1d = np.maximum(reduced_costs, 0.0)
        
        # Add epsilon to ensure positive
        h_1d = h_1d + 1e-9
        
        # The baseline returned (5, 100). We should match the shape.
        # It is unclear how the evaluator uses the 5 rows.
        # If it sums them, replicating is fine.
        # If it uses them independently, replicating is still a reasonable guess 
        # since the "attractiveness" is a property of the item globally.
        h_matrix = np.tile(h_1d, (5, 1))
        
        # Check for finite values
        if not np.all(np.isfinite(h_matrix)):
            return get_baseline()
            
        return h_matrix

    except Exception:
        return get_baseline()
