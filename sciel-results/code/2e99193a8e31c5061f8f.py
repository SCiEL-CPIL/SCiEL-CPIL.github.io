import numpy as np
from scipy.optimize import linprog
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item (larger is more attractive).
    prizes: (5, 100)
    weights: (5, 100)
    Returns: (100,) array
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # Fallback heuristic
    def get_baseline():
        # Simple prize/weight ratio heuristic
        p_total = prizes.sum(axis=0)
        w_total = weights.sum(axis=0)
        h = p_total / (w_total + 1e-9)
        h = np.maximum(h, 1e-9)
        return h

    try:
        # 1. Compute total prize per item
        p_total = prizes.sum(axis=0)  # (100,)
        
        # 2. Setup LP to get dual variables (shadow prices)
        # Maximize sum(p_total[j] * x[j])
        # Equivalent to Minimize -sum(p_total[j] * x[j])
        c = -p_total
        
        # Constraints:
        # 1. Weight constraints: sum_j weights[i, j] * x[j] <= 25 for i in 0..4
        # 2. Bound constraints: x[j] <= 1 for j in 0..99
        A_ub = np.vstack([weights, np.eye(100)])  # (105, 100)
        b_ub = np.concatenate([np.full(5, 25.0), np.ones(100)])  # (105,)
        bounds = [(0, 1) for _ in range(100)]
        
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status != 0:
            return get_baseline()
            
        # Extract duals for the first 5 constraints (weight constraints)
        lambdas = res.marginals[:5]
        
        # Ensure lambdas are non-negative and finite
        if not np.all(np.isfinite(lambdas)) or np.any(lambdas < -1e-6):
            return get_baseline()
            
        # Clamp small negative values to 0
        lambdas = np.maximum(lambdas, 0.0)
        
        # Compute reduced costs
        # RC_j = p_total[j] - sum_i lambda[i] * weights[i, j]
        reduced_costs = p_total - weights.T @ lambdas
        
        # Heuristic should be positive
        h_1d = np.maximum(reduced_costs, 0.0) + 1e-9
        
        if not np.all(np.isfinite(h_1d)):
            return get_baseline()
            
        # Simulate ACO to evaluate and tune the heuristic
        def simulate_aco(h, seed, n_ants=20, n_iter=30):
            np.random.seed(seed)
            n_items = 100
            capacity = 25.0
            n_constraints = 5
            
            best_total_prize = 0.0
            pheromone = np.ones(n_items)
            heuristic = np.where(h > 0, h, 1e-9)
            
            for it in range(n_iter):
                iter_best_prize = 0.0
                
                for ant in range(n_ants):
                    selected = []
                    remaining_capacity = np.full(n_constraints, capacity)
                    current_prize = 0.0
                    available = np.ones(n_items, dtype=bool)
                    
                    while True:
                        # Check which items fit in all constraints
                        fits = np.all(weights.T <= remaining_capacity, axis=1) & available
                        
                        if not np.any(fits):
                            break
                        
                        idx = np.where(fits)[0]
                        probs = pheromone[idx] * heuristic[idx]
                        prob_sum = np.sum(probs)
                        
                        if prob_sum == 0:
                            break
                            
                        probs = probs / prob_sum
                        chosen_local = np.random.choice(len(idx), p=probs)
                        chosen_item = idx[chosen_local]
                        
                        current_prize += p_total[chosen_item]
                        remaining_capacity -= weights[:, chosen_item]
                        available[chosen_item] = False
                        
                    if current_prize > iter_best_prize:
                        iter_best_prize = current_prize
                        
                pheromone *= 0.9
                pheromone += iter_best_prize / 10.0
                
                if iter_best_prize > best_total_prize:
                    best_total_prize = iter_best_prize
                    
            return best_total_prize

        # Generate candidate heuristics
        max_rc = np.max(h_1d)
        candidates = [
            h_1d,                                          # Linear
            np.exp(h_1d - max_rc) + 1e-9,                 # Exponential
            h_1d ** 2 + 1e-9,                             # Quadratic
            np.sqrt(h_1d) + 1e-9,                         # Square root
            h_1d / (np.mean(h_1d) + 1e-9),                # Normalized
        ]
        
        # Evaluate candidates with multiple seeds
        best_h = h_1d
        best_avg_score = -1
        
        seeds = [1, 2, 3]
        
        for h in candidates:
            if not np.all(np.isfinite(h)):
                continue
                
            scores = []
            for seed in seeds:
                if time.time() > deadline - 30.0:
                    break
                score = simulate_aco(h, seed)
                scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_h = h
        
        # Try more seeds if time allows
        if time.time() < deadline - 30.0:
            for h in candidates:
                if not np.all(np.isfinite(h)):
                    continue
                    
                for seed in [4, 5]:
                    if time.time() > deadline - 30.0:
                        break
                    score = simulate_aco(h, seed)
                    # This is a simple refinement; in practice, we'd track per-candidate scores
                    pass
        
        # Final check
        if not np.all(np.isfinite(best_h)):
            return get_baseline()
            
        return best_h

    except Exception:
        return get_baseline()
