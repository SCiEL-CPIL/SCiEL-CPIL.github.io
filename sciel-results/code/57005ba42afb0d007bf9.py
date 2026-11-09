import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in an ACO for MCKP.
    Strategy: Solve the LP relaxation to get dual prices (shadow prices).
    Score items by their reduced cost (prize - sum(dual_j * weight_ij)).
    This guides ants to items that are "cheapest" relative to the theoretical
    continuous optimum, effectively prioritizing high-efficiency items.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Normalize input shapes to (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n_items = P.shape[0]
    n_constraints = w.shape[0]
    capacity = 25.0
    
    # Time management
    t0 = time.time()
    deadline = t0 + 310.0  # Leave 20s buffer for the 330s kill

    # Fast Vectorized Colony Simulation
    def fast_colony_sim(h, seed):
        """
        Simulates the ACO process quickly.
        Returns the best total prize found.
        """
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n_items)
        best_total = 0.0
        
        for _ in range(30):
            ant_totals = []
            for _ in range(20):
                remaining_cap = np.full(n_constraints, capacity)
                used = np.zeros(n_items, dtype=bool)
                total_prize = 0.0
                
                while True:
                    # Check which items fit
                    # An item i fits if w[:, i] <= remaining_cap
                    fits_mask = np.all(w <= remaining_cap[:, None], axis=0) & ~used
                    
                    if not np.any(fits_mask):
                        break
                    
                    fitting_indices = np.where(fits_mask)[0]
                    
                    # Calculate probabilities: pheromone * heuristic
                    probs = pheromone[fitting_indices] * h[fitting_indices]
                    sum_probs = probs.sum()
                    if sum_probs == 0 or not np.isfinite(sum_probs):
                        probs = np.ones(len(fitting_indices))
                        sum_probs = len(fitting_indices)
                    else:
                        probs = probs / sum_probs
                    
                    # Select one item
                    idx = rng.choice(fitting_indices, p=probs)
                    used[idx] = True
                    remaining_cap -= w[:, idx]
                    total_prize += P[idx]
                
            ant_totals.append(total_prize)
            
            # Update pheromone
            max_ant_total = max(ant_totals)
            best_total = max(best_total, max_ant_total)
            pheromone = pheromone * 0.9 + max_ant_total / 10.0
            
        return best_total

    # Candidate 1: LP Dual Prices (Reduced Costs)
    # Maximize P^T x subject to W x <= C, 0 <= x <= 1
    # linprog minimizes c^T x, so we minimize -P^T x
    c = -P
    A_ub = w.T  # shape (5, 100) -> rows are constraints, cols are variables
    b_ub = np.full(n_constraints, capacity)
    bounds = [(0, 1)] * n_items
    
    h_lp = None
    try:
        if time.time() < deadline - 60:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success and res.ineqlin is not None and len(res.ineqlin) > 0:
                # Dual variables (shadow prices) for inequality constraints
                # In scipy's linprog, res.ineqlin contains the duals for A_ub x <= b_ub
                # Note: The sign convention might need checking. 
                # Usually, for min c^T x s.t. A_ub x <= b_ub, duals y >= 0.
                # Objective value = c^T x* = -b^T y*
                # Reduced cost for variable j: c_j - sum_i a_ij y_i
                # Here c_j = -P_j, a_ij = w_ij.
                # Reduced cost = -P_j - sum_i w_ij y_i
                # We want high attractiveness for "good" items.
                # In LP, a variable with 0 reduced cost is in the basis.
                # Variables with positive reduced cost (in minimization) are not in basis.
                # Wait, for maximization (which we converted to min), 
                # "good" items have high P. 
                # Let's look at the "efficiency".
                # Efficiency = P_j - sum_i w_ij * lambda_i
                # If we use lambda = res.ineqlin, and we want to maximize P, 
                # the dual problem is min b^T y s.t. W^T y >= P, y >= 0.
                # So y_i are the prices of resources.
                # The "value" of item j relative to resource prices is P_j - W_j^T y.
                # If this is positive, it's "better than average" or part of optimal mix.
                # If negative, it's worse.
                
                duals = res.ineqlin
                # Ensure duals are non-negative (they should be for <= constraints in min problem)
                duals = np.maximum(duals, 0)
                
                # Compute "profitability" or reduced cost equivalent
                # score = P - W^T * duals
                scores = P - (w * duals[:, None]).sum(axis=0)
                
                # Heuristic should be positive. 
                # Items with high scores are attractive.
                # We can use exp(scores) or just scores clipped.
                # Since P and W are in [0,1], scores are roughly in [-5, 1].
                # We want to emphasize high scores.
                h_lp = np.exp(scores)
                
                # Safety: ensure finite and positive
                h_lp = np.where(np.isfinite(h_lp) & (h_lp > 0), h_lp, 1e-9)
                h_lp = np.clip(h_lp, 1e-9, 1e6)
                
    except Exception:
        h_lp = None

    # Candidate 2: Simple Prize/MaxWeight (Fallback / Baseline)
    V_max = w.max(axis=0)
    h_simple = np.clip(P / (V_max + 1e-9), 1e-9, 1e6)
    
    # Candidate 3: Prize/SumWeight
    V_sum = w.sum(axis=0)
    h_sum = np.clip(P / (V_sum + 1e-9), 1e-9, 1e6)

    # Candidate 4: Hybrid LP + Simple
    # If LP works, maybe blend? Or just pick the best one.
    
    best_h = h_simple
    best_score = -1.0
    
    # Evaluate candidates
    candidates = []
    if h_lp is not None:
        candidates.append(h_lp)
    candidates.append(h_simple)
    candidates.append(h_sum)
    
    # Also try a power version of LP scores
    if h_lp is not None:
        # Try sharpening the distribution
        h_lp_sharp = np.clip(h_lp ** 2, 1e-9, 1e6)
        candidates.append(h_lp_sharp)
        
        # Try log scale? No, exp is already log-linear.
        # Try normalizing LP scores to [0,1] then scaling?
        min_s = h_lp.min()
        max_s = h_lp.max()
        if max_s > min_s:
            h_lp_norm = (h_lp - min_s) / (max_s - min_s)
            # Map to [1, 10] to keep positive
            h_lp_scaled = 1.0 + 9.0 * h_lp_norm
            candidates.append(h_lp_scaled)

    # Add a few more structural heuristics just in case LP fails or is mediocre
    # Prize / (1 + Total Volume)
    h_denom = np.clip(P / (1.0 + V_sum), 1e-9, 1e6)
    candidates.append(h_denom)

    # Evaluate each candidate
    for h in candidates:
        if time.time() > deadline - 10.0:
            break
        # Ensure h is valid
        if not np.all(np.isfinite(h)) or not np.all(h > 0):
            continue
            
        score = fast_colony_sim(h, 42)
        
        if score > best_score:
            best_score = score
            best_h = h

    # If LP heuristic was the best, we might try to refine it with a slight perturbation?
    # Or just return the best found.
    
    # Final sanity check
    if best_h is None or not np.all(np.isfinite(best_h)) or not np.all(best_h > 0):
        best_h = h_simple

    return best_h
