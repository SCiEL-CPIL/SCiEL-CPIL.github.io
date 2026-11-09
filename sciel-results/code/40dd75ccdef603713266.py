import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in an ACO for MCKP.
    Uses Simulated Annealing to optimize the heuristic vector directly
    using a fast, vectorized ACO simulator.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are correct: (5, 100)
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
    # We simulate the ACO exactly as described:
    # 20 ants, 30 iterations.
    # Pheromone starts at 1.
    # Heuristic is fixed (passed as arg).
    # Update: pheromone = pheromone * 0.9 + (best_prize_of_iteration) / 10.
    # Selection prob proportional to pheromone[i] * heuristic[i].
    
    def fast_colony_sim(h, seed):
        """
        Simulates the ACO process quickly.
        Returns the best total prize found by any ant in the entire run.
        """
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n_items)
        best_total_overall = 0.0
        
        for _ in range(30):
            ant_totals = []
            for _ in range(20):
                remaining_cap = np.full(n_constraints, capacity)
                used = np.zeros(n_items, dtype=bool)
                total_prize = 0.0
                
                # Items must fit in ALL constraints
                # w is (5, 100). remaining_cap is (5,).
                # fits_mask[i] is True if w[:, i] <= remaining_cap
                # We can check this vectorially: (w <= remaining_cap[:, None]).all(axis=0)
                
                while True:
                    # Update fits mask
                    # Only check items not yet used
                    # To speed up, we can pre-compute nothing really, but the mask is cheap for N=100
                    fits_mask = np.all(w <= remaining_cap[:, None], axis=0) & ~used
                    
                    if not np.any(fits_mask):
                        break
                    
                    fitting_indices = np.where(fits_mask)[0]
                    
                    # Calculate probabilities
                    # prob[i] = pheromone[i] * h[i]
                    # Note: h[i] must be positive. If h[i] <= 0, it's treated as 1e-9 in task desc?
                    # "Heuristic entries that are not positive become 1e-9"
                    # So we ensure h is positive in the optimization step.
                    
                    probs = pheromone[fitting_indices] * h[fitting_indices]
                    sum_probs = probs.sum()
                    if sum_probs == 0:
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
            if max_ant_total > best_total_overall:
                best_total_overall = max_ant_total
            pheromone = pheromone * 0.9 + max_ant_total / 10.0
            
        return best_total_overall

    # Initial Heuristic: Prize / Max Weight (standard ratio heuristic)
    # This is a good starting point.
    V_max = w.max(axis=0)
    h_init = P / (V_max + 1e-9)
    h_init = np.clip(h_init, 1e-9, 1e6)
    
    best_h = h_init.copy()
    best_score = fast_colony_sim(best_h, 42)
    
    # Simulated Annealing
    # We perturb the heuristic vector and accept/reject based on score change.
    # Since the landscape is likely rugged, SA is appropriate.
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Parameters for SA
    # We have ~300s budget.
    # One simulation takes ~50-100ms (estimated for N=100, C=5, 20 ants, 30 iters).
    # Let's assume 100ms per sim. 300s / 0.1s = 3000 simulations.
    # We can afford a decent number of steps.
    
    # Define a perturbation strategy:
    # Multiply random subset of items by (1 + noise) or add noise?
    # Multiplicative noise preserves scale and positivity better.
    
    T_start = 1.0  # Initial temperature (scale of score difference to accept)
    T_end = 0.01
    alpha_cool = 0.99
    
    T = T_start
    
    # We will run until time runs out or T is very low
    while time.time() < deadline - 5.0:
        # Perturb current_h
        # Create a new candidate by perturbing current_h
        # Use multiplicative noise on a random subset to explore local variations
        noise = np.random.normal(0, 0.1, size=n_items)
        candidate_h = current_h * (1.0 + noise)
        
        # Ensure positivity and reasonable bounds
        candidate_h = np.clip(candidate_h, 1e-9, 1e6)
        
        # Evaluate
        candidate_score = fast_colony_sim(candidate_h, 42)
        
        # Acceptance criterion
        delta_score = candidate_score - current_score
        
        if delta_score > 0:
            # Better, always accept
            current_h = candidate_h
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Worse, accept with probability exp(delta_score / T)
            # Note: delta_score is negative.
            # If T is small, exp(negative/small) is close to 0.
            # If T is large, exp(negative/large) is close to 1.
            prob = np.exp(delta_score / T)
            if np.random.random() < prob:
                current_h = candidate_h
                current_score = candidate_score
        
        # Cool down
        T *= alpha_cool
        
    return best_h
