"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, weights, heuristic, seed=42):
    np.random.seed(seed)
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    tau = np.ones(n_items)
    best_global_prize = 0.0
    
    for iter_count in range(30):
        best_iter_prize = 0.0
        
        for ant_count in range(20):
            remaining_cap = np.full(n_constraints, capacity)
            current_prize = 0.0
            remaining_items = list(range(n_items))
            
            while remaining_items:
                fitting_items = []
                for i in remaining_items:
                    if np.all(weights[:, i] <= remaining_cap):
                        fitting_items.append(i)
                
                if not fitting_items:
                    break
                
                fitting_items = np.array(fitting_items)
                
                probs = tau[fitting_items] * heuristic[fitting_items]
                
                probs[probs <= 0] = 1e-9
                sum_probs = probs.sum()
                if sum_probs == 0:
                    probs = np.ones(len(probs))
                else:
                    probs = probs / sum_probs
                
                chosen_idx = np.random.choice(fitting_items, p=probs)
                
                current_prize += prizes[:, chosen_idx].sum()
                remaining_cap -= weights[:, chosen_idx]
                remaining_items.remove(chosen_idx)
            
        if current_prize > best_iter_prize:
            best_iter_prize = current_prize
        
        if best_iter_prize > best_global_prize:
            best_global_prize = best_iter_prize
            
        tau = tau * 0.9 + (best_iter_prize / 10.0)
        
    return best_global_prize

def _greedy_marginal_utility(prizes, weights, n_starts=50, seed=12345):
    """
    Estimate marginal utility by running many deterministic greedy simulations.
    Each simulation starts with a random permutation of items (or random tie-breaking)
    and greedily adds items if they fit.
    We count how often each item is selected, weighted by the total prize of the solution.
    This captures items that are "good" in the context of the full constraint set.
    """
    np.random.seed(seed)
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    selection_counts = np.zeros(n_items)
    weighted_selection_counts = np.zeros(n_items)
    
    for _ in range(n_starts):
        remaining_cap = np.full(n_constraints, capacity)
        current_prize = 0.0
        remaining_items = list(range(n_items))
        
        # Randomize order to simulate different "ants" or "starts"
        np.random.shuffle(remaining_items)
        
        selected_items = []
        
        for i in remaining_items:
            if np.all(weights[:, i] <= remaining_cap):
                # Item fits
                selected_items.append(i)
                current_prize += prizes[:, i].sum()
                remaining_cap -= weights[:, i]
        
        # If we found a valid solution, credit the items
        if selected_items:
            for i in selected_items:
                selection_counts[i] += 1
                weighted_selection_counts[i] += current_prize
                
    # The heuristic should reflect how valuable an item is.
    # A simple frequency count might be biased towards items that fit easily (low weight).
    # Weighting by the total prize of the solution helps: items in high-value solutions are better.
    # However, if an item is always in high-value solutions, it gets high credit.
    # Let's combine frequency and weighted frequency.
    
    # Normalize
    freq_norm = selection_counts / (selection_counts.max() + 1e-9)
    weight_norm = weighted_selection_counts / (weighted_selection_counts.max() + 1e-9)
    
    # Heuristic is proportional to utility.
    # We want higher scores for items that appear frequently in high-value solutions.
    heuristic = freq_norm + weight_norm
    
    # Ensure positive
    heuristic[heuristic <= 0] = 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    
    return heuristic

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 15.0 # 15s safety margin
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    # Strategy:
    # 1. Generate a candidate heuristic using marginal utility estimation.
    # 2. Also generate a few strong static candidates (density based) as fallbacks.
    # 3. Evaluate them using a fast proxy (smaller ACO simulation) or just pick the best static one if time is tight.
    # Given the 300s limit, we can afford to run the marginal utility estimation (50 starts * 100 items * 5 constraints is very fast).
    # The bottleneck is the ACO simulation for evaluation.
    
    # Generate the simulation-based heuristic
    h_sim = _greedy_marginal_utility(prizes, weights, n_starts=100, seed=999)
    
    # Generate static candidates
    P_sum = prizes.sum(axis=0)
    W_sum = weights.sum(axis=0)
    W_max = weights.max(axis=0)
    
    # Candidate 1: Density
    h_density = P_sum / (W_sum + 1e-9)
    
    # Candidate 2: Max Weight Density (Inspiration 1)
    h_max_density = P_sum / (W_max + 1e-9)
    
    # Candidate 3: Sharpened Max Weight Density (Knowledge from run: squaring helps)
    h_sharp = (P_sum / (W_max + 1e-9)) ** 2
    
    # Candidate 4: Product of Ratios
    prod_ratio = np.ones(n_items)
    for c in range(n_constraints):
        prod_ratio *= (prizes[c] + 1e-9) / (weights[c] + 1e-9)
    
    candidates = [
        ("sim", h_sim),
        ("density", h_density),
        ("max_density", h_max_density),
        ("sharp", h_sharp),
        ("prod", prod_ratio)
    ]
    
    # Evaluate candidates
    # We need to pick the best one.
    # Running full ACO (30 iters, 20 ants) for each candidate is expensive.
    # Let's run a reduced ACO simulation for evaluation: 10 iterations, 10 ants.
    # This should be ~10x faster.
    
    def _fast_simulate_aco(prizes, weights, heuristic, seed=42):
        np.random.seed(seed)
        n_items = weights.shape[1]
        n_constraints = weights.shape[0]
        capacity = 25.0
        
        tau = np.ones(n_items)
        best_global_prize = 0.0
        
        for iter_count in range(10): # Reduced iterations
            best_iter_prize = 0.0
            
            for ant_count in range(10): # Reduced ants
                remaining_cap = np.full(n_constraints, capacity)
                current_prize = 0.0
                remaining_items = list(range(n_items))
                
                while remaining_items:
                    fitting_items = []
                    for i in remaining_items:
                        if np.all(weights[:, i] <= remaining_cap):
                            fitting_items.append(i)
                    
                    if not fitting_items:
                        break
                    
                    fitting_items = np.array(fitting_items)
                    
                    probs = tau[fitting_items] * heuristic[fitting_items]
                    
                    probs[probs <= 0] = 1e-9
                    sum_probs = probs.sum()
                    if sum_probs == 0:
                        probs = np.ones(len(probs))
                    else:
                        probs = probs / sum_probs
                    
                    chosen_idx = np.random.choice(fitting_items, p=probs)
                    
                    current_prize += prizes[:, chosen_idx].sum()
                    remaining_cap -= weights[:, chosen_idx]
                    remaining_items.remove(chosen_idx)
                
            if current_prize > best_iter_prize:
                best_iter_prize = current_prize
            
            if best_iter_prize > best_global_prize:
                best_global_prize = best_iter_prize
                
            tau = tau * 0.9 + (best_iter_prize / 10.0)
            
        return best_global_prize

    best_score = -1
    best_heuristic = candidates[0][1]
    
    seeds = [42, 123, 999]
    
    for name, h in candidates:
        if time.time() > deadline:
            break
            
        h_eval = np.asarray(h).flatten()
        h_eval[h_eval <= 0] = 1e-9
        h_eval[~np.isfinite(h_eval)] = 1e-9
        
        total_score = 0.0
        for s in seeds:
            if time.time() > deadline:
                break
            try:
                score = _fast_simulate_aco(prizes, weights, h_eval, seed=s)
                total_score += score
            except Exception:
                total_score = -1
                break
        
        if total_score > best_score:
            best_score = total_score
            best_heuristic = h_eval
            
    # Final check
    best_heuristic = np.asarray(best_heuristic).flatten()
    best_heuristic[best_heuristic <= 0] = 1e-9
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
# EVOLVE-BLOCK-END
