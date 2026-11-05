import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a static heuristic that favors vertices with
       low degree and low local density, specifically penalizing vertices
       adjacent to hubs less than the hubs themselves to promote 'hub-blocking'.
    2. Refine these scores using an accelerated local search (pair swaps) that
       directly maximizes the size of the independent set produced by the greedy decoder.
       The simulation is vectorized for speed, allowing more iterations within the time limit.
    3. Use multiple restarts with noise to escape local optima.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists as numpy arrays for vectorized deactivation
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # --- 1. Initial Heuristic ---
    # We want to favor vertices that are "safe" to pick.
    # Low degree is generally good.
    # Low "neighbor degree sum" is good (isolated from hubs).
    # However, the "hub-blocking" heuristic suggests that vertices adjacent to
    # high-degree vertices (hubs) are valuable because picking them removes
    # the hub from consideration without the high cost of picking the hub itself.
    
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors (local clustering/density proxy)
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Normalize features to [0, 1] to balance weights
    def normalize(x):
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        else:
            return np.zeros_like(x)
            
    norm_deg = normalize(deg)
    norm_avg_ndeg = normalize(avg_neighbor_deg)
    
    # We want LOW degree and LOW local density.
    # So score = - (w1 * norm_deg + w2 * norm_avg_ndeg)
    # Invert so higher is better:
    scores = -(0.7 * norm_deg + 0.3 * norm_avg_ndeg)
    
    # Normalize scores to [0, 1] for better numerical behavior in local search
    scores = normalize(scores)
    
    def simulate_fast(s):
        """
        Fast greedy simulation.
        Returns the size of the independent set.
        Uses vectorized numpy operations for deactivation.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(N, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if len(nbrs) > 0:
                    # Vectorized deactivation of neighbors
                    active[nbrs] = False
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    # Precompute score order for structured swaps
    score_order = np.argsort(-scores)
    
    # Try multiple restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # For restart 0, use original. For others, perturb from best found.
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Add noise to escape local optima
            # Noise scale decreases slightly for later restarts? Or stays constant?
            # Let's keep it moderate.
            noise_scale = 0.1
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
        
        # Local search loop
        no_improve = 0
        # Allow more iterations since simulation is faster
        max_no_improve = 5000 + 1000 * restart
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose swap candidates
            # 60% structured (close in rank), 40% random
            if rng.random() < 0.6:
                base_idx = rng.integers(0, N - 1)
                # Pick a nearby index
                offset = rng.integers(1, min(20, N - 1))
                i = score_order[base_idx]
                j = score_order[min(base_idx + offset, N - 1)]
            else:
                i, j = rng.integers(0, N, size=2)
            
            if i == j:
                continue
            
            # Swap scores
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                    # Update score order if we find a new best
                    score_order = np.argsort(-current_scores)
            else:
                # Revert swap
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
        # Early stopping if no improvement in this restart
        if current_val <= best_val and restart > 2:
            break
    
    return best_scores
