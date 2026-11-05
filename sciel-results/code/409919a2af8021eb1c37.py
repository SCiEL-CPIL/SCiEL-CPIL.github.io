import numpy as np
import time

def priority(adjacency):
    """
    Calculate priority scores for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a "survival probability" heuristic:
       score ~ -deg - 0.5 * avg_neighbor_degree
    2. Refine these scores using local search (pair swaps) to directly maximize
       the size of the independent set produced by the greedy decoder.
       - Use a fast bitset-based simulation function.
       - Perform random and rank-adjacent swaps.
    """
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
    
    # Ensure symmetry and remove self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    
    # Convert to boolean for faster neighbor lookups
    A_bool = A > 0.5
    
    # Precompute neighbor lists for fast simulation (for ranking/heuristic)
    neighbors_list = [np.where(A_bool[i])[0] for i in range(N)]
    
    # Precompute bitmasks for fast simulation
    # neighbor_masks[i] is an integer with bits set for all neighbors of i
    neighbor_masks = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors_list[i]:
            mask |= (1 << j)
        neighbor_masks[i] = mask
        
    # Also include self in the mask for deactivation
    deactivation_masks = [0] * N
    for i in range(N):
        deactivation_masks[i] = neighbor_masks[i] | (1 << i)

    # --- 1. Initial Heuristic ---
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_deg_sum = A.dot(deg)
    
    # Average degree of neighbors
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Heuristic: Low degree is good. Being near hubs is bad (penalty).
    raw_scores = -(deg + 0.5 * avg_neighbor_deg)
    
    # Normalize to [0, 1]
    if raw_scores.max() > raw_scores.min():
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        scores = np.ones(N)
        
    def simulate_fast(s):
        """
        Fast greedy simulation using bitsets.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active_mask = (1 << N) - 1  # All bits set
        count = 0
        
        for v in order:
            bit = 1 << v
            if active_mask & bit:
                count += 1
                # Deactivate v and its neighbors
                active_mask &= ~deactivation_masks[v]
                if active_mask == 0:
                    break
        return count
    
    # --- 2. Baseline ---
    best_val = simulate_fast(scores)
    best_scores = scores.copy()
    
    # --- 3. Local Search ---
    rng = np.random.default_rng(42)
    
    num_restarts = 20 # Can afford more restarts if simulation is faster
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            current_scores = best_scores.copy()
            current_val = best_val
        else:
            # Perturb the best score found so far
            noise_scale = 0.1 + 0.05 * restart
            noise = rng.normal(0, noise_scale, N)
            current_scores = best_scores + noise
            current_val = simulate_fast(current_scores)
            
        # Local search loop
        no_improve = 0
        max_no_improve = 5000 # Increased due to speedup
        
        # Precompute current ranking for rank-adjacent swaps
        current_rank = np.argsort(-current_scores, kind='stable')
        
        while time.time() - start_time < DEADLINE and no_improve < max_no_improve:
            # Choose move type: 70% random, 30% rank-adjacent
            move_type = rng.random()
            
            if move_type < 0.7:
                # Random swap
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
            else:
                # Rank-adjacent swap
                pos = rng.integers(0, N - 1)
                offset = rng.integers(1, 4)  # 1, 2, or 3
                pos2 = pos + offset if pos + offset < N else pos - offset
                if pos2 < 0:
                    pos2 = pos + 1
                i = current_rank[pos]
                j = current_rank[pos2]
                if i == j:
                    continue
            
            # Swap
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                # Update ranking after improvement
                current_rank = np.argsort(-current_scores, kind='stable')
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
        
    # Final polish pass
    if time.time() - start_time < DEADLINE - 5.0:
        polish_deadline = DEADLINE - 2.0
        current_scores = best_scores.copy()
        current_val = best_val
        current_rank = np.argsort(-current_scores, kind='stable')
        no_improve = 0
        max_no_improve_polish = 5000
        
        while time.time() - start_time < polish_deadline and no_improve < max_no_improve_polish:
            # In polish, use only rank-adjacent swaps for fine-tuning
            pos = rng.integers(0, N - 1)
            offset = rng.integers(1, 3)  # 1 or 2
            pos2 = pos + offset if pos + offset < N else pos - offset
            if pos2 < 0:
                pos2 = pos + 1
            i = current_rank[pos]
            j = current_rank[pos2]
            if i == j:
                continue
            
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate_fast(current_scores)
            
            if new_val > current_val:
                current_val = new_val
                no_improve = 0
                if current_val > best_val:
                    best_val = current_val
                    best_scores = current_scores.copy()
                current_rank = np.argsort(-current_scores, kind='stable')
            else:
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve += 1
    
    return best_scores
