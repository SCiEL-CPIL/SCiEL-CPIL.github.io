import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate candidate vectors with coordinates in {-1, 0, 1}
    # and norm squared <= 3
    candidates = []
    # Norm 1: 1 non-zero
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
            
    # Norm 2: 2 non-zeros
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm 3: 3 non-zeros
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # We will try to find a large subset S such that min_{u,v in S, u!=v} ||u-v||^2 >= max_{u in S} ||u||^2
    # This is equivalent to: For all u, v in S, ||u-v||^2 >= R^2, where R^2 = max_{u in S} ||u||^2.
    
    # Strategy: Simulated Annealing
    # State: boolean mask of selected vectors
    # Objective: count of selected vectors
    # Constraint: For all selected u, v, ||u-v||^2 >= R^2
    
    # Initial state: The seed solution (all Norm 2 vectors)
    # Indices of Norm 2 vectors
    norm2_mask = (norms_sq == 2)
    initial_mask = np.zeros(n_candidates, dtype=bool)
    initial_mask[norm2_mask] = True
    
    def get_R2(mask):
        if not np.any(mask):
            return 0
        return np.max(norms_sq[mask])
        
    def is_valid(mask):
        R2 = get_R2(mask)
        if R2 == 0:
            return True
        # Get indices of selected vectors
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return True
        # Check all pairs
        # Optimization: Only check if R2 > 2, because if R2 <= 2, the seed (Norm 2) is already valid for R2=2.
        # Wait, if R2=2, we need d^2 >= 2.
        # If R2=3, we need d^2 >= 3.
        
        # For speed, we can precompute the conflict matrix for a fixed R2?
        # But R2 changes.
        
        # We can check violations incrementally or in batches.
        # For SA, we need fast evaluation.
        
        # Let's precompute the squared distance matrix for all candidates?
        # n_candidates is ~1562.
        # 1562x1562 matrix is ~2.4M entries.
        # This is feasible.
        
        return None # Placeholder
        
    # Precompute distance matrix
    # dist2[i, j] = ||candidates[i] - candidates[j]||^2
    # Use broadcasting
    # diff = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    # dist2 = np.sum(diff**2, axis=2)
    
    # This might be memory heavy (1562^2 * 8 bytes ~ 20MB). Acceptable.
    diff = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    # Free memory
    del diff
    
    def check_validity(mask):
        R2 = get_R2(mask)
        if R2 == 0:
            return True
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return True
        
        # Get sub-matrix of distances
        d_sub = dist2[np.ix_(idx, idx)]
        # We need min(d_sub) >= R2, excluding diagonal (which is 0)
        # Set diagonal to inf
        np.fill_diagonal(d_sub, np.inf)
        min_d = np.min(d_sub)
        return min_d >= R2

    # Simulated Annealing
    current_mask = initial_mask.copy()
    current_score = np.sum(current_mask)
    best_mask = current_mask.copy()
    best_score = current_score
    
    # Verify initial
    if check_validity(current_mask):
        pass
    else:
        # Should not happen for seed
        current_mask = np.zeros(n_candidates, dtype=bool)
        current_score = 0
        best_mask = current_mask.copy()
        best_score = 0
        
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Random permutation of indices for shuffling
    order = np.arange(n_candidates)
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Pick a random vector to toggle
        i = np.random.choice(order)
        
        # Try adding i
        if not current_mask[i]:
            new_mask = current_mask.copy()
            new_mask[i] = True
            if check_validity(new_mask):
                current_mask = new_mask
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask.copy()
            else:
                # Reject or accept with prob exp(-1/T)
                # Penalty is 1 (loss of validity)
                if np.random.rand() < np.exp(-1.0/T):
                    current_mask = new_mask
                    # Note: current_score is not updated because it's invalid?
                    # In SA, we usually track "energy". Here energy = -count + penalty.
                    # Let's stick to valid states only to simplify?
                    # If we only allow valid states, it's a local search.
                    # To escape local optima, we need to allow invalid states.
                    # Let's define Energy = -count + 100 * (invalid ? 1 : 0)
                    pass
        else:
            # Try removing i
            new_mask = current_mask.copy()
            new_mask[i] = False
            # Removing a vector always maintains validity (or improves it)
            current_mask = new_mask
            current_score -= 1
            
        T *= cooling
        
    # Final polish: Greedy removal of vectors that allow adding others?
    # Or just return best found
    
    # Ensure best_mask is valid
    if not check_validity(best_mask):
        # Fallback to seed
        best_mask = initial_mask.copy()
        best_score = np.sum(best_mask)
        
    return candidates[best_mask]
# EVOLVE-BLOCK-END
