import numpy as np
import time
import random
from itertools import combinations

D = 11

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    safety_margin = 5
    
    # 1. Generate Candidates: All vectors with exactly 4 non-zero entries, each +/- 1
    # Total candidates: C(11, 4) * 2^4 = 330 * 16 = 5280
    candidates = []
    indices = range(D)
    for support in combinations(indices, 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cands = len(candidates)
    
    # Pre-compute norms (all are sqrt(4) = 2, so norm_sq = 4)
    # Condition: min_dist_sq >= 4
    
    # We will use a greedy approach.
    # To speed up, we can pre-compute the distance matrix?
    # 5280 x 5280 matrix of floats32 is ~112 MB. Acceptable.
    # dist_sq[i, j] = ||candidates[i] - candidates[j]||^2
    
    # However, computing this might take time.
    # Let's try to do it efficiently.
    # diff = candidates[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    # dist_sq = np.sum(diff**2, axis=2)
    
    # This creates a 5280x5280x11 array. 5280^2 * 11 * 8 bytes = ~2.5 GB. TOO BIG.
    
    # We must compute distances on the fly or in blocks.
    # Greedy with on-the-fly checking:
    # For each candidate, check against all selected.
    # If selected size is K, cost is K * D.
    # Total cost ~ N * K * D.
    # If K ~ 1000, N ~ 5000. 5000 * 1000 * 11 = 55 million ops. Fast.
    
    best_set = []
    best_size = 0
    
    # Strategy: Randomized Greedy
    # Shuffle candidates, then greedily add if compatible.
    
    def is_compatible(v, current_set, min_dist_sq):
        # current_set is a numpy array of shape (K, D)
        if len(current_set) == 0:
            return True
        # Compute distances from v to all vectors in current_set
        # v: (D,)
        # current_set: (K, D)
        # diff: (K, D)
        diffs = current_set - v
        dists_sq = np.sum(diffs * diffs, axis=1)
        return np.min(dists_sq) >= min_dist_sq

    # We want to find a large set.
    # We can try different random seeds for shuffling.
    
    num_trials = 50
    for trial in range(num_trials):
        if time.time() - start_time > deadline - safety_margin:
            break
            
        # Shuffle candidates
        indices_shuffled = np.random.permutation(n_cands)
        
        current_set = []
        # We assume all vectors have norm_sq = 4.
        # So min_dist_sq must be >= 4.
        
        for idx in indices_shuffled:
            v = candidates[idx]
            # Check compatibility
            if len(current_set) == 0:
                current_set.append(v)
            else:
                # Convert to array for fast numpy ops
                # Optimization: Keep current_set as a list, convert to array only for check?
                # Or keep as array?
                # List append is O(1). Array append is O(K).
                # Let's keep as list, and convert to array for the distance check.
                curr_arr = np.array(current_set)
                diffs = curr_arr - v
                dists_sq = np.sum(diffs * diffs, axis=1)
                if np.min(dists_sq) >= 4:
                    current_set.append(v)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Return the best set found
    if best_size == 0:
        # Fallback to seed if greedy fails (shouldn't happen)
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_set, dtype=np.int64)

# Need to import product
from itertools import product
