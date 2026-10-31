import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate pool of candidate vectors
    # Weight 2: 220 vectors, norm^2 = 2
    # Weight 3: 1320 vectors, norm^2 = 3
    # We will include both to allow for mixed configurations if beneficial.
    
    candidates = []
    
    # Weight 2
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(tuple(v))
            
    # Weight 3
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(tuple(v))
            
    n = len(candidates)
    V = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(V**2, axis=1)
    
    # The constraint is: min pairwise distance >= max vector norm
    # Let M = max(norm_sq). We need dist^2 >= M for all pairs.
    # Note: M depends on the set we select.
    # If we only select Weight 2 vectors, M=2. Constraint: dist^2 >= 2.
    # If we select any Weight 3 vector, M=3. Constraint: dist^2 >= 3.
    
    # Strategy: Farthest-Point Sampling (FPS)
    # We want to select a set S such that for all u, v in S, dist(u, v)^2 >= max(norm(u)^2, norm(v)^2) ??
    # No, the task says "minimum pairwise distance is at least the maximum vector norm".
    # Let $R = \max_{v \in S} \|v\|$.
    # We need $\min_{u \neq v \in S} \|u-v\| \ge R$.
    # Squared: $\min_{u \neq v \in S} \|u-v\|^2 \ge R^2 = \max_{v \in S} \|v\|^2$.
    
    # This is a global constraint. Adding a high-norm vector increases R, making the constraint stricter for ALL pairs.
    # Therefore, it is generally better to stick to the smallest possible norm if the count is high.
    # Weight 2 (norm^2=2) allows 220 vectors.
    # Weight 3 (norm^2=3) requires dist^2 >= 3.
    # Let's check if we can get > 220 with Weight 3 only?
    # Or a mix? If we mix, R^2=3. We need ALL pairs (including W2-W2) to have dist^2 >= 3.
    # But W2-W2 pairs have dist^2 = 2 (if orthogonal) or 4 (if opposite) or 2 (if dot=1).
    # Wait, for W2 vectors $u, v$: $dist^2 = 4 - 2 u \cdot v$.
    # Min dist^2 for distinct W2s is 2 (when $u \cdot v = 1$).
    # If we include any W3 vector, R^2 becomes 3.
    # Then we require $dist^2(u, v) \ge 3$ for ALL pairs.
    # But some W2 pairs have $dist^2 = 2$.
    # So we CANNOT have any W2 pair with dot=1 in our final set if we include W3 vectors.
    # This means we must discard many W2 vectors to accommodate W3 vectors.
    # Likely, the optimal set is either all W2 or all W3 (or higher weights), not a mix, due to the global R constraint.
    
    # Let's try to find a large set of W3 vectors.
    # Constraint: $dist^2(u, v) \ge 3$ for all $u, v \in S \subset W3$.
    # $dist^2 = 3+3-2u \cdot v = 6 - 2u \cdot v$.
    # $6 - 2u \cdot v \ge 3 \implies 2u \cdot v \le 3 \implies u \cdot v \le 1.5 \implies u \cdot v \le 1$.
    
    # So for W3 vectors, we need pairwise dot product <= 1.
    
    # Let's run FPS on the W3 pool specifically, and also on the W2 pool (to ensure we have the 220 baseline).
    
    # 1. Baseline: All W2 vectors are valid (pairwise dist^2 >= 2, R^2=2).
    # Count: 220.
    
    # 2. Search for larger set in W3 pool.
    # Pool W3 indices: 220 to 220+1320-1.
    W3_start = 220
    W3_end = n
    V_w3 = V[W3_start:W3_end]
    n_w3 = len(V_w3)
    
    # Precompute dot products for W3? 
    # n_w3 is 1320. 1320x1320 matrix is ~1.7M floats. ~13MB. Feasible.
    # But FPS is iterative.
    
    best_set_indices = list(range(220)) # Indices in V for W2s
    best_size = 220
    
    # Try FPS on W3
    # Random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Start with a random W3 vector
        start_idx = np.random.randint(n_w3)
        selected_local = [start_idx]
        available = np.ones(n_w3, dtype=bool)
        available[start_idx] = False
        
        # Min distance to selected set for each candidate
        # dist^2(c, s) = ||c||^2 + ||s||^2 - 2 c.s = 6 - 2 c.s
        # We want to maximize min_{s in selected} dist^2(c, s)
        # And we must ensure min dist^2 >= 3.
        # Since all are W3, ||c||^2=3, ||s||^2=3.
        # dist^2 = 6 - 2 c.s.
        # Condition: 6 - 2 c.s >= 3 => c.s <= 1.5 => c.s <= 1.
        
        # We can vectorize the check.
        # Dots = V_w3 @ V_w3.T
        # This is 1320x1320.
        Dots_w3 = V_w3 @ V_w3.T
        # Conflict if Dots > 1
        # Valid if Dots <= 1
        
        # FPS Loop
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find candidate with max min-distance
            # min_dist_sq[c] = min_{s in selected} (6 - 2 * Dots_w3[c, s])
            # = 6 - 2 * max_{s in selected} Dots_w3[c, s]
            
            # Get dots of all candidates against selected
            # Dots_w3[:, selected_local] is (n_w3, len(selected_local))
            dots_sel = Dots_w3[:, selected_local]
            max_dot = np.max(dots_sel, axis=1)
            
            # Candidates must have max_dot <= 1 (to satisfy dist^2 >= 3)
            valid_mask = max_dot <= 1
            # Also must be available
            valid_mask = valid_mask & available
            
            if not np.any(valid_mask):
                break
                
            # Among valid, pick one that maximizes min_dist
            # min_dist = 6 - 2 * max_dot
            # So maximize min_dist <=> minimize max_dot
            # We want the candidate with the smallest max_dot among valid ones.
            
            # Set invalid ones to inf to ignore
            max_dot_valid = np.where(valid_mask, max_dot, np.inf)
            next_idx = np.argmin(max_dot_valid)
            
            if max_dot_valid[next_idx] == np.inf:
                break
                
            selected_local.append(next_idx)
            available[next_idx] = False
            
        # End FPS
        size = len(selected_local)
        if size > best_size:
            best_size = size
            # Map local indices back to global indices
            best_set_indices = [W3_start + i for i in selected_local]
            
        if best_size >= 224:
            break
            
    # Fallback: If we didn't find better than 220, return the 220 W2 vectors
    # Note: The 220 W2 vectors are indices 0..219 in V.
    
    result_vectors = V[best_set_indices]
    
    # Final Verification (Safety Check)
    # Check if result is valid
    if len(result_vectors) > 0:
        norms = np.sum(result_vectors**2, axis=1)
        max_norm = np.max(norms)
        
        # Check pairwise distances
        # For large sets, this is O(k^2 * D). 220^2 * 11 ~ 500k ops. Fast.
        # If k is larger, still fine.
        dists = np.sum(result_vectors**2, axis=1)[:, None] + np.sum(result_vectors**2, axis=1)[None, :] - 2 * (result_vectors @ result_vectors.T)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < max_norm - 1e-9:
            # Invalid result found due to numerical error or logic bug?
            # Fall back to W2
            result_vectors = V[:220]
            
    return result_vectors.astype(np.int64)
