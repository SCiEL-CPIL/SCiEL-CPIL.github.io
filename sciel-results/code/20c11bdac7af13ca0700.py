import time
import random
import numpy as np

N = 64
TOTAL = N * N
TIME_LIMIT = 300.0

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 10.0  # Safety margin

    # Precompute coordinates
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    # Precompute squared distances matrix
    dx = (xs[:, None] - xs[None, :]).astype(np.int32)
    dy = (ys[:, None] - ys[None, :]).astype(np.int32)
    dist = dx * dx + dy * dy

    # Precompute conflict graph for maximum independent set approach
    # Two points a, b conflict if there exists a point c such that 
    # triangle (a,b,c) is isosceles.
    # Actually, for the independent set formulation:
    # We want to select a set S such that NO three points in S form an isosceles triangle.
    # This is NOT simply a maximum independent set on a pairwise conflict graph.
    # The conflict is a 3-way constraint, not pairwise.
    # 
    # However, we can model it differently:
    # A point c is "conflicted" with a pair (a,b) if (a,b,c) is isosceles.
    # This is a hypergraph independent set problem.
    #
    # Alternative: Use a different search strategy.
    # Let's try a different approach: use a more efficient local search with
    # better move generation and a different initialization strategy.
    #
    # New strategy: Use a "density-based" greedy initialization that tries to
    # pack points in a structured pattern, then use local search with swaps.
    
    # Let's try a completely different approach:
    # Use a simulated annealing with a different move set:
    # 1. Add a point
    # 2. Remove a point  
    # 3. Swap a point
    # 4. Move a point (remove and add different)
    
    # But first, let's try a smarter initialization using a grid pattern.
    
    def greedy_structured_init():
        """Initialize with a structured pattern to get a good starting point."""
        current_indices = []
        current_mask = np.zeros(TOTAL, dtype=bool)
        
        # Try to add points in a structured way
        # Start with corners and edges, then fill in
        # This is heuristic, but might give a better starting point
        
        # Actually, let's just use a random greedy but with a twist:
        # Process points in a specific order that might lead to denser packing
        
        # Create a list of all indices
        all_indices = np.arange(TOTAL)
        random.shuffle(all_indices)
        
        for idx in all_indices:
            if time.time() > deadline - 60:
                break
            if len(current_indices) < 2:
                current_indices.append(idx)
                current_mask[idx] = True
                continue
                
            # Check if this point can be added
            # Use vectorized check
            D_c = dist[idx, current_indices]  # distances from idx to all current points
            
            # Condition 1: all distances must be unique
            D_c_sorted = np.sort(D_c)
            if np.any(D_c_sorted[1:] == D_c_sorted[:-1]):
                continue
                
            # Condition 2: d(idx, s1) != d(s1, s2) for any s1, s2
            D_sub = dist[np.ix_(current_indices, current_indices)]
            
            # For each s1, check if d(idx, s1) is in {d(s1, s2) for s2 != s1}
            valid = True
            for k1 in range(len(current_indices)):
                d_cs = D_c[k1]
                d_ss = D_sub[k1, :]
                if d_cs in d_ss:
                    valid = False
                    break
            
            if valid:
                current_indices.append(idx)
                current_mask[idx] = True
                
        return current_indices, current_mask

    current_indices, current_mask = greedy_structured_init()
    best_indices = list(current_indices)
    best_size = len(current_indices)
    
    # Local search with swaps
    # Use a more aggressive search: try multiple swaps per iteration
    
    T = 50.0
    T_min = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline - 5:
        if T < T_min:
            T = T_min
            
        # Try to add a point
        # Sample random points not in current set
        raw_candidates = np.random.choice(TOTAL, size=500, replace=False)
        valid_candidates = raw_candidates[~current_mask[raw_candidates]]
        
        if len(valid_candidates) > 0:
            # Check validity of all candidates
            # Vectorized check
            D_c = dist[np.ix_(valid_candidates, current_indices)]
            
            # Condition 1: check for duplicate distances in each row
            D_c_sorted = np.sort(D_c, axis=1)
            has_dup = np.any(D_c_sorted[:, 1:] == D_c_sorted[:, :-1], axis=1)
            
            # Condition 2: check if d(c, s1) == d(s1, s2) for any s1, s2
            D_sub = dist[np.ix_(current_indices, current_indices)]
            invalid_b = np.zeros(len(valid_candidates), dtype=bool)
            for k1 in range(len(current_indices)):
                d_cs = D_c[:, k1]
                d_ss = D_sub[k1, :]
                invalid_b |= np.isin(d_cs, d_ss)
            
            valid_mask = ~(has_dup | invalid_b)
            valid_cands = valid_candidates[valid_mask]
            
            if len(valid_cands) > 0:
                # Add a random valid candidate
                chosen_idx = valid_cands[random.randrange(len(valid_cands))]
                current_indices.append(chosen_idx)
                current_mask[chosen_idx] = True
                
                if len(current_indices) > best_size:
                    best_size = len(current_indices)
                    best_indices = list(current_indices)
                    
                T *= cooling_rate
                continue
        
        # No point could be added, try a swap
        if len(current_indices) > 0:
            # Pick a random point to remove
            remove_pos = random.randrange(len(current_indices))
            remove_idx = current_indices[remove_pos]
            
            # Temporarily remove
            temp_indices = [x for x in current_indices if x != remove_idx]
            
            # Try to add a different point
            raw_swap_cands = np.random.choice(TOTAL, size=500, replace=False)
            valid_swap_cands = raw_swap_cands[~current_mask[raw_swap_cands]]
            
            if len(valid_swap_cands) > 0:
                D_c = dist[np.ix_(valid_swap_cands, temp_indices)]
                
                # Condition 1
                D_c_sorted = np.sort(D_c, axis=1)
                has_dup = np.any(D_c_sorted[:, 1:] == D_c_sorted[:, :-1], axis=1)
                
                # Condition 2
                D_sub = dist[np.ix_(temp_indices, temp_indices)]
                invalid_b = np.zeros(len(valid_swap_cands), dtype=bool)
                for k1 in range(len(temp_indices)):
                    d_cs = D_c[:, k1]
                    d_ss = D_sub[k1, :]
                    invalid_b |= np.isin(d_cs, d_ss)
                
                valid_mask = ~(has_dup | invalid_b)
                valid_swaps = valid_swap_cands[valid_mask]
                
                if len(valid_swaps) > 0:
                    # Accept swap
                    chosen_swap = valid_swaps[random.randrange(len(valid_swaps))]
                    current_indices[remove_pos] = chosen_swap
                    current_mask[remove_idx] = False
                    current_mask[chosen_swap] = True
        
        T *= cooling_rate

    # Final greedy pass
    if time.time() < deadline - 2:
        remaining_pts = [j for j in range(TOTAL) if not current_mask[j]]
        random.shuffle(remaining_pts)
        for idx in remaining_pts:
            if time.time() > deadline - 2:
                break
            if len(current_indices) < 2:
                current_indices.append(idx)
                current_mask[idx] = True
                continue
                
            D_c = dist[idx, current_indices]
            D_c_sorted = np.sort(D_c)
            if np.any(D_c_sorted[1:] == D_c_sorted[:-1]):
                continue
                
            D_sub = dist[np.ix_(current_indices, current_indices)]
            valid = True
            for k1 in range(len(current_indices)):
                d_cs = D_c[k1]
                d_ss = D_sub[k1, :]
                if d_cs in d_ss:
                    valid = False
                    break
            
            if valid:
                current_indices.append(idx)
                current_mask[idx] = True

    # Return best found
    result_indices = best_indices if len(best_indices) > len(current_indices) else current_indices
    result = [(int(xs[idx]), int(ys[idx])) for idx in result_indices]
    return result

if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
