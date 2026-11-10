import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in Z^11 such that 
    min pairwise squared distance >= max squared norm.
    """
    start_time = time.time()
    deadline = 590.0
    
    d = 11
    
    # Fallback: Norm 2 points (220 points)
    def get_norm2_points():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        p = [0] * d
                        p[i] = si
                        p[j] = sj
                        points.append(p)
        return np.array(points, dtype=np.int64)

    # Generate Norm 3 candidates
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            p = [0] * d
                            p[i] = si
                            p[j] = sj
                            p[k] = sk
                            candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute dot products
    dot_prods = candidates @ candidates.T
    
    # Conflict if dot >= 2
    conflict_mask = dot_prods >= 2
    
    best_set = []
    best_size = 0
    
    # Run multiple random greedy searches
    # Use a time limit for the search phase
    search_deadline = deadline - 10.0
    
    # We can use a faster greedy by precomputing neighbor lists?
    # conflict_mask is 1320x1320.
    
    attempts = 0
    while time.time() < search_deadline:
        attempts += 1
        if attempts % 1000 == 0:
            # Check progress
            pass
            
        order = np.random.permutation(n_cand)
        available = np.ones(n_cand, dtype=bool)
        selected = []
        
        for idx in order:
            if available[idx]:
                selected.append(idx)
                available[idx] = False
                # Mark neighbors as unavailable
                # conflict_mask[idx] is a boolean array of length n_cand
                available[conflict_mask[idx]] = False
        
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = selected.copy()
            
            # Optional: Local search improvement
            # Try to add any remaining node
            selected_set = set(selected)
            # Check if any available node can be added? 
            # In greedy, we marked neighbors as unavailable. 
            # But if we didn't pick a node, it might still be available 
            # if it wasn't a neighbor of any picked node.
            # Wait, the greedy loop iterates through ALL nodes.
            # If a node was available at its turn, it would have been picked.
            # So if the loop finishes, no available node exists that was 
            # "available" at its turn. 
            # However, "available" status changes.
            # A node might become available? No, we only set to False.
            # So if the loop finishes, `available` contains nodes that were 
            # never picked and were blocked by some picked node.
            # So we cannot add any more nodes directly.
            
            # Local Search: Try swaps
            # This is complex to implement efficiently in numpy within time limit.
            # Let's just do many random restarts.
            
    # Verify best set
    if best_size > 220:
        points = candidates[best_set]
        # Verify
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        n_points = len(points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return points
            
    # Fallback
    return get_norm2_points()
