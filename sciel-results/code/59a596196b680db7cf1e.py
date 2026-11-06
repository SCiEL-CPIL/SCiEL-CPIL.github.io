import numpy as np
import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Generate all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute squared distances between all pairs
    # This is a 4096x4096 matrix, which is ~16M entries - manageable
    # But we can compute on the fly for memory efficiency
    
    # Convert to numpy array for fast distance computation
    pts_array = np.array(points, dtype=np.int32)
    
    def get_sq_dists(idx):
        """Get squared distances from point idx to all other points"""
        return np.sum((pts_array - pts_array[idx])**2, axis=1)
    
    best_solution = []
    best_size = 0
    
    # Try multiple random restarts
    max_restarts = 500  # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:  # Stop if close to deadline
            break
        
        random.seed(restart * 7919 + 13)
        
        # Random order of points
        order = list(range(num_points))
        random.shuffle(order)
        
        # Greedily add points
        selected = []
        selected_set = set()
        existing_dists = set()  # All squared distances among selected points
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            # Check if we can add this point
            can_add = True
            sq_dists_to_selected = []
            
            for sel_idx in selected:
                dx = pts_array[idx][0] - pts_array[sel_idx][0]
                dy = pts_array[idx][1] - pts_array[sel_idx][1]
                sd = dx*dx + dy*dy
                sq_dists_to_selected.append(sd)
            
            # Check 1: No two distances from new point to selected points are equal
            if len(sq_dists_to_selected) != len(set(sq_dists_to_selected)):
                can_add = False
            else:
                # Check 2: No new distance equals any existing pair distance
                for sd in sq_dists_to_selected:
                    if sd in existing_dists:
                        can_add = False
                        break
            
            if can_add:
                selected.append(idx)
                selected_set.add(idx)
                # Update existing_dists with new distances
                for sel_idx in selected[:-1]:
                    dx = pts_array[idx][0] - pts_array[sel_idx][0]
                    dy = pts_array[idx][1] - pts_array[sel_idx][1]
                    sd = dx*dx + dy*dy
                    existing_dists.add(sd)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = [tuple(pts_array[i]) for i in selected]
        
        # If we've found a good solution, we can stop early
        if best_size >= 20:
            break
    
    return best_solution
