import time
import random
import math
import numpy as np
from collections import Counter
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def evaluate(points):
        """
        Calculate the max multiplicity of a positive doubled area.
        points: list of (x, y) tuples
        """
        if len(points) < 3:
            return 0
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        # Get indices for all unique triples i < j < k
        # We can use itertools.combinations to get indices, then vectorize the calculation
        # Or use np.triu_indices
        # For n=30, number of triples is 4060. This is small enough to just loop in Python if NumPy overhead is high,
        # but let's try a NumPy approach.
        
        # Actually, for n=30, a pure Python loop might be competitive if optimized, 
        # but NumPy is generally safer for "fast" claims.
        # Let's stick to a highly optimized Python loop first? 
        # No, the directive says NumPy is available. Let's use it.
        
        # Vectorized approach:
        # Create arrays of x and y coords
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Use itertools.combinations to get indices of all triples i < j < k
        idxs = list(itertools.combinations(range(n), 3))
        if not idxs: return 0
        
        # Convert to numpy arrays of indices
        idx_array = np.array(idxs)
        i_idx = idx_array[:, 0]
        j_idx = idx_array[:, 1]
        k_idx = idx_array[:, 2]
        
        xi = x[i_idx]
        yi = y[i_idx]
        xj = x[j_idx]
        yj = y[j_idx]
        xk = x[k_idx]
        yk = y[k_idx]
        
        # Determinant formula: xi*(yj - yk) + xj*(yk - yi) + xk*(yi - yj)
        dets = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
        abs_dets = np.abs(dets)
        
        # Filter out zero areas
        non_zero = abs_dets[abs_dets > 0]
        if len(non_zero) == 0:
            return 0
            
        # Count multiplicities
        # np.unique with return_counts is efficient
        unique_vals, counts = np.unique(non_zero, return_counts=True)
        return int(np.max(counts))

    # Initial Candidates
    candidates = []
    
    def make_grid(w, h):
        ox = -w // 2
        oy = -h // 2
        return [(ox + i, oy + j) for i in range(w) for j in range(h)]

    # Strong grid candidates
    grid_shapes = [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30)]
    for w, h in grid_shapes:
        pts = make_grid(w, h)
        # Check bounds
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(pts)
            
    # Add some random perturbations of the 6x5 grid
    base_grid = make_grid(6, 5)
    for _ in range(5):
        perturbed = base_grid[:]
        for _ in range(5): # Move 5 points
            idx = random.randint(0, 29)
            x, y = perturbed[idx]
            new_x = x + random.randint(-2, 2)
            new_y = y + random.randint(-2, 2)
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                # Check collision
                if (new_x, new_y) not in perturbed:
                    perturbed[idx] = (new_x, new_y)
        candidates.append(perturbed)

    best_points = None
    best_score = -1
    
    # Evaluate candidates
    for pts in candidates:
        if time.time() > deadline:
            break
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]

    if best_points is None:
        best_points = make_grid(6, 5)
        best_score = evaluate(best_points)

    current_points = best_points[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    max_iter = 200000 # Can be higher due to fast eval
    
    point_set = set(current_points)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, 29)
        x, y = current_points[idx]
        
        # Random move
        # Try small steps
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = x + dx
        new_y = y + dy
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        
        # Collision check
        if new_pt in point_set:
            continue
            
        # Apply move
        point_set.remove((x, y))
        point_set.add(new_pt)
        current_points[idx] = new_pt
        
        new_score = evaluate(current_points)
        
        # Acceptance
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = current_score - new_score
            # Avoid log(0) if delta is large and T is small, but T_min prevents T=0
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                point_set.remove(new_pt)
                point_set.add((x, y))
                current_points[idx] = (x, y)
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_points
