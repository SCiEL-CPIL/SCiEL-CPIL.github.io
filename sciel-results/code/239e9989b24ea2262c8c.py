"""Improved circle packing in perimeter-4 rectangle using Simulated Annealing and Delaunay Triangulation."""
import numpy as np
import time
from scipy.spatial import Delaunay
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # State: w, x, y, r
    # w is width, h = 2 - w
    # x, y, r are arrays of length N
    
    def get_neighbors(centers):
        """
        Compute Delaunay neighbors for the given centers.
        Returns a list of sets of neighbor indices for each circle.
        """
        try:
            tri = Delaunay(centers)
            neighbors = [set() for _ in range(N)]
            for simplex in tri.simplices:
                i, j, k = simplex
                neighbors[i].add(j)
                neighbors[i].add(k)
                neighbors[j].add(i)
                neighbors[j].add(k)
                neighbors[k].add(i)
                neighbors[k].add(j)
            # Convert to lists for faster iteration
            return [list(s) for s in neighbors]
        except Exception:
            # Fallback if Delaunay fails (e.g., collinear points)
            # Use all pairs
            return [list(range(N)) for _ in range(N)]

    def check_validity(w, x, y, r, neighbors):
        """
        Check if the configuration is valid.
        Returns True if valid, False otherwise.
        """
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False
        
        # Check boundaries
        for i in range(N):
            if r[i] < 1e-9:
                return False
            if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
                return False
            if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
                return False
                
        # Check overlaps using neighbor list
        # Note: If the neighbor list is stale, we might miss overlaps.
        # However, we update it periodically. For strict validity check in final result,
        # we should check all pairs. But for SA steps, neighbor list is an approximation 
        # to speed things up. We will do a full check for the best solution update.
        for i in range(N):
            for j in neighbors[i]:
                if j > i: # Avoid double checking
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    if dist_sq < (min_dist - 1e-9)**2:
                        return False
        return True

    def full_check_validity(w, x, y, r):
        """
        Strict check for all pairs. Used for final result verification.
        """
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False
        
        for i in range(N):
            if r[i] < 1e-9:
                return False
            if x[i] - r[i] < -1e-8 or x[i] + r[i] > w + 1e-8:
                return False
            if y[i] - r[i] < -1e-8 or y[i] + r[i] > h + 1e-8:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - 1e-9)**2:
                    return False
        return True

    def calculate_value(x, y, r):
        return np.sum(r)

    # Initial Configuration
    w = 1.4
    h = 0.6
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.full(N, 0.09)
    
    # Grid initialization
    a, b = 7, 3
    for i in range(N):
        col = i % a
        row = i // a
        x[i] = (col + 0.5) * w / a
        y[i] = (row + 0.5) * h / b
        
    # Ensure initial validity
    # Adjust radii if needed to fit
    min_spacing_x = w / a
    min_spacing_y = h / b
    max_r = min(min_spacing_x, min_spacing_y) / 2
    # If grid is too tight, reduce r
    # For 7x3 in 1.4x0.6: spacing x = 0.2, y = 0.2. r=0.1 fits exactly.
    # Let's start slightly smaller to allow movement.
    r[:] = 0.095 
    
    if not full_check_validity(w, x, y, r):
         # Fallback to smaller radii
         r[:] = 0.08
         if not full_check_validity(w, x, y, r):
             r[:] = 0.07
             
    centers = np.column_stack((x, y))
    neighbors = get_neighbors(centers)
    
    best_value = calculate_value(x, y, r)
    best_state = (w, x.copy(), y.copy(), r.copy())
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial Temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling factor
    iterations = 0
    max_iterations = 500000 # Cap iterations, but time is the real limit
    
    step_size = 0.02
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        # Copy current state for trial
        trial_w = w
        trial_x = x.copy()
        trial_y = y.copy()
        trial_r = r.copy()
        
        if move_type < 0.4:
            # Move a single circle
            i = random.randint(0, N-1)
            trial_x[i] += random.gauss(0, step_size)
            trial_y[i] += random.gauss(0, step_size)
            
        elif move_type < 0.7:
            # Push two neighbors together
            # Pick a random circle i, then a random neighbor j
            i = random.randint(0, N-1)
            if neighbors[i]:
                j = random.choice(neighbors[i])
                
                # Calculate vector from i to j
                dx = trial_x[j] - trial_x[i]
                dy = trial_y[j] - trial_y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-6:
                    # Randomize direction
                    angle = random.uniform(0, 2*np.pi)
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    dist = 1.0
                else:
                    dx /= dist
                    dy /= dist
                    
                # Target distance is r[i] + r[j]
                target_dist = trial_r[i] + trial_r[j]
                
                # Move both circles towards each other
                # We want to reduce the distance between them to target_dist
                # Current dist is 'dist'. We need to close (dist - target_dist)
                # Let's move each by half the difference
                diff = dist - target_dist
                if diff > 0:
                    # They are too far apart? No, if they are neighbors in Delaunay, 
                    # they might not be touching. We want to push them TOGETHER.
                    # Wait, if dist > target_dist, they are separated. Pushing together means reducing dist.
                    # Move i towards j and j towards i.
                    shift = diff / 2.0
                    trial_x[i] += dx * shift
                    trial_y[i] += dy * shift
                    trial_x[j] -= dx * shift
                    trial_y[j] -= dy * shift
                else:
                    # They are overlapping or touching. Push them apart slightly? 
                    # Or just skip? SA usually handles overlaps by rejecting.
                    # Let's add a small random perturbation to escape local min
                    trial_x[i] += random.gauss(0, step_size/2)
                    trial_y[i] += random.gauss(0, step_size/2)
                    trial_x[j] += random.gauss(0, step_size/2)
                    trial_y[j] += random.gauss(0, step_size/2)
                    
        elif move_type < 0.9:
            # Adjust radii
            i = random.randint(0, N-1)
            # Small increase or decrease
            delta_r = random.gauss(0, 0.005)
            trial_r[i] += delta_r
            if trial_r[i] < 0.01:
                trial_r[i] = 0.01
            if trial_r[i] > 0.5:
                trial_r[i] = 0.5
                
        else:
            # Adjust aspect ratio
            trial_w += random.gauss(0, 0.01)
            if trial_w < 0.5:
                trial_w = 0.5
            if trial_w > 1.5:
                trial_w = 1.5

        # Check validity of trial state
        # Use neighbor list for speed, but be aware it might be stale
        # If the move was large, neighbors might have changed.
        # For now, use the current neighbor list. We update it every 1000 steps.
        if check_validity(trial_w, trial_x, trial_y, trial_r, neighbors):
            trial_value = calculate_value(trial_x, trial_y, trial_r)
            delta_val = trial_value - best_value
            
            if delta_val > 0:
                # Accept
                w, x, y, r = trial_w, trial_x, trial_y, trial_r
                best_value = trial_value
                best_state = (w, x.copy(), y.copy(), r.copy())
            else:
                # Metropolis criterion
                if random.random() < np.exp(delta_val / T):
                    w, x, y, r = trial_w, trial_x, trial_y, trial_r
                    # Note: We don't update best_value here if it's worse, 
                    # but we move the state to explore.
                    
        iterations += 1
        
        # Update neighbors periodically
        if iterations % 500 == 0:
            centers = np.column_stack((x, y))
            neighbors = get_neighbors(centers)
            
        # Cool down
        T *= alpha
        
    # Final Polish: Try to increase radii slightly if possible
    # This is a greedy local improvement
    for _ in range(50):
        improved = False
        for i in range(N):
            # Try increasing r[i]
            old_r = r[i]
            max_possible = 1.0
            # Binary search for max r
            low = old_r
            high = min(max_possible, old_r * 1.5)
            
            # Simple linear probe
            new_r = old_r * 1.01
            if new_r > 1.0: new_r = 1.0
            
            r[i] = new_r
            if full_check_validity(w, x, y, r):
                improved = True
            else:
                r[i] = old_r
                
        if not improved:
            break
            
    # Final check and format
    if full_check_validity(w, x, y, r):
        final_value = calculate_value(x, y, r)
        if final_value > best_value:
            best_value = final_value
            best_state = (w, x.copy(), y.copy(), r.copy())
            
    bw, bx, by, br = best_state
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [bx[i], by[i], br[i]]
        
    return result

# EVOLVE-BLOCK-END
