# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using Simulated Annealing.
    
    Instead of greedily adding points to the excluded set S, we start with a 
    valid baseline and use SA to swap points between S and N. The objective 
    is to maximize the size of S (minimize N) while maintaining the Nikodym 
    property. We use a soft constraint approach where the "energy" is the 
    number of violated constraints, allowing the search to temporarily 
    accept invalid states to escape local minima.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Deadline guard: 630s limit, use 600s for safety
    start_time = time.time()
    deadline = start_time + 600
    
    # Field operations
    # Element (a, b) represents a + b*alpha, alpha^2 = 3
    k_const = 3  # 3 is a non-residue mod 7: 3^3 = 27 = 6 = -1 mod 7
    
    def field_mul(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        a_res = (a1 * a2 + b1 * b2 * k_const) % p
        b_res = (a1 * b2 + a2 * b1) % p
        return a_res * p + b_res

    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # Total 50 directions
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    num_dirs = len(directions)
    
    # Precompute line neighbors for each point and direction
    # line_neighbors[point_idx][dir_idx] = array of 48 other points on the line
    # We store these as a flat array of shape (total_points, num_dirs, q-1)
    # 2401 * 50 * 48 * 4 bytes = ~23 MB. Acceptable.
    
    line_neighbors = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            
            for d_idx, (dx_idx, dy_idx) in enumerate(directions):
                dx_a, dx_b = divmod(dx_idx, p)
                dy_a, dy_b = divmod(dy_idx, p)
                
                count = 0
                for t_idx in range(1, q): # t != 0
                    t_a, t_b = divmod(t_idx, p)
                    
                    # t * dx
                    tx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                    tx_b = (t_a * dx_b + t_b * dx_a) % p
                    
                    # t * dy
                    ty_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    ty_b = (t_a * dy_b + t_b * dx_a) % p # Note: dy_b * dx_a? No, t*dy = (t_a*dy_a + t_b*dy_b*k, t_a*dy_b + t_b*dy_a)
                    # Correction:
                    ty_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    ty_b = (t_a * dy_b + t_b * dy_a) % p
                    
                    new_x_a = (x_a + tx_a) % p
                    new_x_b = (x_b + tx_b) % p
                    new_y_a = (y_a + ty_a) % p
                    new_y_b = (y_b + ty_b) % p
                    
                    new_x_idx = new_x_a * p + new_x_b
                    new_y_idx = new_y_a * p + new_y_b
                    new_point_idx = new_x_idx * q + new_y_idx
                    
                    line_neighbors[point_idx, d_idx, count] = new_point_idx
                    count += 1
    
    # Initialize S (excluded set)
    # Baseline: Use the unital construction from the seed, or start empty and grow?
    # Starting empty is too slow for SA to find a large set. 
    # Let's start with the seed's baseline S.
    
    S = np.zeros(total_points, dtype=bool)
    
    # Seed baseline: Unital points where y is NOT in y_special
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            in_unital = (2 * x_a + y_a * y_a - y_b * y_b * k_const) % p == 0
            in_y_special = (y_a, y_b) in y_special
            if in_unital and not in_y_special:
                point_idx = x_idx * q + y_idx
                S[point_idx] = True
                
    # Precompute counts of S-neighbors for each point and direction
    # count_s[point_idx, d_idx] = number of neighbors in S on the line
    # This is O(1) to update when S changes (if we track changes carefully)
    # But for SA, we might just recalculate or update incrementally.
    # Let's use an array of counts.
    counts = np.zeros((total_points, num_dirs), dtype=np.int32)
    
    # Initialize counts
    for point_idx in range(total_points):
        for d_idx in range(num_dirs):
            neighbors = line_neighbors[point_idx, d_idx]
            counts[point_idx, d_idx] = np.sum(S[neighbors])
    
    # Function to calculate energy (number of violations)
    # A point x is "satisfied" in direction d if counts[x, d] > 0
    # Energy = number of (x, d) pairs where counts[x, d] == 0
    # We want to minimize energy (ideally 0) while maximizing sum(S)
    
    def get_energy(counts):
        # Number of zeros in counts
        return np.sum(counts == 0)
    
    current_energy = get_energy(counts)
    current_s_size = np.sum(S)
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iterations = 100000
    
    best_s_size = current_s_size
    best_S = S.copy()
    best_energy = current_energy
    
    # List of indices in S and not in S for fast swapping
    indices_in_S = np.where(S)[0]
    indices_not_in_S = np.where(~S)[0]
    
    # Precompute random permutation or just use random indices
    
    # Helper to update counts when a point flips
    def update_counts_on_flip(point_idx, added_to_S):
        # If point_idx is added to S, it contributes +1 to the count of all its neighbors
        # If removed, -1.
        # Neighbors of point_idx in direction d are line_neighbors[point_idx, d]
        # But wait: the counts array is counts[x, d] = # of neighbors of x in S.
        # If we add point P to S, it affects counts[x, d] for all x such that P is a neighbor of x in direction d.
        # P is a neighbor of x in direction d if x is a neighbor of P in direction d.
        # Lines are symmetric. So we need to update counts for all neighbors of P.
        
        delta = 1 if added_to_S else -1
        
        for d_idx in range(num_dirs):
            neighbors = line_neighbors[point_idx, d_idx]
            counts[neighbors, d_idx] += delta
            
            # Also, the point P itself has a count in direction d.
            # The count for P in direction d does NOT change when P flips, 
            # because the count excludes P itself.
            # So we only update neighbors.
            
    # Main SA Loop
    iteration = 0
    while iteration < max_iterations and time.time() < deadline - 10:
        # Choose a random point to flip
        # We can bias towards flipping points that are "bad" (contributing to energy)
        # Or just random.
        
        # Randomly pick a point from S to remove OR from not-S to add
        if random.random() < 0.5:
            # Remove from S
            if len(indices_in_S) == 0: continue
            pos = random.randint(0, len(indices_in_S) - 1)
            point_idx = indices_in_S[pos]
            indices_in_S[pos] = indices_in_S[-1]
            indices_in_S = indices_in_S[:-1]
            indices_not_in_S = np.append(indices_not_in_S, point_idx)
            
            # Flip
            S[point_idx] = False
            update_counts_on_flip(point_idx, False)
            
            # Calculate delta energy
            # Old energy contribution of point_idx and its neighbors?
            # Energy = sum(counts == 0).
            # We need to recalculate or track changes.
            # Recalculating full energy is O(2401*50) = 120k ops. Too slow for 100k iters?
            # 120k * 100k = 12 billion ops. Too slow.
            
            # We need incremental energy update.
            # Let's store energy as a variable.
            # When we flip point P, which counts change?
            # counts[neighbors_of_P, d] change.
            # Also, does counts[P, d] change? No.
            # So only the counts of the neighbors change.
            # For each neighbor x of P, and each direction d, counts[x, d] changes by delta.
            # We need to know if counts[x, d] was 0 before and is now >0 (energy -1) or was >0 and is now 0 (energy +1).
            
            # Let's revert the update logic to compute delta first, then apply.
            
            # This is getting complex to implement incrementally correctly inside the loop without precomputing more.
            # Given the time constraint, let's try a simpler heuristic: 
            # Greedy local search with random restarts?
            # Or just accept the slowness if we reduce iterations?
            # 1000 iterations of full energy check: 120M ops. ~1-2 seconds.
            # 10000 iterations: ~20 seconds.
            # We have 600 seconds. So we can do ~20,000-50,000 full energy checks.
            
            # Let's do that: Full energy check each step.
            
            new_energy = get_energy(counts)
            
            # Acceptance criterion
            delta_E = new_energy - current_energy
            
            if delta_E < 0:
                # Better energy
                accept = True
            else:
                # Worse energy
                # Accept with probability exp(-delta_E / T)
                if T > 0:
                    accept = random.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if not accept:
                # Revert
                S[point_idx] = True
                update_counts_on_flip(point_idx, True)
                current_energy = get_energy(counts) # Restore state (should be same)
                # Re-add to indices
                indices_in_S = np.append(indices_in_S, point_idx)
                indices_not_in_S = indices_not_in_S[indices_not_in_S != point_idx]
            else:
                current_energy = new_energy
                current_s_size = np.sum(S)
                if current_s_size > best_s_size and current_energy == 0:
                    best_s_size = current_s_size
                    best_S = S.copy()
                    best_energy = 0
                elif current_s_size > best_s_size:
                    # Track best even if invalid? No, we need valid result.
                    # But maybe an invalid state with high size can lead to valid later.
                    # We only save best if valid (energy 0).
                    pass
                    
        else:
            # Add to S
            if len(indices_not_in_S) == 0: continue
            pos = random.randint(0, len(indices_not_in_S) - 1)
            point_idx = indices_not_in_S[pos]
            indices_not_in_S[pos] = indices_not_in_S[-1]
            indices_not_in_S = indices_not_in_S[:-1]
            indices_in_S = np.append(indices_in_S, point_idx)
            
            # Flip
            S[point_idx] = True
            update_counts_on_flip(point_idx, True)
            
            new_energy = get_energy(counts)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0:
                accept = True
            else:
                if T > 0:
                    accept = random.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if not accept:
                # Revert
                S[point_idx] = False
                update_counts_on_flip(point_idx, False)
                current_energy = get_energy(counts)
                indices_not_in_S = np.append(indices_not_in_S, point_idx)
                indices_in_S = indices_in_S[indices_in_S != point_idx]
            else:
                current_energy = new_energy
                current_s_size = np.sum(S)
                if current_s_size > best_s_size and current_energy == 0:
                    best_s_size = current_s_size
                    best_S = S.copy()
                    best_energy = 0
                    
        # Cooling
        T *= cooling
        iteration += 1
        
        # Periodic restart if stuck?
        if iteration % 5000 == 0 and current_energy > 0:
            # Try to repair current state greedily
            # Or just continue.
            pass

    # Final Result: Use best_S found
    # If best_S is empty (shouldn't be, baseline is valid), use baseline
    if best_s_size == 0:
        best_S = S.copy() # Fallback to current if no improvement recorded? 
        # Actually, if we never found a valid improvement, we should return the initial valid one.
        # Let's ensure we have a valid set.
        # The initial S was valid. So best_S should at least be the initial S if we initialized best_S correctly.
        # I initialized best_S = S.copy() at start. So it's the initial valid set.
        
    # Construct output N = complement of best_S
    points = []
    for point_idx in range(total_points):
        if not best_S[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    if not points:
        # Fallback: return all points
        for point_idx in range(total_points):
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
