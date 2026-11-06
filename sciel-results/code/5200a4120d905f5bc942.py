import numpy as np
import random
import time
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    num_cells = N * N
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances
    # dist_sq[i, j]
    diff = coords[:, np.newaxis, :].astype(np.int32) - coords[np.newaxis, :, :].astype(np.int32)
    dist_sq = (diff[:, :, 0]**2 + diff[:, :, 1]**2).astype(np.int32)
    max_dist = int(np.max(dist_sq))
    
    # We will use a list of dictionaries for distance counts for each cell
    # dist_counts[i] = {distance: count}
    # This is efficient for sparse data, but arrays might be faster if we know max_dist
    # Let's use arrays of size max_dist+1, initialized to 0.
    # Memory: 4096 * 8193 * 4 bytes = ~134 MB. This is acceptable.
    dist_counts = [np.zeros(max_dist + 1, dtype=np.int32) for _ in range(num_cells)]
    
    selected = [False] * num_cells
    
    def delta_add(p):
        """Calculate change in energy if we add point p (currently not selected)."""
        if selected[p]:
            return 0 # Should not happen
        
        # 1. Apex at p: pairs (i, j) in S such that d(p,i) == d(p,j)
        # Count distances from p to all selected points
        counts_p = np.zeros(max_dist + 1, dtype=np.int32)
        for i in range(num_cells):
            if selected[i]:
                d = dist_sq[p, i]
                counts_p[d] += 1
        
        # Sum C(cnt, 2)
        energy_apex_p = 0
        for c in counts_p:
            if c >= 2:
                energy_apex_p += c * (c - 1) // 2
        
        # 2. Apex at i in S: j in S such that d(p,i) == d(i,j)
        # For each i in S, the number of such j is dist_counts[i][d(p,i)]
        energy_apex_i = 0
        for i in range(num_cells):
            if selected[i]:
                d_pi = dist_sq[p, i]
                # dist_counts[i] already contains counts of OTHER selected points at each distance from i
                # Note: p is not selected yet, so it is not in dist_counts[i]
                energy_apex_i += dist_counts[i][d_pi]
        
        return energy_apex_p + energy_apex_i

    def delta_remove(p):
        """Calculate change in energy if we remove point p (currently selected)."""
        if not selected[p]:
            return 0 # Should not happen
        
        # Removing p breaks all isosceles triangles involving p.
        # Energy decreases by the number of isosceles triangles that included p.
        
        # 1. Apex at p: pairs (i, j) in S (excluding p) such that d(p,i) == d(p,j)
        # We need counts of distances from p to other selected points (excluding p itself, which is 0)
        counts_p = np.zeros(max_dist + 1, dtype=np.int32)
        for i in range(num_cells):
            if selected[i] and i != p:
                d = dist_sq[p, i]
                counts_p[d] += 1
        
        energy_apex_p = 0
        for c in counts_p:
            if c >= 2:
                energy_apex_p += c * (c - 1) // 2
        
        # 2. Apex at i in S (excluding p): j in S (excluding p) such that d(p,i) == d(i,j)
        # When p was added, it contributed to energy_apex_i by dist_counts[i][d(p,i)].
        # Wait, dist_counts[i][d(p,i)] counts ALL selected j at that distance.
        # One of those j's is p itself? No, dist_counts[i] is updated when p is added.
        # When p is added, we increment dist_counts[i][d(p,i)] for all i in S.
        # So currently, dist_counts[i][d(p,i)] includes p.
        # The number of OTHER points j at that distance is dist_counts[i][d(p,i)] - 1.
        # These are the triangles (i, j, p) with apex i that will be removed.
        
        energy_apex_i = 0
        for i in range(num_cells):
            if selected[i] and i != p:
                d_pi = dist_sq[p, i]
                # Subtract 1 because p is one of the points at this distance from i
                count_others = dist_counts[i][d_pi] - 1
                if count_others > 0:
                    energy_apex_i += count_others
        
        # Total energy removed
        return energy_apex_p + energy_apex_i

    def add_point(p):
        if selected[p]:
            return
        selected[p] = True
        # Update dist_counts for all other selected points
        for i in range(num_cells):
            if selected[i] and i != p:
                d = dist_sq[i, p]
                dist_counts[i][d] += 1
        # Note: dist_counts[p] is not updated for p itself, but we don't use dist_counts[p] 
        # in delta calculations for other points' apex logic in the same way?
        # Actually, delta_add for a future point q will use dist_counts[i] for i in S.
        # If p is in S, and we check apex at p for triangle (q, p, r), 
        # we need dist_counts[p][d(q,p)].
        # So we MUST update dist_counts[p] as well.
        # Let's update dist_counts[p] for all i in S.
        for i in range(num_cells):
            if selected[i] and i != p:
                d = dist_sq[p, i]
                dist_counts[p][d] += 1

    def remove_point(p):
        if not selected[p]:
            return
        # Update dist_counts for all other selected points
        for i in range(num_cells):
            if selected[i] and i != p:
                d = dist_sq[i, p]
                dist_counts[i][d] -= 1
        # Update dist_counts[p]
        for i in range(num_cells):
            if selected[i] and i != p:
                d = dist_sq[p, i]
                dist_counts[p][d] -= 1
        selected[p] = False

    def simulated_annealing(seed_val, time_limit):
        random.seed(seed_val)
        
        # Initialize with empty set or small random set
        selected = [False] * num_cells
        # Reset dist_counts
        for i in range(num_cells):
            dist_counts[i].fill(0)
            
        # Start with a few random points to give it a head start? 
        # Or start empty and add.
        # Let's start empty.
        
        current_energy = 0
        best_energy = float('inf')
        best_size = 0
        best_selected = [False] * num_cells
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        start_sa = time.time()
        
        while time.time() - start_sa < time_limit:
            if time.time() > deadline - 5.0:
                break
                
            # Pick random point
            p = random.randint(0, num_cells - 1)
            
            if selected[p]:
                # Try to remove
                dE = -delta_remove(p) # Energy change is negative of removed energy
                # Actually, delta_remove returns the energy that is REMOVED.
                # So new_energy = current_energy - delta_remove(p)
                # dE = -delta_remove(p)
                if dE < 0 or random.random() < np.exp(-dE / T):
                    remove_point(p)
                    current_energy += dE
                    size = sum(selected)
                    if current_energy == 0:
                        if size > best_size:
                            best_size = size
                            best_selected = selected[:]
                            best_energy = 0
            else:
                # Try to add
                dE = delta_add(p)
                if dE < 0 or random.random() < np.exp(-dE / T):
                    add_point(p)
                    current_energy += dE
                    size = sum(selected)
                    if current_energy == 0:
                        if size > best_size:
                            best_size = size
                            best_selected = selected[:]
                            best_energy = 0
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # Occasionally reheat if stuck?
            # For simplicity, just cool.
            
        # Return best found in this run
        return best_selected, best_size

    best_overall_size = 0
    best_overall_sel = []
    
    # Run multiple restarts
    n_restarts = 10
    time_per_restart = (deadline - start_time) / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Give each restart a portion of the time
        t_limit = time_per_restart - 2.0
        if t_limit < 5.0:
            t_limit = 5.0
            
        sel, size = simulated_annealing(42 + i, t_limit)
        
        if size > best_overall_size:
            best_overall_size = size
            best_overall_sel = sel
    
    # Convert indices to points
    result_points = []
    for idx in range(num_cells):
        if best_overall_sel[idx]:
            x = idx // N
            y = idx % N
            result_points.append((x, y))
            
    return result_points

