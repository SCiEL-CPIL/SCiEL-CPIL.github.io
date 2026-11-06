import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Try n from 34 downwards
    for n in range(34, 1, -1):
        if time.time() > deadline - 2.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Map each edge to the quads and pents it belongs to
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        n_quads = len(quads)
        n_pents = len(pents)
        
        # For each edge, which quads contain it?
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            for e in combinations(q, 3):
                edge_to_quads[edge_to_idx[e]].append(q_idx)
                
        # For each edge, which pents contain it?
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            for e in combinations(p, 3):
                edge_to_pents[edge_to_idx[e]].append(p_idx)

        # Convert to numpy arrays for faster iteration? 
        # Actually, lists of lists are fine, but let's try to optimize the inner loop.
        # We can precompute the list of quads/pents for each edge as numpy arrays?
        # No, standard lists are okay. The bottleneck is the Python loop.
        
        best_violations = float('inf')
        best_colors = None
        
        # Run local search for the remaining time
        search_deadline = deadline
        
        restart_count = 0
        max_restarts = 10000 # Will be limited by time
        
        while time.time() < search_deadline:
            restart_count += 1
            
            # Initialization strategies:
            # 1. Random
            # 2. Algebraic (sum % 2)
            # 3. Algebraic (sum % 3)
            # 4. Hybrid
            
            strategy = restart_count % 4
            if strategy == 0:
                colors = np.random.randint(0, 2, m).astype(np.int8)
            elif strategy == 1:
                # Color red (0) if sum % 2 == 0, else blue (1)
                # Note: In our convention, 0 is Red, 1 is Blue.
                # Red K4 forbidden, Blue K5 forbidden.
                # Let's try sum % 2 == 0 -> Red (0)
                sums = np.sum(np.array(edges), axis=1)
                colors = (sums % 2).astype(np.int8)
            elif strategy == 2:
                # Color red (0) if sum % 3 == 0, else blue (1)
                sums = np.sum(np.array(edges), axis=1)
                colors = (sums % 3 == 0).astype(np.int8)
            else:
                # Random with bias
                colors = (np.random.rand(m) < 0.5).astype(np.int8)
            
            # Calculate initial violations
            # We need to count red edges in each quad and blue edges in each pent.
            # Red is 0, Blue is 1.
            
            quad_red_counts = np.zeros(n_quads, dtype=np.int32)
            pent_blue_counts = np.zeros(n_pents, dtype=np.int32)
            
            # This loop is still in Python. Can we vectorize?
            # For each edge i, if color[i]==0, increment quad_red_counts[edge_to_quads[i]]
            # If color[i]==1, increment pent_blue_counts[edge_to_pents[i]]
            
            # Let's try to make this faster by iterating over edges and updating counts.
            # It's O(m * avg_degree).
            
            for i in range(m):
                if colors[i] == 0:
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else:
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = np.sum(quad_red_counts == 4) + np.sum(pent_blue_counts == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
                
            # Local search loop
            # We want to minimize violations.
            # A move is flipping an edge i.
            # If we flip Red->Blue:
            #   - Red counts in quads containing i decrease by 1.
            #   - Blue counts in pents containing i increase by 1.
            #   - Violation change:
            #     - For quads: if count was 4, it becomes 3 (violation removed, -1).
            #     - For pents: if count was 9, it becomes 10 (violation added, +1).
            # If we flip Blue->Red:
            #   - Red counts in quads containing i increase by 1.
            #   - Blue counts in pents containing i decrease by 1.
            #   - Violation change:
            #     - For quads: if count was 3, it becomes 4 (violation added, +1).
            #     - For pents: if count was 10, it becomes 9 (violation removed, -1).
            
            local_deadline = min(search_deadline, time.time() + 2.0)
            no_improve_count = 0
            max_no_improve = 5000
            
            while time.time() < local_deadline:
                if no_improve_count > max_no_improve:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    # Quads: red count decreases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1
                    # Pents: blue count increases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1
                else: # Blue -> Red
                    # Quads: red count increases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1
                    # Pents: blue count decreases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1
                
                # Accept if improving or neutral (with some probability)
                # Simple hill climbing: accept if delta_v < 0
                # To escape local minima, accept delta_v == 0 with some prob, or even delta_v > 0 rarely.
                # Let's stick to delta_v <= 0 for stability, but add a bit of randomness.
                if delta_v < 0 or (delta_v == 0 and random.random() < 0.1):
                    # Apply move
                    colors[i] = new_c
                    if new_c == 0: # Now Red
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    no_improve_count = 0
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                else:
                    no_improve_count += 1

    # If no valid coloring found for n > 1, return n=1
    return {'n': 1, 'colors': []}
