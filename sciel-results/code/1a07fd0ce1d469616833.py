import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin for safety (evaluator kills at 150s)
    
    n = 16
    # Precompute all K4s and their edges
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Map edge (i, j) with i < j to index 0..119
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i, j)] = idx
            idx += 1
            
    # For each K4, store the list of edge indices
    k4_edges = []
    for combo in k4s:
        edges = []
        for u, v in itertools.combinations(combo, 2):
            edges.append(edge_map[(u, v)])
        k4_edges.append(edges)
        
    # For each edge, store the list of K4 indices that contain it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Precompute for each edge, the other 5 edges in each K4 it belongs to
    # This helps in fast distinct color counting
    # For edge e in K4 k, the other edges are k4_edges[k] excluding e
    # But we can just recompute distinct colors from the 6 edges.
    # To speed up, let's maintain color counts per K4? 
    # No, distinct count is just len(set).
    
    # Helper to count distinct colors in a K4 given the colors array
    def count_distinct(k4_idx, colors):
        # colors is a list or numpy array of length 120
        edges = k4_edges[k4_idx]
        # Unpack for speed?
        c0 = colors[edges[0]]
        c1 = colors[edges[1]]
        c2 = colors[edges[2]]
        c3 = colors[edges[3]]
        c4 = colors[edges[4]]
        c5 = colors[edges[5]]
        # Use a set or manual check?
        # Manual check for 6 elements is fast
        if c0 == c1:
            if c0 == c2:
                if c0 == c3:
                    if c0 == c4:
                        if c0 == c5:
                            return 1
                        return 2
                    if c0 == c5:
                        return 2
                    return 3
                if c0 == c4:
                    if c0 == c5:
                        return 2
                    return 3
                if c0 == c5:
                    return 3
                return 4
            if c0 == c3:
                if c0 == c4:
                    if c0 == c5:
                        return 2
                    return 3
                if c0 == c5:
                    return 3
                return 4
            if c0 == c4:
                if c0 == c5:
                    return 3
                return 4
            if c0 == c5:
                return 4
            return 5
        # c0 != c1
        if c2 == c0:
            if c3 == c0:
                if c4 == c0:
                    if c5 == c0:
                        return 2
                    if c5 == c1:
                        return 3
                    return 4
                if c4 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    return 4
                if c4 == c5:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    return 4
                return 5
            if c3 == c1:
                if c4 == c0:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    return 4
                if c4 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    return 4
                if c4 == c5:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    return 4
                return 5
            if c3 == c4:
                if c3 == c0:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c3:
                        return 3
                    return 4
                if c3 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c3:
                        return 3
                    return 4
                if c5 == c0:
                    return 4
                if c5 == c1:
                    return 4
                if c5 == c3:
                    return 4
                return 5
            if c3 == c5:
                if c3 == c0:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    if c4 == c3:
                        return 3
                    return 4
                if c3 == c1:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    if c4 == c3:
                        return 3
                    return 4
                if c4 == c0:
                    return 4
                if c4 == c1:
                    return 4
                if c4 == c3:
                    return 4
                return 5
            if c4 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c5:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                    # c4 != c0 and c4 != c1
                return 4
            if c5 == c0:
                return 4
            if c5 == c1:
                return 4
            return 5
        # c0 != c1, c2 != c0
        if c2 == c1:
            if c3 == c0:
                if c4 == c0:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c4:
                        return 3
                    return 4
                if c4 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c4:
                        return 3
                    return 4
                if c4 == c5:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    return 4
                return 5
            if c3 == c1:
                if c4 == c0:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c4:
                        return 3
                    return 4
                if c4 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c4:
                        return 3
                    return 4
                if c4 == c5:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    return 4
                return 5
            if c3 == c4:
                if c3 == c0:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c3:
                        return 3
                    return 4
                if c3 == c1:
                    if c5 == c0:
                        return 3
                    if c5 == c1:
                        return 3
                    if c5 == c3:
                        return 3
                    return 4
                if c5 == c0:
                    return 4
                if c5 == c1:
                    return 4
                if c5 == c3:
                    return 4
                return 5
            if c3 == c5:
                if c3 == c0:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    if c4 == c3:
                        return 3
                    return 4
                if c3 == c1:
                    if c4 == c0:
                        return 3
                    if c4 == c1:
                        return 3
                    if c4 == c3:
                        return 3
                    return 4
                if c4 == c0:
                    return 4
                if c4 == c1:
                    return 4
                if c4 == c3:
                    return 4
                return 5
            if c4 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c5:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                return 4
            if c5 == c0:
                return 4
            if c5 == c1:
                return 4
            return 5
        # c0 != c1, c2 != c0, c2 != c1
        # c2 is distinct from c0, c1
        if c3 == c0:
            if c4 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c2:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c5:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                return 4
            return 5
        if c3 == c1:
            if c4 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c2:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c5:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                return 4
            return 5
        if c3 == c2:
            if c4 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c2:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c4:
                    return 3
                return 4
            if c4 == c5:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                return 4
            return 5
        if c3 == c4:
            if c3 == c0:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c3:
                    return 3
                return 4
            if c3 == c1:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c3:
                    return 3
                return 4
            if c3 == c2:
                if c5 == c0:
                    return 3
                if c5 == c1:
                    return 3
                if c5 == c2:
                    return 3
                if c5 == c3:
                    return 3
                return 4
            if c5 == c0:
                return 4
            if c5 == c1:
                return 4
            if c5 == c2:
                return 4
            if c5 == c3:
                return 4
            return 5
        if c3 == c5:
            if c3 == c0:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                if c4 == c3:
                    return 3
                return 4
            if c3 == c1:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                if c4 == c3:
                    return 3
                return 4
            if c3 == c2:
                if c4 == c0:
                    return 3
                if c4 == c1:
                    return 3
                if c4 == c2:
                    return 3
                if c4 == c3:
                    return 3
                return 4
            if c4 == c0:
                return 4
            if c4 == c1:
                return 4
            if c4 == c2:
                return 4
            if c4 == c3:
                return 4
            return 5
        if c4 == c0:
            if c5 == c0:
                return 3
            if c5 == c1:
                return 3
            if c5 == c2:
                return 3
            if c5 == c3:
                return 3
            if c5 == c4:
                return 3
            return 4
        if c4 == c1:
            if c5 == c0:
                return 3
            if c5 == c1:
                return 3
            if c5 == c2:
                return 3
            if c5 == c3:
                return 3
            if c5 == c4:
                return 3
            return 4
        if c4 == c2:
            if c5 == c0:
                return 3
            if c5 == c1:
                return 3
            if c5 == c2:
                return 3
            if c5 == c3:
                return 3
            if c5 == c4:
                return 3
            return 4
        if c4 == c3:
            if c5 == c0:
                return 3
            if c5 == c1:
                return 3
            if c5 == c2:
                return 3
            if c5 == c3:
                return 3
            if c5 == c4:
                return 3
            return 4
        if c5 == c0:
            return 4
        if c5 == c1:
            return 4
        if c5 == c2:
            return 4
        if c5 == c3:
            return 4
        if c5 == c4:
            return 4
        return 5

    # This manual function is too complex and error-prone. Let's use a simpler, faster set-based approach
    # but optimized with numpy if possible, or just standard set which is C-implemented and fast enough.
    # 1820 K4s * 6 edges = 10920 lookups.
    # Local search step: 91 K4s * 6 edges = 546 lookups.
    # This is very fast in Python.
    
    def count_distinct_fast(k4_idx, colors):
        edges = k4_edges[k4_idx]
        # Using a set is generally fast enough for 6 elements
        return len({colors[e] for e in edges})
    
    # Let's stick with the set version for correctness and reasonable speed.
    # The previous manual version was a mistake.
    
    best_colors = None
    best_num_colors = 17
    
    # Try to find a solution with K colors
    # We will use a simulated annealing or hill climbing approach.
    # Objective: Minimize number of violations (K4s with < 5 colors).
    # Secondary: Minimize number of colors used.
    
    for K in [14, 13, 15, 16]:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for this K
        found_valid = False
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize random coloring
            colors = np.random.randint(0, K, 120)
            
            # Calculate initial violations
            distinct_counts = np.array([count_distinct_fast(i, colors) for i in range(num_k4s)])
            current_viol = np.sum(distinct_counts < 5)
            
            if current_viol == 0:
                best_colors = colors.tolist()
                best_num_colors = K
                found_valid = True
                break
                
            # Local Search / Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            steps = 0
            max_steps = 50000
            
            # Precompute list of invalid K4s for fast selection?
            # We'll just pick a random K4 and check if it's invalid, or maintain a list.
            # Maintaining a list is better.
            invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
            
            while steps < max_steps and T > T_min:
                if time.time() > deadline - 10:
                    break
                    
                if not invalid_k4s:
                    break # Found valid
                    
                # Pick random invalid K4
                k4_idx = random.choice(invalid_k4s)
                edges = k4_edges[k4_idx]
                e_idx = random.choice(edges)
                
                old_color = colors[e_idx]
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                
                # Update distinct counts for affected K4s
                affected_k4s = edge_to_k4s[e_idx]
                old_distincts = {}
                delta_viol = 0
                
                for k4 in affected_k4s:
                    old_dist = distinct_counts[k4]
                    new_dist = count_distinct_fast(k4, colors)
                    old_distincts[k4] = old_dist
                    
                    old_inv = old_dist < 5
                    new_inv = new_dist < 5
                    
                    if old_inv and not new_inv:
                        delta_viol -= 1
                    elif not old_inv and new_inv:
                        delta_viol += 1
                        
                    distinct_counts[k4] = new_dist
                    
                new_viol = current_viol + delta_viol
                
                if new_viol < current_viol:
                    current_viol = new_viol
                    # Update invalid_k4s list
                    # Remove those that became valid, add those that became invalid
                    # This is tricky to maintain efficiently.
                    # Let's just rebuild it occasionally or accept the O(N) cost if we do it often.
                    # Given N=1820, rebuilding is fast.
                    if delta_viol != 0:
                        invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
                    steps += 1
                else:
                    # Metropolis acceptance
                    if new_viol == current_viol:
                        accept = True
                    else:
                        # new_viol > current_viol
                        accept = random.random() < np.exp(-(new_viol - current_viol) / T)
                        
                    if accept:
                        current_viol = new_viol
                        if delta_viol != 0:
                            invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
                        steps += 1
                    else:
                        # Revert
                        colors[e_idx] = old_color
                        for k4 in affected_k4s:
                            distinct_counts[k4] = old_distincts[k4]
                        steps += 1
                        
                T *= cooling
                
            if current_viol == 0:
                best_colors = colors.tolist()
                best_num_colors = K
                found_valid = True
                break
                
        if found_valid:
            # If we found a valid coloring with K, we should try smaller K if we haven't yet.
            # The loop order is [14, 13, 15, 16].
            # If K=14 works, we break this inner loop and the outer loop continues to K=13.
            # If K=13 works, we update best and break.
            # If K=14 fails, we try 13. If 13 fails, we try 15.
            # This logic is slightly flawed if we want the absolute minimum.
            # We should try 13 first, then 14, then 15.
            # Let's reorder the K loop.
            pass

    # Reorder K loop to try smaller K first
    best_colors = None
    best_num_colors = 17
    
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 15:
            break
            
        found_valid = False
        max_restarts = 3
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            colors = np.random.randint(0, K, 120)
            distinct_counts = np.array([count_distinct_fast(i, colors) for i in range(num_k4s)])
            current_viol = np.sum(distinct_counts < 5)
            
            if current_viol == 0:
                best_colors = colors.tolist()
                best_num_colors = K
                found_valid = True
                break
                
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            steps = 0
            max_steps = 20000
            invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
            
            while steps < max_steps and T > T_min:
                if time.time() > deadline - 15:
                    break
                    
                if not invalid_k4s:
                    break
                    
                k4_idx = random.choice(invalid_k4s)
                edges = k4_edges[k4_idx]
                e_idx = random.choice(edges)
                
                old_color = colors[e_idx]
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                
                affected_k4s = edge_to_k4s[e_idx]
                old_distincts = {}
                delta_viol = 0
                
                for k4 in affected_k4s:
                    old_dist = distinct_counts[k4]
                    new_dist = count_distinct_fast(k4, colors)
                    old_distincts[k4] = old_dist
                    
                    old_inv = old_dist < 5
                    new_inv = new_dist < 5
                    
                    if old_inv and not new_inv:
                        delta_viol -= 1
                    elif not old_inv and new_inv:
                        delta_viol += 1
                        
                    distinct_counts[k4] = new_dist
                    
                new_viol = current_viol + delta_viol
                
                if new_viol < current_viol:
                    current_viol = new_viol
                    if delta_viol != 0:
                        invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
                    steps += 1
                else:
                    if new_viol == current_viol:
                        accept = True
                    else:
                        accept = random.random() < np.exp(-(new_viol - current_viol) / T)
                        
                    if accept:
                        current_viol = new_viol
                        if delta_viol != 0:
                            invalid_k4s = [i for i in range(num_k4s) if distinct_counts[i] < 5]
                        steps += 1
                    else:
                        colors[e_idx] = old_color
                        for k4 in affected_k4s:
                            distinct_counts[k4] = old_distincts[k4]
                        steps += 1
                        
                T *= cooling
                
            if current_viol == 0:
                best_colors = colors.tolist()
                best_num_colors = K
                found_valid = True
                break
                
        if found_valid:
            break # Found the smallest K that works, so we can stop
            
    # Fallback
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_num_colors = 17
        
    return best_colors
