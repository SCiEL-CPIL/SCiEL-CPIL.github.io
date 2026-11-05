import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s) # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_distinct_in_k4s(colors):
        """Count distinct colors in each K4 using numpy for speed."""
        k4_distinct = np.zeros(num_k4s, dtype=np.int8)
        k4_deficit = np.zeros(num_k4s, dtype=np.int8)
        for k4_idx, e_indices in enumerate(k4_edge_indices):
            edge_colors = [colors[idx] for idx in e_indices]
            distinct = len(set(edge_colors))
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                k4_deficit[k4_idx] = 5 - distinct
        return k4_distinct, k4_deficit

    def total_deficit_from_k4s(k4_deficit):
        return int(np.sum(k4_deficit))

    def local_search_smooth(max_colors, time_limit, initial_colors=None):
        """
        Local search using a smooth objective:
        Minimize total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        """
        if initial_colors is not None:
            colors = initial_colors[:]
        else:
            colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        k4_counts = []
        k4_distinct = []
        k4_deficit = []
        total_def = 0
        
        for e_indices in k4_edge_indices:
            counts = np.zeros(max_colors, dtype=np.int8)
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = int(np.count_nonzero(counts))
            deficit = 5 - distinct if distinct < 5 else 0
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            k4_deficit.append(deficit)
            total_def += deficit
            
        best_deficit = total_def
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 20.0
        min_temp = 0.005
        cool_rate = 0.9995
        
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_def == 0:
                return colors, 0
            
            if steps_since_improve > max_steps_no_improve:
                # Restart with new random init
                colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
                k4_counts = []
                k4_distinct = []
                k4_deficit = []
                total_def = 0
                for e_indices in k4_edge_indices:
                    counts = np.zeros(max_colors, dtype=np.int8)
                    for idx in e_indices:
                        counts[colors[idx]] += 1
                    distinct = int(np.count_nonzero(counts))
                    deficit = 5 - distinct if distinct < 5 else 0
                    k4_counts.append(counts)
                    k4_distinct.append(distinct)
                    k4_deficit.append(deficit)
                    total_def += deficit
                temp = 20.0
                steps_since_improve = 0
                continue
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick a new color different from old
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Deficit
            delta_deficit = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old states for revert if needed
            old_states = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_def = k4_deficit[k4_idx]
                old_states.append((k4_idx, old_d, old_def))
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                # Update deficit
                new_def = 5 - new_d if new_d < 5 else 0
                
                k4_distinct[k4_idx] = new_d
                k4_deficit[k4_idx] = new_def
                
                delta_deficit += new_def - old_def
            
            # Acceptance Criteria
            accept = False
            if delta_deficit <= 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_deficit < 100 * temp:
                    if random.random() < math.exp(-delta_deficit / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_def += delta_deficit
                if total_def < best_deficit:
                    best_deficit = total_def
                    best_colors = colors[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                # Revert counts
                for k4_idx, old_d, old_def in old_states:
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    k4_deficit[k4_idx] = old_def
                steps_since_improve += 1
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_colors, best_deficit

    def vertex_label_search(max_colors, time_limit):
        """
        Try to find a valid coloring by assigning vertex labels and deriving edge colors
        from a function of the labels. This can find structured solutions.
        """
        # Try various functional forms for edge color from vertex labels
        # Form: color(i,j) = f(label[i], label[j]) mod max_colors
        
        best_sol = None
        best_def = float('inf')
        
        end_time = start_time + time_limit
        
        # Try different label ranges and functional forms
        label_range_options = [max_colors, max_colors + 1, max_colors + 2, max_colors * 2]
        
        for label_range in label_range_options:
            if time.time() > end_time:
                break
            
            # Try several functional forms - more diverse non-linear mappings
            functional_forms = [
                lambda a, b, m: (a + b) % m,
                lambda a, b, m: (a * b) % m,
                lambda a, b, m: (a ^ b) % m,  # XOR
                lambda a, b, m: (a + 2*b) % m,
                lambda a, b, m: (2*a + b) % m,
                lambda a, b, m: (a * 3 + b * 5) % m,
                lambda a, b, m: (a * 7 + b * 11) % m,
                lambda a, b, m: ((a - b) * 3) % m,
                lambda a, b, m: ((a + b) * 3) % m,
                # Quadratic forms
                lambda a, b, m: (a*a + b*b) % m,
                lambda a, b, m: (a*a + a*b + b*b) % m,
                lambda a, b, m: (a*a - b*b) % m,
                lambda a, b, m: (a*b + a + b) % m,
                lambda a, b, m: (a*b - a - b) % m,
                # Modular with offset
                lambda a, b, m: ((a+1)*(b+1)) % m,
                lambda a, b, m: ((a+1)*(b+2)) % m,
                lambda a, b, m: ((a+2)*(b+2)) % m,
                # Higher-degree
                lambda a, b, m: (a*a*a + b*b*b) % m,
                lambda a, b, m: (a*a*b + a*b*b) % m,
                # XOR combined with linear
                lambda a, b, m: ((a ^ b) + a + b) % m,
                lambda a, b, m: ((a ^ b) * 3 + a + b) % m,
                lambda a, b, m: ((a ^ b) + a * b) % m,
                # Product of differences
                lambda a, b, m: ((a - b) * (a + b)) % m,
                lambda a, b, m: ((a - b) * (a - b + 1)) % m,
            ]
            
            for func in functional_forms:
                if time.time() > end_time:
                    break
                
                # Try multiple random label assignments
                for attempt in range(50):
                    if time.time() > end_time:
                        break
                    
                    labels = [random.randint(0, label_range - 1) for _ in range(n)]
                    
                    # Compute edge colors
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            colors.append(func(labels[i], labels[j], max_colors))
                    
                    # Check validity
                    deficit = 0
                    for e_indices in k4_edge_indices:
                        distinct = len(set(colors[idx] for idx in e_indices))
                        if distinct < 5:
                            deficit += 5 - distinct
                    
                    if deficit < best_def:
                        best_def = deficit
                        best_sol = colors[:]
                    
                    if deficit == 0:
                        return colors, 0
                        
                    # If close, try local search refinement
                    if deficit <= 5 and time.time() < end_time - 5:
                        refined, ref_def = local_search_smooth(max_colors, min(5.0, end_time - time.time() - 1.0), initial_colors=colors)
                        if ref_def < best_def:
                            best_def = ref_def
                            best_sol = refined[:]
                        if ref_def == 0:
                            return refined, 0
                    elif deficit <= 20 and time.time() < end_time - 10:
                        # Also try vertex-label local refinement for moderate deficits
                        # Try small perturbations to vertex labels
                        for pert_attempt in range(20):
                            if time.time() > end_time - 10:
                                break
                            new_labels = labels[:]
                            # Perturb 1-2 random vertex labels
                            num_perturb = random.randint(1, 2)
                            for _ in range(num_perturb):
                                v = random.randint(0, n - 1)
                                new_labels[v] = random.randint(0, label_range - 1)
                            
                            # Compute edge colors with new labels
                            pert_colors = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    pert_colors.append(func(new_labels[i], new_labels[j], max_colors))
                            
                            # Check validity
                            pert_deficit = 0
                            for e_indices in k4_edge_indices:
                                distinct = len(set(pert_colors[idx] for idx in e_indices))
                                if distinct < 5:
                                    pert_deficit += 5 - distinct
                            
                            if pert_deficit < deficit:
                                deficit = pert_deficit
                                labels = new_labels[:]
                                colors = pert_colors[:]
                                if deficit < best_def:
                                    best_def = deficit
                                    best_sol = colors[:]
                                if deficit == 0:
                                    return colors, 0
        
        if best_sol is not None:
            return best_sol, best_def
        return None, float('inf')

    # Strategy: 
    # 1. First try vertex label search for C=13 (fast, structured)
    # 2. Then local search for C=13
    # 3. If fail, try C=14, then C=15
    
    # Try C=13 with vertex label approach first - give it more time
    if time.time() < deadline - 35:
        sol, deficit = vertex_label_search(13, min(25.0, deadline - time.time() - 35))
        if deficit == 0:
            return sol
    
    # Try C=13 with local search
    if time.time() < deadline - 25:
        sol, deficit = local_search_smooth(13, min(25.0, deadline - time.time() - 25))
        if deficit == 0:
            return sol
    
    # Try C=14
    if time.time() < deadline - 15:
        sol, deficit = vertex_label_search(14, min(12.0, deadline - time.time() - 15))
        if deficit == 0:
            return sol
    
    if time.time() < deadline - 8:
        sol, deficit = local_search_smooth(14, min(12.0, deadline - time.time() - 8))
        if deficit == 0:
            return sol
    
    # Try C=15
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, deficit = local_search_smooth(15, remaining - 2.0)
        if deficit == 0:
            return sol
    
    # Fallback: Return the seed solution (17 colors)
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
