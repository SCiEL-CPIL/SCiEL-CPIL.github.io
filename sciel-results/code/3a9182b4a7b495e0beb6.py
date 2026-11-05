import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer before 120s kill
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)  # 1820
    
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
    
    def compute_total_deficit(colors, max_colors):
        """Compute total deficit = sum(max(0, 5 - distinct_colors) for all K4s)"""
        total = 0
        for e_indices in k4_edge_indices:
            distinct = len(set(colors[idx] for idx in e_indices))
            if distinct < 5:
                total += 5 - distinct
        return total
    
    def local_search(max_colors, time_limit, seed=None):
        """
        Local search using smooth objective:
        Minimize total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        """
        if seed is not None:
            random.seed(seed)
        
        # Initialize with random colors
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        # k4_counts[k4_idx] is a numpy array of size max_colors
        k4_counts = np.zeros((num_k4s, max_colors), dtype=np.int8)
        k4_distinct = np.zeros(num_k4s, dtype=np.int8)
        
        for k4_idx, e_indices in enumerate(k4_edge_indices):
            for idx in e_indices:
                k4_counts[k4_idx, colors[idx]] += 1
            k4_distinct[k4_idx] = np.count_nonzero(k4_counts[k4_idx])
        
        total_deficit = int(np.sum(np.maximum(0, 5 - k4_distinct)))
        best_deficit = total_deficit
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        min_temp = 0.05
        cool_rate = 0.9995
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_deficit == 0:
                return colors, 0
            
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
                old_d = int(k4_distinct[k4_idx])
                old_def = max(0, 5 - old_d)
                old_states.append((k4_idx, old_d, old_def))
                
                # Update counts
                k4_counts[k4_idx, old_color] -= 1
                k4_counts[k4_idx, new_color] += 1
                
                # Update distinct
                new_d = old_d
                if k4_counts[k4_idx, old_color] == 0:
                    new_d -= 1
                if k4_counts[k4_idx, new_color] == 1:  # It was 0 before
                    new_d += 1
                
                # Update deficit
                new_def = max(0, 5 - new_d)
                
                k4_distinct[k4_idx] = new_d
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
                total_deficit += delta_deficit
                if total_deficit < best_deficit:
                    best_deficit = total_deficit
                    best_colors = colors[:]
            else:
                # Revert counts
                for k4_idx, old_d, old_def in old_states:
                    k4_counts[k4_idx, old_color] += 1
                    k4_counts[k4_idx, new_color] -= 1
                    k4_distinct[k4_idx] = old_d
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
                
        return best_colors, best_deficit
    
    # Try C=13 first (most valuable)
    # We have ~115s. Try 13 for 40s with multiple restarts.
    best_sol = None
    best_deficit = float('inf')
    
    # Try C=13 with multiple restarts
    for attempt in range(3):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_per_attempt = min(15.0, remaining - 5.0)
        if time_per_attempt < 5:
            break
        sol, deficit = local_search(13, time_per_attempt, seed=42 + attempt * 100)
        if deficit < best_deficit:
            best_deficit = deficit
            best_sol = sol
        if deficit == 0:
            return sol
    
    # Try C=14
    if time.time() < deadline - 20.0:
        remaining = deadline - time.time()
        for attempt in range(2):
            if time.time() < deadline - 10.0:
                time_per_attempt = min(10.0, deadline - time.time() - 5.0)
                if time_per_attempt < 3:
                    break
                sol, deficit = local_search(14, time_per_attempt, seed=1000 + attempt * 100)
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_sol = sol
                if deficit == 0:
                    return sol
    
    # Try C=15
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, deficit = local_search(15, remaining - 2.0, seed=2000)
        if deficit < best_deficit:
            best_deficit = deficit
            best_sol = sol
        if deficit == 0:
            return sol
    
    # Fallback: Return the seed solution (17 colors)
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
