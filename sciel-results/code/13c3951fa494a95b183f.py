import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
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

    def is_valid(colors):
        """Check if a coloring is valid (all K4s have >= 5 colors)."""
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                return False
        return True

    def local_search_smooth(max_colors, time_limit):
        """
        Local search using a smooth objective:
        Minimize total_deficit = sum(max(0, 5 - distinct_colors) for all K4s)
        """
        # Initialize with random colors
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        # k4_counts[k4_idx] is a numpy array of size max_colors
        # k4_distinct[k4_idx] is the number of distinct colors
        # k4_deficit[k4_idx] is max(0, 5 - distinct)
        
        k4_counts = []
        k4_distinct = []
        k4_deficit = []
        total_deficit = 0
        
        for e_indices in k4_edge_indices:
            counts = np.zeros(max_colors, dtype=np.int8)
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = np.count_nonzero(counts)
            deficit = 5 - distinct if distinct < 5 else 0
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            k4_deficit.append(deficit)
            total_deficit += deficit
            
        best_deficit = total_deficit
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        # Initial temp should be high enough to accept bad moves
        temp = 10.0
        min_temp = 0.01
        cool_rate = 0.999
        
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
                total_deficit += delta_deficit
                if total_deficit < best_deficit:
                    best_deficit = total_deficit
                    best_colors = colors[:]
            else:
                # Revert counts
                for k4_idx, old_d, old_def in old_states:
                    counts = k4_counts[k4_idx]
                    # We need to revert counts[old_color] and counts[new_color]
                    # But we don't have the old counts values stored, only distinct and deficit.
                    # We can reverse the operations:
                    # counts[old_color] += 1
                    # counts[new_color] -= 1
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    k4_deficit[k4_idx] = old_def
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp
                # Optional: Restart if stuck? 
                # For now, just continue with min_temp.
                
        return best_colors, best_deficit

    # Try C=13
    # We have ~110s.
    # Try 13 for 50s.
    sol, deficit = local_search_smooth(13, 50.0)
    if deficit == 0:
        return sol

    # Try C=14 for 30s
    if time.time() < deadline - 30.0:
        sol, deficit = local_search_smooth(14, 30.0)
        if deficit == 0:
            return sol

    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, deficit = local_search_smooth(15, remaining - 2.0)
        if deficit == 0:
            return sol

    # Fallback: Return the seed solution (17 colors)
    # Note: The task asks to minimize colors. If we fail to find a valid 13/14/15, 
    # we should return the best valid one found, or a known valid one.
    # The seed (i+j)%17 is valid. 
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return seed_colors
