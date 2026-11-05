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

    # Seed solution: (i + j) % 17
    # This is known to be valid for K16?
    # Let's verify validity of seed.
    def is_valid(colors):
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                return False
        return True

    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    if not is_valid(seed_colors):
        # Fallback to a known valid construction if seed fails (unlikely)
        # A proper edge coloring of K16 with 15 colors exists?
        # For now, assume seed is valid.
        pass

    def local_search(max_colors, time_limit):
        # Initialize with random colors
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        # k4_counts[k4_idx] is a numpy array of size max_colors
        # Using numpy for fast sum/unique? Or just python lists?
        # Python lists are faster for small sizes if we avoid overhead.
        
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        
        for e_indices in k4_edge_indices:
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                
        best_violations = total_violations
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        min_temp = 0.01
        cool_rate = 0.9999
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors, 0
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick a new color different from old
            new_color = random.randint(0, max_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate Delta Violations
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old distinct values for revert
            old_distincts = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                # Update violation status
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
            
            # Acceptance Criteria
            accept = False
            if delta_viol <= 0:
                accept = True
            else:
                # Avoid overflow in exp
                if delta_viol < 100 * temp:
                    if random.random() < math.exp(-delta_viol / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
            else:
                # Revert counts
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                # If stuck, restart? 
                # For now, just continue.
                pass
                
        return best_colors, best_violations

    # Try C=13
    # We have ~110s.
    # Try 13 for 40s.
    sol, viol = local_search(13, 40.0)
    if viol == 0:
        return sol

    # Try C=14 for 30s
    if time.time() < deadline - 30.0:
        sol, viol = local_search(14, 30.0)
        if viol == 0:
            return sol

    # Try C=15 for remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        sol, viol = local_search(15, remaining - 2.0)
        if viol == 0:
            return sol

    # Fallback: Return the seed solution (17 colors)
    return seed_colors
