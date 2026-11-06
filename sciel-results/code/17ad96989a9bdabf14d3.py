import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try to find a solution for n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n == 34:
            max_time_for_n = 45
        elif n == 33:
            max_time_for_n = 20
        elif n == 32:
            max_time_for_n = 10
        else:
            max_time_for_n = 5
            
        found = False
        start_n_time = time.time()
        
        while time.time() - start_n_time < max_time_for_n:
            if time.time() > deadline - 5:
                break
                
            # Each attempt gets a slice of the remaining time for this n
            time_limit = 5.0
            if time.time() + time_limit > deadline - 5:
                 time_limit = max(1.0, deadline - 5 - time.time())
                 
            result = solve_fast(n, time_limit)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        # Fallback to a known small solution
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}

def solve_fast(n, time_limit=5):
    """
    Optimized Simulated Annealing using count-based updates.
    """
    start_time = time.time()
    
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Map edge to list of K4 indices and K5 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    # Store edges for each clique to initialize counts
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        es = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(es)
        for e in es:
            edge_to_k5s[e].append(len(k5_edges) - 1)
            
    # Use biased probability for initial coloring (slightly favor red to avoid blue K5s)
    p_red = 0.6  # Probability of red (0)
    colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    
    # Calculate initial counts
    red_counts = [0] * len(k4s)
    blue_counts = [0] * len(k5s)
    
    for i, e_list in enumerate(k4_edges):
        for e in e_list:
            if colors[e] == 0:
                red_counts[i] += 1
                
    for i, e_list in enumerate(k5_edges):
        for e in e_list:
            if colors[e] == 1:
                blue_counts[i] += 1
                
    total_violations = 0
    for rc in red_counts:
        if rc == 4:
            total_violations += 1
    for bc in blue_counts:
        if bc == 10:
            total_violations += 1
            
    if total_violations == 0:
        return colors.copy()
        
    # Faster cooling schedule
    T = 0.5
    cooling = 0.99
    best_colors = colors.copy()
    best_violations = total_violations
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Update K4s
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                # Red -> Blue
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3: # Was 4, now 3 (violation removed)
                    delta_v -= 1
            else:
                # Blue -> Red
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4: # Was 3, now 4 (violation added)
                    delta_v += 1
                    
        # Update K5s
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                # Blue -> Red
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9: # Was 10, now 9 (violation removed)
                    delta_v -= 1
            else:
                # Red -> Blue
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10: # Was 9, now 10 (violation added)
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        # Acceptance with exponential decay
        if delta_v <= 0 or random.random() < math.exp(-delta_v / T):
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return colors.copy()
                
        else:
            # Revert counts
            for k4_idx in edge_to_k4s[e_idx]:
                if old_color == 0:
                    red_counts[k4_idx] += 1
                else:
                    red_counts[k4_idx] -= 1
            for k5_idx in edge_to_k5s[e_idx]:
                if old_color == 1:
                    blue_counts[k5_idx] += 1
                else:
                    blue_counts[k5_idx] -= 1
                    
        T *= cooling
        
    if best_violations == 0:
        return best_colors
    return None
