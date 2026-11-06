import time
import random
import math
from itertools import combinations

# Cache for structural data
_structure_cache = {}

def get_structure(n):
    if n in _structure_cache:
        return _structure_cache[n]
        
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
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
            
    _structure_cache[n] = {
        'num_edges': num_edges,
        'edge_to_k4s': edge_to_k4s,
        'edge_to_k5s': edge_to_k5s,
        'k4_edges': k4_edges,
        'k5_edges': k5_edges,
        'num_k4s': len(k4s),
        'num_k5s': len(k5s)
    }
    return _structure_cache[n]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Priority is n=34. If we can't find it, we try 33, etc.
    # But we should spend most time on 34.
    
    # Try n=34 with multiple restarts
    time_remaining = deadline - 5 - time.time()
    if time_remaining > 0:
        # Allocate 80% of time to n=34
        time_for_34 = time_remaining * 0.8
        result = solve_fast(34, time_for_34, greedy_phase=True)
        if result is not None:
            return {'n': 34, 'colors': result}
            
    # If 34 failed, try 33 with remaining time
    time_remaining = deadline - 5 - time.time()
    if time_remaining > 0:
        result = solve_fast(33, time_remaining, greedy_phase=True)
        if result is not None:
            return {'n': 33, 'colors': result}
            
    # Fallback
    # ... (keep existing fallback logic)
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

def solve_fast(n, time_limit=5, greedy_phase=False):
    """
    Optimized Simulated Annealing with optional Greedy Phase.
    """
    start_time = time.time()
    
    struct = get_structure(n)
    num_edges = struct['num_edges']
    edge_to_k4s = struct['edge_to_k4s']
    edge_to_k5s = struct['edge_to_k5s']
    k4_edges = struct['k4_edges']
    k5_edges = struct['k5_edges']
    num_k4s = struct['num_k4s']
    num_k5s = struct['num_k5s']
    
    # Initial Random Coloring
    colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    # Calculate initial counts
    red_counts = [0] * num_k4s
    blue_counts = [0] * num_k5s
    
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
        
    best_colors = colors.copy()
    best_violations = total_violations
    
    # Greedy Phase: Only accept moves that reduce violations
    if greedy_phase:
        greedy_time = time_limit * 0.2 # Use 20% of time for greedy
        greedy_deadline = start_time + greedy_time
        no_improve_count = 0
        max_no_improve = 10000 # Stop greedy if stuck
        
        while time.time() < greedy_deadline and no_improve_count < max_no_improve:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            for k4_idx in edge_to_k4s[e_idx]:
                if old_color == 0:
                    red_counts[k4_idx] -= 1
                    if red_counts[k4_idx] == 3:
                        delta_v -= 1
                else:
                    red_counts[k4_idx] += 1
                    if red_counts[k4_idx] == 4:
                        delta_v += 1
                        
            for k5_idx in edge_to_k5s[e_idx]:
                if old_color == 1:
                    blue_counts[k5_idx] -= 1
                    if blue_counts[k5_idx] == 9:
                        delta_v -= 1
                else:
                    blue_counts[k5_idx] += 1
                    if blue_counts[k5_idx] == 10:
                        delta_v += 1
                        
            if delta_v < 0:
                colors[e_idx] = new_color
                total_violations += delta_v
                no_improve_count = 0
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    if total_violations == 0:
                        return colors.copy()
            else:
                # Revert
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
                no_improve_count += 1
                
    # SA Phase
    T = 1.0
    cooling = 0.995
    exp = math.exp
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        for k4_idx in edge_to_k4s[e_idx]:
            if old_color == 0:
                red_counts[k4_idx] -= 1
                if red_counts[k4_idx] == 3:
                    delta_v -= 1
            else:
                red_counts[k4_idx] += 1
                if red_counts[k4_idx] == 4:
                    delta_v += 1
                    
        for k5_idx in edge_to_k5s[e_idx]:
            if old_color == 1:
                blue_counts[k5_idx] -= 1
                if blue_counts[k5_idx] == 9:
                    delta_v -= 1
            else:
                blue_counts[k5_idx] += 1
                if blue_counts[k5_idx] == 10:
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        accepted = False
        if delta_v <= 0:
            accepted = True
        else:
            if T > 0.001:
                if random.random() < exp(-delta_v / T):
                    accepted = True
                    
        if accepted:
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return colors.copy()
                
        else:
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
