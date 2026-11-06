import time
import random
import math
from itertools import combinations

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

def greedy_construct(n, p_red=0.6, seed=None):
    """Greedy sequential construction: assign colors in lexicographic order,
    choosing each color to minimize partial monochromatic clique formation."""
    if seed is not None:
        random.seed(seed)
    
    struct = get_structure(n)
    num_edges = struct['num_edges']
    edge_to_k4s = struct['edge_to_k4s']
    edge_to_k5s = struct['edge_to_k5s']
    k4_edges = struct['k4_edges']
    k5_edges = struct['k5_edges']
    
    colors = [None] * num_edges
    red_counts = [0] * struct['num_k4s']
    blue_counts = [0] * struct['num_k5s']
    
    for e_idx in range(num_edges):
        # Score each color choice
        best_color = None
        best_score = float('inf')
        
        for c in [0, 1]:
            # Count how many K4s would be completed if we set this color to red
            # and how many K5s would be completed if blue
            danger = 0.0
            
            for k4_idx in edge_to_k4s[e_idx]:
                if c == 0:  # red
                    # This edge is red; check if other 3 edges in this K4 are also red
                    k4e = k4_edges[k4_idx]
                    other_reds = 0
                    for e in k4e:
                        if e != e_idx:
                            if colors[e] == 0:
                                other_reds += 1
                            elif colors[e] is None:
                                # Uncolored - potential future red
                                other_reds += 0.3
                    if other_reds >= 3:
                        danger += 10.0  # Would complete a red K4
                    elif other_reds >= 2:
                        danger += 1.0   # Near completion
                    
            for k5_idx in edge_to_k5s[e_idx]:
                if c == 1:  # blue
                    k5e = k5_edges[k5_idx]
                    other_blues = 0
                    for e in k5e:
                        if e != e_idx:
                            if colors[e] == 1:
                                other_blues += 1
                            elif colors[e] is None:
                                other_blues += 0.15
                    if other_blues >= 9:
                        danger += 15.0  # Would complete a blue K5
                    elif other_blues >= 7:
                        danger += 1.5   # Near completion
            
            # Add small random tiebreaker
            danger += random.random() * 0.1
            
            if danger < best_score:
                best_score = danger
                best_color = c
        
        colors[e_idx] = best_color
        
        # Update counts
        for k4_idx in edge_to_k4s[e_idx]:
            if best_color == 0:
                red_counts[k4_idx] += 1
        for k5_idx in edge_to_k5s[e_idx]:
            if best_color == 1:
                blue_counts[k5_idx] += 1
    
    # Check validity
    total_violations = 0
    for rc in red_counts:
        if rc == 4:
            total_violations += 1
    for bc in blue_counts:
        if bc == 10:
            total_violations += 1
    
    if total_violations == 0:
        return colors
    return None

def solve_fast(n, time_limit=5, p_red=0.6, use_greedy_init=False, seed=None):
    """Optimized Simulated Annealing using count-based updates."""
    start_time = time.time()
    
    struct = get_structure(n)
    num_edges = struct['num_edges']
    edge_to_k4s = struct['edge_to_k4s']
    edge_to_k5s = struct['edge_to_k5s']
    k4_edges = struct['k4_edges']
    k5_edges = struct['k5_edges']
    num_k4s = struct['num_k4s']
    num_k5s = struct['num_k5s']
    
    # Initial coloring
    if use_greedy_init:
        colors = greedy_construct(n, p_red=p_red, seed=seed)
        if colors is None:
            # Greedy failed, fall back to random
            colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    else:
        colors = [0 if random.random() < p_red else 1 for _ in range(num_edges)]
    
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
        
    T = 0.5
    cooling = 0.995
    best_colors = colors.copy()
    best_violations = total_violations
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
        elif T > 0.001:
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

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 0
    best_colors = None
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
        
        if n == 34:
            max_time_for_n = 50
        elif n == 33:
            max_time_for_n = 25
        elif n == 32:
            max_time_for_n = 12
        else:
            max_time_for_n = 5
            
        found = False
        start_n_time = time.time()
        
        # First try greedy construction (fast, deterministic-ish)
        if n <= 34:
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                result = greedy_construct(n, p_red=0.55 + 0.05 * attempt, seed=42 + attempt)
                if result is not None:
                    best_n = n
                    best_colors = result
                    found = True
                    break
        
        if found:
            break
        
        # Then try SA with various initializations
        p_reds = [0.6, 0.55, 0.65, 0.5, 0.7, 0.55, 0.6, 0.55]
        for attempt in range(len(p_reds)):
            if time.time() > deadline - 5 or time.time() - start_n_time >= max_time_for_n:
                break
                
            time_limit = 4.0
            if time.time() + time_limit > deadline - 5:
                time_limit = max(1.0, deadline - 5 - time.time())
                
            p_red = p_reds[attempt]
            use_greedy = (attempt % 3 == 0)  # Every 3rd attempt uses greedy init
            
            result = solve_fast(n, time_limit, p_red=p_red, use_greedy_init=use_greedy, seed=100 + attempt)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break
            
    if best_n == 0:
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}
