import time
import random
import itertools
from collections import Counter

def construct():
    deadline = time.time() + 140  # safety margin
    
    n = 16
    # Precompute all K4s and their edges
    vertices = list(range(n))
    k4s = []
    k4_edges = []
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(combo)
        k4_edges.append(edges)
    
    num_k4s = len(k4s)
    
    # Edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_list.append((i,j))
            idx += 1
    num_edges = len(edge_list)
    
    # Precompute for each edge, which K4s it belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[edge_index[e]].append(k4_idx)
    
    # Precompute k4 edge indices for fast access
    k4_edge_indices = []
    for edges in k4_edges:
        k4_edge_indices.append([edge_index[e] for e in edges])
    
    # Start with (i+j) mod 17 coloring
    colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    def count_colors_used(c):
        return len(set(c))
    
    def is_valid(c):
        for k4_idx in range(num_k4s):
            edges = k4_edge_indices[k4_idx]
            k4_colors = set()
            for e in edges:
                k4_colors.add(c[e])
            if len(k4_colors) < 5:
                return False
        return True
    
    def count_violations(c):
        violations = 0
        for k4_idx in range(num_k4s):
            edges = k4_edge_indices[k4_idx]
            k4_colors = set()
            for e in edges:
                k4_colors.add(c[e])
            if len(k4_colors) < 5:
                violations += 1
        return violations
    
    # Initial check
    if is_valid(colors):
        best_colors = colors[:]
        best_count = count_colors_used(colors)
    else:
        best_colors = colors[:]
        best_count = count_colors_used(colors)
    
    # Greedy color merging
    current_colors = colors[:]
    current_color_count = count_colors_used(current_colors)
    
    # Try to reduce colors
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        
        colors_used = sorted(set(current_colors))
        merged = False
        for i in range(len(colors_used)):
            for j in range(i+1, len(colors_used)):
                if time.time() > deadline - 10:
                    break
                new_c = current_colors[:]
                for k in range(num_edges):
                    if new_c[k] == colors_used[j]:
                        new_c[k] = colors_used[i]
                if is_valid(new_c):
                    current_colors = new_c
                    current_color_count = count_colors_used(current_colors)
                    merged = True
                    break
            if merged or time.time() > deadline - 10:
                break
        if not merged:
            break
    
    # Simulated annealing to reduce colors
    def simulated_annealing_reduce_colors(colors_arr, max_iters=5000):
        """Use simulated annealing to reduce color count."""
        c = colors_arr[:]
        current_count = count_colors_used(c)
        current_violations = count_violations(c)
        
        if current_violations > 0:
            return c, current_count
        
        best_c = c[:]
        best_count = current_count
        
        T = 1.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if current_count <= 13:
                break
            
            colors_used = list(set(c))
            color_freq = Counter(c)
            sorted_colors = sorted(color_freq.items(), key=lambda x: x[1])
            
            # Pick a color to eliminate (least frequent)
            elim_color, freq = sorted_colors[0]
            
            # Pick a target color
            target_color = random.choice([col for col in colors_used if col != elim_color])
            
            # Create new coloring
            new_c = c[:]
            for k in range(num_edges):
                if new_c[k] == elim_color:
                    new_c[k] = target_color
            
            new_violations = count_violations(new_c)
            new_count = count_colors_used(new_c)
            
            # Accept if better or with probability based on temperature
            if new_violations < current_violations or (new_violations == current_violations and new_count < current_count):
                c = new_c
                current_violations = new_violations
                current_count = new_count
                if current_count < best_count:
                    best_c = c[:]
                    best_count = current_count
            elif new_violations == current_violations:
                # Accept with some probability to explore
                if random.random() < 0.1:
                    c = new_c
                    current_count = new_count
            else:
                # Worse violations, accept with small probability
                delta_v = new_violations - current_violations
                if random.random() < (1.0 / (1.0 + delta_v)) * T:
                    c = new_c
                    current_violations = new_violations
                    current_count = new_count
            
            T *= cooling
        
        return best_c, best_count
    
    # Apply simulated annealing
    improved_colors, improved_count = simulated_annealing_reduce_colors(current_colors)
    if improved_count < best_count:
        best_colors = improved_colors
        best_count = improved_count
    
    # Final validation
    if not is_valid(best_colors):
        # Fallback to original
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
        best_count = 17
    
    return best_colors
