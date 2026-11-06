import time
import random
import math
import numpy as np
from itertools import combinations

def _compute_violations(colors, edge_to_quads, edge_to_pents, n_quads, n_pents):
    """Compute total violations from a coloring."""
    m = len(colors)
    quad_red_counts = np.zeros(n_quads, dtype=np.int32)
    pent_blue_counts = np.zeros(n_pents, dtype=np.int32)
    
    for i in range(m):
        if colors[i] == 0:
            for q_idx in edge_to_quads[i]:
                quad_red_counts[q_idx] += 1
        else:
            for p_idx in edge_to_pents[i]:
                pent_blue_counts[p_idx] += 1
    
    violations = int(np.sum(quad_red_counts == 4) + np.sum(pent_blue_counts == 10))
    return violations, quad_red_counts, pent_blue_counts

def _delta_violations(i, old_c, edge_to_quads, edge_to_pents, quad_red_counts, pent_blue_counts):
    """Compute delta in violations when flipping edge i from old_c to 1-old_c."""
    delta_v = 0
    if old_c == 0:  # Red -> Blue
        for q_idx in edge_to_quads[i]:
            if quad_red_counts[q_idx] == 4:
                delta_v -= 1
        for p_idx in edge_to_pents[i]:
            if pent_blue_counts[p_idx] == 9:
                delta_v += 1
    else:  # Blue -> Red
        for q_idx in edge_to_quads[i]:
            if quad_red_counts[q_idx] == 3:
                delta_v += 1
        for p_idx in edge_to_pents[i]:
            if pent_blue_counts[p_idx] == 10:
                delta_v -= 1
    return delta_v

def _apply_flip(i, new_c, colors, edge_to_quads, edge_to_pents, quad_red_counts, pent_blue_counts):
    """Apply a flip to edge i, updating counts."""
    colors[i] = new_c
    if new_c == 0:  # Now Red
        for q_idx in edge_to_quads[i]:
            quad_red_counts[q_idx] += 1
        for p_idx in edge_to_pents[i]:
            pent_blue_counts[p_idx] -= 1
    else:  # Now Blue
        for q_idx in edge_to_quads[i]:
            quad_red_counts[q_idx] -= 1
        for p_idx in edge_to_pents[i]:
            pent_blue_counts[p_idx] += 1

def _algebraic_init(edges, m, n):
    """Deterministic algebraic initialization using finite field-like structure."""
    # Use a hash-like function based on vertex indices
    colors = np.zeros(m, dtype=np.int8)
    for i, (a, b, c) in enumerate(edges):
        # Use a combination of modular arithmetic
        # Try: red if (a*b + b*c + c*a) % 2 == 0, else blue
        val = (a * b + b * c + c * a) % 2
        colors[i] = val
    return colors

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

        best_violations = float('inf')
        best_colors = None
        
        search_deadline = deadline
        
        # Simulated annealing parameters
        T_start = 1.0
        T_min = 0.001
        cooling_rate = 0.999  # Per step
        max_steps_per_run = 50000
        
        # Multiple restarts with simulated annealing
        restart_count = 0
        max_restarts = 200  # Limited by time
        
        while time.time() < search_deadline:
            restart_count += 1
            
            # Initialization strategies
            strategy = restart_count % 3
            if strategy == 0:
                colors = _algebraic_init(edges, m, n)
            elif strategy == 1:
                colors = np.random.randint(0, 2, m).astype(np.int8)
            else:
                # Algebraic with perturbation
                colors = _algebraic_init(edges, m, n)
                # Randomly flip 10% of edges
                flip_indices = np.random.choice(m, size=max(1, m // 10), replace=False)
                colors[flip_indices] = 1 - colors[flip_indices]
            
            # Calculate initial violations
            violations, quad_red_counts, pent_blue_counts = _compute_violations(
                colors, edge_to_quads, edge_to_pents, n_quads, n_pents
            )
            
            if violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
            
            # Simulated annealing loop
            T = T_start
            steps = 0
            no_improve_count = 0
            max_no_improve = 10000
            
            while time.time() < search_deadline and steps < max_steps_per_run:
                if no_improve_count > max_no_improve:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = _delta_violations(i, old_c, edge_to_quads, edge_to_pents, 
                                           quad_red_counts, pent_blue_counts)
                
                # Simulated annealing acceptance
                if delta_v < 0:
                    # Always accept improving moves
                    accept = True
                else:
                    # Accept worse moves with probability exp(-delta_v / T)
                    if T > T_min:
                        prob = math.exp(-delta_v / T)
                        accept = (random.random() < prob)
                    else:
                        accept = False
                
                if accept:
                    _apply_flip(i, new_c, colors, edge_to_quads, edge_to_pents,
                               quad_red_counts, pent_blue_counts)
                    violations += delta_v
                    steps += 1
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    # Cool down
                    T *= cooling_rate
                    if T < T_min:
                        T = T_min
                else:
                    no_improve_count += 1
                    steps += 1

    # If no valid coloring found, return best effort or n=1
    if best_colors is not None and best_violations == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    return {'n': 1, 'colors': []}
