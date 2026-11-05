import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # List of all edges (i, j) in lexicographic order
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

    # For each pair of edges within a K4, precompute which K4s contain both
    # Actually, let's precompute for each edge pair (e1, e2) the set of K4s containing both
    # This will help with 2-move operations
    # But that's expensive. Let's just use 1-move with good heuristics.

    def count_violations(colors):
        viol = 0
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                viol += 1
        return viol

    def count_colors_used(colors):
        return len(set(colors))

    # Fast local search with delta updates
    def local_search(max_colors, time_limit, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        # Initialize with random colors
        colors = [rng.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        violating_k4s = set()
        
        for k4_idx, e_indices in enumerate(k4_edge_indices):
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                violating_k4s.add(k4_idx)
                
        best_violations = total_violations
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 3.0
        min_temp = 0.001
        cool_rate = 0.99999
        
        # Track stagnation
        stagnation_counter = 0
        last_improvement_time = time.time()
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors, 0
            
            # Pick a move with bias toward violating K4s
            if total_violations > 0 and rng.random() < 0.5:
                # Pick a random violating K4
                v_k4_idx = rng.choice(list(violating_k4s))
                # Pick a random edge from that K4
                e_idx = rng.choice(k4_edge_indices[v_k4_idx])
            elif total_violations > 0 and rng.random() < 0.3:
                # Pick an edge that appears in many violating K4s
                # Heuristic: pick edge with max impact
                candidate_edges = []
                for e_idx2 in range(num_edges):
                    if k4_distinct[edge_to_k4s[e_idx2][0]] < 5 or True:
                        candidate_edges.append(e_idx2)
                # Just pick randomly from all edges but with some structure
                e_idx = rng.randint(0, num_edges - 1)
            else:
                e_idx = rng.randint(0, num_edges - 1)
                
            old_color = colors[e_idx]
            new_color = rng.randint(0, max_colors - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate Delta Violations
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old distinct values for potential revert
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
                if counts[new_color] == 1:  # It was 0 before
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                # Update violation status
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
                
                # Update violating_k4s set
                if was_viol and not is_viol:
                    violating_k4s.discard(k4_idx)
                elif not was_viol and is_viol:
                    violating_k4s.add(k4_idx)
            
            # Acceptance Criteria
            accept = False
            if delta_viol <= 0:
                accept = True
            else:
                if rng.random() < math.exp(-delta_viol / max(temp, 1e-10)):
                    accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    stagnation_counter = 0
                    last_improvement_time = time.time()
                else:
                    stagnation_counter += 1
            else:
                # Revert counts and distinct
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    
                    # Revert violating_k4s set
                    if old_d < 5:
                        violating_k4s.add(k4_idx)
                    else:
                        violating_k4s.discard(k4_idx)
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                if stagnation_counter > 10000:
                    temp = 1.0
                    stagnation_counter = 0
                    # Try a small perturbation: change a few random edges
                    for _ in range(3):
                        pe = rng.randint(0, num_edges - 1)
                        old_c = colors[pe]
                        new_c = rng.randint(0, max_colors - 1)
                        if new_c != old_c:
                            # Quick update without full delta calc for perturbation
                            for k4_idx in edge_to_k4s[pe]:
                                k4_counts[k4_idx][old_c] -= 1
                                k4_counts[k4_idx][new_c] += 1
                                od = k4_distinct[k4_idx]
                                nd = od
                                if k4_counts[k4_idx][old_c] == 0:
                                    nd -= 1
                                if k4_counts[k4_idx][new_c] == 1:
                                    nd += 1
                                k4_distinct[k4_idx] = nd
                                if (od < 5) != (nd < 5):
                                    if nd < 5:
                                        violating_k4s.add(k4_idx)
                                    else:
                                        violating_k4s.discard(k4_idx)
                            colors[pe] = new_c
                    # Recount total_violations
                    total_violations = sum(1 for d in k4_distinct if d < 5)
                
        return best_colors, best_violations

    # Strategy: Try to find a solution with as few colors as possible
    # Bounds: 13 <= f(16) <= 15
    
    best_solution = None
    best_num_colors = 999
    
    # Time allocation:
    # 13 colors: 50s
    # 14 colors: 30s
    # 15 colors: 25s
    # 16 colors: 15s (should be easy)
    # 17 colors: 10s (fallback)
    
    for C in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 5.0:
            break
        
        if C == 13:
            t_limit = 50.0
        elif C == 14:
            t_limit = 30.0
        elif C == 15:
            t_limit = 25.0
        elif C == 16:
            t_limit = 15.0
        else:
            t_limit = 10.0
            
        t_limit = min(t_limit, deadline - time.time() - 5.0)
        if t_limit < 5.0:
            break
            
        # Run multiple restarts
        best_sol_C = None
        best_viol_C = float('inf')
        
        num_restarts = max(1, int(t_limit // 10))
        per_restart_time = t_limit / num_restarts
        
        for r in range(num_restarts):
            rem_time = min(per_restart_time, deadline - time.time() - 5.0)
            if rem_time < 3.0:
                break
            sol, viol = local_search(C, rem_time, seed=1000 + C * 100 + r)
            if viol < best_viol_C:
                best_viol_C = viol
                best_sol_C = sol
            if viol == 0:
                if C < best_num_colors:
                    best_solution = sol
                    best_num_colors = C
                return sol
        
        if best_sol_C is not None and best_viol_C == 0:
            if C < best_num_colors:
                best_solution = best_sol_C
                best_num_colors = C
            return best_sol_C
        
    # Fallback: generate a valid coloring with 17 colors
    # Random coloring with enough colors should work
    random.seed(42)
    for attempt in range(100):
        fallback_colors = [random.randint(0, 16) for _ in range(num_edges)]
        if count_violations(fallback_colors) == 0:
            if best_solution is None or count_colors_used(fallback_colors) < best_num_colors:
                best_solution = fallback_colors
                best_num_colors = count_colors_used(fallback_colors)
            return fallback_colors
    
    # If we have any solution, return it
    if best_solution is not None:
        return best_solution
    
    # Last resort: use a structured coloring
    # (i + 2*j) mod 19 for proper edge coloring won't work (gives 3 colors per K4)
    # Use a random coloring with many colors
    random.seed(12345)
    fallback_colors = [random.randint(0, 19) for _ in range(num_edges)]
    while count_violations(fallback_colors) > 0:
        fallback_colors = [random.randint(0, 19) for _ in range(num_edges)]
    return fallback_colors
