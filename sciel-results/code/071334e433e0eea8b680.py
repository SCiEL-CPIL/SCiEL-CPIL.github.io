import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 120s)
    best_result = {'n': 1, 'colors': []}
    
    # Try n from 34 downwards
    for n in range(34, 1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Map each edge to the quads and pents it belongs to
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            for e in combinations(q, 3):
                edge_to_quads[edge_to_idx[e]].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            for e in combinations(p, 3):
                edge_to_pents[edge_to_idx[e]].append(p_idx)

        best_violations = float('inf')
        best_colors = None
        
        # Run local search for the remaining time
        search_deadline = deadline
        
        restart_count = 0
        max_restarts = 50
        while time.time() < search_deadline and restart_count < max_restarts:
            restart_count += 1
            
            # Algebraic Initialization: Color by parity of sum of vertices
            # This is a linear construction over F2
            colors = []
            for e in edges:
                a, b, c = e
                colors.append((a + b + c) % 2)
            
            # Calculate initial violations
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else: # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Local search loop
            local_deadline = min(search_deadline, time.time() + 1.5)
            no_improve_count = 0
            max_no_improve = 3000
            
            while time.time() < local_deadline:
                if no_improve_count > max_no_improve:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1
                else: # Blue -> Red
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1
                
                # Accept if improving or neutral
                if delta_v < 0 or (delta_v == 0 and random.random() < 0.2):
                    # Apply move
                    colors[i] = new_c
                    if new_c == 0: # Now Red
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    no_improve_count = 0
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                else:
                    no_improve_count += 1

        # If we found a valid coloring for this n, we would have returned.
        # If not, we continue to smaller n.
        # We update best_result if we found a better n? No, we only return valid ones.
        # If we didn't find a valid one for n, we don't update best_result with invalid colors.
        
    # If no valid coloring found for n > 1, return n=1
    return best_result
