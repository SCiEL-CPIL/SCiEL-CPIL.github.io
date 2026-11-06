import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
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
        
        # For each edge, which quads contain it?
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        # For each edge, which pents contain it?
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        # Use a structured heuristic initialization: color based on sum of vertices mod 3
        # This often breaks symmetries better than pure random
        # Try multiple random seeds for the heuristic
        best_violations = float('inf')
        best_colors = None
        
        # Run local search with time budget
        search_deadline = min(deadline - 5.0, start_time + 100.0)
        
        while time.time() < search_deadline:
            # Heuristic initialization: color based on (a+b+c) % 3
            # If sum % 3 == 0 or 1, red (0); if 2, blue (1)
            # Or randomize the mapping
            perm = random.sample([0, 1, 2], 3)
            colors = []
            for (a, b, c) in edges:
                s = (a + b + c) % 3
                # Map s to color using perm
                # perm[s] gives which "group" s belongs to
                # Let's say group 0,1 -> red, group 2 -> blue
                if perm[s] == 2:
                    colors.append(1)  # Blue
                else:
                    colors.append(0)  # Red
            
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
                
            # Local search loop with simulated annealing-like acceptance
            local_deadline = min(search_deadline, time.time() + 2.0)
            temp = 1.0  # Initial temperature
            no_improve = 0
            max_no_improve = 2000
            
            while time.time() < local_deadline and no_improve < max_no_improve:
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    # Quads: red count decreases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1 # Violation removed
                    # Pents: blue count increases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1 # Violation added
                else: # Blue -> Red
                    # Quads: red count increases
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1 # Violation added
                    # Pents: blue count decreases
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1 # Violation removed
                
                # Acceptance criterion
                accepted = False
                if delta_v < 0:
                    accepted = True
                elif delta_v == 0:
                    accepted = random.random() < 0.5
                else:
                    # Accept with probability exp(-delta_v / temp)
                    if random.random() < pow(2.718281828, -delta_v / temp):
                        accepted = True
                
                if accepted:
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
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                # Cool down
                if no_improve > 500:
                    temp *= 0.9
                    if temp < 0.01:
                        break

        # If we didn't find a solution for this n in the allotted time, try smaller n
        # We don't return best_colors because it's not valid.
        
    # If no n >= 2 found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
