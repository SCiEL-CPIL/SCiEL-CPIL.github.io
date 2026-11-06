import numpy as np
from itertools import combinations
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    # Precompute all triples
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    # Precompute quad indices: for each 4-set, which 4 triples form it
    def build_quad_map(n):
        triples = get_triples(n)
        tri_idx = {t: i for i, t in enumerate(triples)}
        quads = list(combinations(range(n), 4))
        quad_tris = []
        for q in quads:
            tris_in_quad = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (q[i], q[j], q[k])
                        tris_in_quad.append(tri_idx[t])
            quad_tris.append(tris_in_quad)
        return quad_tris, len(quads)
    
    # Precompute pent indices: for each 5-set, which 10 triples form it
    def build_pent_map(n):
        triples = get_triples(n)
        tri_idx = {t: i for i, t in enumerate(triples)}
        pents = list(combinations(range(n), 5))
        pent_tris = []
        for p in pents:
            tris_in_pent = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (p[i], p[j], p[k])
                        tris_in_pent.append(tri_idx[t])
            pent_tris.append(tris_in_pent)
        return pent_tris, len(pents)
    
    def count_violations(colors, quad_tris, pent_tris):
        """Count red K4 violations and blue K5 violations."""
        red_viol = 0
        blue_viol = 0
        for tris in quad_tris:
            if all(colors[t] == 0 for t in tris):
                red_viol += 1
        for tris in pent_tris:
            if all(colors[t] == 1 for t in tris):
                blue_viol += 1
        return red_viol, blue_viol
    
    def is_valid(colors, quad_tris, pent_tris):
        rv, bv = count_violations(colors, quad_tris, pent_tris)
        return rv == 0 and bv == 0
    
    # Try to find a valid coloring for a given n using simulated annealing
    def sa_search(n, max_time):
        triples = get_triples(n)
        num_tris = len(triples)
        quad_tris, _ = build_quad_map(n)
        pent_tris, _ = build_pent_map(n)
        
        # Initialize with random coloring biased towards blue (since blue K5 is harder to avoid? Actually red K4 is 4 triples, blue K5 is 10 triples - blue is harder to make all blue, so random 50/50 might be okay, but let's try slightly favoring red to avoid blue K5? No, we want to AVOID blue K5, so we want FEWER all-blue pents. Let's start with all red? No, that creates red K4s. Let's start random.)
        colors = np.random.randint(0, 2, num_tris, dtype=np.int8)
        
        # Check initial
        rv, bv = count_violations(colors, quad_tris, pent_tris)
        current_cost = rv * 10 + bv  # weight red violations more? Or equal? Let's use equal weight but track separately.
        # Actually, let's define cost as total violations
        current_cost = rv + bv
        
        best_colors = colors.copy()
        best_cost = current_cost
        
        # Simulated Annealing parameters
        T = 50.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute for each triple, which quads and pents it belongs to
        # This is expensive for large n, but let's do it once per n
        tri_to_quads = [[] for _ in range(num_tris)]
        for qi, tris in enumerate(quad_tris):
            for t in tris:
                tri_to_quads[t].append(qi)
        
        tri_to_pents = [[] for _ in range(num_tris)]
        for pi, tris in enumerate(pent_tris):
            for t in tris:
                tri_to_pents[t].append(pi)
        
        # We need to track quad_red_count and pent_blue_count for fast updates
        quad_red_count = np.zeros(len(quad_tris), dtype=np.int32)
        pent_blue_count = np.zeros(len(pent_tris), dtype=np.int32)
        
        # Initialize counts
        for qi, tris in enumerate(quad_tris):
            quad_red_count[qi] = np.sum(colors[tris] == 0)
        for pi, tris in enumerate(pent_tris):
            pent_blue_count[pi] = np.sum(colors[tris] == 1)
        
        current_red_viol = np.sum(quad_red_count == 4)
        current_blue_viol = np.sum(pent_blue_count == 10)
        current_cost = current_red_viol + current_blue_viol
        best_cost = current_cost
        best_colors = colors.copy()
        
        elapsed = time.time() - start_time
        steps = 0
        while T > min_T and (time.time() - start_time) < max_time:
            # Pick a random triple to flip
            t = np.random.randint(num_tris)
            old_color = colors[t]
            new_color = 1 - old_color
            
            # Update counts
            # For each quad containing t
            for qi in tri_to_quads[t]:
                if old_color == 0:
                    quad_red_count[qi] -= 1
                else:
                    quad_red_count[qi] += 1
            
            for pi in tri_to_pents[t]:
                if old_color == 1:
                    pent_blue_count[pi] -= 1
                else:
                    pent_blue_count[pi] += 1
            
            # Compute new cost
            new_red_viol = np.sum(quad_red_count == 4)
            new_blue_viol = np.sum(pent_blue_count == 10)
            new_cost = new_red_viol + new_blue_viol
            
            delta_cost = new_cost - current_cost
            
            # Acceptance
            if delta_cost < 0 or (T > 0 and random.random() < np.exp(-delta_cost / T)):
                colors[t] = new_color
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
            else:
                # Revert
                for qi in tri_to_quads[t]:
                    if old_color == 0:
                        quad_red_count[qi] += 1
                    else:
                        quad_red_count[qi] -= 1
                for pi in tri_to_pents[t]:
                    if old_color == 1:
                        pent_blue_count[pi] += 1
                    else:
                        pent_blue_count[pi] -= 1
            
            T *= cooling
            steps += 1
            
            # Periodic check
            if steps % 10000 == 0:
                if time.time() - start_time > max_time:
                    break
        
        return best_colors, best_cost, is_valid(best_colors, quad_tris, pent_tris)
    
    # Try increasing n from a reasonable start
    # The published construction is 34, so let's try up to 36 or 37
    best_n = 0
    best_result = None
    
    # Start with a smaller n to get a valid solution quickly, then try larger
    for n in range(10, 38):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple restarts for each n
        found_valid = False
        for restart in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            colors, cost, valid = sa_search(n, max_time=min(20, deadline - (time.time() - start_time) - 2))
            
            if valid:
                best_n = n
                best_result = colors
                found_valid = True
                break
            elif cost == 0:
                # Should be valid if cost is 0
                best_n = n
                best_result = colors
                found_valid = True
                break
        
        if not found_valid and n > 12:
            # If we can't find a valid one for this n, try next
            continue
    
    if best_result is not None:
        triples = get_triples(best_n)
        color_list = best_result.tolist()
        return {'n': best_n, 'colors': color_list}
    else:
        # Fallback to a small valid coloring
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
