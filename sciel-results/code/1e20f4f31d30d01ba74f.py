"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin before 120s limit

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        num_quads = len(all_quads)
        quad_triple_idx = []
        for a, b, c, d in all_quads:
            quad_triple_idx.append([triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                                     triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]])

        all_quints = list(combinations(range(target_n), 5))
        num_quints = len(all_quints)
        quint_triple_idx = []
        for q5 in all_quints:
            quint_triple_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])

        # Precompute incidence: triple -> list of quads, triple -> list of quints
        triple_to_quads = [[] for _ in range(num_triples)]
        for qi in range(num_quads):
            for ti in quad_triple_idx[qi]:
                triple_to_quads[ti].append(qi)

        triple_to_quints = [[] for _ in range(num_triples)]
        for qi in range(num_quints):
            for ti in quint_triple_idx[qi]:
                triple_to_quints[ti].append(qi)

        found = False
        # Use randomized backtracking with forward checking
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random order of triples
            order = list(range(num_triples))
            random.shuffle(order)
            
            # State
            colors = [-1] * num_triples
            red_quad_count = [0] * num_quads
            blue_quint_count = [0] * num_quints
            
            # Stack for backtracking: (triple_idx, next_color_to_try)
            # We process in the 'order'
            # We will use a recursive-like stack
            # Item: (step, triple_idx, color)
            # If color is -1, it means "try to assign this triple"
            # If color is 0 or 1, it means "this assignment was made, revert if backtracking"
            
            # Actually, simpler iterative DFS:
            # stack = [(triple_idx, 0)] where 0 is the first color to try
            # We need to track the path to revert.
            
            # Let's use a standard backtracking algorithm
            # We maintain a list of assignments made so far to revert them.
            
            assignments = []
            success = False
            
            def backtrack(idx_in_order):
                if time.time() > deadline - 2:
                    return False
                if idx_in_order == num_triples:
                    return True
                
                t = order[idx_in_order]
                
                # Try colors 0 (Red) and 1 (Blue)
                # Heuristic: try the color that creates fewer "almost" violations?
                # For now, randomize or fixed order. Fixed 0,1 is fine.
                for c in [0, 1]:
                    # Check if assigning c to t is valid
                    valid = True
                    
                    if c == 0: # Red
                        for qi in triple_to_quads[t]:
                            if red_quad_count[qi] + 1 > 3: # If it becomes 4, invalid
                                valid = False
                                break
                    else: # Blue
                        for qi in triple_to_quints[t]:
                            if blue_quint_count[qi] + 1 > 9: # If it becomes 10, invalid
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Assign
                    colors[t] = c
                    if c == 0:
                        for qi in triple_to_quads[t]:
                            red_quad_count[qi] += 1
                    else:
                        for qi in triple_to_quints[t]:
                            blue_quint_count[qi] += 1
                    
                    assignments.append((t, c))
                    
                    if backtrack(idx_in_order + 1):
                        return True
                    
                    # Unassign
                    assignments.pop()
                    colors[t] = -1
                    if c == 0:
                        for qi in triple_to_quads[t]:
                            red_quad_count[qi] -= 1
                    else:
                        for qi in triple_to_quints[t]:
                            blue_quint_count[qi] -= 1
                
                return False
            
            if backtrack(0):
                found = True
                best_colors = colors[:]
                break

        if found:
            best_n = target_n
            # best_colors is already set in the loop

    # Final verification
    if best_n > 9:
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]

        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            best_n = 9
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
