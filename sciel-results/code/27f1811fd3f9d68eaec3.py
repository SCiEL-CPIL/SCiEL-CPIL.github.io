import random
import time
import itertools
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Three distinct 4-cycles on 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    # State
    edge_colors = {}
    
    # Initialize with dense random graph
    p_init = 0.95
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p_init:
                edge_colors[(u,v)] = random.randint(0, num_colors - 1)
                
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    missing_edges = [e for e in all_edges if e not in edge_colors]
    
    # Helper to calculate score and violations efficiently
    # We maintain a count of violations per cycle? No, just recompute for simplicity first, 
    # but we need speed. 
    # Let's maintain a list of "bad" cycles.
    
    def get_bad_cycles():
        bad = []
        for i, cycle in enumerate(all_cycles):
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    bad.append(i)
        return bad

    # Simulated Annealing
    # Score = num_edges * 10 - num_bad_cycles
    # We want to maximize this.
    
    def get_score():
        bad = get_bad_cycles()
        return len(edge_colors) * 10 - len(bad)
    
    current_score = get_score()
    best_score = current_score
    best_state = {k: v for k, v in edge_colors.items()}
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Pre-compute list of all edges for random selection
    all_edges_list = all_edges
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add edge
            if not missing_edges:
                # If no missing edges, force recolor or remove
                move_type = 0.5 
            else:
                e = random.choice(missing_edges)
                c = random.randint(0, num_colors - 1)
                # Tentative move
                edge_colors[e] = c
                missing_edges.remove(e)
                
                new_score = get_score()
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = {k: v for k, v in edge_colors.items()}
                else:
                    # Revert
                    del edge_colors[e]
                    missing_edges.append(e)
                
        elif move_type < 0.6:
            # Remove edge
            if edge_colors:
                e = random.choice(list(edge_colors.keys()))
                c = edge_colors[e]
                del edge_colors[e]
                missing_edges.append(e)
                
                new_score = get_score()
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = {k: v for k, v in edge_colors.items()}
                else:
                    # Revert
                    edge_colors[e] = c
                    missing_edges.remove(e)
                    
        else:
            # Recolor edge
            if edge_colors:
                e = random.choice(list(edge_colors.keys()))
                old_c = edge_colors[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c:
                    edge_colors[e] = new_c
                    
                    new_score = get_score()
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = {k: v for k, v in edge_colors.items()}
                    else:
                        # Revert
                        edge_colors[e] = old_c

        T *= cooling_rate
        
    # Restore best state
    edge_colors = best_state
    
    # Final Repair: Ensure validity
    # If the best state is invalid, we must repair it to return a valid solution.
    # The score function penalizes violations, so best_state should be valid or near-valid.
    # But we must guarantee validity for the return value.
    
    bad = get_bad_cycles()
    while bad:
        if time.time() > deadline:
            break
        ci = random.choice(bad)
        cycle = all_cycles[ci]
        # Remove a random edge from the bad cycle
        e_to_remove = random.choice(cycle)
        if e_to_remove in edge_colors:
            del edge_colors[e_to_remove]
        bad = get_bad_cycles()
        
    # Update missing edges for potential further improvement? 
    # We are out of time or close to it. 
    # Let's do a quick greedy add-back if time permits.
    
    current_edges = set(edge_colors.keys())
    missing_edges = [e for e in all_edges if e not in current_edges]
    
    def can_add_with_color(e, c):
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            if not all(other_e in edge_colors for other_e in cycle if other_e != e):
                continue
            other_colors = [edge_colors[other_e] for other_e in cycle if other_e != e]
            if len(set(other_colors)) < 3:
                return False
            if c in other_colors:
                return False
        return True

    # Try to add missing edges greedily
    random.shuffle(missing_edges)
    for e in missing_edges:
        if time.time() > deadline:
            break
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if can_add_with_color(e, c):
                edge_colors[e] = c
                break
                
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result
