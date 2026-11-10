import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # 1. Precompute edges and cycles
    edges = []
    edge_index = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    # 4-cycles are sets of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
    # In K16, a 4-cycle is defined by 4 vertices. There are 3 pairings of 4 vertices into 2 edges.
    # The task says "all three cycles on every four-vertex subset are checked".
    # The three cycles are:
    # 1. (a,b,c,d) -> edges (a,b), (b,c), (c,d), (d,a)
    # 2. (a,b,d,c) -> edges (a,b), (b,d), (d,c), (c,a)
    # 3. (a,c,b,d) -> edges (a,c), (c,b), (b,d), (d,a)
    # Basically, for every set of 4 vertices, there are 3 distinct 4-cycles.
    
    cycles = []
    # edge_to_cycles: list of lists, edge_to_cycles[edge_idx] = list of cycle_indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    # Iterate over all combinations of 4 vertices
    import itertools
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_index[(a, b)] if (a, b) in edge_index else edge_index[(b, a)]
        e2 = edge_index[(b, c)] if (b, c) in edge_index else edge_index[(c, b)]
        e3 = edge_index[(c, d)] if (c, d) in edge_index else edge_index[(d, c)]
        e4 = edge_index[(d, a)] if (d, a) in edge_index else edge_index[(a, d)]
        cyc_idx = len(cycles)
        cycles.append([e1, e2, e3, e4])
        edge_to_cycles[e1].append(cyc_idx)
        edge_to_cycles[e2].append(cyc_idx)
        edge_to_cycles[e3].append(cyc_idx)
        edge_to_cycles[e4].append(cyc_idx)
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_index[(a, b)] if (a, b) in edge_index else edge_index[(b, a)]
        e2 = edge_index[(b, d)] if (b, d) in edge_index else edge_index[(d, b)]
        e3 = edge_index[(d, c)] if (d, c) in edge_index else edge_index[(c, d)]
        e4 = edge_index[(c, a)] if (c, a) in edge_index else edge_index[(a, c)]
        cyc_idx = len(cycles)
        cycles.append([e1, e2, e3, e4])
        edge_to_cycles[e1].append(cyc_idx)
        edge_to_cycles[e2].append(cyc_idx)
        edge_to_cycles[e3].append(cyc_idx)
        edge_to_cycles[e4].append(cyc_idx)
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_index[(a, c)] if (a, c) in edge_index else edge_index[(c, a)]
        e2 = edge_index[(c, b)] if (c, b) in edge_index else edge_index[(b, c)]
        e3 = edge_index[(b, d)] if (b, d) in edge_index else edge_index[(d, b)]
        e4 = edge_index[(d, a)] if (d, a) in edge_index else edge_index[(a, d)]
        cyc_idx = len(cycles)
        cycles.append([e1, e2, e3, e4])
        edge_to_cycles[e1].append(cyc_idx)
        edge_to_cycles[e2].append(cyc_idx)
        edge_to_cycles[e3].append(cyc_idx)
        edge_to_cycles[e4].append(cyc_idx)

    num_cycles = len(cycles)
    
    # Helper to check if a cycle has 4 distinct colors
    def is_violation(cyc, colors):
        c1 = colors[cyc[0]]
        c2 = colors[cyc[1]]
        c3 = colors[cyc[2]]
        c4 = colors[cyc[3]]
        return len({c1, c2, c3, c4}) == 4

    # Initialize state
    # Start with a random permutation of colors? Or random assignment?
    # Random assignment is good for SA.
    colors = np.random.randint(0, MAX_COLOR + 1, size=NUM_EDGES)
    
    # Track distinct colors
    color_counts = np.zeros(MAX_COLOR + 1, dtype=int)
    for c in colors:
        color_counts[c] += 1
    distinct_colors = np.sum(color_counts > 0)
    
    # Track violations
    violations = 0
    for i in range(num_cycles):
        if is_violation(cycles[i], colors):
            violations += 1
            
    best_colors = colors.copy()
    best_distinct = distinct_colors if violations == 0 else -1
    best_violations = violations
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # Objective function:
    # We want to minimize violations, then maximize distinct colors.
    # Score = -1000 * violations + distinct_colors
    
    def get_score(viol, distinct):
        return -1000 * viol + distinct

    current_score = get_score(violations, distinct_colors)
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random edge
        e_idx = np.random.randint(0, NUM_EDGES)
        old_color = colors[e_idx]
        
        # Pick a new color
        # To encourage diversity, we might pick a color not currently used, or a random one.
        # Random from 0..119 is fine.
        new_color = np.random.randint(0, MAX_COLOR + 1)
        
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        delta_viol = 0
        # For each cycle containing e_idx, check status before and after
        for c_idx in edge_to_cycles[e_idx]:
            cyc = cycles[c_idx]
            was_viol = is_violation(cyc, colors)
            
            # Temporarily change
            colors[e_idx] = new_color
            is_viol = is_violation(cyc, colors)
            
            if was_viol and not is_viol:
                delta_viol -= 1
            elif not was_viol and is_viol:
                delta_viol += 1
            # else no change
            
        # Revert if we don't accept, but we need to calculate distinct colors delta first?
        # It's easier to just compute new distinct colors and score, and revert if rejected.
        
        # Update distinct colors count
        old_count = color_counts[old_color]
        new_count = color_counts[new_color]
        
        # Tentatively update counts
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        new_distinct = distinct_colors
        if old_count == 1 and color_counts[old_color] == 0:
            new_distinct -= 1
        if new_count == 0 and color_counts[new_color] == 1:
            new_distinct += 1
            
        new_violations = violations + delta_viol
        new_score = get_score(new_violations, new_distinct)
        delta_score = new_score - current_score
        
        accept = False
        if delta_score >= 0:
            accept = True
        else:
            # Accept with probability exp(delta_score / T)
            # delta_score is negative
            prob = np.exp(delta_score / T)
            if np.random.random() < prob:
                accept = True
                
        if accept:
            # Commit
            violations = new_violations
            distinct_colors = new_distinct
            current_score = new_score
            
            # Update best
            if violations == 0:
                if distinct_colors > best_distinct:
                    best_distinct = distinct_colors
                    best_colors = colors.copy()
            elif violations < best_violations:
                # If we haven't found a valid one yet, track best invalid too?
                # But we must return valid. So only track valid best.
                pass
                
        else:
            # Revert
            colors[e_idx] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
            
        T *= cooling
        
        # Check deadline periodically
        if time.time() > deadline:
            break
            
    # If we never found a valid coloring, return all zeros (which is valid)
    if best_violations != 0:
         # This shouldn't happen if we start from a valid state or if SA works well.
         # But to be safe, if best_colors has violations, we should return a known valid one.
         # Let's just check if best_colors is valid.
         # Actually, if best_violations > 0, best_distinct is -1.
         # We should return the coloring with min violations? No, task requires valid.
         # If we fail to find valid, return all 0s.
         return [0] * NUM_EDGES
         
    return best_colors.tolist()
