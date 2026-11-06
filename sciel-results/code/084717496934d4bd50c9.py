import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 20
    colors_count = 20
    deadline = time.time() + 110.0 # Safety margin
    
    # 1. Precompute edges
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edges.append((u, v))
            idx += 1
    num_edges = len(edges)
    
    # 2. Precompute 4-cycles
    # A 4-cycle is represented by a tuple of 4 edge indices.
    # We generate all sets of 4 vertices, and for each, the 3 distinct 4-cycles.
    cycles = []
    vertex_combos = []
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    u, v, w, x = i, j, k, l
                    # 3 cycles on {u, v, w, x}
                    # 1. u-v-w-x-u
                    e1 = edge_map.get((min(u,v), max(u,v)))
                    e2 = edge_map.get((min(v,w), max(v,w)))
                    e3 = edge_map.get((min(w,x), max(w,x)))
                    e4 = edge_map.get((min(x,u), max(x,u)))
                    cycles.append((e1, e2, e3, e4))
                    
                    # 2. u-v-x-w-u
                    e1 = edge_map.get((min(u,v), max(u,v)))
                    e2 = edge_map.get((min(v,x), max(v,x)))
                    e3 = edge_map.get((min(x,w), max(x,w)))
                    e4 = edge_map.get((min(w,u), max(w,u)))
                    cycles.append((e1, e2, e3, e4))
                    
                    # 3. u-w-v-x-u
                    e1 = edge_map.get((min(u,w), max(u,w)))
                    e2 = edge_map.get((min(w,v), max(w,v)))
                    e3 = edge_map.get((min(v,x), max(v,x)))
                    e4 = edge_map.get((min(x,u), max(x,u)))
                    cycles.append((e1, e2, e3, e4))
    
    num_cycles = len(cycles)
    
    # Map edge index to list of cycle indices it participates in
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)

    # 3. Local Search with smarter initialization and move selection
    # Strategy: Try multiple random restarts with different initializations.
    # For each restart, use a greedy initialization followed by local search.
    
    best_result = None
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies per restart
        if restart % 3 == 0:
            # Random initialization
            colours = [random.randint(0, colors_count - 1) for _ in range(num_edges)]
        elif restart % 3 == 1:
            # Structured initialization: use edge index mod colors_count
            colours = [i % colors_count for i in range(num_edges)]
        else:
            # Another structured pattern
            colours = [(i * 7) % colors_count for i in range(num_edges)]
        
        # Calculate initial violations
        violated = set()
        for c_idx, cycle in enumerate(cycles):
            c1, c2, c3, c4 = cycle
            if len({colours[c1], colours[c2], colours[c3], colours[c4]}) < 4:
                violated.add(c_idx)
        
        if not violated:
            return [[edges[i][0], edges[i][1], colours[i]] for i in range(num_edges)]
        
        # Local search loop for this restart
        # Use a more aggressive strategy: prioritize cycles with more conflicts
        search_steps = 0
        max_steps_per_restart = 50000
        
        while time.time() < deadline - 10.0 and search_steps < max_steps_per_restart:
            if not violated:
                return [[edges[i][0], edges[i][1], colours[i]] for i in range(num_edges)]
            
            # Pick a violated cycle - bias toward cycles that are "more violated"
            # We'll just pick randomly but with a slight bias
            c_idx = random.choice(list(violated))
            cycle = cycles[c_idx]
            
            # Try to find a color change that reduces violations
            best_new_col = None
            best_violation_delta = 0
            best_e_idx = None
            
            # For each edge in the cycle, try all colors and pick the one that reduces violations most
            for e_idx in cycle:
                old_col = colours[e_idx]
                
                # Try a few candidate colors
                candidates = [random.randint(0, colors_count - 1) for _ in range(5)]
                # Ensure we try at least one different color
                if all(c == old_col for c in candidates):
                    candidates[0] = (old_col + 1) % colors_count
                
                for new_col in candidates:
                    if new_col == old_col:
                        continue
                    
                    # Simulate the change
                    colours[e_idx] = new_col
                    
                    # Count violations in affected cycles
                    current_violations = 0
                    for cyc_idx in edge_to_cycles[e_idx]:
                        c1, c2, c3, c4 = cycles[cyc_idx]
                        if len({colours[c1], colours[c2], colours[c3], colours[c4]}) < 4:
                            current_violations += 1
                    
                    # Revert
                    colours[e_idx] = old_col
                    
                    delta = best_violation_delta - current_violations + 1  # +1 for the base violation being fixed
                    if delta > best_violation_delta:
                        best_violation_delta = delta
                        best_new_col = new_col
                        best_e_idx = e_idx
            
            if best_e_idx is not None:
                colours[best_e_idx] = best_new_col
                e_idx = best_e_idx
                old_col = colours[e_idx]
            else:
                # Fallback: random move
                e_idx = random.choice(cycle)
                old_col = colours[e_idx]
                new_col = random.randint(0, colors_count - 1)
                if new_col == old_col:
                    new_col = (new_col + 1) % colors_count
                colours[e_idx] = new_col
                best_new_col = new_col
            
            # Update violations for cycles involving e_idx
            to_remove = []
            to_add = []
            
            for cyc_idx in edge_to_cycles[e_idx]:
                in_violated = cyc_idx in violated
                
                c1, c2, c3, c4 = cycles[cyc_idx]
                is_violated = len({colours[c1], colours[c2], colours[c3], colours[c4]}) < 4
                
                if in_violated and not is_violated:
                    to_remove.append(cyc_idx)
                elif not in_violated and is_violated:
                    to_add.append(cyc_idx)
            
            for idx in to_remove:
                violated.discard(idx)
            for idx in to_add:
                violated.add(idx)
            
            search_steps += 1
    
    # If we ran out of time, return a valid subgraph
    # Fallback to a known valid structure if local search fails:
    # A star is valid.
    return [[0, v, 0] for v in range(1, 20)]

# EVOLVE-BLOCK-END
