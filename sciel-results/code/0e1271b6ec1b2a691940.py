import random
import time
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill

    # All possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    m = len(all_edges)  # 190
    
    # Edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Find all 4-cycles in K_20
    # A 4-cycle is (a, b, c, d) with edges (a,b), (b,c), (c,d), (d,a)
    # We'll represent each 4-cycle by its 4 edge indices
    cycles = []
    cycle_set = set()
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Check if (a,b), (b,c), (c,d), (d,a) form a cycle
                    # Need all 4 edges to exist in K_n (they do)
                    # But we need to check they form a valid 4-cycle
                    # (a,b)-(b,c)-(c,d)-(d,a)
                    edges_4 = [(a, b), (b, c), (c, d), (d, a)]
                    edge_idxs = tuple(sorted(edge_index[e] for e in edges_4))
                    if edge_idxs not in cycle_set:
                        cycle_set.add(edge_idxs)
                        cycles.append(list(edge_idxs))
    
    num_cycles = len(cycles)
    
    # For each edge, find which cycles contain it
    edge_to_cycles = {i: [] for i in range(m)}
    for ci, cyc in enumerate(cycles):
        for ei in cyc:
            edge_to_cycles[ei].append(ci)
    
    # For each edge, find which other edges in each cycle could conflict
    # Precompute for each edge, the list of (other_edge_in_same_cycle, cycle_index)
    edge_conflicts = {i: [] for i in range(m)}
    for ci, cyc in enumerate(cycles):
        for j in range(4):
            e1 = cyc[j]
            for k in range(4):
                if k != j:
                    e2 = cyc[k]
                    edge_conflicts[e1].append((e2, ci))
    
    # Random restart local search
    best_score = -1
    best_solution = None
    
    max_restarts = 30
    restart = 0
    
    while time.time() < deadline - 2.0 and restart < max_restarts:
        restart += 1
        
        # Initialize: random subset of edges with random colors
        # Start with a moderate density
        edge_present = [False] * m
        edge_color = [0] * m
        
        # Random initialization with some density
        density = random.uniform(0.3, 0.7)
        for i in range(m):
            if random.random() < density:
                edge_present[i] = True
                edge_color[i] = random.randint(0, num_colors - 1)
        
        cur_edges = sum(edge_present)
        
        # Compute violations: for each 4-cycle, check if all 4 edges present
        # and colors are not all distinct
        def count_violations(present, colors):
            violations = 0
            for cyc in cycles:
                if all(present[e] for e in cyc):
                    cols = [colors[e] for e in cyc]
                    if len(set(cols)) < 4:
                        violations += 1
            return violations
        
        cur_violations = count_violations(edge_present, edge_color)
        
        # Simulated annealing
        T = 10.0
        alpha = 0.9995
        steps_per_restart = 50000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2.0:
                break
            
            # Random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Flip an edge's presence
                i = random.randint(0, m - 1)
                old_val = edge_present[i]
                edge_present[i] = not old_val
                new_edges = sum(edge_present)
                
                # Recompute violations only for affected cycles
                # This is expensive; for simplicity, recompute all
                # But let's try incremental
                affected_cycles = edge_to_cycles[i]
                
                # Count violations in affected cycles
                # Old violations in these cycles
                old_v_in_affected = 0
                new_v_in_affected = 0
                
                for ci in affected_cycles:
                    cyc = cycles[ci]
                    if not old_val:
                        # Edge was absent before, now present
                        # Old: cycle couldn't be violated (edge absent)
                        # New: check if all edges present and colors not distinct
                        if all(edge_present[e] for e in cyc):
                            cols = [edge_color[e] for e in cyc]
                            if len(set(cols)) < 4:
                                new_v_in_affected += 1
                    else:
                        # Edge was present before, now absent
                        # Old: might have been violated
                        if all(edge_present[e] for e in cyc if e != i):
                            # All other edges present
                            cols = [edge_color[e] for e in cyc]
                            if len(set(cols)) < 4:
                                old_v_in_affected += 1
                        # New: edge absent, can't be violated
                
                delta_v = new_v_in_affected - old_v_in_affected
                delta_e = new_edges - cur_edges
                
                # Acceptance
                if delta_v < 0 or (delta_v == 0 and delta_e >= 0):
                    cur_violations += delta_v
                    cur_edges = new_edges
                elif delta_v > 0 or (delta_v == 0 and delta_e < 0):
                    # Compute acceptance probability
                    score_delta = -delta_v * 10 - delta_e
                    if random.random() < pow(2.718281828, -score_delta / max(T, 0.01)):
                        cur_violations += delta_v
                        cur_edges = new_edges
                    else:
                        edge_present[i] = old_val
                # Update T
                T *= alpha
                
            elif move_type < 0.7:
                # Change color of a present edge
                i = random.randint(0, m - 1)
                if not edge_present[i]:
                    continue
                old_color = edge_color[i]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                edge_color[i] = new_color
                
                # Recompute violations for affected cycles
                affected_cycles = edge_to_cycles[i]
                old_v_in_affected = 0
                new_v_in_affected = 0
                
                for ci in affected_cycles:
                    cyc = cycles[ci]
                    if all(edge_present[e] for e in cyc):
                        # Old colors
                        old_cols = [edge_color[e] if e != i else old_color for e in cyc]
                        if len(set(old_cols)) < 4:
                            old_v_in_affected += 1
                        # New colors
                        new_cols = [edge_color[e] for e in cyc]
                        if len(set(new_cols)) < 4:
                            new_v_in_affected += 1
                
                delta_v = new_v_in_affected - old_v_in_affected
                
                if delta_v <= 0:
                    cur_violations += delta_v
                else:
                    if random.random() < pow(2.718281828, -delta_v * 10 / max(T, 0.01)):
                        cur_violations += delta_v
                    else:
                        edge_color[i] = old_color
                T *= alpha
                
            else:
                # Remove an edge (to reduce violations)
                # Only if we have many violations
                if cur_violations > 0:
                    # Pick a random violated cycle and remove one of its edges
                    # Find a violated cycle
                    violated_cycles = []
                    for cyc in cycles:
                        if all(edge_present[e] for e in cyc):
                            cols = [edge_color[e] for e in cyc]
                            if len(set(cols)) < 4:
                                violated_cycles.append(cyc)
                    if violated_cycles:
                        cyc = random.choice(violated_cycles)
                        i = random.choice(cyc)
                        edge_present[i] = False
                        cur_edges -= 1
                        T *= alpha
                else:
                    # No violations, try to add an edge
                    absent_edges = [i for i in range(m) if not edge_present[i]]
                    if absent_edges:
                        i = random.choice(absent_edges)
                        edge_present[i] = True
                        edge_color[i] = random.randint(0, num_colors - 1)
                        cur_edges += 1
                        T *= alpha
        
        # Check if this solution is valid and better
        if cur_violations == 0 and cur_edges > best_score:
            best_score = cur_edges
            best_solution = (edge_present[:], edge_color[:])
        
        # Also try to improve by greedily adding edges
        if cur_violations == 0:
            # Try to add more edges
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                absent_edges = [i for i in range(m) if not edge_present[i]]
                random.shuffle(absent_edges)
                for i in absent_edges:
                    if time.time() > deadline - 2.0:
                        break
                    # Try adding edge i with each color
                    for c in range(num_colors):
                        edge_present[i] = True
                        edge_color[i] = c
                        # Check if this creates violations
                        # Only need to check cycles containing edge i
                        ok = True
                        for ci in edge_to_cycles[i]:
                            cyc = cycles[ci]
                            if all(edge_present[e] for e in cyc):
                                cols = [edge_color[e] for e in cyc]
                                if len(set(cols)) < 4:
                                    ok = False
                                    break
                        if ok:
                            improved = True
                            cur_edges += 1
                            break
                        else:
                            edge_present[i] = False
                    else:
                        edge_present[i] = False
                
                if cur_edges > best_score:
                    best_score = cur_edges
                    best_solution = (edge_present[:], edge_color[:])
    
    # Return best solution
    if best_solution is None:
        # Fallback: return empty or minimal valid solution
        # A star graph is always valid (no 4-cycles)
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    present, colors = best_solution
    result = []
    for i in range(m):
        if present[i]:
            u, v = all_edges[i]
            result.append([u, v, colors[i]])
    
    return result
