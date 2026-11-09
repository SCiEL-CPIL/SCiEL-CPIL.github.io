import numpy as np
import time
import random
import itertools

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 115.0
    
    # 1. Precompute edge indices and cycles
    edge_index = {}
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
    m = len(all_edges) # 190
    
    # Precompute all 4-cycles as tuples of 4 edge indices
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d
    # The 3 possible cycles on {a,b,c,d} are:
    # 1: a-b-c-d-a -> edges (a,b), (b,c), (c,d), (d,a)
    # 2: a-b-d-c-a -> edges (a,b), (b,d), (d,c), (c,a)
    # 3: a-c-b-d-a -> edges (a,c), (c,b), (b,d), (d,a)
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1
        e1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
        cycles.append(tuple(e1))
        # Cycle 2
        e2 = [edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]]
        cycles.append(tuple(e2))
        # Cycle 3
        e3 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
        cycles.append(tuple(e3))
        
    num_cycles = len(cycles)
    
    # Precompute edge_to_cycles: for each edge, list of cycle indices it belongs to
    edge_to_cycles = [[] for _ in range(m)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)
            
    # State
    # colors: array of length m, colors[i] is the color of edge i
    # active: boolean array of length m, active[i] is True if edge i is present
    # conflict_count: array of length m, number of conflicting cycles for each edge
    # is_conflict: boolean array of length num_cycles, True if cycle is a conflict
    
    def check_cycle_conflict(cyc, colors, active):
        """Check if a specific cycle (tuple of edge indices) is a conflict."""
        # If any edge in the cycle is not active, it's not a conflict (we only care about existing edges)
        # But in our local search, we usually keep all edges active and just recolor, 
        # or remove edges. If an edge is removed, cycles involving it are no longer "active cycles".
        # For simplicity in the local search step where we try to recolor an existing edge,
        # we assume all 4 edges are present.
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        return not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)

    def calculate_initial_conflicts(colors, active):
        conflict_count = np.zeros(m, dtype=int)
        is_conflict = np.zeros(num_cycles, dtype=bool)
        
        for i, cyc in enumerate(cycles):
            # Check if all edges in cycle are active
            if not all(active[e] for e in cyc):
                continue
            if check_cycle_conflict(cyc, colors, active):
                is_conflict[i] = True
                for e in cyc:
                    conflict_count[e] += 1
                    
        return conflict_count, is_conflict

    best_edges = 0
    best_adj = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize
        # Start with a 1-factorization-like coloring to reduce initial conflicts
        # Or just random
        random.seed(restart)
        np.random.seed(restart)
        
        colors = np.random.randint(0, num_colors, m)
        active = np.ones(m, dtype=bool)
        
        # Initial conflict calculation
        conflict_count, is_conflict = calculate_initial_conflicts(colors, active)
        
        # Local Search Phase 1: Reduce Conflicts (Recolor or Remove)
        # We want to reach 0 conflicts.
        # We prioritize recoloring. If recoloring doesn't help, remove.
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            if np.sum(is_conflict) == 0:
                break
                
            # Find the edge with the most conflicts
            # Only consider active edges
            active_conflicts = conflict_count.copy()
            active_conflicts[~active] = -1
            worst_edge = np.argmax(active_conflicts)
            
            if active_conflicts[worst_edge] <= 0:
                break
                
            current_conflicts = np.sum(is_conflict)
            
            # Try to recolor worst_edge
            best_new_color = -1
            best_conflicts_after = current_conflicts
            best_delta = 0
            
            # We need to simulate the change for each color
            # To do this efficiently, we look at the cycles containing worst_edge
            affected_cycles = edge_to_cycles[worst_edge]
            
            # Current state of affected cycles
            # We need to know which ones are currently conflicts
            current_affected_conflicts = 0
            for ci in affected_cycles:
                if is_conflict[ci]:
                    current_affected_conflicts += 1
            
            old_color = colors[worst_edge]
            
            for new_c in range(num_colors):
                if new_c == old_color:
                    continue
                
                # Simulate
                # For each affected cycle, check if it becomes a conflict or stays one
                new_affected_conflicts = 0
                for ci in affected_cycles:
                    cyc = cycles[ci]
                    # Check if all other edges are active (they should be, since we only recolor active edges)
                    # And check if this specific cycle becomes a conflict with new_c
                    # We need to get the colors of the other 3 edges
                    # This is O(4) per cycle, so O(4 * 153) ~ 600 ops per color
                    # 20 colors * 600 = 12000 ops per step.
                    # 5000 steps * 12000 = 60M ops. Might be slow in Python?
                    # Let's optimize: precompute the "other" edges' colors? No, they change.
                    # But we can just check the 4 colors.
                    
                    e0, e1, e2, e3 = cyc
                    c0 = new_c if e0 == worst_edge else colors[e0]
                    c1 = new_c if e1 == worst_edge else colors[e1]
                    c2 = new_c if e2 == worst_edge else colors[e2]
                    c3 = new_c if e3 == worst_edge else colors[e3]
                    
                    # Check distinctness
                    if not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3):
                        new_affected_conflicts += 1
                
                delta = new_affected_conflicts - current_affected_conflicts
                total_conflicts_after = current_conflicts + delta
                
                if total_conflicts_after < best_conflicts_after:
                    best_conflicts_after = total_conflicts_after
                    best_new_color = new_c
                    best_delta = delta
            
            if best_new_color != -1:
                # Apply recoloring
                colors[worst_edge] = best_new_color
                
                # Update conflict counts and is_conflict array
                # For each affected cycle, update its status
                for ci in affected_cycles:
                    cyc = cycles[ci]
                    was_conflict = is_conflict[ci]
                    is_now_conflict = check_cycle_conflict(cyc, colors, active)
                    
                    if was_conflict and not is_now_conflict:
                        # Conflict resolved
                        is_conflict[ci] = False
                        for e in cyc:
                            conflict_count[e] -= 1
                    elif not was_conflict and is_now_conflict:
                        # New conflict
                        is_conflict[ci] = True
                        for e in cyc:
                            conflict_count[e] += 1
                # Note: The edge worst_edge's count is updated via the loop above
                # But we also need to ensure consistency. The loop updates all edges in the cycle.
                
            else:
                # No recoloring improves the situation (or keeps it same).
                # Remove the edge.
                active[worst_edge] = False
                colors[worst_edge] = -1 # Mark as inactive
                
                # Update conflicts for affected cycles
                for ci in affected_cycles:
                    cyc = cycles[ci]
                    # If the cycle had this edge, and it's now inactive, it's no longer a "conflict" in the sense of a bad cycle
                    # Because the cycle doesn't exist anymore.
                    if is_conflict[ci]:
                        is_conflict[ci] = False
                        for e in cyc:
                            if active[e]: # Only update active edges? 
                                # Actually, if the edge is removed, the cycle is gone.
                                # The conflict count for the removed edge should be 0.
                                # The conflict counts for other edges in this cycle should decrease by 1
                                # because this cycle no longer contributes to their conflict count.
                                conflict_count[e] -= 1
                conflict_count[worst_edge] = 0

        # Check if we achieved 0 conflicts
        if np.sum(is_conflict) == 0:
            current_edge_count = np.sum(active)
            if current_edge_count > best_edges:
                best_edges = current_edge_count
                # Reconstruct adj matrix
                adj = -np.ones((n, n), dtype=int)
                for i, (u, v) in enumerate(all_edges):
                    if active[i]:
                        adj[u, v] = colors[i]
                        adj[v, u] = colors[i]
                best_adj = adj.copy()
                
                # Phase 2: Try to add back removed edges
                # Iterate through inactive edges
                inactive_indices = [i for i in range(m) if not active[i]]
                random.shuffle(inactive_indices)
                
                for i in inactive_indices:
                    if time.time() > deadline:
                        break
                    
                    u, v = all_edges[i]
                    
                    # Try all colors
                    found_valid_color = False
                    for c in range(num_colors):
                        # Simulate adding edge i with color c
                        # This edge participates in cycles in edge_to_cycles[i]
                        # All these cycles are currently NOT conflicts (since we have 0 conflicts)
                        # We need to check if adding this edge creates any new conflicts.
                        # A cycle becomes a conflict if:
                        # 1. All 4 edges are active.
                        # 2. The 4 colors are not distinct.
                        
                        # Since we are adding edge i, the other 3 edges must be active for the cycle to be "active".
                        # If any of the other 3 edges are inactive, the cycle is not active, so no conflict.
                        
                        valid = True
                        for ci in edge_to_cycles[i]:
                            cyc = cycles[ci]
                            # Check if other 3 edges are active
                            other_edges = [e for e in cyc if e != i]
                            if not all(active[e] for e in other_edges):
                                continue
                            
                            # Check colors
                            c0 = c if cyc[0] == i else colors[cyc[0]]
                            c1 = c if cyc[1] == i else colors[cyc[1]]
                            c2 = c if cyc[2] == i else colors[cyc[2]]
                            c3 = c if cyc[3] == i else colors[cyc[3]]
                            
                            if not (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3):
                                valid = False
                                break
                        
                        if valid:
                            # Add the edge
                            active[i] = True
                            colors[i] = c
                            found_valid_color = True
                            # Update conflict counts (should remain 0)
                            # We don't strictly need to update is_conflict if we verified validity,
                            # but for consistency in case we continue searching:
                            # Since we verified no new conflicts, is_conflict remains all False.
                            break
                    
                    if found_valid_color:
                        # We added an edge. This might open up possibilities for other edges?
                        # No, adding an edge only increases constraints.
                        # However, it might allow a different edge to be added if we had removed it due to conflict?
                        # No, we only removed edges to resolve conflicts. Adding an edge back is safe if it doesn't create conflicts.
                        pass

        # If we found a valid graph with 190 edges, we can stop
        if best_edges == 190:
            break

    # Final Result
    if best_adj is None:
        # Fallback: return a valid small graph
        # A tree is always valid
        adj = -np.ones((n, n), dtype=int)
        for v in range(1, n):
            c = v % num_colors
            adj[0, v] = c
            adj[v, 0] = c
        best_adj = adj

    res = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
