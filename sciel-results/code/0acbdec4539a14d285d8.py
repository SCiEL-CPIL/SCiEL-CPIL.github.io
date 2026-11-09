import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20
    
    # 1. Generate all possible edges in K_20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # 2. Assign algebraic coloring: c(u, v) = (u + v) % 20
    color_map = {}
    edge_colors = np.zeros(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(all_edges):
        c = (u + v) % num_colors
        color_map[(u, v)] = c
        edge_colors[i] = c

    # 3. Precompute all 4-cycles and identify bad ones
    # A 4-cycle is defined by 4 distinct vertices a, b, c, d.
    # There are 3 distinct 4-cycles for each set of 4 vertices.
    bad_cycles = []
    
    # Helper to get edge index
    def get_eidx(u, v):
        if u > v:
            u, v = v, u
        return edge_index[(u, v)]

    # Iterate through all combinations of 4 vertices
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Cycle 1: a-b-c-d-a
        e1 = [get_eidx(a,b), get_eidx(b,c), get_eidx(c,d), get_eidx(d,a)]
        cols = [edge_colors[e] for e in e1]
        if len(set(cols)) < 4:
            bad_cycles.append(e1)
            
        # Cycle 2: a-b-d-c-a
        e2 = [get_eidx(a,b), get_eidx(b,d), get_eidx(d,c), get_eidx(c,a)]
        cols = [edge_colors[e] for e in e2]
        if len(set(cols)) < 4:
            bad_cycles.append(e2)
            
        # Cycle 3: a-c-b-d-a
        e3 = [get_eidx(a,c), get_eidx(c,b), get_eidx(b,d), get_eidx(d,a)]
        cols = [edge_colors[e] for e in e3]
        if len(set(cols)) < 4:
            bad_cycles.append(e3)

    # 4. Greedy Hitting Set
    # active_constraints: indices of bad cycles not yet hit
    active_constraints = set(range(len(bad_cycles)))
    
    # edge_count[j]: number of active constraints containing edge j
    edge_count = np.zeros(num_edges, dtype=np.int32)
    for cycle in bad_cycles:
        for e in cycle:
            edge_count[e] += 1
            
    removed = np.zeros(num_edges, dtype=bool)
    
    # Precompute for each edge, the list of cycles it belongs to (for fast update)
    # This might be memory heavy but num_edges=190, num_cycles~14000.
    # 190 * 14000 is too big for a dense matrix.
    # Use list of lists.
    edge_to_cycles = [[] for _ in range(num_edges)]
    for i, cycle in enumerate(bad_cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)

    # Greedy loop
    while active_constraints:
        if time.time() > deadline:
            break
        
        # Find edge with max count among non-removed
        # We can just find max in edge_count where removed is False
        # To speed up, we could maintain a priority queue, but simple scan is O(190)
        max_count = -1
        best_edge = -1
        for j in range(num_edges):
            if not removed[j] and edge_count[j] > max_count:
                max_count = edge_count[j]
                best_edge = j
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed[best_edge] = True
        
        # Update active constraints
        for c_idx in edge_to_cycles[best_edge]:
            if c_idx in active_constraints:
                active_constraints.remove(c_idx)
                # Decrement count for other edges in this cycle
                for e in bad_cycles[c_idx]:
                    if e != best_edge:
                        edge_count[e] -= 1

    # 5. Local Search: Try to add back edges
    # We have a valid set of edges. Can we add any?
    # For each removed edge, check if adding it creates new bad cycles.
    # If it creates K bad cycles, we must remove edges from those cycles.
    # This is a complex repair. 
    # Simple heuristic: Try to add back edges in order of "popularity" (original edge_count).
    
    # Sort removed edges by original edge_count descending
    removed_indices = np.where(removed)[0]
    removed_indices = removed_indices[np.argsort(-edge_count[removed_indices])]
    
    for e_idx in removed_indices:
        if time.time() > deadline:
            break
        
        # Tentatively add edge
        removed[e_idx] = False
        
        # Find new bad cycles
        # A cycle is bad if all its edges are present (not removed) AND it was originally bad?
        # No, adding an edge might create a NEW bad cycle that wasn't in bad_cycles?
        # No, bad_cycles is fixed by the coloring.
        # So we just need to check if any bad cycle is now "satisfied" (all edges present).
        
        new_violations = []
        for c_idx in edge_to_cycles[e_idx]:
            # Check if all edges in this cycle are present
            is_valid = True
            for e in bad_cycles[c_idx]:
                if removed[e]:
                    is_valid = False
                    break
            if is_valid:
                new_violations.append(c_idx)
                
        if not new_violations:
            # Success, keep the edge
            continue
        else:
            # We need to remove edges to hit these new_violations
            # Try to find a minimal set of edges to remove from the union of these cycles
            # Simple greedy: pick the edge in these cycles that hits the most
            # But we must not re-add e_idx?
            
            # If the number of violations is small, we can try to resolve
            # If it's too complex, revert
            if len(new_violations) > 3:
                removed[e_idx] = True
                continue
                
            # Greedy repair
            # Count frequency of edges in new_violations
            freq = {}
            for c_idx in new_violations:
                for e in bad_cycles[c_idx]:
                    if not removed[e]:
                        freq[e] = freq.get(e, 0) + 1
                        
            # Remove edges with highest freq
            # We must keep e_idx? No, we can remove e_idx too.
            # But we just added it.
            
            # Let's just revert if it's not trivial
            removed[e_idx] = True

    # 6. Construct result
    result = []
    for i, (u, v) in enumerate(all_edges):
        if not removed[i]:
            result.append([u, v, color_map[(u, v)]])
            
    return result

# EVOLVE-BLOCK-END
