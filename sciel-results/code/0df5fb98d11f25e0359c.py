import time
import random
from itertools import combinations

def is_not_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    # Build adjacency: for each vertex, which edges contain it
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_to_edges[v].append(i)
    
    # State: color assignment for each vertex (-1=unassigned, 0, 1)
    # Edge state: for each edge, count of 0s and 1s assigned
    edge_zero_count = [0] * len(edges)
    edge_one_count = [0] * len(edges)
    
    def propagate(colors, edge_zc, edge_oc, queue):
        """Unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite."""
        changed = True
        while changed:
            changed = False
            while queue:
                v, c = queue.pop()
                for ei in vert_to_edges[v]:
                    e = edges[ei]
                    if colors[e[0]] == -1 or colors[e[1]] == -1 or colors[e[2]] == -1 or colors[e[3]] == -1 or colors[e[4]] == -1:
                        # Check if we can force the last vertex
                        unassigned = [x for x in e if colors[x] == -1]
                        if len(unassigned) == 1:
                            # Count colors in this edge
                            z = sum(1 for x in e if colors[x] == 0)
                            o = sum(1 for x in e if colors[x] == 1)
                            if z == 4:
                                # All other 4 are 0, so this must be 1
                                if colors[unassigned[0]] != 1:
                                    if colors[unassigned[0]] == 0:
                                        return False
                                    colors[unassigned[0]] = 1
                                    edge_oc[ei] += 1
                                    queue.append((unassigned[0], 1))
                                    changed = True
                            elif o == 4:
                                # All other 4 are 1, so this must be 0
                                if colors[unassigned[0]] != 0:
                                    if colors[unassigned[0]] == 1:
                                        return False
                                    colors[unassigned[0]] = 0
                                    edge_zc[ei] += 1
                                    queue.append((unassigned[0], 0))
                                    changed = True
            # Check for contradictions: any edge with 5 of same color
            for ei in range(len(edges)):
                if edge_zc[ei] == 5 or edge_oc[ei] == 5:
                    return False
        return True
    
    def backtrack(colors, edge_zc, edge_oc, depth):
        if time.time() > deadline:
            return None
        # Find next unassigned vertex
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            # All assigned, check if valid
            for ei in range(len(edges)):
                if edge_zc[ei] == 5 or edge_oc[ei] == 5:
                    return None
            return colors[:]
        
        # Choose vertex with most constraints
        v = max(unassigned, key=lambda x: len(vert_to_edges[x]))
        
        for c in [0, 1]:
            new_colors = colors[:]
            new_ezc = edge_zc[:]
            new_eoc = edge_oc[:]
            new_ezc[v] = edge_zc[v]  # not quite right, let me redo
            
            # Try assigning v = c
            new_colors[v] = c
            queue = [(v, c)]
            
            # Copy state
            new_ezc = edge_zc[:]
            new_eoc = edge_oc[:]
            
            if c == 0:
                for ei in vert_to_edges[v]:
                    new_ezc[ei] += 1
            else:
                for ei in vert_to_edges[v]:
                    new_eoc[ei] += 1
            
            if propagate(new_colors, new_ezc, new_eoc, queue):
                result = backtrack(new_colors, new_ezc, new_eoc, depth + 1)
                if result is not None:
                    return result
        
        return None
    
    colors = [-1] * n
    edge_zc = [0] * len(edges)
    edge_oc = [0] * len(edges)
    
    # Try both initial assignments for vertex 0
    for c0 in [0, 1]:
        colors = [-1] * n
        colors[0] = c0
        edge_zc = [0] * len(edges)
        edge_oc = [0] * len(edges)
        for ei in vert_to_edges[0]:
            if c0 == 0:
                edge_zc[ei] += 1
            else:
                edge_oc[ei] += 1
        
        queue = [(0, c0)]
        if propagate(colors, edge_zc, edge_oc, queue):
            result = backtrack(colors, edge_zc, edge_oc, 0)
            if result is not None:
                # Found a valid 2-coloring, so it IS 2-colorable
                return False
    
    # No valid 2-coloring found
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try n from 9 to 15
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        # Precompute vertex masks for set cover approach
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        M0 = [0] * n
        M1 = [0] * n
        
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
                    
        # Generate all edges
        all_edges = list(combinations(range(n), 5))
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        # Greedy Set Cover with restarts and local search
        best_local_count = float('inf')
        best_local_edges = None
        
        restarts = 30 if n <= 11 else 15
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected_indices = []
            remaining = list(range(len(all_edges)))
            random.shuffle(remaining)
            
            # Greedy
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        
                if best_gain == 0:
                    break
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones:
                # Verify with exact checker
                candidate_edges = [all_edges[i] for i in selected_indices]
                if is_not_2_colorable(n, candidate_edges):
                    # Local search: try to remove edges
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(len(selected_indices) - 1, -1, -1):
                            e_idx = selected_indices[i]
                            new_covered = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    new_covered |= edge_masks[selected_indices[j]]
                            
                            if new_covered == all_ones:
                                # Check if still not 2-colorable
                                new_edges = [all_edges[selected_indices[j]] for j in range(len(selected_indices)) if j != i]
                                if is_not_2_colorable(n, new_edges):
                                    selected_indices.pop(i)
                                    covered = new_covered
                                    improved = True
                                    break
                            
                    if len(selected_indices) < best_local_count:
                        best_local_count = len(selected_indices)
                        best_local_edges = [all_edges[i] for i in selected_indices]
                        
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                print(f"Found: n={n}, edges={best_count}")
                
    if best_edges is None:
        # Fallback
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
