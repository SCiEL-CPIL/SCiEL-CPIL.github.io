"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_not_3_colorable_masks(masks, n):
    """Check if graph is NOT 3-colorable using bitmask-based backtracking with MRV."""
    degrees = [bin(masks[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    pos_to_vertex = [order[i] for i in range(n)]
    
    def backtrack(coloring):
        # MRV: pick uncolored vertex with fewest available colors
        best_idx = -1
        best_avail = 4
        for i in range(n):
            v = pos_to_vertex[i]
            if coloring[v] != -1:
                continue
            forbidden = 0
            m = masks[v]
            for u in range(n):
                if (m >> u) & 1:
                    if coloring[u] != -1:
                        forbidden |= (1 << coloring[u])
            avail = 3 - bin(forbidden).count('1')
            if avail < best_avail:
                best_avail = avail
                best_idx = i
                if avail == 0:
                    break
        
        if best_idx == -1:
            return True
        
        v = pos_to_vertex[best_idx]
        forbidden = 0
        m = masks[v]
        for u in range(n):
            if (m >> u) & 1:
                if coloring[u] != -1:
                    forbidden |= (1 << coloring[u])
        
        for c in range(3):
            if not (forbidden >> c) & 1:
                coloring[v] = c
                if backtrack(coloring):
                    return True
                coloring[v] = -1
        
        return False
    
    coloring = [-1] * n
    return not backtrack(coloring)

def _get_edges_from_masks(masks, n):
    edges = []
    for u in range(n):
        m = masks[u]
        v = 0
        while m:
            if m & 1:
                if u < v:
                    edges.append([u, v])
            m >>= 1
            v += 1
    return edges

def _max_independent_set_branch_and_bound(conflict_masks, num_nodes, deadline):
    """Find maximum independent set in conflict graph using branch and bound.
    
    conflict_masks[i] is a bitmask of nodes that conflict with node i.
    Returns the size of the maximum independent set and the set of nodes.
    """
    best_size = [0]
    best_set = [0]  # bitmask representation
    
    def branch(candidates, current_set, current_size):
        if time.time() > deadline:
            return
        if current_size + bin(candidates).count('1') <= best_size[0]:
            return
        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_set
            return
        
        # Branch on a node with minimum degree in the candidate subgraph
        # Pick node with fewest conflicts among candidates
        min_conflicts = num_nodes + 1
        chosen = -1
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            conflicts_in_candidates = bin(conflict_masks[node] & candidates).count('1')
            if conflicts_in_candidates < min_conflicts:
                min_conflicts = conflicts_in_candidates
                chosen = node
            temp ^= lsb
        
        if chosen == -1:
            return
        
        # Option 1: Include chosen node
        new_candidates = candidates & ~conflict_masks[chosen] & ~(1 << chosen)
        branch(new_candidates, current_set | (1 << chosen), current_size + 1)
        
        # Option 2: Exclude chosen node
        branch(candidates & ~(1 << chosen), current_set, current_size)
    
    branch((1 << num_nodes) - 1, 0, 0)
    return best_size[0], best_set[0]

def construct():
    deadline = time.time() + 140  # 140s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 possible edges
    
    # Map edge (u,v) with u < v to index
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    
    # Build conflict graph: two edges conflict if they form a triangle
    # Edges (a,b) and (c,d) conflict if they share a vertex and the third vertex completes a triangle
    # Actually, two edges conflict if adding both creates a triangle
    # (u,v) and (x,y) conflict if:
    # - They share exactly one vertex, say u=x, and v=y is adjacent to both (but we're building from scratch)
    # Wait, in the line graph of K16, two edges are adjacent if they share a vertex.
    # But we want triangle-free, so (u,v) and (v,w) conflict because they'd form a path u-v-w,
    # and if we also had (u,w) that's a triangle. But actually (u,v) and (v,w) don't directly form a triangle.
    # 
    # Let me reconsider: We want to select a set of edges such that no three form a triangle.
    # Two edges (u,v) and (u,w) don't conflict by themselves - they only conflict if (v,w) is also selected.
    # This is a 3-uniform hypergraph problem, not a simple graph MIS problem.
    #
    # Alternative approach: Use greedy + local search with better initialization.
    
    # Let's use a different structural approach: start from a known good structure
    # and use simulated annealing or iterated local search.
    
    import random
    random.seed(42)
    
    # Initialize with a random triangle-free graph
    def random_triangle_free_graph():
        masks = [0] * n
        # Randomly add edges while maintaining triangle-freeness
        possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if (masks[u] & masks[v]) == 0:  # No common neighbor, so no triangle
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks
    
    def count_edges(masks):
        return sum(bin(m).count('1') for m in masks) // 2
    
    # Try multiple random initializations and local search
    best_masks = [0] * n
    best_edge_count = 0
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        masks = random_triangle_free_graph()
        current_count = count_edges(masks)
        
        # Local search: try to add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (masks[u] >> v) & 1:
                        continue
                    if (masks[u] & masks[v]) != 0:
                        continue
                    # Try adding edge (u, v)
                    new_masks = masks[:]
                    new_masks[u] |= (1 << v)
                    new_masks[v] |= (1 << u)
                    if _is_not_3_colorable_masks(new_masks, n):
                        masks = new_masks
                        current_count += 1
                        improved = True
                        if current_count > best_edge_count:
                            best_edge_count = current_count
                            best_masks = masks[:]
        
        # Also try removing edges and re-adding different ones
        if time.time() < deadline:
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Remove a random edge
                edges = _get_edges_from_masks(masks, n)
                if not edges:
                    break
                u, v = random.choice(edges)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                current_count -= 1
                
                # Try to add different edges
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if (masks[u2] >> v2) & 1:
                            continue
                        if (masks[u2] & masks[v2]) != 0:
                            continue
                        new_masks = masks[:]
                        new_masks[u2] |= (1 << v2)
                        new_masks[v2] |= (1 << u2)
                        if _is_not_3_colorable_masks(new_masks, n):
                            masks = new_masks
                            current_count += 1
                            if current_count > best_edge_count:
                                best_edge_count = current_count
                                best_masks = masks[:]
    
    # Final verification
    if best_edge_count == 0:
        # Fallback: return empty graph (shouldn't happen)
        return []
    
    # Verify the best solution is valid
    if not _is_not_3_colorable_masks(best_masks, n):
        # Find any valid solution
        for trial in range(100):
            if time.time() > deadline:
                break
            masks = random_triangle_free_graph()
            if _is_not_3_colorable_masks(masks, n):
                best_masks = masks
                best_edge_count = count_edges(masks)
                break
    
    return _get_edges_from_masks(best_masks, n)
# EVOLVE-BLOCK-END
