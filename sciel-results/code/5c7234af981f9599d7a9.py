"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_triangle_free(adj, edge):
    """Check if adding edge (u,v) to graph with adjacency sets creates a triangle."""
    u, v = edge
    # Check if any common neighbor exists
    common = adj[u] & adj[v]
    return len(common) == 0

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    # adj[i] is a bitmask of neighbors of vertex i
    color = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        # Try colors 0, 1, 2
        used = 0
        for nb in range(n):
            if adj[v] & (1 << nb):
                if color[nb] != -1:
                    used |= (1 << color[nb])
        for c in range(3):
            if not (used & (1 << c)):
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = -1
        return False
    
    # Order vertices by degree (highest first) for better pruning
    order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    # Remap
    pos = {v: i for i, v in enumerate(order)}
    new_adj = [0] * n
    for v in range(n):
        for nb in range(n):
            if adj[v] & (1 << nb):
                new_adj[pos[v]] |= (1 << pos[nb])
    
    color2 = [-1] * n
    def backtrack2(idx):
        if idx == n:
            return True
        v = order[idx]
        used = 0
        for nb in range(n):
            if adj[v] & (1 << nb):
                if color2[nb] != -1:
                    used |= (1 << color2[nb])
        for c in range(3):
            if not (used & (1 << c)):
                color2[v] = c
                if backtrack2(idx + 1):
                    return True
                color2[v] = -1
        return False
    
    return backtrack2(0)

def _count_edges(adj, n):
    """Count edges in adjacency bitmask representation."""
    count = 0
    for i in range(n):
        count += bin(adj[i]).count('1')
    return count // 2

def construct():
    n = 16
    deadline = time.time() + 140  # Safety margin
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(all_edges)
    
    # Start with Grötzsch graph as initial solution
    # Vertices 0-4: cycle, 5-9: middle layer, 10: top, 11-15: isolated
    adj = [0] * n
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Build Grötzsch graph
    for i in range(5):
        add_edge(i, (i+1) % 5)
    for i in range(5):
        u, w = i, (i+1) % 5
        add_edge(u, 5+w)
        add_edge(w, 5+u)
    for i in range(5):
        add_edge(5+i, 10)
    
    best_edges = _count_edges(adj, n)
    best_adj = adj[:]
    
    # Backtracking search: try to add edges
    # We'll use a recursive approach with pruning
    def search(edge_idx, current_adj, current_count):
        nonlocal best_edges, best_adj
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we add all remaining edges, can we beat best?
        remaining = num_edges - edge_idx
        if current_count + remaining <= best_edges:
            return
        
        if edge_idx == num_edges:
            if current_count > best_edges:
                # Check 3-colorability
                if not _is_3_colorable(current_adj, n):
                    best_edges = current_count
                    best_adj = current_adj[:]
            return
        
        u, v = all_edges[edge_idx]
        
        # Option 1: Skip this edge
        search(edge_idx + 1, current_adj, current_count)
        
        # Option 2: Add this edge (if triangle-free)
        if not (current_adj[u] & current_adj[v]):
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            search(edge_idx + 1, new_adj, current_count + 1)
    
    # Use a smarter approach: greedy with local search
    # First, try to add edges greedily
    adj_current = best_adj[:]
    count = best_edges
    
    # Try adding edges in order of potential benefit
    # Sort edges by how many conflicts they have (fewer conflicts = better candidates)
    def edge_conflict_count(adj, u, v):
        """Count vertices that would form triangles if we add edge (u,v)."""
        conflicts = 0
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    # Greedy: repeatedly add the edge with fewest conflicts that doesn't create a triangle
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        best_edge = None
        best_conflicts = float('inf')
        
        for u, v in all_edges:
            if adj_current[u] & (1 << v):
                continue  # Edge already exists
            if adj_current[u] & adj_current[v]:
                continue  # Would create triangle
            conflicts = bin(adj_current[u] & adj_current[v]).count('1')
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj_current[u] |= (1 << v)
        adj_current[v] |= (1 << u)
        count += 1
        
        # Check if still 3-colorable
        if _is_3_colorable(adj_current, n):
            # Revert
            adj_current[u] &= ~(1 << v)
            adj_current[v] &= ~(1 << u)
            count -= 1
            break
        else:
            if count > best_edges:
                best_edges = count
                best_adj = adj_current[:]
    
    # Convert best_adj back to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
# EVOLVE-BLOCK-END
