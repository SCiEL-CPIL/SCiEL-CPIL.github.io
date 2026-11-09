"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_3_colorable(adj, n, deadline=None):
    """Check if graph is 3-colorable using backtracking with bitmasks."""
    # Build neighbor bitmasks
    neighbors = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
    
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: bin(neighbors[v]).count('1'), reverse=True)
    
    color = [-1] * n
    used_colors = [0] * n  # bitmask of colors used by neighbors
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return None  # timeout sentinel
        if idx == n:
            return True
        v = order[idx]
        # Available colors: those not used by colored neighbors
        available = 7 & ~used_colors[v]
        if available == 0:
            return False
        # Try each available color
        while available:
            c = (available & -available).bit_length() - 1
            available &= available - 1
            color[v] = c
            # Update used_colors for neighbors
            update = []
            for u in range(n):
                if (neighbors[v] >> u) & 1:
                    old = used_colors[u]
                    used_colors[u] = old | (1 << c)
                    update.append((u, old))
            if backtrack(idx + 1):
                return True
            # Undo
            for u, old in update:
                used_colors[u] = old
            color[v] = -1
        return False
    
    result = backtrack(0)
    if result is None:
        # Timeout - assume colorable to be safe (we'll verify later if needed)
        return True
    return result

def _is_triangle_free_add(adj, u, v, n):
    """Check if adding edge (u,v) maintains triangle-freeness."""
    # No common neighbor
    nu = 0
    nv = 0
    for w in range(n):
        if adj[u][w]:
            nu |= (1 << w)
        if adj[v][w]:
            nv |= (1 << w)
    return (nu & nv) == 0

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n = 16
    
    # Start with Grötzsch graph (11 vertices, 20 edges) as base
    # Vertices 0-4: outer 5-cycle
    # Vertices 5-9: inner 5-cycle (star)
    # Vertex 10: center connected to 5-9
    adj = [[0]*n for _ in range(n)]
    
    # Outer cycle 0-1-2-3-4-0
    for i in range(5):
        u, v = i, (i + 1) % 5
        adj[u][v] = adj[v][u] = 1
    
    # Inner star: vertex 5+i connected to outer vertices i and i-1 (mod 5)
    for i in range(5):
        adj[i][5+i] = adj[5+i][i] = 1
        adj[(i-1)%5][5+i] = adj[5+i][(i-1)%5] = 1
    
    # Center vertex 10 connected to all inner vertices
    for i in range(5):
        adj[10][5+i] = adj[5+i][10] = 1
    
    # Count edges in base
    base_edges = sum(sum(row) for row in adj) // 2
    
    # Candidate edges to add: all edges involving vertices 11-15
    candidate_edges = []
    for u in range(11, 16):
        for v in range(u):
            candidate_edges.append((u, v))
    
    # Also include edges among 0-10 that might have been missed (none in Grötzsch)
    # Actually let's also try adding edges among 0-10 if possible
    for u in range(11):
        for v in range(u+1, 11):
            if not adj[u][v]:
                candidate_edges.append((u, v))
    
    # Sort by heuristic: prefer edges that connect to high-degree vertices
    # but also maintain flexibility
    def edge_heuristic(e):
        u, v = e
        du = sum(adj[u])
        dv = sum(adj[v])
        # Prefer edges between vertices with moderate degree
        return -(du + dv)
    
    candidate_edges.sort(key=edge_heuristic)
    
    best_edges_count = base_edges
    best_adj = [row[:] for row in adj]
    
    # Greedy addition with backtracking-like improvement
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if adj[u][v]:
            continue
        # Check triangle-freeness
        if not _is_triangle_free_add(adj, u, v, n):
            continue
        # Tentatively add edge
        adj[u][v] = adj[v][u] = 1
        # Check 3-colorability
        if not _is_3_colorable(adj, n, deadline):
            best_edges_count += 1
            best_adj = [row[:] for row in adj]
        else:
            # Revert
            adj[u][v] = adj[v][u] = 0
    
    # Try to improve with local search: remove an edge, try adding different edges
    # Do a few rounds of improvement
    for round_num in range(3):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each edge and see if we can add more
        current_edges = sum(sum(row) for row in adj) // 2
        
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                if time.time() > deadline:
                    break
                
                # Remove edge (u,v)
                adj[u][v] = adj[v][u] = 0
                
                # Try to add other edges
                added = 0
                for u2 in range(11, 16):
                    for v2 in range(u2):
                        if adj[u2][v2]:
                            continue
                        if not _is_triangle_free_add(adj, u2, v2, n):
                            continue
                        adj[u2][v2] = adj[v2][u2] = 1
                        if not _is_3_colorable(adj, n, deadline):
                            added += 1
                        else:
                            adj[u2][v2] = adj[v2][u2] = 0
                
                # Also try edges among 0-10
                for u2 in range(11):
                    for v2 in range(u2+1, 11):
                        if adj[u2][v2]:
                            continue
                        if not _is_triangle_free_add(adj, u2, v2, n):
                            continue
                        adj[u2][v2] = adj[v2][u2] = 1
                        if not _is_3_colorable(adj, n, deadline):
                            added += 1
                        else:
                            adj[u2][v2] = adj[v2][u2] = 0
                
                new_count = sum(sum(row) for row in adj) // 2
                
                if new_count > current_edges:
                    current_edges = new_count
                    improved = True
                else:
                    # Restore original state
                    adj[u][v] = adj[v][u] = 1
                    # Undo added edges
                    for u2 in range(11, 16):
                        for v2 in range(u2):
                            if adj[u2][v2] and not best_adj[u2][v2]:
                                adj[u2][v2] = adj[v2][u2] = 0
                    for u2 in range(11):
                        for v2 in range(u2+1, 11):
                            if adj[u2][v2] and not best_adj[u2][v2]:
                                adj[u2][v2] = adj[v2][u2] = 0
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if not improved:
            break
    
    # Final: try to add any remaining candidate edges
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if adj[u][v]:
            continue
        if not _is_triangle_free_add(adj, u, v, n):
            continue
        adj[u][v] = adj[v][u] = 1
        if not _is_3_colorable(adj, n, deadline):
            pass  # Keep it
        else:
            adj[u][v] = adj[v][u] = 0
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
