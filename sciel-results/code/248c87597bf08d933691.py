import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # 10s margin before 120s limit
    best_edges = 0
    best_adj = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Precompute all 4-cycles (as sets of 4 edges)
    # A 4-cycle is u-v-x-y-u where u<v, and we canonicalize
    # Actually, let's just enumerate all quadruples (u,v,x,y) with u<v, x<y, 
    # u!=v, x!=y, and the four edges exist: (u,v), (v,x), (x,y), (y,u)
    # We need to check that these form a simple 4-cycle: all 4 vertices distinct
    
    # For efficiency, let's precompute for each potential edge (u,v) the list of
    # "constraints" - i.e., triples of other edges that with (u,v) form a 4-cycle.
    # When we add (u,v) with color c, for each such triple, the 4 colors must be distinct.
    
    # Let's build: for each edge (u,v) [u<v], list of (edge1, edge2, edge3) where
    # (u,v), edge1, edge2, edge3 form a 4-cycle.
    # edge1 = (v,x), edge2 = (x,y), edge3 = (y,u) for some x,y distinct from u,v and each other.
    
    # Actually, to be more efficient in checking, let's precompute for each pair (u,v)
    # the list of "witnesses" - each witness is a tuple (vx, xy, yu) representing the
    # other three edges in a 4-cycle with (u,v).
    
    # Let's index edges by a flat index for faster lookup
    # edge_idx[u][v] = flat index, or -1 if not present
    # But for simplicity, let's use the adjacency matrix approach but precompute witnesses.
    
    # Precompute all 4-cycles
    # A 4-cycle can be represented by 4 distinct vertices a,b,c,d in order.
    # The edges are (a,b), (b,c), (c,d), (d,a).
    # We need to enumerate all sets of 4 distinct vertices and all ways to order them into a cycle.
    # For 4 distinct vertices, there are 3 distinct cycles (up to rotation and reflection):
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    
    # Let's precompute: for each edge (u,v), the list of "opposite edge pairs" that form a 4-cycle.
    # If (u,v) is one edge, the other three edges are (v,x), (x,y), (y,u) for distinct x,y.
    
    # For each edge (u,v) with u<v, find all pairs (x,y) such that:
    # - x,y distinct from u,v and from each other
    # - (v,x), (x,y), (y,u) are all valid edge positions (i.e., we can check if they exist)
    # The 4-cycle is u-v-x-y-u.
    
    # Let's precompute for each (u,v) the list of (x,y) pairs.
    # But we also need to check that (v,x), (x,y), (y,u) are present in the graph.
    # So the witness check is: for each (x,y) in witnesses[(u,v)], if adj[v,x]!=-1 and adj[x,y]!=-1 and adj[y,u]!=-1,
    # then check that {c_uv, c_vx, c_xy, c_yu} has 4 distinct colors.
    
    # Let's precompute the witness lists.
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    # Check that (v,x), (x,y), (y,u) are valid edge positions
                    # i.e., v!=x, x!=y, y!=u (already checked) and edges are undirected
                    # We just need to record (x,y) as a potential witness
                    # The edges are (v,x), (x,y), (y,u)
                    wl.append((x, y))
            witness[(u,v)] = wl
            witness[(v,u)] = wl  # symmetric
    
    def can_add_edge_fast(adj, u, v, c):
        """Check if adding edge (u,v) with color c is valid using precomputed witnesses."""
        if u > v:
            u, v = v, u
        for x, y in witness[(u,v)]:
            # Check if edges (v,x), (x,y), (y,u) exist
            # Note: adj is symmetric, adj[a,b] = adj[b,a]
            c_vx = adj[v, x] if v != x else -1
            if c_vx == -1:
                continue
            c_xy = adj[x, y] if x != y else -1
            if c_xy == -1:
                continue
            c_yu = adj[y, u] if y != u else -1
            if c_yu == -1:
                continue
            # Check rainbow
            if len({c, c_vx, c_xy, c_yu}) < 4:
                return False
        return True
    
    def can_recolor_edge_fast(adj, u, v, new_c):
        """Check if recoloring edge (u,v) to new_c is valid."""
        if u > v:
            u, v = v, u
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        
        # Temporarily set
        adj[u, v] = new_c
        adj[v, u] = new_c
        
        valid = True
        for x, y in witness[(u,v)]:
            c_vx = adj[v, x] if v != x else -1
            if c_vx == -1:
                continue
            c_xy = adj[x, y] if x != y else -1
            if c_xy == -1:
                continue
            c_yu = adj[y, u] if y != u else -1
            if c_yu == -1:
                continue
            if len({new_c, c_vx, c_xy, c_yu}) < 4:
                valid = False
                break
        
        # Revert
        adj[u, v] = old_c
        adj[v, u] = old_c
        
        return valid
    
    # Try multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Start with a random tree
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            c = random.randint(0, n-1)
            adj[u, v] = c
            adj[v, u] = c
        
        # Try to add more edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            added = False
            colors = list(range(n))
            random.shuffle(colors)
            
            for c in colors:
                if can_add_edge_fast(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    added = True
                    break
            
            if not added:
                # Try to recolor an edge to make room
                # Try recoloring edges incident to u or v
                for neighbor in [u, v]:
                    for x in range(n):
                        if x == neighbor:
                            continue
                        if adj[neighbor, x] == -1:
                            continue
                        old_c = adj[neighbor, x]
                        for new_c in range(n):
                            if new_c == old_c:
                                continue
                            if can_recolor_edge_fast(adj, neighbor, x, new_c):
                                adj[neighbor, x] = new_c
                                adj[x, neighbor] = new_c
                                
                                # Try to add (u,v) again
                                added_after = False
                                for c2 in range(n):
                                    if can_add_edge_fast(adj, u, v, c2):
                                        adj[u, v] = c2
                                        adj[v, u] = c2
                                        added_after = True
                                        break
                                
                                if added_after:
                                    added = True
                                    break
                                else:
                                    adj[neighbor, x] = old_c
                                    adj[x, neighbor] = old_c
                        
                        if added:
                            break
                    if added:
                        break
        
        cnt = np.sum(adj != -1) // 2
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj.copy()
        
        if best_edges > 160:
            break
    
    # Convert to result format
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
