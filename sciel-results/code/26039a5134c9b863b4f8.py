import numpy as np
import random
import time
from itertools import combinations

def has_triangle(vertices, adj):
    """
    Check if the set of vertices contains a triangle (3-clique).
    adj: 2D numpy array or list of lists (0/1).
    """
    m = len(vertices)
    if m < 3:
        return False
    # Map original indices to local indices for faster access? 
    # Or just use original indices.
    # Using original indices is fine.
    for i in range(m):
        u = vertices[i]
        for j in range(i + 1, m):
            v = vertices[j]
            if not adj[u][v]:
                continue
            for k in range(j + 1, m):
                w = vertices[k]
                if adj[u][w] and adj[v][w]:
                    return True
    return False

def has_independent_triangle(vertices, adj):
    """
    Check if the set of vertices contains an independent set of size 3.
    """
    m = len(vertices)
    if m < 3:
        return False
    for i in range(m):
        u = vertices[i]
        for j in range(i + 1, m):
            v = vertices[j]
            if adj[u][v]:
                continue
            for k in range(j + 1, m):
                w = vertices[k]
                if not adj[u][w] and not adj[v][w]:
                    return True
    return False

def try_greedy_construct(n, time_limit):
    """
    Attempt to construct a (5,5)-free graph on n vertices using greedy edge addition.
    Returns (n, edges) if successful, else None.
    """
    start = time.time()
    # Try multiple random orders
    for attempt in range(20):
        if time.time() - start > time_limit:
            break
            
        adj = [[0] * n for _ in range(n)]
        edges = []
        
        # Generate all pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        success = True
        for u, v in pairs:
            if time.time() - start > time_limit:
                return None
            
            # Get neighbors and non-neighbors
            # Note: u and v are not yet connected.
            # Neighbors of u: those w != v such that adj[u][w] == 1
            # Neighbors of v: those w != u such that adj[v][w] == 1
            
            # Common Neighbors
            cn = []
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] and adj[v][w]:
                    cn.append(w)
            
            # Common Non-Neighbors
            cnn = []
            for w in range(n):
                if w == u or w == v:
                    continue
                if not adj[u][w] and not adj[v][w]:
                    cnn.append(w)
            
            # Check if adding edge (u,v) creates a K5
            # This happens if cn contains a triangle
            creates_k5 = has_triangle(cn, adj)
            
            # Check if NOT adding edge (u,v) creates an I5
            # This happens if cnn contains an independent triangle
            creates_i5 = has_independent_triangle(cnn, adj)
            
            if creates_k5 and creates_i5:
                # Conflict: Adding edge creates K5, not adding creates I5.
                # We are stuck.
                success = False
                break
            elif creates_k5:
                # Must NOT add edge
                pass
            elif creates_i5:
                # MUST add edge
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append((u, v))
            else:
                # Safe either way. Choose randomly.
                if random.random() < 0.5:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    edges.append((u, v))
        
        if success:
            # Final verification just in case
            # (The greedy logic should guarantee it, but let's be safe)
            # We can skip full verification for speed, or do it once.
            # Let's do a quick check for K5/I5 count?
            # Actually, the greedy logic prevents *new* violations.
            # Since we started with empty graph (no violations), and we only add edges
            # if they don't create K5, and we are forced to add if not adding creates I5,
            # we should be safe.
            # However, "creates_i5" check ensures that if we DON'T add, we create I5.
            # So we ADD. Does adding create K5? We checked `creates_k5` is False.
            # So adding is safe.
            # If we DON'T add, we create I5. So we MUST add.
            # Logic holds.
            
            return n, edges
        
    return None

def construct():
    start_time = time.time()
    time_limit = 290
    
    # Fallback: Paley 37
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Try Greedy Construction for n = 42 down to 38
    for n in range(42, 37, -1):
        if time.time() - start_time > time_limit:
            break
            
        result = try_greedy_construct(n, time_limit - (time.time() - start_time))
        if result:
            return result

    # If no better graph found, return seed
    return get_paley_37()
