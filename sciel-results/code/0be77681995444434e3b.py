"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_k5(adj, n):
    """Check if graph with adjacency matrix adj (n x n bool) has a K5."""
    for combo in combinations(range(n), 5):
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                if not adj[combo[i]][combo[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _has_indep5(adj, n):
    """Check if graph has an independent set of size 5."""
    for combo in combinations(range(n), 5):
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                if adj[combo[i]][combo[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _is_valid(adj, n):
    """Check no K5 and no independent 5-set."""
    if _has_k5(adj, n):
        return False
    if _has_indep5(adj, n):
        return False
    return True

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _try_add_vertex(adj, n, deadline):
    """Try to add a new vertex at position n. Return new adj if valid, else None."""
    # Try all 2^n possible connection patterns? No, that's too many for large n.
    # Instead, try random patterns and also greedy patterns.
    # For efficiency, use the fact that adding vertex v with neighbor set S:
    # - Creates K5 if there's a K4 in S (since v connects to all in S)
    # - Creates independent 5-set if there's an independent 4-set in non-neighbors
    
    # Greedy approach: start with no edges to new vertex, add edges one by one
    # or start with all edges, remove one by one.
    
    # Try a few random connection patterns
    for attempt in range(200):
        if time.time() > deadline:
            return None
        new_adj = [row[:] for row in adj]
        new_adj.append([False] * n + [False])
        for i in range(n):
            new_adj[i].append(False)
        
        # Random pattern
        for i in range(n):
            if random.random() < 0.5:
                new_adj[n][i] = True
                new_adj[i][n] = True
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    # Try greedy: start with all connections, remove edges that cause violations
    new_adj = [row[:] for row in adj]
    new_adj.append([False] * n + [False])
    for i in range(n):
        new_adj[i].append(False)
    
    # Start with all edges to new vertex
    for i in range(n):
        new_adj[n][i] = True
        new_adj[i][n] = True
    
    if _is_valid(new_adj, n + 1):
        return new_adj
    
    # Remove edges one by one (random order)
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            return None
        new_adj[n][i] = False
        new_adj[i][n] = False
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    # Try the other way: start with no edges, add one by one
    new_adj = [row[:] for row in adj]
    new_adj.append([False] * n + [False])
    for i in range(n):
        new_adj[i].append(False)
    
    if _is_valid(new_adj, n + 1):
        return new_adj
    
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            return None
        new_adj[n][i] = True
        new_adj[i][n] = True
        if _is_valid(new_adj, n + 1):
            return new_adj
    
    return None

def _try_flip_edge(adj, n, deadline):
    """Try flipping a random edge and check validity. Return new adj if valid, else None."""
    if n < 2:
        return None
    i, j = random.sample(range(n), 2)
    new_adj = [row[:] for row in adj]
    new_adj[i][j] = not new_adj[i][j]
    new_adj[j][i] = new_adj[i][j]
    if _is_valid(new_adj, n):
        return new_adj
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s to leave safety margin
    
    # Start with Paley graph on 37 vertices
    adj = _paley_graph(37)
    n = 37
    best_n = n
    best_adj = [row[:] for row in adj]
    
    random.seed(42)
    
    # Phase 1: Try to add vertices
    while n < 46 and time.time() < deadline:
        new_adj = _try_add_vertex(adj, n, deadline)
        if new_adj is not None:
            adj = new_adj
            n += 1
            best_n = n
            best_adj = [row[:] for row in adj]
        else:
            # Try some edge flips to make room, then retry
            improved = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge(adj, n, deadline)
                if flipped is not None:
                    adj = flipped
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 2: Continue trying to add vertices with more aggressive search
    while n < 46 and time.time() < deadline:
        # Try multiple random patterns for new vertex
        added = False
        for trial in range(500):
            if time.time() > deadline:
                break
            new_adj = [row[:] for row in adj]
            new_adj.append([False] * n + [False])
            for i in range(n):
                new_adj[i].append(False)
            
            # Random with bias
            for i in range(n):
                if random.random() < 0.45:
                    new_adj[n][i] = True
                    new_adj[i][n] = True
            
            if _is_valid(new_adj, n + 1):
                adj = new_adj
                n += 1
                best_n = n
                best_adj = [row[:] for row in adj]
                added = True
                break
        
        if not added:
            # Try edge modifications
            progress = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge(adj, n, deadline)
                if flipped is not None:
                    adj = flipped
                    progress = True
            if not progress:
                break
    
    # Return the best valid graph found
    if best_n >= n:
        adj = best_adj
        n = best_n
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    
    return n, edges

# EVOLVE-BLOCK-END
