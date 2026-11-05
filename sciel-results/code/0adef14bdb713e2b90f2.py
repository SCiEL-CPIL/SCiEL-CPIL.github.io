"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import time
import random
import sys

# Deadline guard
_START_TIME = time.time()
DEADLINE = 110.0  # seconds; leave safety margin before 120s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    """Return a dict with 'n' and 'edges' for a K4-free graph with no independent 6-set."""
    best_n = 0
    best_edges = []
    
    # Start from known good constructions and try to grow them
    # Strategy:
    # 1. Try known small constructions as seeds
    # 2. Use local search / repair to grow the graph
    # 3. Use targeted repair to fix K4s and independent sets
    
    rng = random.Random(42)
    
    # Try multiple random seeds for diversity
    for seed in [42, 123, 999, 7, 31415, 271828]:
        if _time_left() < 5:
            break
        rng = random.Random(seed)
        n_found, edges_found = _try_grow(rng, initial_n=17)
        if n_found > best_n:
            best_n = n_found
            best_edges = edges_found
        if best_n >= 35:  # Exoo's record
            break
    
    if best_n == 0:
        # Fallback: Paley graph on 17 vertices
        squares = {x * x % 17 for x in range(1, 17)}
        best_n = 17
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    return {'n': best_n, 'edges': best_edges}

def _has_k4(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains a K4."""
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return True, (a, b, c, d)
    return False, None

def _has_independent_6(adj, n):
    """Check if graph contains an independent set of size 6."""
    # Independent set in G = clique in complement
    # Complement: ~adj
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a][b]:  # not independent edge
                continue
            for c in range(b + 1, n):
                if adj[a][c] or adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] or adj[b][d] or adj[c][d]:
                        continue
                    for e in range(d + 1, n):
                        if adj[a][e] or adj[b][e] or adj[c][e] or adj[d][e]:
                            continue
                        for f in range(e + 1, n):
                            if adj[a][f] or adj[b][f] or adj[c][f] or adj[d][f] or adj[e][f]:
                                continue
                            return True, (a, b, c, d, e, f)
    return False, None

def _is_valid(adj, n):
    """Check both K4-free and no independent 6-set."""
    has_k4, _ = _has_k4(adj, n)
    if has_k4:
        return False
    has_ind, _ = _has_independent_6(adj, n)
    if has_ind:
        return False
    return True

def _try_grow(rng, initial_n=17):
    """Try to build a large valid graph starting from a seed."""
    
    # Build initial graph: Paley graph on 17 vertices
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i][j] = True
                adj[j][i] = True
    
    # Verify initial graph is valid
    if not _is_valid(adj, n):
        # Try a different initial construction
        return _try_grow_alt(rng, initial_n)
    
    # Try to grow by adding vertices one at a time
    for attempt in range(50):
        if _time_left() < 3:
            break
        
        # Try to add a new vertex
        new_n = n + 1
        new_adj = [[False] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(i + 1, n):
                new_adj[i][j] = adj[i][j]
                new_adj[j][i] = adj[i][j]
        
        # Determine connections of new vertex to maximize validity
        # The new vertex must not create K4 with any 3 existing vertices
        # And must not create independent 6-set with any 5 existing vertices
        
        # Find vertices that would form K4 if connected
        # For each triple (a,b,c) that forms a triangle, new vertex cannot connect to all three
        # Also for each independent 5-set, new vertex must connect to at least one
        
        # Heuristic: connect to a random subset, then repair
        connected = [False] * n
        for i in range(n):
            # Connect with some probability
            connected[i] = rng.random() < 0.5
        
        for i in range(n):
            new_adj[n][i] = connected[i]
            new_adj[i][n] = connected[i]
        
        # Repair loop
        repaired = _repair(new_adj, new_n, rng, max_iters=200)
        if repaired is not None:
            adj = repaired
            n = new_n
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return n, edges

def _try_grow_alt(rng, initial_n):
    """Alternative growth strategy using random initialization."""
    # Start with a random graph and repair
    for trial in range(20):
        if _time_left() < 3:
            break
        
        n = initial_n
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    adj[i][j] = True
                    adj[j][i] = True
        
        repaired = _repair(adj, n, rng, max_iters=500)
        if repaired is not None:
            # Try to grow
            for grow_attempt in range(30):
                if _time_left() < 3:
                    break
                new_n = n + 1
                new_adj = [[False] * new_n for _ in range(new_n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        new_adj[i][j] = adj[i][j]
                        new_adj[j][i] = adj[i][j]
                for i in range(n):
                    new_adj[n][i] = rng.random() < 0.5
                    new_adj[i][n] = new_adj[n][i]
                
                repaired2 = _repair(new_adj, new_n, rng, max_iters=200)
                if repaired2 is not None:
                    adj = repaired2
                    n = new_n
            
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return n, edges
    
    return 0, []

def _repair(adj, n, rng, max_iters=500):
    """Repair a graph to be K4-free with no independent 6-set."""
    for it in range(max_iters):
        if _time_left() < 1:
            return None
        
        # Check for K4
        has_k4, k4 = _has_k4(adj, n)
        if has_k4:
            a, b, c, d = k4
            # Break the K4 by removing one edge randomly
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            pair = pairs[rng.randrange(len(pairs))]
            adj[pair[0]][pair[1]] = False
            adj[pair[1]][pair[0]] = False
            continue
        
        # Check for independent 6-set
        has_ind, ind6 = _has_independent_6(adj, n)
        if has_ind:
            # Add an edge within the independent set to break it
            a, b, c, d, e, f = ind6
            pairs = [(a,b), (a,c), (a,d), (a,e), (a,f),
                     (b,c), (b,d), (b,e), (b,f),
                     (c,d), (c,e), (c,f),
                     (d,e), (d,f),
                     (e,f)]
            pair = pairs[rng.randrange(len(pairs))]
            adj[pair[0]][pair[1]] = True
            adj[pair[1]][pair[0]] = True
            continue
        
        # No violations found - graph is valid
        return adj
    
    # Check final state
    if _is_valid(adj, n):
        return adj
    return None

# Entry point
def construct():
    """Return a dict with 'n' and 'edges' for a K4-free graph with no independent 6-set."""
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 110.0
    
    best_n = 0
    best_edges = []
    
    # Try multiple strategies
    for seed in [42, 123, 999, 7, 31415, 271828, 12345, 67890]:
        if time.time() - _START_TIME > 105:
            break
        rng = random.Random(seed)
        n_found, edges_found = _try_grow(rng, initial_n=17)
        if n_found > best_n:
            best_n = n_found
            best_edges = edges_found
        if best_n >= 35:
            break
    
    if best_n == 0:
        # Fallback: Paley graph on 17 vertices
        squares = {x * x % 17 for x in range(1, 17)}
        best_n = 17
        best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    return {'n': best_n, 'edges': best_edges}
