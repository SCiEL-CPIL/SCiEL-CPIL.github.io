import random
import time
import numpy as np

N = 20
C = 20

def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, we cannot have any 4-cycle u-v-x-y-u
            for y in adj_u:
                if y != v and y != x and x in adj[y]:
                    return False
            continue

        for y in adj_u:
            if y == v or y == x:
                continue
            if x in adj[y]:
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True

def _greedy_construct(seed):
    """Run a single greedy construction pass."""
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj, colour_of, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def _validate(edges):
    """Check if all 4-cycles in the graph are rainbow."""
    if not edges:
        return True
    
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    
    for u, v, c in edges:
        adj[u].add(v)
        adj[v].add(u)
        colour_of[u][v] = c
        colour_of[v][u] = c
    
    # Check all 4-cycles
    for i in range(N):
        for j in range(N):
            if i == j or j not in adj[i]:
                continue
            for k in range(N):
                if k == i or k == j or k not in adj[j]:
                    continue
                for l in range(N):
                    if l == i or l == j or l == k:
                        continue
                    if l not in adj[k] or i not in adj[l]:
                        continue
                    # Check it's a simple 4-cycle (no repeated vertices except start/end)
                    # i-j-k-l-i
                    if k in adj[i] or l in adj[j]:
                        continue  # Not a simple 4-cycle
                    
                    c1 = colour_of[i][j]
                    c2 = colour_of[j][k]
                    c3 = colour_of[k][l]
                    c4 = colour_of[l][i]
                    
                    if len({c1, c2, c3, c4}) != 4:
                        return False
    return True

def _local_search(edges, max_iters=500):
    """Local search to add edges or change colors."""
    if not edges:
        return edges
    
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    
    for u, v, c in edges:
        adj[u].add(v)
        adj[v].add(u)
        colour_of[u][v] = c
        colour_of[v][u] = c
    
    best_count = len(edges)
    best_edges = list(edges)
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        # Try adding a missing edge
        missing = [(i, j) for i in range(N) for j in range(i + 1, N) if j not in adj[i]]
        if missing:
            u, v = rng.choice(missing)
            for c in range(C):
                if _edge_ok(adj, colour_of, u, v, c):
                    adj[u].add(v)
                    adj[v].add(u)
                    colour_of[u][v] = c
                    colour_of[v][u] = c
                    best_count += 1
                    best_edges = [[a, b, colour_of[a][b]] for a in range(N) for b in range(a + 1, N) if b in adj[a]]
                    break
        
        # Try changing a color
        if edges:
            idx = rng.randrange(len(edges))
            u, v, old_c = edges[idx]
            for new_c in range(C):
                if new_c == old_c:
                    continue
                # Check if changing color is valid
                colour_of[u][v] = new_c
                colour_of[v][u] = new_c
                if _validate_from_state(adj, colour_of):
                    best_edges = [[a, b, colour_of[a][b]] for a in range(N) for b in range(a + 1, N) if b in adj[a]]
                    break
                else:
                    colour_of[u][v] = old_c
                    colour_of[v][u] = old_c
    
    return best_edges

def _validate_from_state(adj, colour_of):
    """Validate current state for rainbow 4-cycles."""
    for i in range(N):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k == i:
                    continue
                for l in adj[k]:
                    if l == i or l == j or l == k:
                        continue
                    if i not in adj[l]:
                        continue
                    if k in adj[i] or l in adj[j]:
                        continue
                    c1 = colour_of[i][j]
                    c2 = colour_of[j][k]
                    c3 = colour_of[k][l]
                    c4 = colour_of[l][i]
                    if len({c1, c2, c3, c4}) != 4:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    # Phase 1: Multiple greedy restarts
    seed = 42
    while time.time() < deadline - 30:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 10:
        improved = _local_search(best_edges, max_iters=200)
        if len(improved) > max_edges:
            best_edges = improved
            max_edges = len(improved)
    
    # Phase 3: More greedy restarts if time remains
    while time.time() < deadline:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    return best_edges
