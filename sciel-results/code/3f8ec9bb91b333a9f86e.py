import time
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _precompute_cycles(n):
    """Precompute all 4-cycles in K_n as sets of 4 edge indices."""
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    cycles = []
    for u, v, w, x in combinations(range(n), 4):
        # A 4-cycle on vertices u,v,w,x has 3 possible edge sets:
        # (u-v, v-w, w-x, x-u)
        # (u-v, v-x, x-w, w-u)
        # (u-w, w-v, v-x, x-u)
        
        # Cycle 1: u-v-w-x-u
        e1 = (min(u,v), max(u,v))
        e2 = (min(v,w), max(v,w))
        e3 = (min(w,x), max(w,x))
        e4 = (min(x,u), max(x,u))
        if all(e in edge_to_idx for e in [e1, e2, e3, e4]):
            cycles.append(frozenset([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]))
        
        # Cycle 2: u-v-x-w-u
        e1 = (min(u,v), max(u,v))
        e2 = (min(v,x), max(v,x))
        e3 = (min(x,w), max(x,w))
        e4 = (min(w,u), max(w,u))
        if all(e in edge_to_idx for e in [e1, e2, e3, e4]):
            cycles.append(frozenset([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]))
        
        # Cycle 3: u-w-v-x-u
        e1 = (min(u,w), max(u,w))
        e2 = (min(w,v), max(w,v))
        e3 = (min(v,x), max(v,x))
        e4 = (min(x,u), max(x,u))
        if all(e in edge_to_idx for e in [e1, e2, e3, e4]):
            cycles.append(frozenset([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]]))
    
    return all_edges, cycles

def _validate(edges, n, all_edges, cycles):
    """Check if the given edge coloring is valid (all 4-cycles are rainbow)."""
    # Build color lookup: edge_idx -> color
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    color_of = [-1] * len(all_edges)
    
    for u, v, c in edges:
        e = (min(u,v), max(u,v))
        if e in edge_to_idx:
            color_of[edge_to_idx[e]] = c
    
    # Check all 4-cycles
    for cycle in cycles:
        colors = [color_of[i] for i in cycle]
        if -1 in colors:
            # Not all edges present, skip
            continue
        if len(set(colors)) != 4:
            return False
    
    return True

def _greedy_build(n, num_colors, all_edges, cycles, rng):
    """Greedy construction with random edge ordering."""
    shuffled_edges = all_edges[:]
    rng.shuffle(shuffled_edges)
    
    adj = [[-1] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    edges = []
    
    for (u, v) in shuffled_edges:
        placed = False
        colors = list(range(num_colors))
        rng.shuffle(colors)
        
        for c in colors:
            ok = True
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        
                        if ca == c or cb == c or cv == c:
                            ok = False
                            break
                        if ca == cb or ca == cv or cb == cv:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
                placed = True
                break
    
    return edges

def _local_search(edges, n, num_colors, all_edges, cycles, rng, max_iters=100):
    """Try to improve the solution by adding edges or changing colors."""
    adj = [[-1] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    edge_set = set((min(u,v), max(u,v)) for u, v, c in edges)
    current_edges = list(edges)
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        
        # Try to add missing edges
        for (u, v) in all_edges:
            if (u, v) in edge_set:
                continue
            
            for c in range(num_colors):
                ok = True
                for a in neighbors[u]:
                    if a == v:
                        continue
                    ca = adj[u][a]
                    
                    for b in neighbors[v]:
                        if b == u or b == a:
                            continue
                        
                        if adj[a][b] != -1:
                            cb = adj[a][b]
                            cv = adj[b][v]
                            
                            if ca == c or cb == c or cv == c:
                                ok = False
                                break
                            if ca == cb or ca == cv or cb == cv:
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edge_set.add((u, v))
                    current_edges.append([u, v, c])
                    improved = True
                    break
    
    return current_edges

def construct():
    n = 20
    num_colors = 20
    
    # Precompute cycles
    all_edges, cycles = _precompute_cycles(n)
    
    best_edges = []
    best_count = 0
    
    attempts = 0
    
    while _time_left() > 10.0:
        rng = random.Random(attempts)
        
        # Greedy construction
        edges = _greedy_build(n, num_colors, all_edges, cycles, rng)
        
        # Local search to improve
        edges = _local_search(edges, n, num_colors, all_edges, cycles, rng, max_iters=50)
        
        # Validate
        if _validate(edges, n, all_edges, cycles):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
        else:
            # If invalid, fall back to just the greedy result
            if _validate(_greedy_build(n, num_colors, all_edges, cycles, rng), n, all_edges, cycles):
                greedy_edges = _greedy_build(n, num_colors, all_edges, cycles, rng)
                if len(greedy_edges) > best_count:
                    best_count = len(greedy_edges)
                    best_edges = greedy_edges[:]
        
        attempts += 1
        
    # Fallback if no edges were found (should not happen)
    if not best_edges:
        best_edges = [[0, 1, 0]]
        
    return best_edges
