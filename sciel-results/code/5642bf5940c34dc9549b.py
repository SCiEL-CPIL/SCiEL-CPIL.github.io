"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
import random

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _get_2_core_edges(edges, n):
    """
    Returns the set of edges in the 2-core of the graph defined by 'edges'.
    The 2-core is the maximal subgraph where every vertex has degree at least 2.
    """
    if not edges:
        return set()
    
    # Build degree and adjacency
    deg = defaultdict(int)
    adj = defaultdict(set)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    # Queue for vertices with degree < 2
    from collections import deque
    q = deque([v for v in deg if deg[v] < 2])
    
    removed = set()
    while q:
        v = q.popleft()
        if v in removed:
            continue
        removed.add(v)
        for nb in adj[v]:
            if nb not in removed:
                deg[nb] -= 1
                if deg[nb] < 2:
                    q.append(nb)
                    
    # Reconstruct core edges
    core_edges = set()
    for u, v in edges:
        if u not in removed and v not in removed:
            core_edges.add((u, v))
            
    return core_edges

def _find_longest_cycle_in_subgraph(edges, n, max_search_time=0.05):
    """
    Finds a long cycle in the subgraph defined by 'edges'.
    Uses a heuristic DFS approach.
    Returns a list of vertices representing the cycle, or None if no cycle found.
    """
    if not edges:
        return None
        
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Only consider vertices with degree >= 2 in this subgraph
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
        
    best_cycle = None
    best_len = 0
    start_time = time.time()
    
    # Shuffle candidates to introduce randomness in tie-breaking or order
    random.shuffle(candidates)
    
    for start in candidates:
        if time.time() - start_time > max_search_time:
            break
            
        # DFS to find longest cycle starting/ending at start
        # We limit the depth to avoid exponential blowup, though N is small
        # A cycle must have length >= 3
        
        # State: (current_node, path, visited_set)
        # We want to find a path that returns to start with length > current best
        # To find the *longest*, we can use a simple DFS that explores all options
        # but prunes if current path length + max_possible_remaining < best_len
        
        # For small N, simple DFS with backtracking is okay if we limit it.
        # However, finding the *exact* longest cycle is NP-hard.
        # We will use a heuristic: DFS that prioritizes neighbors with higher degree
        # and tries to extend the path as long as possible.
        
        def dfs(node, path, visited):
            nonlocal best_cycle, best_len
            
            # Pruning: if we can't beat the best, stop
            # Max possible extension is len(candidates) - len(path)
            if len(path) + (len(candidates) - len(path)) <= best_len:
                return
                
            for nb in adj[node]:
                if nb == start:
                    if len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
                    
        # To avoid deep recursion issues or infinite loops in dense graphs,
        # we can just run a limited search.
        # Given N<=30, recursion depth is safe.
        # But to speed up, we can limit the number of branches or use a time check.
        
        # Let's just run the DFS. It might be slow for dense graphs, but 2-core of sparse G(n,p) is manageable.
        # To be safe, we'll add a global time check inside the loop if needed, 
        # but for now, we rely on the outer max_search_time.
        
        # Optimization: Order neighbors by degree descending to find long cycles faster?
        # Actually, to find *long* cycles, we might want to avoid closing early.
        # The DFS above will find *all* cycles starting at start.
        # This could be expensive.
        
        # Alternative: Just find *a* cycle, but prefer longer ones by random restarts?
        # Let's stick to the DFS for now but add a node limit to the DFS itself if it gets too big.
        # For N=30, it's likely fine.
        
        dfs(start, [start], {start})
        
    return best_cycle

def _decompose_graph(edges, n):
    pieces = []
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    
    # 1. Extract cycles from the 2-core iteratively
    while remaining_edges:
        core_edges = _get_2_core_edges(remaining_edges, n)
        if not core_edges:
            break
            
        # Find a long cycle in the core
        cycle = _find_longest_cycle_in_subgraph(core_edges, n, max_search_time=0.1)
        
        if cycle:
            pieces.append(cycle)
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining_edges.discard(e)
        else:
            # Should not happen if core_edges is not empty, but safety break
            break
            
    # 2. Remaining edges are single edges
    for e in remaining_edges:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    deadline = time.time() + 140.0
    
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        
        # Check time
        if time.time() > deadline - 10:
            # If we are running out of time, just return single edges for the rest
            results.append([[u, v] for u, v in edges])
            continue
            
        pieces = _decompose_graph(edges, n)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
