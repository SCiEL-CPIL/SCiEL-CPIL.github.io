import time
import random
from collections import deque

_T0 = time.time()
_DEADLINE = 145.0  # Wall-clock limit with margin (evaluator kills at 150s)

def _instance(n, p, seed):
    """
    Reproduce the instance generation logic from the task description.
    The seed program's _instance was:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)
    """
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    # The path edges
    path_edges = set()
    for i in range(n - 1):
        u, v = order[i], order[i + 1]
        path_edges.add((min(u, v), max(u, v)))
    
    # The random edges
    random_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                random_edges.add((u, v))
                
    all_edges = path_edges | random_edges
    return sorted(all_edges)

def _find_longest_cycle_greedy(adj, start, deadline):
    """
    Find a long cycle starting from 'start' using a greedy longest-path heuristic.
    Returns a list of vertices in the cycle or None.
    """
    # We want to find a path from start to some neighbor of start, then close the cycle.
    # Greedy strategy: At each step, move to the neighbor with the most unvisited neighbors (or highest degree).
    
    path = [start]
    visited = {start}
    
    # To close the cycle, we need to end at a neighbor of start.
    # We can treat neighbors of start as "targets".
    # Let's just find a long simple path starting at start.
    
    current = start
    while True:
        if time.time() > deadline:
            break
        
        # Get unvisited neighbors
        unvisited_neighbors = [w for w in adj[current] if w not in visited]
        
        if not unvisited_neighbors:
            break
        
        # Heuristic: Choose the neighbor with the highest degree (or most unvisited neighbors)
        # Sorting by degree is a common heuristic for longest path.
        # We can also prioritize neighbors that are connected to 'start' to close the loop early if the path is long enough?
        # No, we want the LONGEST path. So we should avoid closing early unless forced.
        # However, if we don't close, we can't form a cycle.
        # The greedy path might get stuck. 
        # A better heuristic: 
        # 1. If current has a neighbor that is 'start' and len(path) >= 3, we could close. 
        #    But we want to maximize length. So we should only close if we are stuck or if we decide to stop.
        # 2. Actually, we should just find a long path. Then check if the end is connected to start.
        #    If not, we might not have a cycle.
        #    To ensure we get a cycle, we can force the last step to be a neighbor of start if possible?
        #    Or, we can run the greedy path, and if it doesn't end at a neighbor of start, we might fail to close.
        #    
        #    Alternative: Run greedy path. If it ends at 'w' and (start, w) is an edge, we have a cycle.
        #    If not, we can try to backtrack or just discard this attempt and try another start.
        #    Given the speed, trying multiple starts is viable.
        
        # Let's refine the heuristic:
        # Prefer neighbors with high degree.
        # If we are at a vertex that is a neighbor of start, and we have already visited many nodes, we might want to close?
        # No, let's just go for the longest path.
        
        # Sort neighbors by degree (descending)
        unvisited_neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
        
        # Pick the best
        next_node = unvisited_neighbors[0]
        visited.add(next_node)
        path.append(next_node)
        current = next_node
        
        # Early exit if we've visited a lot of nodes?
        # If len(path) > n * 0.8, it's a very good cycle candidate.
        
    # Check if we can close the cycle
    if len(path) >= 3 and (start, path[-1]) in set(map(tuple, adj[start])):
         # Verify edge exists
         if path[-1] in adj[start]:
             return path
    return None

def _improve_cycle_2opt(cycle, adj, deadline):
    """
    Try to extend the cycle using 2-opt moves.
    A 2-opt move removes two edges (a,b) and (c,d) and adds (a,c) and (b,d) if they exist,
    potentially creating a longer cycle or a different structure.
    For a single cycle, we can look for "chords" or ways to merge.
    Actually, for a simple cycle, 2-opt isn't directly applicable in the same way as TSP.
    Instead, we can look for "extensions": 
    If we have a cycle C, and there is a path P outside C connecting two vertices of C,
    we can replace the segment of C between those vertices with P to get a longer cycle.
    This is complex. 
    A simpler improvement: 
    If we found a cycle, check if we can swap edges to include more vertices.
    Given time constraints, a simple local search on the path before closing might be better.
    
    Let's stick to the greedy path but with a small local search at the end?
    Or just rely on multiple restarts.
    """
    pass # We will skip complex 2-opt for now and rely on restarts.

def _decompose_graph(n, edges, deadline):
    """
    Decompose the graph into cycles and single edges.
    Returns a list of pieces.
    """
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # We will greedily extract cycles.
    # To get better results, we can try to extract the longest available cycle first.
    # We'll maintain a set of remaining edges.
    
    remaining_edges = set(edges)
    pieces = []
    
    # Map edge to index or just keep the set
    # For efficiency, we'll work with a copy of adjacency and remove edges.
    work_adj = [set(s) for s in adj]
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if any edges remain
        has_edge = False
        for v in range(n):
            if work_adj[v]:
                has_edge = True
                break
        if not has_edge:
            break
        
        # Try to find a long cycle
        # We'll try starting from several high-degree vertices
        best_cycle = None
        best_len = 0
        
        # Sort vertices by degree
        vertices = sorted(range(n), key=lambda v: len(work_adj[v]), reverse=True)
        
        # Try a few starts
        starts_to_try = vertices[:min(10, len(vertices))]
        
        for start in starts_to_try:
            if time.time() > deadline:
                break
            if len(work_adj[start]) < 2:
                continue
                
            # Run greedy cycle finder
            cycle = _find_longest_cycle_greedy(work_adj, start, deadline)
            if cycle and len(cycle) > best_len:
                best_cycle = cycle
                best_len = len(cycle)
                if best_len >= n: # Hamiltonian
                    break
                    
        if best_cycle:
            # Remove edges of the cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                work_adj[u].discard(v)
                work_adj[v].discard(u)
            pieces.append(best_cycle)
        else:
            # No more cycles found by the greedy heuristic
            # Break out and treat remaining edges as single edges
            break
            
    # Add remaining edges as single edges
    for v in range(n):
        for w in work_adj[v]:
            if v < w:
                pieces.append([v, w])
                
    return pieces

def _validate(adj, n, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    for p in pieces:
        k = len(p)
        if k == 2:
            e = tuple(sorted(p))
            if e in seen:
                return False
            seen.add(e)
        elif k >= 3:
            for i in range(k):
                e = tuple(sorted((p[i], p[(i + 1) % k])))
                if e in seen:
                    return False
                seen.add(e)
        else:
            return False
    all_edges = set()
    for v in range(n):
        for w in adj[v]:
            if v < w:
                all_edges.add((v, w))
    return seen == all_edges

def _solve_one(n, p, seed, time_budget):
    edges = _instance(n, p, seed)
    adj = _adj_from_edges(edges, n)
    m = len(edges)
    
    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in edges]
    best_count = m
    
    deadline = _T0 + _DEADLINE - 2 # Safety margin
    
    # Strategy: Run multiple restarts of the greedy decomposition
    # Each restart uses a slightly different heuristic or randomization
    # Since _find_longest_cycle_greedy is deterministic based on degree,
    # we can randomize the tie-breaking in the neighbor selection to get different paths.
    
    # Modify _find_longest_cycle_greedy to accept a random seed or tie-breaker?
    # Or just run the standard one. If it's deterministic, restarts won't help.
    # Let's add a bit of randomness to the neighbor selection in the greedy step.
    
    def _find_longest_cycle_greedy_random(adj, start, deadline, rng):
        path = [start]
        visited = {start}
        current = start
        while True:
            if time.time() > deadline:
                break
            unvisited_neighbors = [w for w in adj[current] if w not in visited]
            if not unvisited_neighbors:
                break
            # Sort by degree, but break ties randomly or just pick random from top K
            unvisited_neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
            # Pick from the top 3 to introduce variation
            k = min(3, len(unvisited_neighbors))
            next_node = rng.choice(unvisited_neighbors[:k])
            visited.add(next_node)
            path.append(next_node)
            current = next_node
            
        if len(path) >= 3 and path[-1] in adj[start]:
            return path
        return None

    best_pieces_found = None
    
    # Try a few random restarts
    for i in range(5): # 5 restarts
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(seed + i)
        # We need a fresh adjacency for each restart
        work_adj = [set(s) for s in adj]
        pieces = []
        
        while True:
            if time.time() > deadline - 2:
                break
            has_edge = False
            for v in range(n):
                if work_adj[v]:
                    has_edge = True
                    break
            if not has_edge:
                break
            
            best_cycle = None
            best_len = 0
            
            vertices = sorted(range(n), key=lambda v: len(work_adj[v]), reverse=True)
            starts_to_try = vertices[:min(10, len(vertices))]
            
            for start in starts_to_try:
                if time.time() > deadline - 2:
                    break
                if len(work_adj[start]) < 2:
                    continue
                cycle = _find_longest_cycle_greedy_random(work_adj, start, deadline, rng)
                if cycle and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len >= n:
                        break
                        
            if best_cycle:
                for i_idx in range(len(best_cycle)):
                    u = best_cycle[i_idx]
                    v = best_cycle[(i_idx + 1) % len(best_cycle)]
                    work_adj[u].discard(v)
                    work_adj[v].discard(u)
                pieces.append(best_cycle)
            else:
                break
                
        for v in range(n):
            for w in work_adj[v]:
                if v < w:
                    pieces.append([v, w])
                    
        if _validate(adj, n, pieces):
            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces

    return best_pieces

def _adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time per graph
    # Total deadline 145s. 
    # Graph 1: 20 nodes. Graph 2: 25 nodes. Graph 3: 30 nodes.
    # Larger graphs take longer.
    # Let's give 40s, 40s, 40s and keep 25s buffer?
    # Or just check deadline in _solve_one.
    
    for spec in specs:
        if time.time() - _T0 > _DEADLINE - 5:
            # Fallback: all single edges
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([[u, v] for (u, v) in edges])
            continue
        try:
            pieces = _solve_one(spec[0], spec[1], spec[2], time_budget=40.0)
        except Exception:
            edges = _instance(spec[0], spec[1], spec[2])
            pieces = [[u, v] for (u, v) in edges]
        results.append(pieces)
    return results
