import time
import random
import heapq
from collections import defaultdict

_T0 = time.time()
_DEADLINE = 145.0  # Wall-clock limit with margin (evaluator kills at 150s)

def _instance(n, p, seed):
    """
    Reproduce the instance generation logic from the task description.
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

def _adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

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

def _find_cycle_dfs(adj, start, deadline):
    """
    Find a cycle starting and ending at 'start' using DFS.
    Returns a list of vertices in the cycle or None.
    """
    # We want to find a path from start to a neighbor of start.
    # This is equivalent to finding a cycle through start.
    
    # To improve the chance of finding a long cycle, we can use a heuristic.
    # But first, let's just find *any* cycle through start.
    
    # Actually, finding the *longest* cycle is NP-hard.
    # We'll use a greedy heuristic that tries to extend the path as much as possible
    # while keeping the option to close the loop.
    
    # A better heuristic:
    # 1. Start at 'start'.
    # 2. At each step, choose a neighbor that:
    #    a. Is not visited.
    #    b. Has the highest degree (or most unvisited neighbors).
    #    c. If possible, prefer neighbors that are connected to 'start' if we are "stuck" elsewhere? No, that closes too early.
    #    d. We must ensure we can eventually return to 'start'.
    
    # A robust method for finding a *long* cycle in sparse graphs:
    # Use a randomized greedy approach with multiple restarts.
    
    # Let's implement a specific heuristic:
    # "Greedy Longest Path with Backtracking" is complex.
    # "Greedy Degree Heuristic":
    #    current = start
    #    path = [start]
    #    visited = {start}
    #    while True:
    #        neighbors = [w for w in adj[current] if w not in visited]
    #        if not neighbors:
    #            break
    #        # Sort by degree descending
    #        neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
    #        # To avoid getting stuck, maybe pick randomly from top 3?
    #        # Or just pick the best.
    #        next_node = neighbors[0]
    #        path.append(next_node)
    #        visited.add(next_node)
    #        current = next_node
    #    # Check if last node is connected to start
    #    if len(path) >= 3 and path[-1] in adj[start]:
    #        return path
    #    return None
    
    # The above greedy often fails to close the loop.
    # A better approach: Force the last step to be a neighbor of start?
    # We can try to find a path from start to a neighbor of start, visiting as many nodes as possible.
    
    # Let's try a different heuristic:
    # 1. Identify all neighbors of start.
    # 2. For each neighbor 't' of start, try to find a longest path from start to 't' in the graph minus the edge (start, t) ? No, that's too slow.
    
    # Let's stick to the greedy degree heuristic but with randomization and multiple attempts.
    # We'll return the best cycle found among several randomized greedy runs.
    
    best_cycle = None
    best_len = 0
    
    # Number of randomized attempts
    attempts = 10
    rng = random.Random(start) # Deterministic per start vertex
    
    for _ in range(attempts):
        if time.time() > deadline:
            break
            
        path = [start]
        visited = {start}
        current = start
        
        while True:
            if time.time() > deadline:
                break
                
            unvisited_neighbors = [w for w in adj[current] if w not in visited]
            
            if not unvisited_neighbors:
                break
            
            # Heuristic: Sort by degree, pick randomly from top K
            unvisited_neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
            k = min(3, len(unvisited_neighbors))
            next_node = rng.choice(unvisited_neighbors[:k])
            
            visited.add(next_node)
            path.append(next_node)
            current = next_node
            
        # Check if we can close the cycle
        if len(path) >= 3 and path[-1] in adj[start]:
            if len(path) > best_len:
                best_len = len(path)
                best_cycle = path[:]
                
    return best_cycle

def _decompose_graph(n, edges, deadline):
    """
    Decompose the graph into cycles and single edges.
    Uses a greedy peeling strategy: repeatedly find a long cycle, remove it, and continue.
    """
    # Build adjacency list
    adj = _adj_from_edges(edges, n)
    
    remaining_edges = set(edges)
    pieces = []
    
    # Work on a copy of adjacency
    work_adj = [set(s) for s in adj]
    
    # To prioritize finding long cycles, we can sort starting vertices by degree.
    # We'll maintain a list of vertices with degree >= 2.
    
    while True:
        if time.time() > deadline:
            break
            
        # Check if any edges remain with degree >= 2
        # Actually, we just need to find *any* cycle.
        # If no cycles exist, we are left with a forest (paths/cycles).
        # A graph with no cycles is a forest.
        
        # Find a vertex with degree >= 2
        candidate_starts = [v for v in range(n) if len(work_adj[v]) >= 2]
        
        if not candidate_starts:
            break
            
        # Sort candidates by degree descending to try high-degree vertices first
        candidate_starts.sort(key=lambda v: len(work_adj[v]), reverse=True)
        
        best_cycle = None
        best_len = 0
        
        # Try a subset of starts to save time
        # If the graph is small, try all. If large, try top K.
        max_starts = min(len(candidate_starts), 20)
        
        for start in candidate_starts[:max_starts]:
            if time.time() > deadline:
                break
                
            cycle = _find_cycle_dfs(work_adj, start, deadline)
            if cycle and len(cycle) > best_len:
                best_cycle = cycle
                best_len = len(cycle)
                
                # If we found a Hamiltonian cycle, we can't do better
                if best_len == n:
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
            # No more cycles found
            break
            
    # Add remaining edges as single edges
    for v in range(n):
        for w in work_adj[v]:
            if v < w:
                pieces.append([v, w])
                
    return pieces

def _solve_one(n, p, seed, time_budget):
    edges = _instance(n, p, seed)
    adj = _adj_from_edges(edges, n)
    m = len(edges)
    
    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in edges]
    best_count = m
    
    deadline = _T0 + _DEADLINE - 2 # Safety margin
    
    # Strategy: Run the greedy decomposition multiple times with different random seeds
    # to explore different cycle orderings.
    
    # We can also try to improve the solution by "swapping" edges between cycles?
    # For now, multiple restarts of the greedy peeling is a good structural change
    # from the previous single-run or simple greedy.
    
    for i in range(20): # 20 restarts
        if time.time() > deadline - 2:
            break
            
        # We need to randomize the process.
        # The current _find_cycle_dfs uses a fixed rng seed based on 'start'.
        # To get different decompositions, we can randomize the order in which
        # we pick starting vertices or the internal randomness of the cycle finder.
        
        # Let's modify the call to use a global random state or pass a seed.
        # For simplicity, let's just rely on the fact that the greedy choice
        # might be deterministic. We need to inject randomness into the selection
        # of the start vertex or the neighbor selection.
        
        # Let's create a wrapper that randomizes the start order.
        rng = random.Random(seed + i)
        
        # Shuffle the candidate starts?
        # The _decompose_graph function sorts by degree.
        # We can add a small random perturbation to the degree to break ties randomly.
        
        # To do this cleanly, let's write a slightly modified decomposer that accepts an rng.
        
        work_adj = [set(s) for s in adj]
        pieces = []
        
        while True:
            if time.time() > deadline - 2:
                break
                
            candidate_starts = [v for v in range(n) if len(work_adj[v]) >= 2]
            if not candidate_starts:
                break
            
            # Sort by degree, but add random noise to break ties or slightly reorder
            # This ensures different cycles are picked in different restarts.
            def sort_key(v):
                return (len(work_adj[v]) - rng.random() * 0.5)
            
            candidate_starts.sort(key=sort_key, reverse=True)
            
            best_cycle = None
            best_len = 0
            max_starts = min(len(candidate_starts), 20)
            
            for start in candidate_starts[:max_starts]:
                if time.time() > deadline - 2:
                    break
                    
                # Use the rng for internal randomness in cycle finding?
                # Currently _find_cycle_dfs uses random.Random(start).
                # We should pass the rng or a seed derived from it.
                
                # Let's refactor _find_cycle_dfs to accept an rng.
                
                cycle = _find_cycle_dfs_with_rng(work_adj, start, deadline, rng)
                if cycle and len(cycle) > best_len:
                    best_cycle = cycle
                    best_len = len(cycle)
                    if best_len == n:
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

def _find_cycle_dfs_with_rng(adj, start, deadline, rng):
    """
    Find a cycle starting and ending at 'start' using DFS with provided RNG.
    """
    best_cycle = None
    best_len = 0
    
    attempts = 10
    
    for _ in range(attempts):
        if time.time() > deadline:
            break
            
        path = [start]
        visited = {start}
        current = start
        
        while True:
            if time.time() > deadline:
                break
                
            unvisited_neighbors = [w for w in adj[current] if w not in visited]
            
            if not unvisited_neighbors:
                break
            
            unvisited_neighbors.sort(key=lambda w: len(adj[w]), reverse=True)
            k = min(3, len(unvisited_neighbors))
            next_node = rng.choice(unvisited_neighbors[:k])
            
            visited.add(next_node)
            path.append(next_node)
            current = next_node
            
        if len(path) >= 3 and path[-1] in adj[start]:
            if len(path) > best_len:
                best_len = len(path)
                best_cycle = path[:]
                
    return best_cycle

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() - _T0 > _DEADLINE - 5:
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
