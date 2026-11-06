"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
import time
import random
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _remove_edge(adj, remaining, u, v):
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)


def _find_cycle_bfs(adj, n, seed, deadline=None):
    """Find a long cycle using BFS-based longest path heuristic."""
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts[:10]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Use BFS to explore paths, preferring longer branches
        queue = deque([(start, [start])])
        visited = {start}
        max_depth = 0
        best_path = [start]

        while queue:
            if deadline is not None and time.time() > deadline:
                break
            current, path = queue.popleft()
            if len(path) > max_depth:
                max_depth = len(path)
                best_path = list(path)

            # Explore neighbors, prioritizing those with more unvisited connections
            neighbors = list(adj[current])
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                if nb not in visited:
                    new_visited = visited | {nb}
                    new_path = path + [nb]
                    queue.append((nb, new_path))
                    visited.add(nb)
                elif nb == start and len(path) >= 3:
                    # Found a cycle
                    cycle = list(path)
                    if len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break

        # Check if best_path can form a cycle
        if len(best_path) >= 3 and start in adj[best_path[-1]]:
            if len(best_path) > len(best_cycle):
                best_cycle = list(best_path)
        else:
            # Try to find a cycle in best_path
            for i in range(len(best_path) - 2):
                for j in range(len(best_path) - 1, i + 1, -1):
                    if best_path[i] in adj[best_path[j]]:
                        cycle = best_path[i:j + 1]
                        if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                            best_cycle = list(cycle)
                        break

    return best_cycle


def _find_cycle_greedy(adj, n, seed, deadline=None):
    """Find a long cycle using greedy longest-path extension."""
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts[:10]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        path = [start]
        visited = {start}
        
        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break

    return best_cycle

def _find_cycle_random_walk(adj, n, seed, deadline=None):
    """Find a long cycle using random walks."""
    if not adj:
        return []
    
    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    
    # Try a few random walks
    for _ in range(10):
        if deadline is not None and time.time() > deadline:
            break
        start = rng.choice(vertices)
        if not adj[start]:
            continue
            
        path = [start]
        visited = {start}
        
        # Limit walk length to avoid infinite loops in dense graphs, 
        # though simple path length is bounded by n
        for _ in range(n): 
            u = path[-1]
            # Prefer unvisited neighbors
            unvisited_neighbors = [v for v in adj[u] if v not in visited]
            if not unvisited_neighbors:
                break
            
            # Randomly pick next
            v = rng.choice(unvisited_neighbors)
            path.append(v)
            visited.add(v)
            
        # Check if we can close the cycle
        if len(path) >= 3:
            u = path[-1]
            for i in range(len(path) - 1):
                if path[i] in adj[u]:
                    cycle = path[i:]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break
                    
    return best_cycle

def _find_best_cycle(adj, n, seed, deadline=None):
    """Find the best cycle using multiple strategies."""
    c1 = _find_cycle_greedy(adj, n, seed, deadline)
    c2 = _find_cycle_random_walk(adj, n, seed + 1000, deadline)
    c3 = _find_cycle_bfs(adj, n, seed + 2000, deadline)
    
    best = max([c1, c2, c3], key=len)
    return best

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    global_deadline = start_time + time_limit

    def greedy_decompose(remaining_edges, n, seed_offset=0):
        remaining = set(remaining_edges)
        pieces = []
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)

        cycle_counter = 0
        while remaining:
            if time.time() > global_deadline:
                for e in remaining:
                    pieces.append(list(e))
                break

            # Find a long cycle
            cycle = _find_best_cycle(adj, n, seed + seed_offset + cycle_counter, deadline=global_deadline)
            
            if cycle and len(cycle) >= 3:
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        valid = False
                        break
                
                if valid:
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        _remove_edge(adj, remaining, u, v)
                    pieces.append(cycle)
                    cycle_counter += 1
                else:
                    pass

            if not cycle or len(cycle) < 3:
                if remaining:
                    # Prioritize edges that are part of potential cycles (high degree endpoints)
                    scored = []
                    for e in remaining:
                        u, v = e
                        # Heuristic: prefer edges where both endpoints have high degree
                        score = len(adj.get(u, set())) + len(adj.get(v, set()))
                        scored.append((score, e))
                    scored.sort(reverse=True)
                    top = scored[:min(5, len(scored))]
                    _, e = random.choice(top)
                    u, v = e
                    _remove_edge(adj, remaining, u, v)
                    pieces.append([u, v])

        return pieces

    # Multi-start search
    # Increase restarts if time permits, but keep it safe
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > global_deadline:
            break

        pieces = greedy_decompose(edges, n, restart)
        current_count = len(pieces)

        if current_count < best_count:
            best_count = current_count
            best_pieces = pieces

    if best_pieces is None:
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_start = time.time()
    total_deadline = total_start + 140.0  # Safety margin for 150s limit
    
    for i, (n, p, seed) in enumerate(specs):
        remaining_time = total_deadline - time.time()
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        # Allocate time proportionally
        time_limit = min(remaining_time / (3 - i), 45.0)
        time_limit = max(time_limit, 5.0)
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=time_limit)
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
