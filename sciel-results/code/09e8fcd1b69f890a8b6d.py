import time
import random
from collections import defaultdict

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def _instance(n, p, seed):
    """Generate the fixed instance graph."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    # The 'path' edges
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    # The random edges
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _remove_edge(adj, remaining, u, v):
    """Remove edge (u,v) from remaining set and adjacency list."""
    e = (min(u, v), max(u, v))
    remaining.discard(e)
    if v in adj[u]:
        adj[u].remove(v)
    if u in adj[v]:
        adj[v].remove(u)

def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    # Shuffle vertices to randomize start order
    rng.shuffle(vertices)

    # Try multiple starting vertices: high degree + random
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start
        # Always extend to unvisited neighbor with highest degree
        # This tends to keep the path "open" for longer cycles
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            # Candidates are unvisited neighbors
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            
            # Sort by degree descending
            candidates.sort(reverse=True)
            
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            
            path.append(next_v)
            visited.add(next_v)

        # 1. Check if we can close the cycle directly at the start
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # 2. Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # We iterate i from 0 upwards, and j from end downwards to find the longest cycle
        # starting at path[i].
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            # If we already found a cycle longer than possible from this i, break?
            # Max possible length from i is len(path) - i.
            # If len(best_cycle) >= len(path) - i, we can't beat it with this i.
            if len(best_cycle) >= len(path) - i:
                break
                
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Found the longest cycle starting at i (largest j)

    return best_cycle

def _decompose_greedy(edges, n, seed, deadline=None):
    """
    Perform a single greedy decomposition pass.
    Returns a list of pieces.
    """
    if not edges:
        return []

    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    cycle_counter = 0

    while remaining:
        if deadline is not None and time.time() > deadline:
            break

        # Find a long cycle
        cycle = _find_longest_cycle_greedy(adj, n, seed + cycle_counter, deadline=deadline)
        
        if cycle and len(cycle) >= 3:
            # Verify cycle edges are in remaining (safety check)
            valid = True
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e not in remaining:
                    valid = False
                    break
                cycle_edges.append(e)
            
            if valid:
                # Remove edges from remaining and adj
                for u, v in cycle_edges:
                    _remove_edge(adj, remaining, u, v)
                pieces.append(cycle)
                cycle_counter += 1
                continue
            else:
                # Cycle invalid, fall through to single edge removal
                pass

        # No valid cycle found or cycle was invalid.
        # Remove a single edge.
        # Heuristic: Remove an edge incident to a vertex with high degree?
        # Or just pick an edge. Picking an edge from a "bridge" or leaf might help?
        # Actually, removing edges randomly or from high-degree nodes is fine.
        # Let's pick an edge randomly to allow different restarts to behave differently
        # if we were to randomize edge selection, but here the cycle finding is randomized.
        # For single edges, order doesn't matter for the final count, only for the path taken.
        # However, which edge we remove affects future cycles.
        # Removing edges incident to high-degree vertices might break up large components.
        # Let's just pick the first edge in the set (arbitrary but deterministic per set state).
        # To make restarts diverse, the cycle finding randomness is the main driver.
        
        e = next(iter(remaining))
        u, v = e
        _remove_edge(adj, remaining, u, v)
        pieces.append([u, v])

    return pieces

def _optimize_graph(edges, n, seed, time_limit):
    """
    Run multiple restarts of the greedy decomposition to find the best (fewest pieces) solution.
    """
    if not edges:
        return []

    best_pieces = None
    best_count = len(edges)
    start_time = time.time()
    
    # If time limit is too small, just do one pass
    if time_limit < 0.1:
        return _decompose_greedy(edges, n, seed, deadline=time.time() + time_limit)

    restart = 0
    while time.time() < start_time + time_limit:
        current_deadline = start_time + time_limit
        pieces = _decompose_greedy(edges, n, seed + restart * 1000, deadline=current_deadline)
        
        if len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces
            
        # Early exit if we've found a perfect decomposition?
        # Min pieces is at least E/3 (if all triangles) but usually higher.
        # If best_count is very low, we might stop.
        # But let's run out of time or restarts.
        restart += 1
        
        # Safety: don't loop too many times if it's extremely fast
        if restart > 1000:
            break

    if best_pieces is None:
        # Fallback: all edges as single pieces
        best_pieces = [[u, v] for u, v in edges]

    return best_pieces

def construct():
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    # Distribute time among graphs
    # Graphs are small, so even 20s each is plenty.
    # Total available: ~140s.
    # Let's give 40s to each, leaving 20s buffer.
    time_per_graph = 40.0
    
    for n, p, seed in specs:
        if time.time() > _DEADLINE - 5.0:
            # Not enough time left, return trivial solution for remaining
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(n, p, seed)
        
        # Calculate remaining time for this graph
        elapsed = time.time() - _START_TIME
        remaining_time = _DEADLINE - elapsed - 2.0 # 2s safety
        if remaining_time < 0:
            remaining_time = 0.1
            
        # Cap time per graph to ensure we don't starve later graphs
        alloc_time = min(time_per_graph, remaining_time)
        
        pieces = _optimize_graph(edges, n, seed, alloc_time)
        results.append(pieces)

    return results
