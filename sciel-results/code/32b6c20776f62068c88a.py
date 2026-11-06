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

def _find_cycle_dfs(adj, n, start, target_len=None, deadline=None):
    """Find a cycle starting from 'start' using DFS. Tries to find long cycles."""
    best_cycle = []
    
    # Sort neighbors by degree to find longer paths
    neighbors = {v: sorted(adj[v], key=lambda x: -len(adj[x])) for v in adj}
    
    def dfs(u, path, visited):
        nonlocal best_cycle
        if deadline is not None and time.time() > deadline:
            return
        
        if len(path) > len(best_cycle) and path[0] in adj[u] and len(path) >= 3:
            best_cycle = list(path)
        
        if target_len and len(path) >= target_len:
            return

        for v in neighbors[u]:
            if v == path[0]:
                if len(path) >= 3:
                    candidate = list(path)
                    if len(candidate) > len(best_cycle):
                        best_cycle = candidate
            elif v not in visited:
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                path.pop()
                visited.remove(v)
            else:
                # Backtrack if we hit a visited node that isn't the start
                # This is a simple DFS, might not find the absolute longest, but is fast
                pass

    if start in adj:
        dfs(start, [start], {start})
        
    return best_cycle

def _find_top_k_cycles(adj, n, k, deadline=None):
    """Find top k distinct cycles by length."""
    if not adj:
        return []
    
    cycles = []
    seen_cycles = set()
    
    # Get vertices with degree > 0
    active_vertices = [v for v in adj if adj[v]]
    if not active_vertices:
        return []
    
    # Sort by degree descending to start from high-degree nodes
    active_vertices.sort(key=lambda v: -len(adj[v]))
    
    # Try from top few vertices
    for start in active_vertices[:min(10, len(active_vertices))]:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue
            
        cycle = _find_cycle_dfs(adj, n, start, deadline=deadline)
        if cycle:
            # Normalize cycle for uniqueness check
            # Find min vertex in cycle to rotate
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            
            # Check both directions? 
            # A cycle 1-2-3-1 is same as 1-3-2-1.
            # Let's just store the sorted edges or a canonical form.
            edges_in_cycle = frozenset((min(u,v), max(u,v)) for u, v in zip(cycle, cycle[1:]+cycle[:1]))
            if edges_in_cycle not in seen_cycles:
                seen_cycles.add(edges_in_cycle)
                cycles.append(cycle)
                
        # Also try to find a different cycle by forcing a different first step?
        # For now, just one DFS per start.
        
    # Sort by length descending
    cycles.sort(key=len, reverse=True)
    return cycles[:k]

def _beam_search(edges, n, K=10, M=5, deadline=None):
    """
    Beam search for decomposition.
    State: (remaining_edges_set, pieces_list, adj_dict)
    """
    start_time = time.time()
    
    # Initial state
    initial_adj = defaultdict(set)
    for u, v in edges:
        initial_adj[u].add(v)
        initial_adj[v].add(u)
        
    beam = [(set(edges), [], initial_adj)]
    
    best_state = None
    best_score = float('inf')
    
    while beam:
        if deadline is not None and time.time() > deadline:
            break
            
        next_beam = []
        
        for remaining, pieces, adj in beam:
            if not remaining:
                # Complete decomposition
                score = len(pieces)
                if score < best_score:
                    best_score = score
                    best_state = (remaining, pieces, adj)
                continue
                
            # Find top M cycles
            top_cycles = _find_top_k_cycles(adj, n, M, deadline=deadline)
            
            if not top_cycles:
                # No cycles found, must take single edges
                # To keep beam size manageable, just take one arbitrary edge
                # Or take the edge with highest degree endpoints
                if remaining:
                    # Pick edge with max degree sum
                    e = max(remaining, key=lambda e: len(adj.get(e[0], set())) + len(adj.get(e[1], set())))
                    new_remaining = set(remaining)
                    new_remaining.discard(e)
                    new_adj = {u: set(v) for u, v in adj.items()}
                    _remove_edge(new_adj, new_remaining, e[0], e[1])
                    next_beam.append((new_remaining, pieces + [[e[0], e[1]]], new_adj))
            else:
                for cycle in top_cycles:
                    # Verify cycle edges are in remaining
                    valid = True
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e not in remaining:
                            valid = False
                            break
                            
                    if valid:
                        new_remaining = set(remaining)
                        new_adj = {u: set(v) for u, v in adj.items()}
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            _remove_edge(new_adj, new_remaining, u, v)
                        
                        next_beam.append((new_remaining, pieces + [cycle], new_adj))
                        
        if not next_beam:
            break
            
        # Score states for beam pruning
        # Heuristic: len(pieces) + len(remaining)/4
        def score_state(state):
            rem, pcs, _ = state
            return len(pcs) + len(rem) / 4.0
            
        next_beam.sort(key=score_state)
        beam = next_beam[:K]
        
    # If beam search didn't find a complete solution, fallback to greedy on the best partial
    if best_state is None:
        if beam:
            rem, pcs, adj = beam[0]
            # Finish greedily
            while rem:
                if deadline is not None and time.time() > deadline:
                    break
                top_cycles = _find_top_k_cycles(adj, n, 1, deadline=deadline)
                if top_cycles:
                    cycle = top_cycles[0]
                    valid = True
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e not in rem:
                            valid = False
                            break
                    if valid:
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            _remove_edge(adj, rem, u, v)
                        pcs.append(cycle)
                    else:
                        e = next(iter(rem))
                        _remove_edge(adj, rem, e[0], e[1])
                        pcs.append([e[0], e[1]])
                else:
                    if rem:
                        e = next(iter(rem))
                        _remove_edge(adj, rem, e[0], e[1])
                        pcs.append([e[0], e[1]])
            best_state = (rem, pcs, adj)
        else:
            # Fallback to all single edges
            pcs = [[u, v] for u, v in edges]
            best_state = (set(), pcs, {})
            
    return best_state[1]

def _greedy_fallback(edges, n, deadline=None):
    """Simple greedy fallback if beam search fails or is too slow."""
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        cycle = _find_top_k_cycles(adj, n, 1, deadline=deadline)
        if cycle:
            valid = True
            for i in range(len(cycle[0])):
                u = cycle[0][i]
                v = cycle[0][(i + 1) % len(cycle[0])]
                e = (min(u, v), max(u, v))
                if e not in remaining:
                    valid = False
                    break
            if valid:
                for i in range(len(cycle[0])):
                    u = cycle[0][i]
                    v = cycle[0][(i + 1) % len(cycle[0])]
                    _remove_edge(adj, remaining, u, v)
                pieces.append(cycle[0])
            else:
                e = next(iter(remaining))
                _remove_edge(adj, remaining, e[0], e[1])
                pieces.append([e[0], e[1]])
        else:
            if remaining:
                e = next(iter(remaining))
                _remove_edge(adj, remaining, e[0], e[1])
                pieces.append([e[0], e[1]])
                
    return pieces

def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges."""
    if not edges:
        return []
        
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try Beam Search first
    # Adjust K and M based on time available
    # For small graphs, larger beam is better
    K = 20
    M = 5
    
    try:
        pieces = _beam_search(edges, n, K, M, deadline)
    except Exception:
        pieces = _greedy_fallback(edges, n, deadline)
        
    # If beam search didn't finish or returned bad result, try greedy with restarts
    # But beam search should be robust.
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_start = time.time()
    total_deadline = total_start + 140.0
    
    for i, (n, p, seed) in enumerate(specs):
        remaining_time = total_deadline - time.time()
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        time_limit = min(remaining_time / (3 - i), 45.0)
        time_limit = max(time_limit, 5.0)
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, seed, time_limit=time_limit)
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-END
