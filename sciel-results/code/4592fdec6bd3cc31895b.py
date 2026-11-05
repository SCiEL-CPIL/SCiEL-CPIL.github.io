"""Decompose three fixed instance graphs into few cycles + single edges."""
import time
import sys
from random import Random
from collections import defaultdict

_DEADLINE = 150.0
_START = None

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed() - 3.0

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _extract_cycles(adj, edge_set):
    """
    Extract all simple cycles from a graph where max degree <= 2.
    Returns list of cycles (lists of vertices).
    """
    cycles = []
    visited = set()
    for u in adj:
        if u in visited:
            continue
        if len(adj[u]) < 2:
            continue
        # Walk the component
        path = [u]
        visited.add(u)
        prev = None
        cur = u
        found_cycle = False
        while True:
            # Find next node
            nxts = [x for x in adj[cur] if x != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt in visited:
                if nxt == u:
                    # Cycle found
                    # The path is u ... cur
                    # Check if it's a valid cycle (len >= 3)
                    if len(path) >= 3:
                        cycles.append(list(path))
                    found_cycle = True
                # If nxt is some other visited node, it's not a simple cycle starting at u
                # In a max-degree-2 graph, if we hit a visited node that isn't start, 
                # it implies we entered a cycle not starting at u, or a path crossed?
                # But max degree 2 means components are disjoint cycles or paths.
                # So if we start at u (deg 2), we must be in a cycle.
                # If we hit a visited node, it must be u.
                break
            else:
                path.append(nxt)
                visited.add(nxt)
                prev = cur
                cur = nxt
                if len(path) > 1000: # Safety
                    break
    return cycles

def _decompose_2matching(edges, time_budget):
    """
    Stochastic 2-matching approach.
    1. Randomly remove edges until max degree <= 2.
    2. Extract cycles.
    3. Remaining edges are singles.
    Repeat to find best solution.
    """
    if not edges:
        return []
    
    best_pieces = [list(e) for e in edges]
    best_score = len(edges)
    
    adj_template = _build_adj(edges)
    n = max(max(e) for e in edges) + 1
    
    rng = Random(42)
    deadline = time.time() + time_budget
    attempts = 0
    
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        current_edges = set(edges)
        adj = defaultdict(set)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Remove edges to ensure max degree <= 2
        # We pick vertices with degree > 2 and remove incident edges
        # To be stochastic, we shuffle the order of processing or edge selection
        
        # Get list of vertices with degree > 2
        # We do this iteratively
        # Optimization: maintain a list of "bad" vertices
        bad_vertices = [v for v in adj if len(adj[v]) > 2]
        
        while bad_vertices:
            # Pick a random bad vertex
            v = rng.choice(bad_vertices)
            
            # Pick a random neighbor to remove edge to
            if not adj[v]:
                bad_vertices.remove(v)
                continue
            
            neighbors = list(adj[v])
            u = rng.choice(neighbors)
            
            # Remove edge (u, v)
            e = tuple(sorted((u, v)))
            if e in current_edges:
                current_edges.remove(e)
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Update bad_vertices list
                # If u or v now has degree <= 2, they might be removed from bad list
                # If they become > 2 (impossible, degree decreases), 
                # We just rebuild or update carefully.
                # For simplicity and speed (N is small), we can just check
                # if they are in bad_vertices and degree <= 2, remove them.
                # If they are not in bad_vertices but degree > 2 (impossible), add them.
                
                # Since degree only decreases, vertices only leave the "bad" set.
                # We don't need to add to bad set.
                
                if v in bad_vertices and len(adj[v]) <= 2:
                    bad_vertices.remove(v)
                if u in bad_vertices and len(adj[u]) <= 2:
                    bad_vertices.remove(u)
            else:
                # Edge already removed? Should not happen if adj is consistent.
                # But if it does, just update bad list
                if v in bad_vertices and len(adj[v]) <= 2:
                    bad_vertices.remove(v)
        
        # Now max degree <= 2. Extract cycles.
        cycles = _extract_cycles(adj, current_edges)
        
        # Count pieces
        num_cycles = len(cycles)
        # Edges in cycles:
        edges_in_cycles = set()
        for c in cycles:
            for i in range(len(c)):
                a, b = c[i], c[(i+1) % len(c)]
                edges_in_cycles.add(tuple(sorted((a, b))))
        
        # Remaining edges are singles
        remaining_edges = current_edges - edges_in_cycles
        # Also, edges that were removed are singles
        removed_edges = set(edges) - current_edges
        all_singles = remaining_edges | removed_edges
        
        total_pieces = num_cycles + len(all_singles)
        
        if total_pieces < best_score:
            best_score = total_pieces
            best_pieces = [list(c) for c in cycles] + [list(e) for e in all_singles]
            
    return best_pieces

def construct():
    global _START
    _START = time.time()
    sys.setrecursionlimit(10000)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    all_pieces = []
    
    # Allocate time budget for each graph
    # Total 150s. Leave 5s for overhead. 145s / 3 graphs ~ 48s each.
    # But 30-node graph is largest. Give more time to it?
    # Let's give equal time or proportional to N^2.
    # N=20, 25, 30. N^2: 400, 625, 900. Sum 1925.
    # T1 = 145 * 400/1925 ~ 30s
    # T2 = 145 * 625/1925 ~ 47s
    # T3 = 145 * 900/1925 ~ 67s
    
    budgets = [30.0, 47.0, 67.0]
    
    for i, spec in enumerate(specs):
        if _time_left() < 5.0:
            # Emergency fallback: all singles
            edges = _instance(*spec)
            all_pieces.append([list(e) for e in edges])
            continue
        
        edges = _instance(*spec)
        
        # Use the stochastic 2-matching decomposition
        pieces = _decompose_2matching(edges, budgets[i])
        
        all_pieces.append(pieces)
        
        # Optional: Local search improvement?
        # The 2-matching heuristic is already quite good.
        # We can try to "swap" edges back in if it forms a cycle?
        # That's complex. Let's stick to the robust fast method first.
        
    return all_pieces
