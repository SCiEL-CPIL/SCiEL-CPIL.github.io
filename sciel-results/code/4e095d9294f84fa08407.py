"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque
from scipy.optimize import linear_sum_assignment

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_2factor(adj, n):
    """Find a maximum 2-factor (vertex-disjoint cycles) using bipartite matching.
    
    Models each edge as a left node that must be assigned to one of its two endpoint vertices.
    Each vertex can be assigned at most 2 edges (one 'in', one 'out' role).
    A perfect matching on the induced subgraph gives a 2-factor.
    """
    # Build edge list from current adjacency
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    
    m = len(edge_list)
    if m == 0:
        return []
    
    # Right side: 2n nodes (vertex v has 'in' slot 2v and 'out' slot 2v+1)
    # Left side: m edge nodes
    # Edge (u,v) can connect to: u_out (2u+1), v_in (2v), v_out (2v+1), u_in (2u)
    # But we need a consistent orientation: if edge (u,v) uses u_out then v must use v_in
    # Actually for 2-factor: each vertex has degree 2 in the subgraph.
    # We model: assign each edge to one of its endpoints as the "head" vertex.
    # Each vertex can be head of at most 1 edge and tail of at most 1 edge.
    # Simpler: create bipartite graph where left = edges, right = vertices (each vertex appears once)
    # An edge (u,v) connects to u and v. We want a matching where each vertex is matched to at most 2 edges.
    # This is a b-matching problem. Use the standard reduction: split each vertex into 2 copies.
    
    INF = 10**9
    # Cost matrix: m x 2n
    cost = [[INF] * (2 * n) for _ in range(m)]
    
    for i, (u, v) in enumerate(edge_list):
        # Edge i can be assigned to vertex u (either in or out slot) or vertex v
        cost[i][2 * u] = 0      # u as slot 0
        cost[i][2 * u + 1] = 0  # u as slot 1
        cost[i][2 * v] = 0      # v as slot 0
        cost[i][2 * v + 1] = 0  # v as slot 1
    
    # Find minimum cost assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Collect matched edges and their assigned vertex slots
    matched = {}  # edge_index -> (vertex, slot)
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] < INF:
            vertex = j // 2
            slot = j % 2
            matched[i] = (vertex, slot)
    
    if not matched:
        return []
    
    # Build the subgraph from matched edges
    # Each matched edge (u,v) is included in the 2-factor
    subgraph_adj = {v: set() for v in range(n)}
    for i, (vertex, slot) in matched.items():
        u, v = edge_list[i]
        subgraph_adj[u].add(v)
        subgraph_adj[v].add(u)
    
    # Extract cycles from the subgraph
    # Each vertex in the subgraph should have degree 0, 1, or 2
    # Vertices with degree 2 are in cycles, degree 1 are in paths (shouldn't happen in 2-factor)
    visited = set()
    cycles = []
    
    for start in range(n):
        if start in visited or len(subgraph_adj[start]) != 2:
            continue
        
        # Walk the cycle
        path = [start]
        visited.add(start)
        prev = -1
        curr = start
        while True:
            neighbors = list(subgraph_adj[curr])
            # Choose the next vertex (not the previous one, unless it's the last step back to start)
            if len(neighbors) == 2:
                if prev == -1:
                    nxt = neighbors[0]
                else:
                    nxt = neighbors[0] if neighbors[0] != prev else neighbors[1]
            else:
                break
            
            if nxt == start and len(path) >= 3:
                cycles.append(path)
                break
            if nxt in visited:
                break
            
            prev = curr
            curr = nxt
            path.append(curr)
            visited.add(curr)
        
        if len(path) >= 3:
            # Verify it's a valid cycle
            valid = True
            for i in range(len(path)):
                u = path[i]
                v = path[(i + 1) % len(path)]
                if v not in subgraph_adj[u]:
                    valid = False
                    break
            if valid:
                cycles.append(path) if path not in cycles else None
            else:
                # Don't double-add
                pass
    
    # Remove duplicates and invalid cycles
    unique_cycles = []
    seen = set()
    for c in cycles:
        key = frozenset(c)
        if key not in seen and len(c) >= 3:
            # Verify edges exist in original adj
            ok = True
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                if v not in adj[u]:
                    ok = False
                    break
            if ok:
                seen.add(key)
                unique_cycles.append(c)
    
    return unique_cycles

def _find_cycle_dfs(adj, n, start, rng, max_depth=60):
    """Find a cycle through start using randomized DFS."""
    visited = {start}
    path = [start]
    stack = [(start, list(adj[start]))]
    
    while stack:
        node, neighbors = stack[-1]
        if not neighbors:
            stack.pop()
            path.pop()
            continue
        nxt = neighbors.pop()
        if nxt == start and len(path) >= 3:
            return list(path)
        if nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            stack.append((nxt, list(adj[nxt])))
            if len(path) > max_depth:
                stack.pop()
                path.pop()
    
    return None

def _find_longest_cycle_greedy(adj, n, rng, max_attempts=30):
    """Find a long cycle using multiple randomized DFS attempts."""
    best_cycle = None
    best_len = 0
    
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    for _ in range(max_attempts):
        start = rng.choice(candidates)
        cycle = _find_cycle_dfs(adj, n, start, rng)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
    
    return best_cycle

def _decompose(edges, n, deadline):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    rng = Random(42)
    max_iterations = 100
    iters = 0
    no_progress = 0
    
    while remaining and iters < max_iterations:
        if time.time() > deadline:
            break
        
        # Try 2-factor approach first for larger graphs
        if len(remaining) >= 6:
            cycles = _find_2factor(adj, n)
            
            if cycles:
                # Use the best set of cycles from the 2-factor
                # Sort by length descending to extract longest first
                cycles.sort(key=lambda c: -len(c))
                
                for cycle in cycles:
                    if time.time() > deadline:
                        break
                    
                    # Validate cycle edges are in remaining
                    cycle_edges = set()
                    valid = True
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        if edge in remaining:
                            cycle_edges.add(edge)
                        else:
                            valid = False
                            break
                    
                    if not valid or len(cycle_edges) == 0:
                        continue
                    
                    # Remove cycle edges
                    for edge in cycle_edges:
                        remaining.discard(edge)
                        u, v = edge
                        adj[u].discard(v)
                        adj[v].discard(u)
                    
                    pieces.append(cycle[:])
                    iters += 1
                
                no_progress = 0
                continue
        
        # Fallback: greedy single cycle extraction
        cycle = _find_longest_cycle_greedy(adj, n, rng, max_attempts=20)
        
        if cycle is None or len(cycle) < 3:
            no_progress += 1
            if no_progress > 3:
                break
            iters += 1
            continue
        
        no_progress = 0
        
        # Validate
        cycle_edges = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                cycle_edges.add(edge)
            else:
                valid = False
                break
        
        if not valid:
            iters += 1
            continue
        
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(cycle[:])
        iters += 1
    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                s = specs[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n, deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
