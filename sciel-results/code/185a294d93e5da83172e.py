import time
import random
from collections import defaultdict, deque

_T0 = time.time()
_DEADLINE = 140.0
_MARGIN = 10.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_shortest_cycle_from(adj, start):
    """Find a shortest cycle through start in the graph adj (adjacency sets)."""
    # BFS from start's neighbors, looking for a path back to start
    # This finds the shortest cycle that includes start.
    
    # BFS from each neighbor of start
    best_cycle = None
    best_len = float('inf')
    
    for nb in sorted(adj[start]):
        # BFS from nb, trying to reach start
        visited = {start: -1, nb: nb}  # node -> parent (for path reconstruction)
        queue = deque([nb])
        found = False
        
        while queue and not found:
            u = queue.popleft()
            for v in sorted(adj[u]):
                if v == start:
                    # Found a cycle: start -> ... -> u -> start
                    # Reconstruct path from nb to u
                    path = [u]
                    cur = u
                    while cur != nb:
                        cur = visited[cur]
                        path.append(cur)
                    path.reverse()
                    # Cycle: start, nb, ..., u
                    cycle = [start] + path
                    if len(cycle) < best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    found = True
                    break
                if v not in visited:
                    visited[v] = u
                    queue.append(v)
        
        if found:
            break
    
    return best_cycle

def _decompose_into_cycles(adj):
    """
    Repeatedly find and remove cycles from an even-degree graph.
    Uses shortest-cycle heuristic to find good cycles.
    Returns list of cycles (each a list of vertices, no closing edge).
    """
    adj = {u: set(vs) for u, vs in adj.items() if vs}
    cycles = []
    
    while adj:
        # Find a vertex with degree >= 2
        start = None
        for u in adj:
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
        
        # Try to find a shortest cycle through start
        cycle = _find_shortest_cycle_from(adj, start)
        
        if cycle is None:
            # Fallback: try another vertex
            found = False
            for u in adj:
                if len(adj[u]) >= 2:
                    cycle = _find_shortest_cycle_from(adj, u)
                    if cycle is not None:
                        start = u
                        found = True
                        break
            if not found:
                break
        
        if cycle is None or len(cycle) < 3:
            break
        
        # Remove the cycle edges
        for i in range(len(cycle) - 1):
            a, b = cycle[i], cycle[i + 1]
            adj[a].discard(b)
            adj[b].discard(a)
        # Also remove the closing edge
        a, b = cycle[-1], cycle[0]
        adj[a].discard(b)
        adj[b].discard(a)
        
        # Remove empty vertices
        adj = {u: vs for u, vs in adj.items() if vs}
        
        cycles.append(cycle)
    
    return cycles

def _decompose(edges):
    """
    Decompose graph into cycles + singles.
    Strategy:
    1. Find a large even-degree subgraph (fix odd degrees by removing edges)
    2. Decompose the even-degree subgraph into cycles using shortest-cycle heuristic
    3. Remaining edges become singles
    4. Local search to merge singles into cycles
    """
    if not edges:
        return []
    
    present = set((min(u, v), max(u, v)) for u, v in edges)
    if not present:
        return []
    
    best_pieces = None
    best_score = len(present)  # all singles
    
    rng = random.Random(42)
    
    # Multiple restarts with different parity-fixing strategies
    for restart in range(30):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in present:
            adj[u].add(v)
            adj[v].add(u)
        
        # Fix odd degrees: find edges to remove so all degrees become even
        # This is a T-join problem. We'll use a greedy approach with some randomization.
        adj_copy = {u: set(vs) for u, vs in adj.items()}
        
        # Greedy: repeatedly find two odd vertices connected by an edge, remove it
        # With some randomization based on restart
        fixed = True
        while fixed:
            fixed = False
            odd = [u for u in adj_copy if adj_copy[u] and len(adj_copy[u]) % 2 == 1]
            if not odd:
                break
            
            # Find edges between odd vertices
            candidate_edges = []
            for i in range(len(odd)):
                for j in range(i + 1, len(odd)):
                    a, b = odd[i], odd[j]
                    if b in adj_copy[a]:
                        candidate_edges.append((a, b))
            
            if candidate_edges:
                # Pick randomly among candidates for diversity
                idx = rng.randrange(len(candidate_edges))
                a, b = candidate_edges[idx]
                adj_copy[a].discard(b)
                adj_copy[b].discard(a)
                fixed = True
            else:
                # No direct edge between odd vertices; remove a random edge incident to an odd vertex
                if odd:
                    u = odd[rng.randrange(len(odd))]
                    if adj_copy[u]:
                        v = next(iter(adj_copy[u]))
                        adj_copy[u].discard(v)
                        adj_copy[v].discard(u)
                        fixed = True
        
        # Verify all degrees are even
        for u in adj_copy:
            if adj_copy[u] and len(adj_copy[u]) % 2 != 0:
                # Clear problematic vertices
                adj_copy[u].clear()
        
        adj_copy = {u: vs for u, vs in adj_copy.items() if vs}
        
        # Decompose into cycles using shortest-cycle heuristic
        cycles = _decompose_into_cycles(adj_copy)
        
        # Build pieces
        used = set()
        pieces = []
        for c in cycles:
            valid = True
            for i in range(len(c) - 1):
                e = (min(c[i], c[i+1]), max(c[i], c[i+1]))
                if e not in present:
                    valid = False
                    break
            # Check closing edge
            e_close = (min(c[-1], c[0]), max(c[-1], c[0]))
            if e_close not in present:
                valid = False
            if valid and len(c) >= 3:
                pieces.append(c)
                for i in range(len(c) - 1):
                    used.add((min(c[i], c[i+1]), max(c[i], c[i+1])))
                used.add(e_close)
        
        # Remaining edges as singles
        for e in present:
            if e not in used:
                pieces.append([e[0], e[1]])
                used.add(e)
        
        score = len(pieces)
        if score < best_score:
            best_score = score
            best_pieces = pieces
        
        if best_score <= len(present) // 2:
            break
    
    if best_pieces is None:
        return [[u, v] for u, v in present]
    
    # Local search: try to convert pairs of singles into cycles
    pieces = best_pieces
    improved = True
    max_iter = 100
    
    while improved and time.time() < _DEADLINE - _MARGIN:
        improved = False
        
        # Identify single edges
        singles = {}
        for i, p in enumerate(pieces):
            if len(p) == 2:
                e = (min(p[0], p[1]), max(p[0], p[1]))
                singles[e] = i
        
        if len(singles) < 2:
            break
        
        # Build adjacency from all pieces
        adj = defaultdict(set)
        for p in pieces:
            for i in range(len(p) - 1):
                u, v = p[i], p[i+1]
                adj[u].add(v)
                adj[v].add(u)
            # Closing edge for cycles
            if len(p) >= 3:
                u, v = p[-1], p[0]
                adj[u].add(v)
                adj[v].add(u)
        
        single_list = list(singles.keys())
        for i in range(len(single_list)):
            if time.time() > _DEADLINE - _MARGIN:
                break
            for j in range(i + 1, len(single_list)):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                e1, e2 = single_list[i], single_list[j]
                u1, v1 = e1
                u2, v2 = e2
                
                # Check if the two single edges share a vertex
                pairs_to_check = []
                if u1 == u2 and v1 != v2:
                    pairs_to_check.append((v1, v2, u1))
                if u1 == v2 and v1 != u2:
                    pairs_to_check.append((v1, u2, u1))
                if v1 == u2 and u1 != v2:
                    pairs_to_check.append((u1, v2, v1))
                if v1 == v2 and u1 != u2:
                    pairs_to_check.append((u1, u2, v1))
                
                if not pairs_to_check:
                    continue
                
                for (a, b, s) in pairs_to_check:
                    if a == b:
                        continue
                    
                    # BFS from a to b avoiding e1 and e2
                    forbidden = {e1, e2}
                    visited = {a}
                    parent = {a: -1}
                    queue = deque([a])
                    found = False
                    
                    while queue and not found:
                        node = queue.popleft()
                        if node == b:
                            found = True
                            break
                        for nb in sorted(adj[node]):
                            e = (min(node, nb), max(node, nb))
                            if e in forbidden:
                                continue
                            if nb not in visited:
                                visited.add(nb)
                                parent[nb] = node
                                queue.append(nb)
                    
                    if found:
                        # Reconstruct path
                        path = [b]
                        cur = b
                        while cur != a:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        
                        # Cycle: s, a, ..., b, s
                        cycle = [s] + path
                        if len(set(cycle)) == len(cycle) and len(cycle) >= 3:
                            idx1 = singles[e1]
                            idx2 = singles[e2]
                            pieces[idx1] = None
                            pieces[idx2] = None
                            pieces.append(cycle)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    result = [p for p in pieces if p is not None]
    return result

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        if time.time() > _DEADLINE - _MARGIN:
            for spec2 in specs[len(results):]:
                e2 = _instance(*spec2)
                results.append([[u, v] for u, v in e2])
            break
    return results
