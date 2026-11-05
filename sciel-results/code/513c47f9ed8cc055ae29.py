"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

_DEADLINE = 110  # seconds; safety margin under 120s limit

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(n, edge_list):
    adj = defaultdict(set)
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_cycles_greedy(n, edges):
    """
    Repeatedly find a cycle in the remaining edge set, remove its edges,
    and collect it as a piece. Continue until no cycle exists.
    Uses a heuristic to find cycles: for each vertex, do a BFS/DFS to find
    the shortest cycle through it, preferring high-degree vertices.
    """
    adj = _build_adjacency(n, edges)
    pieces = []
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    
    while True:
        # Check deadline
        if time.time() > _DEADLINE:
            break
        
        # Find a cycle in the remaining graph
        cycle = _find_one_cycle(n, remaining_edges, adj)
        if cycle is None:
            break
        
        # Remove cycle edges from remaining
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in remaining_edges:
                remaining_edges.discard(e)
            # Update adj
            if v in adj[u]:
                adj[u].discard(v)
            if u in adj[v]:
                adj[v].discard(u)
        
        pieces.append(cycle)
    
    # Remaining edges become single-edge pieces
    for u, v in remaining_edges:
        pieces.append([u, v])
    
    return pieces


def _find_one_cycle(n, remaining_edges, adj):
    """
    Find one cycle in the graph defined by remaining_edges.
    Strategy: try to find a short cycle by exploring from high-degree vertices.
    """
    if not remaining_edges:
        return None
    
    # Get vertices sorted by degree (descending) for better cycle finding
    vertices = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
    
    for start in vertices:
        if len(adj[start]) < 2:
            continue
        
        # BFS to find shortest cycle through 'start'
        cycle = _bfs_shortest_cycle(start, adj)
        if cycle is not None:
            # Verify all edges in cycle are in remaining_edges
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e not in remaining_edges:
                    valid = False
                    break
            if valid:
                return cycle
    
    # Fallback: try DFS from any vertex
    for start in vertices:
        if len(adj[start]) < 2:
            continue
        cycle = _dfs_cycle(start, adj)
        if cycle is not None:
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e not in remaining_edges:
                    valid = False
                    break
            if valid:
                return cycle
    
    return None


def _bfs_shortest_cycle(start, adj):
    """
    BFS from start to find the shortest cycle containing start.
    Returns list of vertices in the cycle, or None.
    """
    # BFS layer by layer from start
    # We want to find a path from a neighbor of start back to start
    # via a different neighbor
    
    # Get neighbors of start
    neighbors = list(adj[start])
    if len(neighbors) < 2:
        return None
    
    # For each neighbor, do BFS and see if we can reach another neighbor of start
    # Actually, standard approach: remove start, then for each pair of neighbors,
    # check if there's a path between them. The shortest such path + start forms a cycle.
    
    # Better: BFS from all neighbors simultaneously, tracking which neighbor we started from
    from collections import deque
    
    # dist[v] = (distance, start_neighbor)
    dist = {}
    for nb in neighbors:
        dist[nb] = (1, nb)
    
    queue = deque(neighbors)
    parent = {nb: None for nb in neighbors}
    start_nb = {nb: nb for nb in neighbors}
    
    while queue:
        u = queue.popleft()
        du, su = dist[u]
        
        for v in adj[u]:
            if v == start:
                # Found a cycle: start -> ... -> u -> start
                # Reconstruct path
                cycle = [start]
                # Go back from u to the start neighbor
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                cycle = [start] + path
                return cycle
            
            if v in dist:
                dv, sv = dist[v]
                if sv != su:
                    # Found a cycle: start -> ... -> v -> u -> ... -> start
                    # Actually: start -> su -> ... -> v and start -> sv -> ... -> u, and u-v edge
                    # Cycle: start, su, ..., v, u, ..., sv, start? No.
                    # Let me think again.
                    # We have path from su to v (length du... no)
                    # dist[nb] = (1, nb) means nb is at distance 1 from start via neighbor nb.
                    # So dist[v] = (dv, sv) means v is at distance dv from start, reached via initial neighbor sv.
                    # u is at distance du from start, reached via su.
                    # Edge u-v exists.
                    # Cycle: start -> (path from start to u via su) -> u -> v -> (path from v to start via sv reversed) -> start
                    # Path from start to u: start, su, ..., u (length du)
                    # Path from start to v: start, sv, ..., v (length dv)
                    # Cycle: start, su, ..., u, v, ..., sv, start
                    # Length = du + dv + 1 (edge u-v) ... wait, need to be careful.
                    
                    # Reconstruct path from start to u
                    path_u = [u]
                    cur = u
                    while cur != su:
                        cur = parent[cur]
                        path_u.append(cur)
                    path_u.reverse()  # su, ..., u
                    # path_u starts with su, ends with u
                    
                    # Reconstruct path from start to v
                    path_v = [v]
                    cur = v
                    while cur != sv:
                        cur = parent[cur]
                        path_v.append(cur)
                    path_v.reverse()  # sv, ..., v
                    # path_v starts with sv, ends with v
                    
                    # Cycle: start, su, ..., u, v, ..., sv, start
                    # = [start] + path_u + path_v
                    cycle = [start] + path_u + path_v
                    # Verify: last element is sv, and sv is a neighbor of start, so edge sv-start closes it.
                    return cycle
            
            if v not in dist:
                dist[v] = (du + 1, su)
                parent[v] = u
                start_nb[v] = su
                queue.append(v)
    
    return None


def _dfs_cycle(start, adj):
    """
    DFS to find any cycle containing start.
    """
    visited = {start: -1}
    parent = {start: None}
    
    def dfs(u):
        for v in adj[u]:
            if v == start:
                # Found cycle
                cycle = [start]
                cur = u
                while cur != start:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.reverse()
                # Remove duplicate start at end
                if cycle[0] == cycle[-1]:
                    cycle.pop()
                return cycle
            if v not in visited:
                visited[v] = visited[u] + 1
                parent[v] = u
                result = dfs(v)
                if result is not None:
                    return result
        return None
    
    return dfs(start)


def _decompose(n, edges):
    """
    Decompose the graph into cycles and single edges, minimizing piece count.
    """
    if not edges:
        return []
    
    return _find_cycles_greedy(n, edges)


def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() - start_time > _DEADLINE:
            # Return what we have plus trivial decompositions for remaining
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(n, p, seed)
        pieces = _decompose(n, edges)
        results.append(pieces)
    
    # Verify and fix if needed
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Check coverage and disjointness
        used_edges = set()
        valid = True
        for piece in results[i]:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                # Check all vertices distinct
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for j in range(len(piece)):
                    u, v = piece[j], piece[(j + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
            if time.time() - start_time > _DEADLINE:
                break
        
        if not valid or used_edges != edge_set:
            # Fallback: trivial decomposition
            results[i] = [list(e) for e in edges]
    
    return results
# EVOLVE-BLOCK-END
