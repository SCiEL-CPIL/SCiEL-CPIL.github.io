"""Path decomposition via greedy + randomized restarts with maximum-matching polish."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        if time.time() > deadline - 5:
            results.append([list(e) for e in edges])
            continue
        result = _decompose(edges, deadline)
        results.append(result)
    return results


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    if not edges:
        return []
    best = None
    best_count = len(edges)
    lb = _lower_bound(edges)

    # Phase 1: randomized greedy restarts
    budget = deadline - time.time()
    n_restarts = max(40, min(200, int(budget * 20)))
    for a in range(n_restarts):
        if time.time() > deadline - 20:
            break
        rng = random.Random(a * 7919 + 13)
        paths = _greedy(edges, rng, deadline)
        if paths:
            paths = _merge(paths, edges)
            c = len(paths)
            if c < best_count:
                best_count = c
                best = [list(p) for p in paths]
                if c <= lb:
                    break

    # Phase 2: maximum-matching polish (exact optimal path count)
    if time.time() < deadline - 8:
        try:
            matched = _matching_decompose(edges)
            if matched is not None:
                mc = len(matched)
                if mc < best_count:
                    best_count = mc
                    best = matched
        except Exception:
            pass

    if best is None:
        best = [list(e) for e in edges]
    return best


def _lower_bound(edges):
    if not edges:
        return 0
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd = sum(1 for d in deg.values() if d % 2 == 1)
    return max(1, (odd + 1) // 2)


def _greedy(edges, rng, deadline):
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    paths = []
    while remaining:
        if time.time() > deadline:
            return None
        start = _pick_start(adj, deg, rng)
        if start is None:
            cands = [v for v in adj if adj[v]]
            if not cands:
                break
            start = rng.choice(cands)
        path = _extend(start, adj, rng, deadline)
        if not path or len(path) < 2:
            if adj[start]:
                path = [start, rng.choice(list(adj[start]))]
            else:
                e = next(iter(remaining))
                path = [e[0], e[1]]
        pe = set()
        for i in range(len(path) - 1):
            e = tuple(sorted((path[i], path[i+1])))
            if e in remaining:
                pe.add(e)
        if not pe:
            break
        for e in pe:
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            deg[u] -= 1
            deg[v] -= 1
        for v in list(adj):
            if not adj[v]:
                del adj[v]
        paths.append(path)
    return paths


def _pick_start(adj, deg, rng):
    odd = [v for v in adj if deg.get(v, 0) % 2 == 1 and adj[v]]
    if odd:
        return rng.choice(odd)
    cands = [v for v in adj if adj[v]]
    if not cands:
        return None
    ws = [max(1, deg.get(v, 0)) for v in cands]
    t = sum(ws)
    r = rng.random() * t
    c = 0
    for v, w in zip(cands, ws):
        c += w
        if r <= c:
            return v
    return cands[-1]


def _extend(start, adj, rng, deadline):
    path = [start]
    visited = {start}
    for _ in range(100):
        if time.time() > deadline:
            break
        cur = path[-1]
        nbrs = [n for n in adj[cur] if n not in visited]
        if not nbrs:
            break
        def sc(n):
            return sum(1 for x in adj[n] if x not in visited)
        nbrs.sort(key=sc)
        k = min(3, len(nbrs))
        choice = rng.choice(nbrs[:k])
        path.append(choice)
        visited.add(choice)
    return path


def _merge(paths, edges):
    if len(paths) <= 1:
        return paths
    es = set(edges)
    changed = True
    while changed:
        changed = False
        for i in range(len(paths)):
            for j in range(len(paths)):
                if i == j:
                    continue
                merged = False
                new_p = None
                if paths[i][-1] == paths[j][0]:
                    c = paths[i] + paths[j][1:]
                    if _simple(c, es):
                        new_p, merged = c, True
                if not merged and paths[i][-1] == paths[j][-1]:
                    c = paths[i] + paths[j][::-1][1:]
                    if _simple(c, es):
                        new_p, merged = c, True
                if not merged and paths[i][0] == paths[j][0]:
                    c = paths[i][::-1] + paths[j][1:]
                    if _simple(c, es):
                        new_p, merged = c, True
                if not merged and paths[j][-1] == paths[i][0]:
                    c = paths[j] + paths[i][1:]
                    if _simple(c, es):
                        new_p, merged = c, True
                if merged:
                    paths[i] = new_p
                    paths.pop(j)
                    changed = True
                    break
            if changed:
                break
    return paths


def _simple(path, es):
    if len(path) < 2:
        return True
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if tuple(sorted((path[i], path[i+1]))) not in es:
            return False
    return True


def _matching_decompose(edges):
    """
    Exact minimum path decomposition via maximum matching.
    
    Model: each edge e=(u,v) has two "half-edges" (endpoints): u-side and v-side.
    A path decomposition pairs consecutive half-edges at shared vertices.
    Maximum matching on the "half-edge graph" minimizes the number of paths.
    Number of paths = |E| - |M| where M is the maximum matching size.
    """
    n_edges = len(edges)
    if n_edges == 0:
        return []
    
    # Build half-edge graph
    # Each edge i=(u,v) gets two nodes: 2*i (u-side) and 2*i+1 (v-side)
    # Two half-edges are matchable if they share a vertex AND the two edges
    # are different AND the shared vertex is not the other endpoint (to keep paths simple)
    
    # Actually, the standard formulation:
    # Vertex set: for each edge (u,v), create two nodes representing the two ends.
    # Edge in matching graph between end_a_of_e1 and end_b_of_e2 if:
    #   - e1 != e2
    #   - they share a common vertex x
    #   - x is the "inner" vertex (not the endpoint that would create a cycle)
    # 
    # Simpler: use the line graph approach.
    # The path decomposition problem = find max matching in a derived graph.
    
    # Let me use a cleaner formulation:
    # For each edge e=(u,v), it has two "ports": port_u and port_v.
    # In a path, consecutive edges share a vertex. If edge e1=(a,x) and e2=(x,b) 
    # are consecutive in a path (going a-x-b), then port_x of e1 is matched to port_x of e2.
    # 
    # So: create a graph where nodes are (edge_id, vertex) for each incidence.
    # Two nodes (e1, x) and (e2, x) are connected if e1 != e2 and both are incident to x.
    # Maximum matching in this graph gives the pairing of consecutive edges.
    # Each matched pair merges two paths into one, so paths = |E| - |matching|.
    
    # Build incidence nodes
    inc_nodes = []  # list of (edge_idx, vertex)
    node_to_edge = {}  # node_id -> edge_idx
    node_to_vert = {}  # node_id -> vertex
    vert_to_nodes = defaultdict(list)  # vertex -> [node_ids]
    
    for ei, (u, v) in enumerate(edges):
        nid_u = len(inc_nodes)
        inc_nodes.append((ei, u))
        node_to_edge[nid_u] = ei
        node_to_vert[nid_u] = u
        vert_to_nodes[u].append(nid_u)
        
        nid_v = len(inc_nodes)
        inc_nodes.append((ei, v))
        node_to_edge[nid_v] = ei
        node_to_vert[nid_v] = v
        vert_to_nodes[v].append(nid_v)
    
    # Build matching graph: connect (e1, x) and (e2, x) if e1 != e2
    # This is a general graph (not bipartite in general), so we need Blossom.
    # For small graphs, we can implement a simple Blossom or use networkx-style approach.
    
    n_nodes = len(inc_nodes)
    
    # Build adjacency for the matching graph
    match_adj = defaultdict(set)
    for x, nodes in vert_to_nodes.items():
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                if node_to_edge[ni] != node_to_edge[nj]:
                    match_adj[ni].add(nj)
                    match_adj[nj].add(ni)
    
    # Find maximum matching using a simple augmenting path algorithm
    # For general graphs, we need Blossom, but let's implement a basic version
    # that works well for these sizes (n_nodes ~ 2*|E| ~ 2*40-60 = 80-120)
    
    matching = _max_matching_blossom(n_nodes, match_adj)
    
    if matching is None:
        return None
    
    # Reconstruct paths from matching
    # Each node (ei, x) that is matched to (ej, x) means edges ei and ej are consecutive at vertex x.
    # Unmatched nodes are path endpoints.
    
    # Build the path structure
    # For each edge, determine which of its two ports are matched and to what.
    edge_port_match = {}  # (edge_idx, vertex) -> matched_node_id or None
    
    for ni in range(n_nodes):
        matched_to = matching.get(ni)
        if matched_to is not None and matched_to > ni:
            ei = node_to_edge[ni]
            xi = node_to_vert[ni]
            ej = node_to_edge[matched_to]
            xj = node_to_vert[matched_to]
            edge_port_match[(ei, xi)] = (ej, xj)
            edge_port_match[(ej, xj)] = (ei, xi)
        elif matched_to is None:
            ei = node_to_edge[ni]
            xi = node_to_vert[ni]
            edge_port_match[(ei, xi)] = None
    
    # Now trace paths
    # A path starts at an unmatched port (endpoint) and follows matches.
    # At each edge, we enter from one port and exit from the other.
    
    # Build: for each edge, which port connects to which next edge
    # port_connect[(ei, vert)] = (ej, vert2) if matched, else None
    
    # Find all path starts (unmatched ports)
    starts = []
    for (ei, xi), target in edge_port_match.items():
        if target is None:
            starts.append((ei, xi))
    
    # Also need to handle: what if an edge's port is matched but we need to know
    # which vertex is the "entry" vs "exit"
    
    # For each edge ei=(u,v), its two ports are (ei,u) and (ei,v).
    # If (ei,u) is matched to (ej,x), then in the path, we go ... -> ej -> ei (entering ei at u).
    # Then we exit ei at v, so we look at (ei,v) to find the next edge.
    
    # Trace paths
    paths = []
    visited_edges = set()
    
    for start_ei, start_xi in starts:
        if start_ei in visited_edges:
            continue
        
        # Start of path: edge start_ei, entering at start_xi
        # The path vertex sequence starts with start_xi
        path_verts = [start_xi]
        current_ei = start_ei
        current_entry = start_xi
        
        while True:
            visited_edges.add(current_ei)
            # Find the other vertex of current_ei
            u, v = edges[current_ei]
            if current_entry == u:
                other = v
            else:
                other = u
            path_verts.append(other)
            
            # Check if (current_ei, other) is matched to continue
            next_match = edge_port_match.get((current_ei, other))
            if next_match is None:
                break
            next_ei, next_x = next_match
            if next_ei in visited_edges:
                break  # Shouldn't happen in a valid matching
            current_ei = next_ei
            current_entry = next_x
        
        if len(path_verts) >= 2:
            paths.append(path_verts)
    
    # Check if all edges are covered
    if len(visited_edges) != n_edges:
        # Some edges not covered - fallback
        return None
    
    # Verify all paths are simple and valid
    es = set(edges)
    for p in paths:
        if not _simple(p, es):
            return None
    
    return paths


def _max_matching_blossom(n, adj):
    """
    Maximum matching in a general graph using a simplified Blossom algorithm.
    For the sizes we deal with (n < 200), this should be fast enough.
    """
    if n == 0:
        return {}
    
    # Use a standard augmenting path approach with blossom contraction
    # For simplicity and reliability, use a well-known implementation pattern
    
    match = [-1] * n  # match[i] = j means i is matched to j
    parent = [-1] * n
    base = list(range(n))
    used = [False] * n
    blossom = [False] * n
    
    def lca(a, b):
        used_lca = [False] * n
        while True:
            a = base[a]
            used_lca[a] = True
            if match[a] == -1:
                break
            a = parent[match[a]]
        while True:
            b = base[b]
            if used_lca[b]:
                return b
            b = parent[match[b]]
    
    def mark_path(v, b, children):
        while base[v] != b:
            blossom[base[v]] = True
            blossom[base[match[v]]] = True
            parent[v] = children
            children = match[v]
            v = parent[match[v]]
    
    def augment(v):
        while match[v] != -1:
            pv = parent[v]
            ppv = match[pv]
            match[v] = pv
            match[pv] = v
            v = ppv
    
    def find_path(root):
        used = [False] * n
        parent = [-1] * n
        base = list(range(n))
        blossom = [False] * n
        queue = [root]
        used[root] = True
        
        while queue:
            u = queue.pop(0)
            for v in adj.get(u, []):
                if base[u] == base[v] or match[u] == v:
                    continue
                if v == root or (match[v] != -1 and parent[match[v]] != -1):
                    curbase = lca(u, v)
                    blossom = [False] * n
                    mark_path(u, curbase, v)
                    mark_path(v, curbase, u)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = curbase
                        if used[i]:
                            used[i] = False
                            queue.append(i)
                else:
                    if match[v] == -1:
                        parent[v] = u
                        augment(v)
                        return True
                    else:
                        w = match[v]
                        parent[w] = v
                        used[w] = True
                        queue.append(w)
        return False
    
    # Try to find augmenting paths from each unmatched vertex
    for i in range(n):
        if match[i] == -1:
            # Reset structures
            parent = [-1] * n
            base = list(range(n))
            blossom = [False] * n
            used = [False] * n
            
            success = find_path(i)
    
    # Build result
    result = {}
    for i in range(n):
        if match[i] != -1 and i < match[i]:
            result[i] = match[i]
            result[match[i]] = i
    return result

# EVOLVE-BLOCK-END
