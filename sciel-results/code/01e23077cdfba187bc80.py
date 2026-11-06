import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(left, right, adj):
    """
    Hopcroft-Karp maximum bipartite matching.
    left: list of left nodes
    right: list of right nodes
    adj: dict mapping left node -> list of right nodes
    Returns: dict mapping left node -> matched right node (or None)
    """
    pair_u = {u: None for u in left}
    pair_v = {v: None for v in right}
    dist = {}

    def bfs():
        queue = deque()
        found = False
        for u in left:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        while queue:
            u = queue.popleft()
            for v in adj.get(u, []):
                pu = pair_v[v]
                if pu is None:
                    found = True
                elif dist[pu] == float('inf'):
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj.get(u, []):
            pu = pair_v[v]
            if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching = 0
    while bfs():
        for u in left:
            if pair_u[u] is None:
                if dfs(u):
                    matching += 1

    return pair_u


def _decompose_via_matching(edges, n, deadline):
    """
    Decompose edges into cycles using maximum matching on the line graph.
    Each undirected edge (u,v) becomes two directed arcs: (u->v) and (v->u).
    In the line graph, arc (u->v) can be followed by arc (v->w) if they share vertex v.
    We build a bipartite graph: left = arcs as "tails", right = arcs as "heads".
    Edge in bipartite graph: left (u->v) connected to right (v->w) for all w != u.
    Maximum matching gives us a set of directed cycles (and paths).
    Cycles of length >= 3 in the original graph correspond to valid cycle pieces.
    """
    if not edges:
        return []

    # Create directed arcs: each undirected edge (u,v) with u<v becomes (u,v) and (v,u)
    arcs = []  # list of (u, v) directed
    arc_id = {}  # (u,v) -> index
    for u, v in edges:
        arcs.append((u, v))
        arcs.append((v, u))
        arc_id[(u, v)] = len(arcs) - 2
        arc_id[(v, u)] = len(arcs) - 1

    m_arcs = len(arcs)

    # Build bipartite graph for line graph
    # Left node i represents arc i as a "start"
    # Right node j represents arc j as an "end"
    # Left i connected to right j if arc i ends at vertex that arc j starts at
    # i.e., arcs[i][1] == arcs[j][0] and i != j (can't follow itself)
    
    # Group arcs by their starting vertex
    arcs_by_start = defaultdict(list)
    for i, (u, v) in enumerate(arcs):
        arcs_by_start[u].append(i)
    
    # Group arcs by their ending vertex
    arcs_by_end = defaultdict(list)
    for i, (u, v) in enumerate(arcs):
        arcs_by_end[v].append(i)

    # Build adjacency: left arc i -> right arcs j where arcs[i][1] == arcs[j][0]
    adj = {}
    for i, (u, v) in enumerate(arcs):
        # Find all arcs j that start at v (i.e., arcs[j][0] == v)
        candidates = arcs_by_start[v]
        # Exclude i itself (can't have (u,v) followed by (v,u) directly? Actually we can, but that's a 2-cycle which isn't valid)
        # Actually, (u,v) followed by (v,u) would be a 2-cycle in the line graph, which corresponds to going u->v->u, not a valid cycle
        # So we should exclude the reverse arc
        rev = arc_id.get((v, u))
        adj[i] = [j for j in candidates if j != i and j != rev]

    left_nodes = list(range(m_arcs))
    right_nodes = list(range(m_arcs))

    # Run Hopcroft-Karp
    pair_u = _hopcroft_karp(left_nodes, right_nodes, adj)

    # Extract cycles from the matching
    # A matched pair (i -> j) means arc i is followed by arc j in a cycle/path
    # Build next pointer
    next_arc = {}
    for i in range(m_arcs):
        j = pair_u.get(i)
        if j is not None:
            next_arc[i] = j

    # Find all cycles in the functional graph
    visited = set()
    cycles = []
    
    for i in range(m_arcs):
        if i in visited:
            continue
        # Follow the chain
        path = []
        cur = i
        while cur is not None and cur not in visited:
            if cur in path:
                # Found a cycle
                idx = path.index(cur)
                cycle_arcs = path[idx:]
                if len(cycle_arcs) >= 3:
                    cycles.append(cycle_arcs)
                break
            path.append(cur)
            visited.add(cur)
            cur = next_arc.get(cur)
        # Mark remaining path as visited
        for node in path:
            visited.add(node)

    # Convert cycle arcs to vertex cycles
    pieces = []
    used_arcs = set()
    
    for cycle_arcs in cycles:
        # Extract vertices
        verts = []
        valid = True
        for arc_idx in cycle_arcs:
            u, v = arcs[arc_idx]
            if not verts or verts[-1] != u:
                verts.append(u)
            # v should be next
        # Check validity: first vertex should equal last vertex's "next"
        # Actually, let's reconstruct properly
        verts = []
        for arc_idx in cycle_arcs:
            u, v = arcs[arc_idx]
            if not verts:
                verts.append(u)
            elif verts[-1] != u:
                valid = False
                break
            verts.append(v)
        
        if not valid or len(verts) < 4:  # Need at least 3 distinct vertices, so 4 in this representation
            continue
        
        # Check if it's a valid cycle: first vertex == last vertex
        if verts[0] != verts[-1]:
            continue
        
        # Extract unique vertices (remove the last which equals first)
        cycle_verts = verts[:-1]
        
        # Check all vertices are distinct
        if len(set(cycle_verts)) != len(cycle_verts):
            continue
        
        # Check all edges exist in the original graph
        all_edges_exist = True
        for i in range(len(cycle_verts)):
            u = cycle_verts[i]
            v = cycle_verts[(i + 1) % len(cycle_verts)]
            e = (min(u, v), max(u, v))
            if e not in set(edges):
                all_edges_exist = False
                break
        
        if all_edges_exist:
            # Mark these arcs as used
            for arc_idx in cycle_arcs:
                used_arcs.add(arc_idx)
                used_arcs.add(arc_id.get((arcs[arc_idx][1], arcs[arc_idx][0]), -1))  # reverse arc
            pieces.append(cycle_verts)

    # Find remaining edges not covered by cycles
    covered_edges = set()
    for piece in pieces:
        for i in range(len(piece)):
            u = piece[i]
            v = piece[(i + 1) % len(piece)]
            covered_edges.add((min(u, v), max(u, v)))

    remaining_edges = [e for e in edges if e not in covered_edges]
    
    # Add remaining as single edges
    for u, v in remaining_edges:
        pieces.append([u, v])

    return pieces


def _decompose_greedy(edges, n, seed, deadline):
    """Greedy fallback: find cycles greedily."""
    if not edges:
        return []
    
    remaining = set(edges)
    pieces = []
    adj = defaultdict(list)
    for u, v in remaining:
        adj[u].append(v)
        adj[v].append(u)

    while remaining:
        if time.time() > deadline:
            break
        
        # Find a cycle using DFS
        cycle = None
        for start in list(adj.keys()):
            if not adj[start]:
                continue
            # Try to find a cycle starting from start
            stack = [(start, [start])]
            visited_in_path = {start}
            found = False
            while stack and not found:
                node, path = stack.pop()
                for nb in adj[node]:
                    if nb == start and len(path) >= 3:
                        cycle = list(path)
                        found = True
                        break
                    if nb not in visited_in_path:
                        visited_in_path.add(nb)
                        stack.append((nb, path + [nb]))
            if cycle:
                break
        
        if cycle:
            # Extract cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            pieces.append(cycle)
        else:
            # No cycle, take an edge
            u, v = next(iter(remaining))
            remaining.discard((u, v))
            if v in adj[u]:
                adj[u].remove(v)
            if u in adj[v]:
                adj[v].remove(u)
            pieces.append([u, v])

    return pieces


def _decompose(edges, n, seed, time_limit=40.0):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    if not edges:
        return []

    start_time = time.time()
    deadline = start_time + time_limit

    # Try the matching-based approach first
    try:
        pieces = _decompose_via_matching(edges, n, deadline)
        if pieces:
            return pieces
    except Exception:
        pass

    # Fallback to greedy
    return _decompose_greedy(edges, n, seed, deadline)


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s kill
    per_graph_budget = total_budget / len(specs)
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        remaining_time = min(per_graph_budget, total_budget - (time.time() - start_time))
        if remaining_time <= 5:
            # Not enough time, just return all edges as single pieces
            results.append([[u, v] for u, v in edges])
            continue
        pieces = _decompose(edges, n, seed, time_limit=remaining_time)
        results.append(pieces)

    return results
