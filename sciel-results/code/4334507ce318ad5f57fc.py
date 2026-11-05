import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Minimum path decomposition via pairing-based local search.
        
        Representation: for each vertex, we partition its incident edges into
        pairs (path continuations) and singles (path endpoints).
        Number of paths = (number of singles) / 2.
        
        Local search: swap pairings at vertices to reduce path count while
        maintaining simplicity (no cycles).
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency: for each vertex, list of (neighbor, edge_id)
        adj = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        
        # Initial pairing: greedily pair up incident edges at each vertex
        # Minimize singles (path endpoints)
        def initial_pairing():
            """Create initial pairing: pair as many edges as possible at each vertex."""
            # pairing[v] = list of pairs of edge_ids, plus singles
            # Represent as: for each vertex, a dict mapping edge_id -> partner_edge_id or -1 if single
            partner = {}  # edge_id -> partner edge_id (or -1)
            paired = set()
            
            # Process vertices in order of degree (highest first) to maximize pairing
            vertices_by_deg = sorted(range(n_vertices), key=lambda v: -len(adj[v]))
            
            for v in vertices_by_deg:
                incident = [eid for (_, eid) in adj[v] if eid not in paired]
                # Greedily pair them up
                for i in range(0, len(incident) - 1, 2):
                    e1, e2 = incident[i], incident[i + 1]
                    partner[e1] = e2
                    partner[e2] = e1
                    paired.add(e1)
                    paired.add(e2)
                # Remaining edge (if odd count) is single
                if len(incident) % 2 == 1:
                    partner[incident[-1]] = -1
            
            # Ensure all edges have a partner entry
            for eid in range(len(edges)):
                if eid not in partner:
                    partner[eid] = -1
            
            return partner
        
        def count_paths(partner):
            """Count number of paths given a pairing."""
            singles = sum(1 for eid in range(len(edges)) if partner[eid] == -1)
            return singles // 2
        
        def build_paths(partner):
            """Reconstruct paths from pairing. Returns list of paths or None if cycle detected."""
            visited_edge = set()
            paths = []
            
            for start_eid in range(len(edges)):
                if start_eid in visited_edge:
                    continue
                
                # Start a path from this edge
                u, v = edges[start_eid]
                path = [u, v]
                visited_edge.add(start_eid)
                
                # Extend from v side
                cur_v = v
                while True:
                    # Find the partner edge at cur_v
                    cur_eid = start_eid if cur_v == v else None
                    # Actually, we need to track which edge we came in on
                    break
                
                # Let me redo this more carefully
                # A path is determined by following: at each vertex, if we enter via edge e1,
                # we leave via partner[e1] (if not single)
                
                # Start: pick an unvisited edge, determine its two endpoints
                # One endpoint is a "start" (single at that vertex) or we just pick a direction
                pass
            
            # Better approach: find path starts (vertices where an edge is single)
            # Then follow the pairing chain
            path_starts = []  # (vertex, edge_id) where edge is single at vertex
            for eid in range(len(edges)):
                u, v = edges[eid]
                if partner[eid] == -1:
                    # This edge is single at both u and v? No, single means it's an endpoint
                    # Actually partner[eid] = -1 means at BOTH endpoints, this edge is not paired
                    # Wait no. Let me reconsider.
                    # partner[eid] is the edge that eid is paired with at EACH vertex it touches.
                    # But an edge touches 2 vertices. At each vertex, the edge may or may not be paired.
                    # I need a different representation.
                    pass
            
            return None  # placeholder
        
        # Let me use a cleaner representation:
        # For each vertex v, pairing[v] is a list of pairs of edge_ids that are connected through v.
        # An edge is a "path end" at vertex v if it's not in any pair at v.
        
        def initial_pairing_v2():
            """
            pairing[v] = list of (eid1, eid2) pairs at vertex v.
            An edge eid is a path endpoint at v if eid not in any pair at v.
            Total path endpoints = sum over all v of (deg(v) - 2*len(pairing[v]))
            Number of paths = total_endpoints / 2
            """
            pairing = {v: [] for v in range(n_vertices)}
            paired_at = defaultdict(set)  # edge_id -> set of vertices where it's paired
            
            vertices_by_deg = sorted(range(n_vertices), key=lambda v: -len(adj[v]))
            
            for v in vertices_by_deg:
                incident = [eid for (_, eid) in adj[v]]
                # Sort by whether already paired elsewhere (prefer unpaired)
                incident.sort(key=lambda eid: (len(paired_at[eid]), eid))
                for i in range(0, len(incident) - 1, 2):
                    e1, e2 = incident[i], incident[i + 1]
                    pairing[v].append((e1, e2))
                    paired_at[e1].add(v)
                    paired_at[e2].add(v)
            
            return pairing, paired_at
        
        def count_paths_v2(pairing):
            """Count paths from pairing representation."""
            total_ends = 0
            for v in range(n_vertices):
                deg_v = len(adj[v])
                total_ends += deg_v - 2 * len(pairing[v])
            return total_ends // 2
        
        def build_paths_v2(pairing):
            """
            Reconstruct paths. Returns list of paths (vertex lists) or None if invalid (cycle).
            """
            # For each edge, determine at each endpoint whether it's paired and with which edge
            # edge_endpoint_partner[eid][vertex] = partner_eid or None
            edge_partner_at = {}
            for v in range(n_vertices):
                for e1, e2 in pairing[v]:
                    if e1 not in edge_partner_at:
                        edge_partner_at[e1] = {}
                    edge_partner_at[e1][v] = e2
                    if e2 not in edge_partner_at:
                        edge_partner_at[e2] = {}
                    edge_partner_at[e2][v] = e1
            
            # Find path starts: (vertex, edge_id) where edge is not paired at vertex
            starts = []
            for eid in range(len(edges)):
                u, v = edges[eid]
                is_single_u = (eid not in edge_partner_at) or (u not in edge_partner_at.get(eid, {}))
                is_single_v = (eid not in edge_partner_at) or (v not in edge_partner_at.get(eid, {}))
                if is_single_u:
                    starts.append((u, eid))
                if is_single_v:
                    starts.append((v, eid))
            
            # Follow paths from each start
            visited = set()  # set of (vertex, eid) visited
            paths = []
            
            for start_v, start_eid in starts:
                if (start_v, start_eid) in visited:
                    continue
                
                u, v = edges[start_eid]
                # Determine direction: we're at start_v, going through start_eid to the other end
                if start_v == u:
                    other = v
                else:
                    other = u
                
                path = [start_v, other]
                visited.add((start_v, start_eid))
                visited.add((other, start_eid))
                
                cur_v = other
                cur_eid = start_eid
                valid = True
                
                while True:
                    # At cur_v, find partner of cur_eid
                    partner_eid = edge_partner_at.get(cur_eid, {}).get(cur_v, None)
                    if partner_eid is None:
                        # End of path
                        break
                    
                    # Move to next vertex via partner_eid
                    pu, pv = edges[partner_eid]
                    if pu == cur_v:
                        next_v = pv
                    else:
                        next_v = pu
                    
                    if (next_v, partner_eid) in visited:
                        # Cycle detected!
                        valid = False
                        break
                    
                    path.append(next_v)
                    visited.add((next_v, partner_eid))
                    visited.add((cur_v, partner_eid))
                    
                    cur_v = next_v
                    cur_eid = partner_eid
                
                if valid:
                    paths.append(path)
            
            if not valid:
                return None
            
            # Verify all edges are covered
            used_edges = set()
            for p in paths:
                for i in range(len(p) - 1):
                    used_edges.add(tuple(sorted((p[i], p[i+1]))))
            
            if used_edges != edge_set:
                return None
            
            return paths
        
        def is_valid_pairing(pairing):
            """Check if pairing produces valid simple paths (no cycles)."""
            paths = build_paths_v2(pairing)
            return paths is not None
        
        def local_search(pairing, max_iter=5000):
            """
            Local search: try swapping pairings at vertices to reduce path count.
            A move: at vertex v, take two pairs (a,b) and (c,d), replace with (a,c) and (b,d).
            Or: unpair (a,b) and pair a with c, b with d where c,d were previously unpaired.
            """
            best_pairing = pairing
            best_count = count_paths_v2(pairing)
            best_paths = build_paths_v2(pairing)
            if best_paths is None:
                # Invalid, need to fix
                return pairing, best_count, None
            
            rng = random.Random(42)
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random vertex and try a swap
                v = rng.choice([x for x in range(n_vertices) if len(pairing[x]) >= 2])
                if len(pairing[v]) < 2:
                    continue
                
                # Pick two pairs to swap
                i, j = rng.sample(range(len(pairing[v])), 2)
                a, b = pairing[v][i]
                c, d = pairing[v][j]
                
                # Try swap: (a,b),(c,d) -> (a,c),(b,d)
                pairing[v][i] = (a, c)
                pairing[v][j] = (b, d)
                
                new_count = count_paths_v2(pairing)
                if new_count < best_count:
                    paths = build_paths_v2(pairing)
                    if paths is not None:
                        best_count = new_count
                        best_pairing = {vv: list(pairs) for vv, pairs in pairing.items()}
                        best_paths = paths
                    else:
                        # Revert
                        pairing[v][i] = (a, b)
                        pairing[v][j] = (c, d)
                elif new_count == best_count:
                    # Try to keep if it doesn't create cycles (for diversity)
                    paths = build_paths_v2(pairing)
                    if paths is not None:
                        # Accept with some probability for exploration
                        if rng.random() < 0.3:
                            best_pairing = {vv: list(pairs) for vv, pairs in pairing.items()}
                            best_paths = paths
                    else:
                        pairing[v][i] = (a, b)
                        pairing[v][j] = (c, d)
                else:
                    # Revert
                    pairing[v][i] = (a, b)
                    pairing[v][j] = (c, d)
                
                # Also try: unpair a pair and pair with singles
                # At vertex v, if there are singles, try to pair them
                incident_eids = [eid for (_, eid) in adj[v]]
                paired_eids = set()
                for e1, e2 in pairing[v]:
                    paired_eids.add(e1)
                    paired_eids.add(e2)
                singles = [eid for eid in incident_eids if eid not in paired_eids]
                
                if len(singles) >= 2 and len(pairing[v]) >= 1:
                    # Unpair one pair and pair its members with singles
                    pi = rng.choice(range(len(pairing[v])))
                    x, y = pairing[v][pi]
                    s1, s2 = rng.sample(singles, 2)
                    
                    # Remove pair (x,y), add (x,s1) and (y,s2)
                    pairing[v].pop(pi)
                    pairing[v].append((x, s1))
                    pairing[v].append((y, s2))
                    
                    new_count = count_paths_v2(pairing)
                    if new_count < best_count:
                        paths = build_paths_v2(pairing)
                        if paths is not None:
                            best_count = new_count
                            best_pairing = {vv: list(pairs) for vv, pairs in pairing.items()}
                            best_paths = paths
                        else:
                            # Revert
                            pairing[v].append((x, y))
                            pairing[v].remove((x, s1))
                            pairing[v].remove((y, s2))
                    else:
                        # Revert
                        pairing[v].append((x, y))
                        pairing[v].remove((x, s1))
                        pairing[v].remove((y, s2))
            
            return best_pairing, best_count, best_paths
        
        # Run initial + local search with multiple restarts
        best_count = float('inf')
        best_paths = None
        
        for restart in range(20):
            if time.time() > deadline - 3:
                break
            
            rng = random.Random(restart * 1000 + 7)
            
            # Create initial pairing with some randomness
            pairing = {v: [] for v in range(n_vertices)}
            paired_at = defaultdict(set)
            
            vertices_order = list(range(n_vertices))
            rng.shuffle(vertices_order)
            # Sort by degree descending with some randomness
            vertices_order.sort(key=lambda v: (-len(adj[v]), rng.random()))
            
            for v in vertices_order:
                incident = [eid for (_, eid) in adj[v]]
                rng.shuffle(incident)
                # Prefer edges not yet paired elsewhere
                incident.sort(key=lambda eid: (len(paired_at[eid]), rng.random()))
                for i in range(0, len(incident) - 1, 2):
                    e1, e2 = incident[i], incident[i + 1]
                    pairing[v].append((e1, e2))
                    paired_at[e1].add(v)
                    paired_at[e2].add(v)
            
            # Local search
            pairing, count, paths = local_search(pairing, max_iter=3000)
            
            if paths is not None and count < best_count:
                best_count = count
                best_paths = paths
            
            # If we hit the theoretical lower bound, stop
            odd_vertices = sum(1 for v in range(n_vertices) if len(adj[v]) % 2 == 1)
            lower_bound = max(odd_vertices // 2, 1) if edges else 0
            if best_count <= lower_bound:
                break
        
        if best_paths is None:
            # Fallback: greedy path building
            remaining = set(edges)
            paths = []
            while remaining:
                # Pick a random edge to start
                eid = rng.choice(list(remaining))
                u, v = edges[eid]
                remaining.remove(tuple(sorted((u, v))))
                path = [u, v]
                
                # Extend from both ends
                changed = True
                while changed:
                    changed = False
                    for end_pos in [0, -1]:
                        if end_pos == 0:
                            cur = path[0]
                        else:
                            cur = path[-1]
                        nbrs = [w for w in adj[cur] if (min(cur,w), max(cur,w)) in remaining and w not in path]
                        if nbrs:
                            nxt = nbrs[0]
                            remaining.remove(tuple(sorted((cur, nxt))))
                            if end_pos == 0:
                                path.insert(0, nxt)
                            else:
                                path.append(nxt)
                            changed = True
                
                paths.append(path)
            best_paths = paths
        
        return best_paths

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != edge_set:
            return False
        
        return True

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        # Validate
        if not validate_paths(paths, edges):
            # Fallback: each edge as a path
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
    
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
