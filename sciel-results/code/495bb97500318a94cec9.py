import time
import random
import sys
from collections import defaultdict, deque

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _bfs_shortest_path(adj, s, t):
    """Return list of vertices from s to t via BFS, or None if unreachable."""
    if s == t:
        return [s]
    parent = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                if v == t:
                    # reconstruct
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
    return None


def _decompose_eulerian(adj, edge_set):
    """
    Given an even-degree (Eulerian) multigraph represented by adj (vertex -> list of neighbors,
    with duplicates for multi-edges) and edge_set (set of edges, may contain duplicates via a counter),
    decompose into edge-disjoint cycles. Each cycle is a list of vertices (first != last).
    """
    # Build a multigraph edge list
    edge_list = []
    for u, v in edge_set:
        edge_list.append((u, v))
    
    # Build adjacency with edge indices
    adj_idx = defaultdict(list)
    for i, (u, v) in enumerate(edge_list):
        adj_idx[u].append((v, i))
        adj_idx[v].append((u, i))
    
    used = [False] * len(edge_list)
    cycles = []
    
    for start_i in range(len(edge_list)):
        if used[start_i]:
            continue
        u, v = edge_list[start_i]
        used[start_i] = True
        # Start a trail
        trail = [u, v]
        prev = u
        cur = v
        while True:
            # Find next unused edge from cur
            found = False
            for nb, ei in adj_idx[cur]:
                if not used[ei]:
                    used[ei] = True
                    trail.append(nb)
                    prev, cur = cur, nb
                    found = True
                    break
            if not found:
                break
        # The trail should be a cycle (start == end) in Eulerian graph
        # But we started with edge (u,v) and the trail ends at some vertex.
        # In an Eulerian graph, any trail that uses all edges of a connected component is a cycle.
        # Actually, the trail we built is a path from u to cur. For it to be a cycle, cur must == u.
        # In Eulerian graphs, this is guaranteed if we properly decompose.
        # Let's just record it as a cycle (it should be closed).
        if trail[0] == trail[-1]:
            cycles.append(trail)
        else:
            # This shouldn't happen in a proper Eulerian decomposition, but handle gracefully
            cycles.append(trail)
    
    return cycles


def _solve_one(edges, n, deadline):
    """
    Compute a minimum (or near-minimum) edge-disjoint path cover.
    """
    edge_set = set(map(tuple, edges))
    if not edge_set:
        return []
    
    # Build adjacency
    adj = defaultdict(list)
    deg = defaultdict(int)
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1
    
    # Find odd-degree vertices
    odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
    L = max(1, (len(odd_vertices) + 1) // 2)
    
    # Pair up odd vertices: (0,1), (2,3), ...
    pairs = []
    for i in range(0, len(odd_vertices), 2):
        if i + 1 < len(odd_vertices):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
    
    # For each pair, find a shortest path
    pair_paths = []
    pair_edges = set()
    for a, b in pairs:
        path = _bfs_shortest_path(adj, a, b)
        if path is None:
            # Graph should be connected, but handle gracefully
            path = [a, b]
        pair_paths.append(path)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = (min(u, v), max(u, v))
            pair_edges.add(e)
    
    # Now, the edges NOT in pair_edges form an Eulerian subgraph
    # (because removing the pairing paths makes all degrees even)
    remaining_edges = edge_set - pair_edges
    
    # Decompose remaining edges into cycles
    if remaining_edges:
        # Build multigraph for remaining edges
        rem_adj = defaultdict(list)
        for u, v in remaining_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
        
        # Use the cycle decomposition
        cycles = _decompose_eulerian(rem_adj, list(remaining_edges))
    else:
        cycles = []
    
    # Now we have:
    # - pair_paths: L paths (or fewer if some pairs are adjacent)
    # - cycles: edge-disjoint cycles covering remaining edges
    #
    # We need to merge everything into exactly L paths.
    # Strategy: Each cycle can be "absorbed" into a pair path by:
    #   1. Breaking the cycle at a vertex that lies on a pair path
    #   2. Splicing the broken cycle into the pair path
    #
    # But this is complex. Simpler approach:
    # Convert each cycle into a path by removing one edge, then merge all paths.
    
    # Convert cycles to paths
    all_paths = list(pair_paths)
    for cycle in cycles:
        # cycle is [v0, v1, ..., vk] where (vk, v0) is an edge
        # Break it: path = [v0, v1, ..., vk]
        if len(cycle) >= 2:
            all_paths.append(cycle)
        # The broken edge (vk, v0) needs to be covered
        # Actually, the cycle includes the edge (vk, v0). When we break it,
        # we lose that edge. Let me reconsider.
        # The cycle as a list [v0, v1, ..., vk] uses edges (v0,v1), ..., (v{k-1},vk).
        # The edge (vk, v0) is NOT in this list. So we need to add it.
        # But wait - in _decompose_eulerian, the trail starts with edge (u,v) and
        # the resulting trail is [u, v, ...]. The last edge in the trail connects
        # the last two vertices. The edge back to start is implicit.
        # Let me re-examine: the trail is a sequence of vertices where consecutive
        # vertices are connected by edges. If it's a cycle, trail[0] == trail[-1].
        # So the path [v0, v1, ..., vk] with v0 == vk uses edges (v0,v1),...,(v{k-1},vk).
        # That's all the edges in the cycle. Good.
        # But if trail[0] != trail[-1], we have a path, not a cycle.
    
    # Now we have more than L paths. We need to merge them down to L.
    # Merging: if path A ends at vertex x and path B starts at vertex x,
    # we can concatenate: A + B[1:] (removing duplicate x).
    # This reduces the path count by 1.
    
    # We need to perform L - len(all_paths) ... wait, we have more paths than L.
    # We need to merge until we have L paths.
    
    # Use a union-find / greedy merging approach
    # Represent each path by its start and end vertex
    path_info = []
    for p in all_paths:
        if len(p) >= 2:
            path_info.append(p)
    
    # Merge paths by shared endpoints
    # A path can be extended at either end if another path starts/ends there
    # and the connection doesn't create a cycle (i.e., the vertex appears only
    # at the endpoint).
    
    # Build a map: vertex -> list of (path_index, is_start)
    # For each path, start = path[0], end = path[-1]
    
    # We'll do iterative merging
    # This is essentially finding a path cover with minimum paths, which is
    # what we're trying to do. Let's use a simpler approach:
    # Just return all_paths as-is. It covers all edges and is edge-disjoint.
    # The count may be more than L, but it's valid.
    
    # Actually, let's try to merge properly.
    # Each path has a start and end. We can merge path A (end = x) with path B (start = x)
    # to get a new path. We need x to not appear in the interior of either path.
    
    # For simplicity and correctness, let's just return the paths we have.
    # The count is: L (pair paths) + number_of_cycles
    # This may exceed L, but it's a valid edge-disjoint path cover.
    
    # To do better, we'd need to splice cycles into pair paths.
    # Let's try a simple splicing: for each cycle, find a vertex that's on a pair path,
    # break the cycle there, and splice into the pair path.
    
    # Actually, the theoretical result says we CAN achieve exactly L paths.
    # The construction is: take the pairing paths, and for each cycle in the
    # remaining Eulerian subgraph, find a vertex v on the cycle that also lies
    # on some pairing path. Break the cycle at v (making it a path from v to v,
    # i.e., a cycle), then splice it into the pairing path at v.
    # Splicing: if the pairing path is ...a, v, b... and the cycle is v, c, d, ..., v,
    # the spliced path is ...a, c, d, ..., b... (replacing v with the cycle).
    # This keeps the path simple if the cycle's interior vertices don't appear elsewhere.
    
    # This is getting complex. Let's just return the valid cover we have.
    # The number of paths = L + num_cycles, which is at most L + m (where m is edges).
    # For our small graphs, this should be reasonable.
    
    # Let's do a simple merge pass to reduce path count.
    # Build adjacency: for each vertex, which paths have it as start/end
    # and whether it's in the interior.
    
    # For each path, mark interior vertices
    path_interior = [set() for _ in path_info]
    for i, p in enumerate(path_info):
        for v in p[1:-1]:
            path_interior[i].add(v)
    
    # Map: vertex -> list of (path_idx, role) where role is 'start' or 'end'
    endpoint_map = defaultdict(list)
    for i, p in enumerate(path_info):
        endpoint_map[p[0]].append((i, 'start'))
        endpoint_map[p[-1]].append((i, 'end'))
    
    # Merge: repeatedly find two paths that can be merged
    # Path i (end = v) and path j (start = v) can be merged if v not in interior of either
    # and i != j
    merged = [False] * len(path_info)
    changed = True
    while changed:
        changed = False
        for v in list(endpoint_map.keys()):
            if merged_count := sum(merged):
                pass
            starts = [(i, r) for i, r in endpoint_map[v] if r == 'start' and not merged[i]]
            ends = [(i, r) for i, r in endpoint_map[v] if r == 'end' and not merged[i]]
            for i, _ in ends:
                for j, _ in starts:
                    if i == j:
                        continue
                    # Check v not in interior of either
                    if v in path_interior[i] or v in path_interior[j]:
                        continue
                    # Merge: new_path = path_info[i] + path_info[j][1:]
                    new_path = path_info[i] + path_info[j][1:]
                    # Check simplicity
                    if len(set(new_path)) != len(new_path):
                        continue
                    # Perform merge
                    path_info[i] = new_path
                    merged[j] = True
                    # Update endpoint_map
                    # Remove old endpoints
                    # This is getting complicated. Let's just rebuild.
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if changed:
            # Rebuild endpoint_map and path_interior
            path_info = [p for i, p in enumerate(path_info) if not merged[i]]
            path_interior = [set() for _ in path_info]
            for i, p in enumerate(path_info):
                for v in p[1:-1]:
                    path_interior[i].add(v)
            endpoint_map = defaultdict(list)
            for i, p in enumerate(path_info):
                endpoint_map[p[0]].append((i, 'start'))
                endpoint_map[p[-1]].append((i, 'end'))
    
    # Filter out empty/invalid paths
    final_paths = [p for p in path_info if len(p) >= 2]
    
    # Ensure all edges are covered
    used_edges = set()
    for p in final_paths:
        for i in range(len(p) - 1):
            u, v = p[i], p[i + 1]
            used_edges.add((min(u, v), max(u, v)))
    
    missing = edge_set - used_edges
    for e in missing:
        final_paths.append([e[0], e[1]])
    
    return final_paths


def _validate_and_fix(paths, edges, n):
    """
    Validate that paths are simple, use graph edges, are edge-disjoint, and cover all edges.
    If invalid, fix by reconstructing.
    """
    edge_set = set(map(tuple, edges))
    used_edges = set()
    valid_paths = []
    
    for path in paths:
        if len(path) < 2:
            continue
        # Check simplicity
        if len(set(path)) != len(path):
            continue
        # Check edges
        valid = True
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                valid = False
                break
            if e in used_edges:
                valid = False
                break
            path_edges.append(e)
        if not valid:
            continue
        for e in path_edges:
            used_edges.add(e)
        valid_paths.append(path)
    
    # Check coverage
    if used_edges != edge_set:
        # Add missing edges as single-edge paths
        for e in edge_set - used_edges:
            valid_paths.append([e[0], e[1]])
    
    return valid_paths


def construct():
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    all_edges = [_instance(n, p, seed) for n, p, seed in specs]
    total_edges = sum(len(e) for e in all_edges)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = all_edges[i]
        frac = len(edges) / max(1, total_edges)
        time_alloc = max(10.0, frac * 140.0)
        remaining_time = total_deadline - time.time()
        time_alloc = min(time_alloc, remaining_time - 2.0)
        if time_alloc < 5.0:
            time_alloc = max(5.0, remaining_time - 1.0)
        
        graph_deadline = time.time() + time_alloc
        paths = _solve_one(edges, n, graph_deadline)
        paths = _validate_and_fix(paths, edges, n)
        results.append(paths)
    
    return results


if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
