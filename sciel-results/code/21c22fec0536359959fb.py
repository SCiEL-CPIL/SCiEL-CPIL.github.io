"""Minimum edge path cover via Gallai's theorem: odd pairing + Eulerian circuit + cut."""
import time
from collections import deque, defaultdict, Counter

_DEADLINE = 148.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _bfs_path(adj, s, t, forbidden):
    """Shortest simple path from s to t, internal vertices not in forbidden."""
    if s == t:
        return [s]
    if s in forbidden or t in forbidden:
        return None
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if len(path) >= 3 and set(path[1:-1]) & forbidden:
        return None
    return path


def _pair_odd(adj, odd):
    """Find vertex-disjoint paths pairing up all odd vertices. Returns list of paths."""
    remaining = list(odd)
    used_internal = set()
    paths = []
    while len(remaining) >= 2 and _remaining() > 3.0:
        best = None
        best_len = float('inf')
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                s, t = remaining[i], remaining[j]
                p = _bfs_path(adj, s, t, used_internal)
                if p is None:
                    continue
                if len(p) < best_len:
                    best = p
                    best_len = len(p)
        if best is None:
            break
        used_internal |= set(best[1:-1])
        paths.append(best)
        remaining.remove(best[0])
        remaining.remove(best[-1])
    while len(remaining) >= 2:
        s, t = remaining[0], remaining[1]
        p = _bfs_path(adj, s, t, used_internal)
        if p is None:
            p = _bfs_path(adj, s, t, set())
        if p is None:
            remaining.remove(s)
            remaining.remove(t)
            continue
        used_internal |= set(p[1:-1])
        paths.append(p)
        remaining.remove(s)
        remaining.remove(t)
    return paths


def _solve_one(edges):
    if not edges:
        return []
    
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)

    degree = {u: len(adj[u]) for u in adj}
    odd = sorted([u for u in adj if degree[u] % 2 == 1])

    if len(odd) == 0:
        # Eulerian: one circuit
        hadj = defaultdict(set)
        for (u, v) in edges:
            hadj[u].add(v)
            hadj[v].add(u)
        # Find Eulerian circuit using Hierholzer
        if not hadj:
            return []
        stack = [next(iter(hadj))]
        circuit = []
        while stack:
            v = stack[-1]
            if hadj[v]:
                u = next(iter(hadj[v]))
                hadj[v].discard(u)
                hadj[u].discard(v)
                stack.append(u)
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        return [circuit]

    # Pair odd vertices with vertex-disjoint paths
    pair_paths = _pair_odd(adj, odd)
    k = len(pair_paths)
    if k == 0:
        # Fallback: just do greedy longest path
        remaining_edges = set(edges)
        final_paths = []
        while remaining_edges and _remaining() > 2.0:
            radj = defaultdict(set)
            for (u, v) in remaining_edges:
                radj[u].add(v)
                radj[v].add(u)
            verts = list(radj.keys())
            if not verts:
                break
            verts.sort(key=lambda x: len(radj[x]))
            start = verts[0]
            path = [start]
            visited_v = {start}
            cur = start
            while True:
                nxt = None
                for v in radj[cur]:
                    if v not in visited_v and tuple(sorted((cur, v))) in remaining_edges:
                        nxt = v
                        break
                if nxt is None:
                    break
                remaining_edges.discard(tuple(sorted((cur, nxt))))
                path.append(nxt)
                visited_v.add(nxt)
                cur = nxt
            if len(path) >= 2:
                final_paths.append(path)
            else:
                break
        return final_paths

    # Build multigraph H = G + pair_paths (as multiset of edges)
    edge_list = list(edges)
    for pp in pair_paths:
        for i in range(len(pp) - 1):
            edge_list.append((pp[i], pp[i + 1]))

    # Build adjacency for Hierholzer with edge indices
    hadj = defaultdict(list)
    for idx, (u, v) in enumerate(edge_list):
        hadj[u].append((v, idx))
        hadj[v].append((u, idx))

    # Hierholzer on multigraph
    used_edge = [False] * len(edge_list)
    if not edge_list:
        return []
    stack = [(edge_list[0][0], None)]
    edge_seq = []  # list of (u, v, edge_index) in circuit order
    while stack:
        v, _ = stack[-1]
        advanced = False
        for u, idx in hadj[v]:
            if not used_edge[idx]:
                used_edge[idx] = True
                hadj[v].remove((u, idx))
                edge_seq.append((v, u, idx))
                stack.append((u, idx))
                advanced = True
                break
        if not advanced:
            stack.pop()
    # edge_seq is in reverse-circuit order; reverse it
    edge_seq.reverse()

    # Now edge_seq[i] = (u, v, idx) means the circuit goes u->v using edge_list[idx].
    # Identify added edges (from pair_paths) by checking if (u,v) is in added_counter.
    added_counter = Counter()
    for pp in pair_paths:
        for i in range(len(pp) - 1):
            added_counter[(pp[i], pp[i + 1])] += 1
            added_counter[(pp[i + 1], pp[i])] += 1

    # Walk the circuit, tracking which added edges we cross.
    added_positions = []
    for pos, (u, v, idx) in enumerate(edge_seq):
        if added_counter[(u, v)] > 0:
            added_counter[(u, v)] -= 1
            added_positions.append(pos)

    # The circuit is cyclic. Cut at added_positions.
    m = len(edge_seq)
    if not added_positions:
        # Shouldn't happen if k>0
        # Just return the whole circuit as a path (remove last vertex if it equals first)
        circuit_verts = [edge_seq[0][0]]
        for u, v, idx in edge_seq:
            circuit_verts.append(v)
        if len(circuit_verts) >= 2 and circuit_verts[0] == circuit_verts[-1]:
            circuit_verts.pop()
        return [circuit_verts]

    results = []
    for i in range(len(added_positions)):
        a = added_positions[i]
        b = added_positions[(i + 1) % len(added_positions)]
        if b < a:
            b += m
        # Path vertices from after edge a to before edge b
        path_verts = []
        for pos in range(a + 1, b):
            path_verts.append(edge_seq[pos][0])
        path_verts.append(edge_seq[(b - 1) % m][1])
        # Ensure simple path: remove consecutive duplicates if any
        if len(path_verts) >= 2:
            # Verify it's a valid simple path
            # Check all vertices are distinct
            if len(set(path_verts)) == len(path_verts):
                results.append(path_verts)
            else:
                # If not simple, try to fix by removing repeated vertices
                # This should rarely happen
                seen = set()
                new_path = []
                for v in path_verts:
                    if v not in seen:
                        new_path.append(v)
                        seen.add(v)
                if len(new_path) >= 2:
                    results.append(new_path)

    # Verify that all original edges are covered
    covered_edges = set()
    for p in results:
        for k_idx in range(len(p) - 1):
            covered_edges.add(tuple(sorted((p[k_idx], p[k_idx + 1]))))

    # If some edges are missing, add them as separate paths or extend existing paths
    missing_edges = set(edges) - covered_edges
    if missing_edges:
        # For each missing edge, add it as a path of length 1
        for u, v in missing_edges:
            # Try to extend an existing path
            extended = False
            for i, p in enumerate(results):
                if p[0] == u:
                    results[i] = [v] + p
                    extended = True
                    break
                elif p[-1] == u:
                    results[i] = p + [v]
                    extended = True
                    break
                elif p[0] == v:
                    results[i] = [u] + p
                    extended = True
                    break
                elif p[-1] == v:
                    results[i] = p + [u]
                    extended = True
                    break
            if not extended:
                results.append([u, v])

    return results


def construct():
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(*spec)
        paths = _solve_one(edges)
        results.append(paths)
    return results
