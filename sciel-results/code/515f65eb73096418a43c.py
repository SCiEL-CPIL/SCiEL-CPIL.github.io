"""Edge-disjoint path cover via maximum matching on the line graph."""
import time
import random
from itertools import combinations

# Deadline guard: kill before the 150s wall-clock limit.
_START = time.time()
_DEADLINE = 150.0 - 20.0  # 130s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def construct():
    """Return three lists of paths (vertex lists) for the fixed instance graphs."""
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _decompose(edges, n, time_budget):
        """Decompose edges into edge-disjoint paths minimizing the count.

        Strategy: build the line graph where each original edge is a vertex and
        two such vertices are adjacent when the edges share an endpoint.  A
        matching in this line graph pairs adjacent edges into a continuous
        segment; each matched pair reduces the path count by one.  We seek a
        large matching (greedy + local improvement) and then assemble paths.
        """
        if not edges:
            return []

        m = len(edges)
        # Map each edge to an index 0..m-1
        edge_to_idx = {e: i for i, e in enumerate(edges)}

        # Build adjacency in the line graph: edges that share a vertex.
        # adj[i] = list of j such that edges[i] and edges[j] share an endpoint.
        adj = [[] for _ in range(m)]
        # For each vertex, collect incident edges and connect them pairwise.
        inc = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            inc[u].append(i)
            inc[v].append(i)
        for u in range(n):
            lst = inc[u]
            L = len(lst)
            for a in range(L):
                ia = lst[a]
                for b in range(a + 1, L):
                    ib = lst[b]
                    adj[ia].append(ib)
                    adj[ib].append(ia)

        # Greedy matching on the line graph.
        # matched[i] = j or -1
        matched = [-1] * m

        # Greedy: sort candidate pairs by some heuristic and add if both free.
        # Use a simple greedy that iterates edges and picks an available neighbor.
        # For better quality, do multiple random-order passes and keep the best.
        best_matched = None
        best_size = 0

        rng = Random(0)
        # Number of greedy restarts limited by time.
        max_restarts = 200
        for r in range(max_restarts):
            if _time_left() < time_budget / max(1, max_restarts - r) - 0.5:
                break
            matched = [-1] * m
            order = list(range(m))
            rng.shuffle(order)
            for i in order:
                if matched[i] != -1:
                    continue
                # Pick a free neighbor; prefer one with fewest free neighbors (to avoid dead-ends)
                best_j = -1
                best_deg = None
                for j in adj[i]:
                    if matched[j] == -1:
                        # count free neighbors of j
                        deg = sum(1 for k in adj[j] if matched[k] == -1)
                        if best_deg is None or deg < best_deg:
                            best_deg = deg
                            best_j = j
                            if deg == 1:
                                break
                if best_j != -1:
                    matched[i] = best_j
                    matched[best_j] = i
            size = sum(1 for i in range(m) if matched[i] != -1) // 2
            if size > best_size:
                best_size = size
                best_matched = matched[:]

        # Local improvement: try to increase matching size via alternating paths of length 3.
        # If we have matched (a,b) and (c,d) and edges a-c and b-d exist (in line graph)
        # we can swap to (a,c),(b,d) -- same size, not helpful for size.
        # To increase size we need an augmenting path of odd length.
        # Simpler: try to find an unmatched edge (u,v) and see if we can reassign.
        # For time, do a limited 1-optimization: for each unmatched pair (u,v),
        # check if we can match u to some w and free v's partner... this is complex.
        # Instead, do a few rounds of "try to match an unmatched vertex to a free neighbor
        # by breaking an existing match if it leads to net gain".
        if best_matched is not None:
            matched = best_matched[:]
            # A few improvement passes: for each unmatched vertex u, look for a neighbor w
            # that is matched to x; if u-w edge exists and w-x is the only match, and
            # u has another free neighbor, we could swap. Net gain only if we can match
            # both u and x. This is a 2-vertex swap: currently w-x matched. If we match
            # u-w and x-(some free y), we gain 1. Try: for unmatched u, for neighbor w
            # matched to x, if there exists free y adjacent to x, then swap to (u,w),(x,y).
            improved = True
            passes = 0
            while improved and passes < 5 and _time_left() > 5.0:
                improved = False
                passes += 1
                for u in range(m):
                    if matched[u] != -1:
                        continue
                    for w in adj[u]:
                        x = matched[w]
                        if x == -1:
                            # w is free? shouldn't happen since matched is symmetric; w free means we could just match u-w
                            matched[u] = w
                            matched[w] = u
                            improved = True
                            break
                        # x is matched to w. Try to find a free y adjacent to x.
                        found = False
                        for y in adj[x]:
                            if matched[y] == -1:
                                # Swap: break (w,x), make (u,w) and (x,y)
                                matched[u] = w
                                matched[w] = u
                                matched[x] = y
                                matched[y] = x
                                improved = True
                                found = True
                                break
                        if found:
                            break
                # After a pass, also try to match remaining free vertices directly.
                for u in range(m):
                    if matched[u] != -1:
                        continue
                    for w in adj[u]:
                        if matched[w] == -1:
                            matched[u] = w
                            matched[w] = u
                            improved = True
                            break

        # Assemble paths from the matching.
        # Each matched pair (i,j) means edges[i] and edges[j] are consecutive in a path.
        # Build directed adjacency: for each matched pair, we need to orient them.
        # Edge i = (u_i, v_i). If matched with edge j = (u_j, v_j), they share a vertex.
        # We need to form a sequence of edges where consecutive edges share a vertex.
        # Build a graph on edge-indices where matched pairs are "links". Each component
        # is a path (since matching, max degree 1 in the link graph... wait, matching means
        # each vertex in line graph has degree <=1 in the matching, so components are
        # single edges or isolated vertices in the match graph -> each component is either
        # a matched pair or a single unmatched edge).
        # So each path is either 1 edge (unmatched) or 2 edges (matched pair).
        # That gives path count = m - best_size.
        # Now produce vertex lists.
        paths = []
        used = [False] * m
        for i in range(m):
            if used[i]:
                continue
            if matched[i] != -1:
                j = matched[i]
                used[i] = True
                used[j] = True
                # Two edges sharing a vertex; form a path of 3 vertices.
                e1 = edges[i]
                e2 = edges[j]
                # Determine shared vertex and order.
                if e1[1] == e2[0]:
                    path = [e1[0], e1[1], e2[1]]
                elif e1[1] == e2[1]:
                    path = [e1[0], e1[1], e2[0]]
                elif e1[0] == e2[0]:
                    path = [e1[1], e1[0], e2[1]]
                else:
                    # e1[0] == e2[1]
                    path = [e1[1], e1[0], e2[0]]
                paths.append(path)
            else:
                used[i] = True
                e1 = edges[i]
                paths.append([e1[0], e1[1]])
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    # Allocate time budget per instance.
    result = []
    n_specs = len(specs)
    for k, spec in enumerate(specs):
        if _time_left() < 1.0:
            # Fallback: trivial decomposition (each edge its own path) for remaining.
            for kk in range(k, n_specs):
                e = instance(*specs[kk])
                result.append([list(ed) for ed in e])
            break
        n, p, seed = spec
        edges = instance(n, p, seed)
        # Give this instance most of the remaining time, but leave room for others.
        remaining_specs = n_specs - k
        per = max(1.0, _time_left() / remaining_specs - 0.5)
        paths = _decompose(edges, n, per)
        result.append(paths)

    return result
