"""Set-cover formulation: precompute which vertex pairs each candidate edge covers."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}

def _build_adj(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_triangle_free(added):
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    return True

def _diameter(added):
    adj = _build_adj(added)
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best

def _valid(added):
    if not _is_triangle_free(added):
        return False
    d = _diameter(added)
    return d <= P['diameter']

def _all_pairs():
    """All unordered vertex pairs (u,v) with u < v."""
    return [(u, v) for u in range(_N) for v in range(u + 1, _N)]

def _pairs_at_dist_gt_d(adj, d):
    """Return set of pairs (u,v) with shortest distance > d in graph with adjacency adj."""
    pairs = set()
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        for v in range(_N):
            if dist[v] == -1 or dist[v] > d:
                if s < v:
                    pairs.add((s, v))
                else:
                    pairs.add((v, s))
    return pairs

def _precompute_coverage():
    """
    For each candidate edge (u,v) (non-base), compute which pairs (a,b) with dist_base(a,b) > 4
    would be brought to distance <= 4 if we add edge (u,v) to the base path.
    Also precompute triangle-freedom constraints.
    """
    base_adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        base_adj[a].add(b)
        base_adj[b].add(a)
    
    # Pairs that need to be covered (distance > 4 in base path)
    need_cover = _pairs_at_dist_gt_d(base_adj, 4)
    
    # For each candidate edge, determine which "need_cover" pairs it helps
    # A pair (a,b) is helped by edge (u,v) if adding (u,v) makes dist(a,b) <= 4
    # We compute this by: for pair (a,b), check if there exists a path of length <= 4
    # in base + {(u,v)}. This is expensive, so we approximate:
    # Adding edge (u,v) creates shortcuts. dist_new(a,b) = min(dist_base(a,b),
    #   dist_base(a,u)+1+dist_base(v,b), dist_base(a,v)+1+dist_base(u,b))
    # where dist_base is distance in the path graph.
    
    def path_dist(x, y):
        return abs(x - y)
    
    candidate_edges = []
    coverage = {}  # edge -> frozenset of pairs it covers
    
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) in _BASE_SET:
                continue
            candidate_edges.append((u, v))
            
            # For each pair in need_cover, check if adding (u,v) brings it to dist <= 4
            covered = set()
            for (a, b) in need_cover:
                # Shortest path using edge (u,v) at most once
                # Option 1: a -> ... -> u -> v -> ... -> b
                d1 = path_dist(a, u) + 1 + path_dist(v, b)
                # Option 2: a -> ... -> v -> u -> ... -> b
                d2 = path_dist(a, v) + 1 + path_dist(u, b)
                # Option 3: just base path
                d3 = path_dist(a, b)
                if min(d1, d2, d3) <= 4:
                    covered.add((a, b))
            coverage[(u, v)] = frozenset(covered)
    
    return candidate_edges, coverage, need_cover

def _triangle_free_conflicts():
    """
    Return a dict mapping each candidate edge to the set of other candidate edges
    that conflict with it (would form a triangle if both added).
    Two edges (u,v) and (w,x) conflict if they share a vertex AND the other two
    endpoints are adjacent in the base path, OR if they share a vertex and the
    other endpoints are adjacent to the shared vertex.
    
    Actually, two edges (u,v) and (w,x) form a triangle if:
    - They share exactly one vertex, say u, and (v,x) is an edge in base+added.
    - Or they are the same edge (not applicable here).
    
    More precisely: edges (u,v) and (w,x) create a triangle if:
    - u==w and (v,x) is in base edges, or v==x and (u,w) is in base, etc.
    - Or u==x and (v,w) is in base, etc.
    - Or u==v... no, edges are simple.
    
    Let me think again. Triangle = 3 vertices with all 3 edges present.
    If we add edges (u,v) and (w,x), a triangle forms if there exist 3 distinct
    vertices a,b,c such that all of (a,b),(b,c),(c,a) are in base ∪ {(u,v),(w,x)}.
    
    Since base is a path, the only triangles can involve at most 2 added edges
    (a path has no triangles). So a triangle would need:
    - 2 added edges + 1 base edge, or
    - 1 added edge + 2 base edges (impossible since path has no cycles of length 3)
    
    So: edges (u,v) and (w,x) conflict if there's a base edge that completes a triangle.
    Case: u==w (shared vertex). Then triangle would be (u,v),(u,x),(v,x). Need (v,x) in base.
    Case: u==x. Triangle: (u,v),(w,x=u),(v,w). Need (v,w) in base.
    Case: v==w. Triangle: (u,v=w),(w,x),(u,x). Need (u,x) in base.
    Case: v==x. Triangle: (u,v=x),(w,x),(u,w). Need (u,w) in base.
    Case: u==v or w==x: impossible (simple edges).
    
    Also: what if the two added edges don't share a vertex? Then no triangle from just
    these two + base (since base is a path, can't form a 3-cycle with two disjoint chords).
    
    Wait, actually I need to be more careful. Let me just check all cases.
    """
    base_adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        base_adj[a].add(b)
        base_adj[b].add(a)
    
    candidate_edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) not in _BASE_SET:
                candidate_edges.append((u, v))
    
    conflicts = {e: set() for e in candidate_edges}
    
    for i in range(len(candidate_edges)):
        for j in range(i + 1, len(candidate_edges)):
            e1 = candidate_edges[i]
            e2 = candidate_edges[j]
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if these two edges + some base edge form a triangle
            # The triangle must use both added edges and exactly one base edge.
            # Vertices involved: {u1,v1} ∪ {u2,v2}
            
            # Case 1: shared vertex u1 == u2
            if u1 == u2:
                # Triangle: (u1,v1), (u2,v2), (v1,v2). Need (v1,v2) in base.
                if v1 in base_adj[v2]:
                    conflicts[e1].add(e2)
                    conflicts[e2].add(e1)
            # Case 2: shared vertex u1 == v2
            if u1 == v2:
                # Triangle: (u1,v1), (u2,v2), (v1,u2). Need (v1,u2) in base.
                if v1 in base_adj[u2]:
                    conflicts[e1].add(e2)
                    conflicts[e2].add(e1)
            # Case 3: shared vertex v1 == u2
            if v1 == u2:
                # Triangle: (u1,v1), (u2,v2), (u1,v2). Need (u1,v2) in base.
                if u1 in base_adj[v2]:
                    conflicts[e1].add(e2)
                    conflicts[e2].add(e1)
            # Case 4: shared vertex v1 == v2
            if v1 == v2:
                # Triangle: (u1,v1), (u2,v2), (u1,u2). Need (u1,u2) in base.
                if u1 in base_adj[u2]:
                    conflicts[e1].add(e2)
                    conflicts[e2].add(e1)
    
    return candidate_edges, conflicts

def _set_cover_greedy(seed, deadline):
    """Greedy set cover: pick edge covering most uncovered pairs, subject to triangle-free."""
    rng = random.Random(seed)
    candidate_edges, conflicts = _triangle_free_conflicts()
    candidate_edges, coverage, need_cover = _precompute_coverage()
    
    # Rebuild conflicts for the candidate_edges list from _precompute_coverage
    # (they should be the same set)
    base_adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        base_adj[a].add(b)
        base_adj[b].add(a)
    
    conflicts = {e: set() for e in candidate_edges}
    for i in range(len(candidate_edges)):
        for j in range(i + 1, len(candidate_edges)):
            e1 = candidate_edges[i]
            e2 = candidate_edges[j]
            u1, v1 = e1
            u2, v2 = e2
            if u1 == u2 and v1 in base_adj[v2]:
                conflicts[e1].add(e2); conflicts[e2].add(e1)
            if u1 == v2 and v1 in base_adj[u2]:
                conflicts[e1].add(e2); conflicts[e2].add(e1)
            if v1 == u2 and u1 in base_adj[v2]:
                conflicts[e1].add(e2); conflicts[e2].add(e1)
            if v1 == v2 and u1 in base_adj[u2]:
                conflicts[e1].add(e2); conflicts[e2].add(e1)
    
    uncovered = set(need_cover)
    selected = []
    selected_set = set()
    
    while uncovered and time.time() < deadline:
        # Find best edge: max new coverage, triangle-free with selected
        best_edge = None
        best_count = -1
        for e in candidate_edges:
            if e in selected_set:
                continue
            # Check triangle-free with all selected
            if any(s in conflicts[e] for s in selected_set):
                continue
            new_cover = coverage[e] & uncovered
            cnt = len(new_cover)
            if cnt > best_count:
                best_count = cnt
                best_edge = e
            elif cnt == best_count and best_edge is not None:
                # Tie-break randomly
                if rng.random() < 0.5:
                    best_edge = e
        
        if best_edge is None or best_count <= 0:
            break
        
        selected.append(best_edge)
        selected_set.add(best_edge)
        uncovered -= coverage[best_edge]
    
    # Verify and try to reduce
    result = list(selected)
    if not _valid(result):
        return None
    result = _try_remove(result, seed)
    return result

def _try_remove(added, seed):
    cur = list(added)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur[:i] + cur[i+1:]
            if _valid(trial):
                cur = trial
                improved = True
                break
    return cur

def construct():
    deadline = time.time() + 55.0
    best = None
    
    # Try set-cover greedy with multiple seeds
    for seed in range(50):
        if time.time() >= deadline:
            break
        result = _set_cover_greedy(seed, deadline)
        if result is not None:
            if best is None or len(result) < len(best):
                best = result
        if best is not None and len(best) <= 5:
            break
    
    if best is None:
        # Fallback: bipartite-like construction
        return [[u, v] for u in range(_N) for v in range(u + 1, _N)
                if (u - v) % 2 and (u, v) not in _BASE_SET]
    
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
