"""Path decomposition via greedy extension + directed matching-based merging."""
import time
from random import Random
from collections import defaultdict

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def _greedy_extend(paths, adj, remaining, deadline, rng=None):
    """Extend each path greedily at both ends using unused edges."""
    if rng is None:
        rng = Random(0)
    for path in paths:
        if time.time() > deadline:
            break
        extended = True
        while extended and time.time() < deadline:
            extended = False
            for pos in (0, -1):
                cur = path[0] if pos == 0 else path[-1]
                nbrs = list(adj.get(cur, []))
                rng.shuffle(nbrs)
                for (w, ek) in nbrs:
                    if ek in remaining and w not in path:
                        remaining.discard(ek)
                        adj[cur] = [(x, y) for (x, y) in adj[cur] if y != ek]
                        adj[w] = [(x, y) for (x, y) in adj[w] if y != ek]
                        if pos == 0:
                            path.insert(0, w)
                        else:
                            path.append(w)
                        extended = True
                        break
    return paths

def _matching_merge(paths, edges_set, deadline):
    """
    Model merging as maximum matching on directed auxiliary graph.
    Nodes: (path_index, endpoint_pos) where endpoint_pos in {0, 1} (start, end).
    Directed edge from (i, pi) to (j, pj) if connecting endpoint i*pi to j*pj
    via an unused edge yields a valid simple merged path.
    A matching selects pairs of endpoints to connect (each endpoint used once).
    """
    n = len(paths)
    if n < 2:
        return paths, False
    # Build adjacency list of unused edges per vertex for fast lookup
    adj_unused = defaultdict(set)
    for e in edges_set:
        u, v = e
        adj_unused[u].add(v)
        adj_unused[v].add(u)

    # For each path, precompute vertex set for simplicity check
    path_sets = [set(p) for p in paths]

    # Auxiliary directed edges: list of (src, dst) where src=(i,pi), dst=(j,pj)
    # We only need to consider connecting endpoint of path i to endpoint of path j
    # where i != j. The merged path is valid if:
    #   - there's an unused edge between the two endpoints
    #   - the concatenation is simple (no repeated vertices)
    #
    # To keep it tractable, we build a bipartite-like structure:
    # Left side: (i, pi) for all i, pi in {0,1}
    # Right side: (j, pj) for all j, pj in {0,1}
    # Edge exists if i != j and the merge is valid.
    # We want a matching where each left node matches at most one right node,
    # and each right node at most one left node. But we also need to ensure
    # that we don't match two endpoints of the same path (i.e., (i,0) and (i,1)
    # both matched would split path i). So we add constraint: for each path i,
    # at most one of (i,0), (i,1) is matched on left AND at most one on right.
    # This is a general matching problem; we'll use a simple greedy matching
    # with randomization, then try to augment.

    # For efficiency, precompute for each pair of endpoints whether they can connect
    # and what the merged path would look like.
    # We'll build a list of candidate merges: (i, pi, j, pj, merged_path)
    # where pi, pj in {0,1}.

    candidates = []  # list of (i, pi, j, pj)
    merge_cache = {}  # (i, pi, j, pj) -> merged_path or None

    for i in range(n):
        for pi in (0, 1):
            vi = paths[i][0] if pi == 0 else paths[i][-1]
            for j in range(n):
                if j == i:
                    continue
                for pj in (0, 1):
                    vj = paths[j][0] if pj == 0 else paths[j][-1]
                    if vj not in adj_unused.get(vi, set()):
                        continue
                    # Check if merged path is simple
                    # Determine concatenation order
                    # If we connect vi to vj, the new path goes through vi-vj.
                    # We need to orient both paths so that vi is at the "connecting" end
                    # and vj is at the "connecting" end.
                    # Path i: if pi==0, vi is at start; orient as paths[i]
                    #          if pi==1, vi is at end; orient as paths[i]
                    # Path j: if pj==0, vj is at start; orient as paths[j][::-1]
                    #          if pj==1, vj is at end; orient as paths[j]
                    # The merged path: (path i oriented so vi is at the end we connect from)
                    #                 + (path j oriented so vj is at the start we connect to)
                    # But we must ensure vi is at the end of path_i_segment and vj at start of path_j_segment.
                    # Let's define:
                    #   seg_i = paths[i] if pi==0 else paths[i]  # vi is at index 0 if pi==0, else at end
                    #   Actually, we want the segment to end at vi (so we can append vj).
                    #   If pi==0: vi is at index 0. We want segment ending at vi, so seg_i = paths[i][::-1]
                    #   If pi==1: vi is at end. seg_i = paths[i]
                    #   seg_j: we want segment starting at vj.
                    #   If pj==0: vj is at index 0. seg_j = paths[j]
                    #   If pj==1: vj is at end. seg_j = paths[j][::-1]
                    # merged = seg_i + seg_j[1:]  (vj is last of seg_i? No, vi is last of seg_i, vj is first of seg_j)
                    # Wait: seg_i ends at vi, seg_j starts at vj. The edge is vi-vj.
                    # So merged = seg_i + seg_j[1:]
                    if pi == 0:
                        seg_i = paths[i][::-1]
                    else:
                        seg_i = paths[i]
                    if pj == 0:
                        seg_j = paths[j]
                    else:
                        seg_j = paths[j][::-1]
                    # seg_i[-1] should be vi, seg_j[0] should be vj
                    if seg_i[-1] != vi or seg_j[0] != vj:
                        continue
                    merged = seg_i + seg_j[1:]
                    # Check simplicity
                    if len(set(merged)) != len(merged):
                        continue
                    # Check all edges in merged are valid (in edges_set)
                    valid = True
                    for k in range(len(merged) - 1):
                        ek = _edge_key(merged[k], merged[k+1])
                        if ek not in edges_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    candidates.append((i, pi, j, pj))
                    merge_cache[(i, pi, j, pj)] = merged

    if not candidates:
        return paths, False

    # Now find a matching: select a set of candidates such that:
    # - No two candidates share the same (i, pi) on the "left" (i.e., same path endpoint used as source)
    # - No two candidates share the same (j, pj) on the "right" (i.e., same path endpoint used as target)
    # - For each path i, at most one of (i,0), (i,1) is used as left, and at most one as right.
    #   (This is automatically handled if we treat left and right separately, but we need
    #    to ensure that a path isn't used as both source and target in a way that splits it.
    #    Actually, if path i is used as source (left) in one merge and as target (right) in another,
    #    that would mean path i is being connected at both ends, which is fine as long as the
    #    resulting structure is a single path. But in our model, each merge combines two paths
    #    into one. If path i is merged with path j (i as source, j as target) and also with path k
    #    (i as target, k as source), that would be a 3-path merge, which our simple model doesn't
    #    handle directly. To keep it simple, we'll restrict to matchings where each path appears
    #    in at most one candidate (either as source or target, not both).)
    #
    # Simplify: we'll do a greedy matching with randomization, ensuring each path is used at most once.

    # Build a bipartite graph: left nodes = (i, pi), right nodes = (j, pj)
    # Edge if candidate exists.
    # We want a matching that maximizes the number of edges, with the constraint that
    # for each path index, at most one left node and at most one right node are matched.
    # This is a standard bipartite matching with additional constraints.
    # For simplicity, we'll use a greedy approach: shuffle candidates, add if no conflict.

    left_used = set()  # (i, pi)
    right_used = set()  # (j, pj)
    path_left_used = {}  # i -> True if path i has a left node used
    path_right_used = {}  # i -> True if path i has a right node used
    selected = []

    rng = Random(42)
    cand_list = candidates[:]
    rng.shuffle(cand_list)

    for (i, pi, j, pj) in cand_list:
        if (i, pi) in left_used or (j, pj) in right_used:
            continue
        if path_left_used.get(i) or path_right_used.get(j):
            # Path i already used as left, or path j already used as right
            # This would mean path i or j is involved in another merge, which we don't allow
            continue
        # Check if path i is used as right or path j as left
        if path_right_used.get(i) or path_left_used.get(j):
            continue
        left_used.add((i, pi))
        right_used.add((j, pj))
        path_left_used[i] = True
        path_right_used[j] = True
        selected.append((i, pi, j, pj))

    if not selected:
        return paths, False

    # Apply the merges: for each selected candidate, merge paths i and j
    # We need to be careful: if multiple merges involve the same path, we'd have conflicts,
    # but our constraint ensures each path is in at most one merge.
    new_paths = []
    used_paths = set()
    for (i, pi, j, pj) in selected:
        merged = merge_cache[(i, pi, j, pj)]
        new_paths.append(merged)
        used_paths.add(i)
        used_paths.add(j)

    # Add paths not involved in any merge
    for k in range(n):
        if k not in used_paths:
            new_paths.append(paths[k])

    return new_paths, True

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count."""
    edges = set(tuple(sorted(e)) for e in edges)
    if not edges:
        return []

    # Build adjacency
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        adj[u].append((v, e))
        adj[v].append((u, e))

    remaining = set(edges)
    paths = []

    # Initial greedy: create paths from edges
    for e in sorted(edges):
        if time.time() > deadline:
            break
        if e not in remaining:
            continue
        u, v = e
        remaining.discard(e)
        adj[u] = [(w, ek) for (w, ek) in adj[u] if ek != e]
        adj[v] = [(w, ek) for (w, ek) in adj[v] if ek != e]
        paths.append([u, v])

    # Greedy extension
    _greedy_extend(paths, adj, remaining, deadline)

    # Iteratively try to merge paths via matching
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(50):  # multiple matching rounds
            if time.time() > deadline:
                break
            new_paths, merged = _matching_merge(paths, remaining, deadline)
            if merged:
                paths = new_paths
                improved = True
                # After merging, try greedy extension again
                _greedy_extend(paths, adj, remaining, deadline)

    # Any remaining edges: add as single-edge paths
    for e in sorted(remaining):
        if time.time() > deadline:
            break
        remaining.discard(e)
        u, v = e
        paths.append([u, v])

    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 140
    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    for idx, spec in enumerate(specs):
        if time.time() > deadline:
            # Fallback: trivial decomposition for remaining
            for rem_spec in specs[len(results):]:
                results.append([[list(e) for e in instance(*rem_spec)]])
            break
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
    return results
