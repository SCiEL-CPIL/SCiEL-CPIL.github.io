"""Combined approach: Warnsdorff-guided DFS with strong pruning + maximum path cover
via bipartite matching + 2-opt local search + beam search + path stitching."""

import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].add(v)
    adj[v].add(u)

DEG = [len(adj[i]) for i in range(N)]

# Precompute sorted neighbors by degree (Warnsdorff ordering)
NEIGHBORS_SORTED = [sorted(adj[i], key=lambda w: (DEG[w], w)) for i in range(N)]


def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj[path[i]]:
            return False
    return True


def greedy_grow(start):
    """Grow a path greedily using Warnsdorff-like heuristic (least remaining degree)."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj[w] - seen), DEG[w], w))
        path.append(v)
        seen.add(v)
    return path


def try_extend_both_ends(path):
    """Extend a path from both ends repeatedly, always choosing the end with the
    most constrained (fewest) candidate to avoid dead-ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        cur_l = path[0]
        cands_l = adj[cur_l] - seen
        cur_r = path[-1]
        cands_r = adj[cur_r] - seen
        if cands_l and cands_r:
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.append(v)
            seen.add(v)
            changed = True
    return path


def local_search_improve(path, deadline):
    """Improve a path via 2-opt style moves: try reversing segments and re-extending."""
    best = list(path)
    best_len = len(path)
    improved = True

    while improved and time.time() < deadline:
        improved = False
        n = len(best)

        # Try 2-opt: reverse a segment [i..j]
        # Only check valid 2-opt moves (where the reversal creates valid connections)
        for i in range(n - 1):
            if time.time() > deadline:
                return best
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best
                left_ok = (i == 0) or (best[j] in adj[best[i - 1]])
                right_ok = (j == n - 1) or (best[i] in adj[best[j + 1]])
                if left_ok and right_ok:
                    new_path = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    # The 2-opt reversal is valid by construction if left_ok and right_ok
                    # Just need to verify internal edges are preserved (they are, since we reverse)
                    ext = try_extend_both_ends(new_path)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best = ext
                        improved = True

        # Try 1-opt: remove an internal vertex and reconnect
        n = len(best)
        for i in range(1, n - 1):
            if time.time() > deadline:
                return best
            if best[i + 1] in adj[best[i - 1]]:
                new_path = best[:i] + best[i + 1:]
                ext = try_extend_both_ends(new_path)
                if len(ext) > best_len:
                    best_len = len(ext)
                    best = ext
                    improved = True

        # Try segment swap: swap two adjacent segments
        n = len(best)
        for i in range(1, n - 1):
            if time.time() > deadline:
                return best
            for j in range(i + 1, n - 1):
                if time.time() > deadline:
                    return best
                # Swap best[i..j] with best[j+1..k] for some k
                # This is complex; skip for now and rely on 2-opt
                break

    return best


def dfs_longest_path(start, deadline, best_len):
    """
    Depth-first search for longest simple path starting at `start`.
    Uses time-limited search with strong pruning including
    articulation-point-like checks on the remaining graph.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = max(1, best_len)

    path = [start]
    seen = [False] * N
    seen[start] = True

    # Track unvisited count for pruning
    unvisited_count = N - 1

    def _count_unvisited_component(start_node, exclude):
        """Count size of unvisited connected component containing start_node,
        not passing through any seen vertex or exclude."""
        count = 0
        stack = [start_node]
        local_visited = set()
        while stack:
            node = stack.pop()
            if node in local_visited:
                continue
            if seen[node] or node == exclude:
                continue
            local_visited.add(node)
            count += 1
            for nb in adj[node]:
                if nb not in local_visited and not seen[nb] and nb != exclude:
                    stack.append(nb)
        return count

    def _dfs(cur, depth):
        nonlocal best_path, best_len_local, unvisited_count
        if time.time() > deadline:
            return

        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)

        # Pruning: if we can't beat best even visiting all remaining
        if depth + unvisited_count <= best_len_local:
            return

        # Try extending from current end
        for nxt in NEIGHBORS_SORTED[cur]:
            if time.time() > deadline:
                return
            if not seen[nxt]:
                # Check if taking nxt would isolate a large unvisited component
                # from the rest of the unvisited graph
                # Count unvisited neighbors of nxt (excluding cur)
                unvisited_neighbors_of_nxt = 0
                for nb in adj[nxt]:
                    if not seen[nb] and nb != cur:
                        unvisited_neighbors_of_nxt += 1

                # If nxt is a cut vertex in the remaining graph, we might lose
                # access to a large component. Check component sizes.
                if unvisited_neighbors_of_nxt > 0 and depth + 1 < best_len_local:
                    # Count how many unvisited vertices are reachable from each
                    # unvisited neighbor of nxt (through unvisited vertices, not through cur)
                    # This is expensive, so only do it for high-degree nxt
                    if unvisited_neighbors_of_nxt <= 4:
                        comp_sizes = []
                        for nb in adj[nxt]:
                            if not seen[nb] and nb != cur:
                                sz = _count_unvisited_component(nb, nxt)
                                comp_sizes.append(sz)
                        # The largest component we can still reach after going to nxt
                        # is the sum of all components (since we enter via nxt)
                        # But if we go to nxt, we can only continue into ONE branch
                        # unless we come back (which we can't in a simple path)
                        # So the max we can visit from nxt onward is:
                        # 1 (nxt itself) + max(comp_sizes) if we must pick one branch
                        # Actually we can visit multiple branches if they reconnect
                        # For safety, use the total as upper bound
                        total_unvisited_from_nxt = sum(comp_sizes)
                        if depth + 1 + total_unvisited_from_nxt <= best_len_local:
                            continue

                seen[nxt] = True
                path.append(nxt)
                unvisited_count -= 1
                _dfs(nxt, depth + 1)
                path.pop()
                seen[nxt] = False
                unvisited_count += 1

    _dfs(start, 1)
    return best_path


def beam_search_path(start, beam_width, max_depth, deadline):
    """Beam search for longest path starting from start."""
    beam = [([start], {start})]
    best_path = [start]

    for depth in range(max_depth):
        if time.time() > deadline:
            break
        if not beam:
            break
        new_beam = []
        for p, seen in beam:
            cur = p[-1]
            candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            for v in candidates:
                new_path = p + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        # Sort by potential: prefer paths where the end has fewer unvisited neighbors
        new_beam.sort(key=lambda x: len(adj[x[0][-1]] - x[1]))
        beam = new_beam[:beam_width]

    return best_path


def find_max_path_cover():
    """
    Finds a maximum set of vertex-disjoint paths covering the maximum number of vertices.
    Uses maximum bipartite matching on a split graph.
    Each vertex v is split into v_in (left) and v_out (right).
    An edge u-v creates arcs u_in->v_out and v_in->u_out.
    A matching selects at most one outgoing arc per vertex and at most one incoming arc per vertex.
    Vertices with no matched incoming arc are path starts; no matched outgoing arc are path ends.
    """
    rows = []
    cols = []

    for u, v in EDGES:
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)

    data = np.ones(len(rows))
    bipartite_graph = csr_matrix((data, (rows, cols)), shape=(N, N))

    try:
        match_left, match_right = maximum_bipartite_matching(bipartite_graph)
    except Exception:
        return []

    # match_left[i] = j means vertex i's outgoing edge goes to j
    # match_right[j] = i means vertex j's incoming edge comes from i
    # A vertex is a path start if no one maps to it via match_right
    # i.e., match_right[v] == -1 for all... actually match_right[v] gives the predecessor

    # Build successor and predecessor arrays
    succ = [int(match_left[i]) for i in range(N)]
    pred = [int(match_right[i]) for i in range(N)]

    # Find path starts: vertices with no predecessor (pred[v] == -1)
    starts = [v for v in range(N) if pred[v] == -1]

    paths = []
    for s in starts:
        path = [s]
        curr = s
        while True:
            nxt = succ[curr]
            if nxt == -1 or nxt == curr:
                break
            path.append(nxt)
            curr = nxt
        if len(path) > 1:
            paths.append(path)

    # Also check for isolated vertices (no edges at all) - they form trivial paths
    # But more importantly, check for vertices not in any path above
    # (vertices that are in cycles or were missed)
    in_path = set()
    for p in paths:
        in_path.update(p)

    # Check for 2-cycles or longer cycles in the matching
    visited_cycle = set()
    for v in range(N):
        if v not in in_path and v not in visited_cycle:
            # Follow successor chain
            cycle = []
            curr = v
            while curr not in visited_cycle and curr not in in_path:
                cycle.append(curr)
                visited_cycle.add(curr)
                nxt = succ[curr]
                if nxt == -1 or nxt == curr:
                    break
                curr = nxt
            if len(cycle) > 1:
                # It's a cycle; break it into a path
                paths.append(cycle)
                in_path.update(cycle)

    return paths


def stitch_paths(paths):
    """
    Merges disjoint paths into a single path if possible.
    Returns the longest path found.
    Uses iterative greedy merging with multiple strategies.
    """
    if not paths:
        return []

    active_paths = [list(p) for p in paths]

    # Precompute endpoints adjacency for fast lookup
    def can_connect(p1, p2):
        """Return the combined path if p1 and p2 can be connected, else None."""
        # p1_end -> p2_start
        if p1[-1] in adj[p2[0]]:
            return p1 + p2
        # p2_end -> p1_start
        if p2[-1] in adj[p1[0]]:
            return p2 + p1
        # p1_end -> p2_end (reverse p2)
        if p1[-1] in adj[p2[-1]]:
            return p1 + p2[::-1]
        # p1_start -> p2_end (reverse p1)
        if p1[0] in adj[p2[-1]]:
            return p1[::-1] + p2
        return None

    changed = True
    while changed:
        changed = False
        n_paths = len(active_paths)
        i = 0
        while i < n_paths:
            if i >= len(active_paths):
                break
            j = i + 1
            while j < len(active_paths):
                combined = can_connect(active_paths[i], active_paths[j])
                if combined is not None:
                    active_paths[i] = combined
                    active_paths.pop(j)
                    changed = True
                    n_paths = len(active_paths)
                    j = i + 1
                    continue
                j += 1
            i += 1

    if not active_paths:
        return []
    return max(active_paths, key=len)


def combine_paths(path1, path2, deadline):
    """Try to combine two paths into a longer path."""
    best = max(path1, path2, key=len)
    best_len = len(best)

    # Try to connect end of path1 to start of path2
    if path1[-1] in adj[path2[0]]:
        combined = path1 + path2[1:]
        if len(combined) > best_len and len(set(combined)) == len(combined):
            best = combined
            best_len = len(combined)

    # Try to connect end of path2 to start of path1
    if path2[-1] in adj[path1[0]]:
        combined = path2 + path1[1:]
        if len(combined) > best_len and len(set(combined)) == len(combined):
            best = combined
            best_len = len(combined)

    # Try inserting path1 into path2 at various positions
    for i in range(len(path2) - 1):
        if time.time() > deadline:
            break
        if path1[0] in adj[path2[i]] and path1[-1] in adj[path2[i + 1]]:
            combined = path2[:i + 1] + path1 + path2[i + 1:]
            if len(combined) > best_len and len(set(combined)) == len(combined):
                best = combined
                best_len = len(combined)
        if path1[-1] in adj[path2[i]] and path1[0] in adj[path2[i + 1]]:
            combined = path2[:i + 1] + path1[::-1] + path2[i + 1:]
            if len(combined) > best_len and len(set(combined)) == len(combined):
                best = combined
                best_len = len(combined)

    # Try inserting path2 into path1 at various positions
    for i in range(len(path1) - 1):
        if time.time() > deadline:
            break
        if path2[0] in adj[path1[i]] and path2[-1] in adj[path1[i + 1]]:
            combined = path1[:i + 1] + path2 + path1[i + 1:]
            if len(combined) > best_len and len(set(combined)) == len(combined):
                best = combined
                best_len = len(combined)
        if path2[-1] in adj[path1[i]] and path2[0] in adj[path1[i + 1]]:
            combined = path1[:i + 1] + path2[::-1] + path1[i + 1:]
            if len(combined) > best_len and len(set(combined)) == len(combined):
                best = combined
                best_len = len(combined)

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    best_path = []
    best_len = 0
    all_paths = []

    # Phase 0: Maximum Path Cover via Matching
    if time.time() < deadline - 50:
        cover_paths = find_max_path_cover()
        if cover_paths:
            stitched = stitch_paths(cover_paths)
            if stitched:
                extended = try_extend_both_ends(stitched)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = extended
                if len(extended) >= max(1, best_len - 3):
                    all_paths.append(extended)

            # Also try individual long paths from the cover as seeds
            for cp in cover_paths:
                if len(cp) > 1:
                    ext_cp = try_extend_both_ends(cp)
                    if len(ext_cp) > best_len:
                        best_len = len(ext_cp)
                        best_path = list(ext_cp)
                    if len(ext_cp) >= max(1, best_len - 3):
                        all_paths.append(list(ext_cp))

    # Phase 1: Greedy from all starts, keep top paths
    for start in range(N):
        if time.time() > deadline:
            break
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        if len(path) >= max(1, best_len - 3):
            all_paths.append(list(path))

    # Phase 2: DFS from multiple promising starting points (with time limit)
    if best_len > 0 and time.time() < deadline - 20:
        dfs_starts = list(set([best_path[0], best_path[-1]]))
        high_deg = sorted(range(N), key=lambda x: -DEG[x])[:5]
        dfs_starts.extend(high_deg)
        dfs_starts = list(set(dfs_starts))

        dfs_time_budget = deadline - 20 - time.time()
        per_start_time = max(2, dfs_time_budget / max(1, len(dfs_starts)))

        for dfs_start in dfs_starts:
            if time.time() > deadline - 20:
                break
            dfs_path = dfs_longest_path(dfs_start, min(time.time() + per_start_time, deadline - 20), best_len)
            if len(dfs_path) > best_len:
                best_len = len(dfs_path)
                best_path = list(dfs_path)

    # Phase 3: Beam search from promising starts
    if time.time() < deadline - 15:
        promising_starts = list(set([best_path[0], best_path[-1], 0, 3, 8, 9, 19, 24]))
        for start in promising_starts:
            if time.time() > deadline - 15:
                break
            path = beam_search_path(start, beam_width=80, max_depth=N, deadline=deadline - 15)
            path = try_extend_both_ends(path)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            if len(path) >= max(1, best_len - 3):
                all_paths.append(list(path))

    # Phase 4: Local search (2-opt) on the best path, then final extension
    if best_len > 0 and time.time() < deadline - 5:
        improved = local_search_improve(best_path, deadline - 5)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 4b: Try local search on second-best path too
    if best_len > 0 and len(all_paths) > 1 and time.time() < deadline - 3:
        second_best = max(all_paths, key=len, default=[])
        if second_best != best_path:
            improved2 = local_search_improve(second_best, deadline - 3)
            if len(improved2) > best_len:
                best_len = len(improved2)
                best_path = improved2

    # Phase 5: Try combining different paths
    if len(all_paths) > 1 and time.time() < deadline - 10:
        for i in range(len(all_paths)):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, len(all_paths)):
                if time.time() > deadline - 10:
                    break
                combined = combine_paths(all_paths[i], all_paths[j], deadline - 10)
                if len(combined) > best_len:
                    best_len = len(combined)
                    best_path = combined

    # Phase 6: Final end-extension
    if best_len > 0 and time.time() < deadline:
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 7: One more round of local search if we have time
    if best_len > 0 and time.time() < deadline - 2:
        improved = local_search_improve(best_path, deadline - 2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 8: Final polish - try more aggressive extension
    if best_len > 0 and time.time() < deadline:
        improved = list(best_path)
        seen = set(improved)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cur_l = improved[0]
            cands_l = adj[cur_l] - seen
            if cands_l:
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                improved.insert(0, v)
                seen.add(v)
                changed = True
            cur_r = improved[-1]
            cands_r = adj[cur_r] - seen
            if cands_r:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                improved.append(v)
                seen.add(v)
                changed = True
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Ensure valid nonempty path
    if not best_path:
        best_path = [0]

    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]

    return best_path
