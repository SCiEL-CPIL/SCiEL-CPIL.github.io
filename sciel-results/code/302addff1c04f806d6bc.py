"""Longest Path via Directed Line Graph Matching + DFS Refinement."""

import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.csgraph import connected_components

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

# Build directed line graph for matching formulation
# Nodes in line graph = directed edges (u,v) where (u,v) is an edge in original graph
# Two directed edges (u,v) and (w,x) are adjacent in line graph if v == w
# We want a maximum matching in this bipartite graph:
#   Left side: directed edges as "outgoing from u" 
#   Right side: directed edges as "incoming to v"
#   Edge exists if the outgoing edge's head = incoming edge's tail
# Actually, let me think more carefully.
#
# A path v1-v2-...-vk corresponds to directed edges (v1,v2), (v2,v3), ..., (v_{k-1},vk)
# Constraints: each vertex appears at most once, so:
#   - At most one directed edge enters each vertex (except start)
#   - At most one directed edge leaves each vertex (except end)
#   - For internal vertices, exactly one enters and one leaves
#
# This is equivalent to finding a maximum set of directed edges such that:
#   - No two share a tail (outgoing constraint)
#   - No two share a head (incoming constraint)
# This is a bipartite matching problem!
#
# Left nodes: vertices (as tails)
# Right nodes: vertices (as heads)  
# Edge (u_left, v_right) exists if (u,v) is an edge in original graph
# Maximum matching gives us a set of vertex-disjoint directed edges
# These form a collection of paths and cycles. We need to extract the longest path.

def build_line_graph_matching():
    """
    Build bipartite graph for matching formulation.
    Left: vertices 0..N-1 (as tails)
    Right: vertices 0..N-1 (as heads)
    Edge (u_left, v_right) if (u,v) in original graph.
    """
    # Build sparse matrix
    rows = []
    cols = []
    for u, v in EDGES:
        rows.append(u)
        cols.append(v)
        rows.append(v)
        cols.append(u)
    
    data = [1] * len(rows)
    mat = csr_matrix((data, (rows, cols)), shape=(N, N))
    return mat


def extract_paths_from_matching(matching):
    """
    Given a matching (list where matching[i] = j means left i matched to right j),
    extract all paths and cycles formed by the directed edges.
    Returns list of paths (each path is a list of vertices).
    """
    # matching[i] = j means directed edge (i, j) is in the path set
    # Build forward and backward maps
    forward = {}  # u -> v if (u,v) is in matching
    backward = {}  # v -> u if (u,v) is in matching
    
    for u in range(N):
        v = matching[u]
        if v != -1:
            forward[u] = v
            backward[v] = u
    
    # Find all paths and cycles
    visited = set()
    paths = []
    
    for start in range(N):
        if start in visited:
            continue
        if start not in forward:
            # start is a path endpoint (no outgoing edge)
            # Walk backward to find the full path
            path = [start]
            cur = start
            while cur in backward:
                cur = backward[cur]
                path.append(cur)
                visited.add(cur)
            path.reverse()
            # Mark all as visited
            for v in path:
                visited.add(v)
            if len(path) > 1:
                paths.append(path)
        elif start not in backward:
            # start has outgoing but no incoming - it's a path start
            path = [start]
            cur = start
            while cur in forward:
                cur = forward[cur]
                path.append(cur)
                visited.add(cur)
            for v in path:
                visited.add(v)
            if len(path) > 1:
                paths.append(path)
    
    # Check for cycles (shouldn't happen if we have a valid matching, but just in case)
    # Actually, a matching can form cycles. Let's handle that.
    for start in range(N):
        if start in visited:
            continue
        # Try to find a cycle
        cycle = [start]
        cur = start
        in_cycle = {start}
        while True:
            if cur not in forward:
                break
            cur = forward[cur]
            if cur in in_cycle:
                # Found a cycle, extract it
                idx = cycle.index(cur)
                cycle = cycle[idx:]
                for v in cycle:
                    visited.add(v)
                if len(cycle) > 1:
                    paths.append(cycle)
                break
            if cur in visited:
                break
            cycle.append(cur)
            in_cycle.add(cur)
    
    return paths


def combine_paths(paths, adj):
    """
    Try to combine multiple paths into longer paths by finding connections
    between path endpoints.
    """
    if not paths:
        return []
    
    # Sort by length descending
    paths = sorted(paths, key=len, reverse=True)
    
    # For each path, record its endpoints
    # Try to merge paths by connecting endpoints
    
    merged = True
    while merged:
        merged = False
        new_paths = []
        used = [False] * len(paths)
        
        for i in range(len(paths)):
            if used[i]:
                continue
            pi = paths[i]
            if len(pi) < 2:
                new_paths.append(pi)
                continue
            
            start_i, end_i = pi[0], pi[-1]
            
            for j in range(i + 1, len(paths)):
                if used[j]:
                    continue
                pj = paths[j]
                if len(pj) < 2:
                    continue
                
                start_j, end_j = pj[0], pj[-1]
                
                # Try all 4 combinations of connecting endpoints
                # 1. end_i -> start_j
                if end_i in adj[start_j]:
                    new_paths.append(pi + pj)
                    used[i] = True
                    used[j] = True
                    merged = True
                    break
                # 2. end_i -> end_j (reverse pj)
                if end_i in adj[end_j]:
                    new_paths.append(pi + list(reversed(pj)))
                    used[i] = True
                    used[j] = True
                    merged = True
                    break
                # 3. start_i -> start_j (reverse pi)
                if start_i in adj[start_j]:
                    new_paths.append(list(reversed(pi)) + pj)
                    used[i] = True
                    used[j] = True
                    merged = True
                    break
                # 4. start_i -> end_j (reverse both)
                if start_i in adj[end_j]:
                    new_paths.append(list(reversed(pi)) + list(reversed(pj)))
                    used[i] = True
                    used[j] = True
                    merged = True
                    break
            
            if not used[i]:
                new_paths.append(pi)
        
        paths = new_paths
    
    return paths


def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj[path[i]]:
            return False
    return True


def local_search_improve(path, max_iters=200):
    """
    Try to improve a path using 2-opt style moves and endpoint extensions.
    Returns improved path.
    """
    if not path:
        return path
    
    path = list(path)
    best_len = len(path)
    improved = True
    
    for _ in range(max_iters):
        if not improved:
            break
        improved = False
        
        # Try extending from both ends
        cur_path = list(path)
        seen = set(cur_path)
        
        # Try to extend left
        for _ in range(10):
            cur = cur_path[0]
            candidates = adj[cur] - seen
            if not candidates:
                break
            # Choose candidate with fewest unvisited neighbors (Warnsdorff)
            v = min(candidates, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            cur_path.insert(0, v)
            seen.add(v)
            improved = True
        
        # Try to extend right
        for _ in range(10):
            cur = cur_path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                break
            v = min(candidates, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            cur_path.append(v)
            seen.add(v)
            improved = True
        
        if len(cur_path) > best_len:
            path = cur_path
            best_len = len(path)
            improved = True
        else:
            break
    
    return path


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


def beam_search_path(start, beam_width=50, deadline=None):
    """
    Beam search for longest path starting from `start`.
    Maintains a beam of partial paths, expanding the most promising ones.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    # Each beam entry: (path, seen_set)
    beam = [([start], {start})]
    best_path = [start]
    best_len = 1
    
    while time.time() < deadline:
        new_beam = []
        
        for path, seen in beam:
            cur = path[-1]
            candidates = adj[cur] - seen
            
            if not candidates:
                # Dead end, keep as candidate
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                continue
            
            # Score candidates
            scored = []
            for v in candidates:
                # Prefer vertices with fewer unvisited neighbors (Warnsdorff)
                unv_count = sum(1 for nb in adj[v] if nb not in seen and nb != cur)
                scored.append((unv_count, DEG[v], v))
            
            scored.sort()
            
            # Add top candidates to new beam
            for _, _, v in scored[:3]:  # Keep top 3 per path
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)
        
        if not new_beam:
            break
        
        # Prune beam to keep only top beam_width
        new_beam.sort(key=lambda x: (-len(x[0]), sum(1 for nb in adj[x[0][-1]] if nb not in x[1])))
        beam = new_beam[:beam_width]
        
        # Early stop if we found a very long path
        if best_len >= N - 1:
            break
    
    return best_path


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


def dfs_from_both_ends(start, deadline, best_len):
    """
    DFS that grows a path from both ends simultaneously.
    At each step, choose which end to extend based on which has fewer options.
    """
    best_path = [start]
    best_len_local = max(1, best_len)
    
    path = [start]
    seen = [False] * N
    seen[start] = True
    
    def _dfs():
        nonlocal best_path, best_len_local
        
        if time.time() > deadline:
            return
        
        if len(path) > best_len_local:
            best_len_local = len(path)
            best_path = list(path)
        
        # Check if we can extend either end
        cur_l = path[0]
        cands_l = [v for v in adj[cur_l] if not seen[v]]
        cur_r = path[-1]
        cands_r = [v for v in adj[cur_r] if not seen[v]]
        
        if not cands_l and not cands_r:
            return
        
        # Choose which end to extend
        if cands_l and cands_r:
            # Extend the end with fewer candidates (more constrained)
            if len(cands_l) <= len(cands_r):
                extend_left = True
            else:
                extend_left = False
        elif cands_l:
            extend_left = True
        else:
            extend_left = False
        
        if extend_left:
            # Sort candidates by Warnsdorff
            cands_l.sort(key=lambda w: (len(adj[w] - {x for x in path}), DEG[w], w))
            for v in cands_l:
                if time.time() > deadline:
                    return
                seen[v] = True
                path.insert(0, v)
                _dfs()
                path.pop(0)
                seen[v] = False
        else:
            cands_r.sort(key=lambda w: (len(adj[w] - {x for x in path}), DEG[w], w))
            for v in cands_r:
                if time.time() > deadline:
                    return
                seen[v] = True
                path.append(v)
                _dfs()
                path.pop()
                seen[v] = False
    
    _dfs()
    return best_path


def check_connectivity(unvisited_mask, current_node):
    """
    Check if all unvisited nodes are in the same connected component as current_node
    in the graph induced by unvisited nodes + current_node.
    If not, moving to a neighbor that separates them is bad.
    """
    if not any(unvisited_mask):
        return True

    visited = set()
    stack = [current_node]
    visited.add(current_node)
    
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb in visited:
                continue
            if unvisited_mask[nb]:
                visited.add(nb)
                stack.append(nb)
    
    for i in range(N):
        if unvisited_mask[i] and i not in visited:
            return False
    return True


def dfs_longest_path(start, deadline, best_len):
    """
    Depth-first search for longest simple path starting at `start`.
    Uses time-limited search with strong connectivity pruning.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = max(1, best_len)

    path = [start]
    seen = [False] * N
    seen[start] = True
    unvisited_mask = [True] * N
    unvisited_mask[start] = False
    unvisited_count = N - 1

    def _dfs(cur, depth):
        nonlocal best_path, best_len_local, unvisited_count

        if time.time() > deadline:
            return

        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)

        if depth + unvisited_count <= best_len_local:
            return

        candidates = []
        for nxt in NEIGHBORS_SORTED[cur]:
            if not seen[nxt]:
                candidates.append(nxt)
        
        if not candidates:
            return

        def count_unvisited_neighbors(node):
            count = 0
            for nb in adj[node]:
                if unvisited_mask[nb]:
                    count += 1
            return count

        candidates.sort(key=lambda x: count_unvisited_neighbors(x))

        for nxt in candidates:
            if time.time() > deadline:
                return
            
            if unvisited_count > 0:
                unv_nb_count = 0
                for nb in adj[nxt]:
                    if unvisited_mask[nb] and nb != cur:
                        unv_nb_count += 1
                
                if unv_nb_count == 0 and unvisited_count > 1:
                    continue

                reachable = set()
                stack = [nxt]
                reachable.add(nxt)
                
                while stack:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb in reachable:
                            continue
                        if unvisited_mask[nb]:
                            reachable.add(nb)
                            stack.append(nb)
                
                if len(reachable) < unvisited_count:
                    if depth + len(reachable) <= best_len_local:
                        continue

            seen[nxt] = True
            unvisited_mask[nxt] = False
            path.append(nxt)
            unvisited_count -= 1
            _dfs(nxt, depth + 1)
            path.pop()
            unvisited_mask[nxt] = True
            seen[nxt] = False
            unvisited_count += 1

    _dfs(start, 1)
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    best_path = []
    best_len = 0

    # Phase 0: Matching-based approach to get initial path segments
    try:
        mat = build_line_graph_matching()
        matching = maximum_bipartite_matching(mat)
        
        # matching[i] = j means directed edge (i, j) is selected
        # Convert to list format
        matching_list = [-1] * N
        for i in range(N):
            if matching[i] >= 0:
                matching_list[i] = int(matching[i])
        
        paths = extract_paths_from_matching(matching_list)
        paths = combine_paths(paths, adj)
        
        for p in paths:
            if len(p) > best_len:
                best_len = len(p)
                best_path = list(p)
    except Exception:
        pass

    # Phase 1: Greedy from all nodes
    for start in range(N):
        if time.time() > deadline:
            break
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
    high_deg = sorted(range(N), key=lambda x: -DEG[x])[:10]
    for start in high_deg:
        if time.time() > deadline:
            break
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    # Phase 2: Beam search from promising starts
    beam_starts = [best_path[0], best_path[-1]] + high_deg[:5]
    for v in range(N):
        if DEG[v] <= 2:
            beam_starts.append(v)
    beam_starts = list(set(beam_starts))
    
    beam_deadline = start_time + 80 * 0.4
    for start in beam_starts:
        if time.time() > beam_deadline:
            break
        bs_path = beam_search_path(start, beam_width=30, deadline=beam_deadline)
        bs_path = local_search_improve(bs_path, max_iters=50)
        if len(bs_path) > best_len:
            best_len = len(bs_path)
            best_path = list(bs_path)

    # Phase 3: DFS with Pruning
    dfs_starts = [best_path[0], best_path[-1]]
    dfs_starts.extend(high_deg[:5])
    for v in range(N):
        if DEG[v] == 1:
            dfs_starts.append(v)
    dfs_starts = list(set(dfs_starts))
    
    dfs_deadline = start_time + 80 * 0.75
    for start in dfs_starts:
        if time.time() > dfs_deadline:
            break
        dfs_path = dfs_longest_path(start, dfs_deadline, best_len)
        dfs_path = local_search_improve(dfs_path, max_iters=100)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = list(dfs_path)

    # Phase 4: DFS from both ends
    if time.time() < start_time + 80 * 0.9:
        two_end_deadline = start_time + 80 * 0.9
        for start in [best_path[0], best_path[-1]] + high_deg[:3]:
            if time.time() > two_end_deadline:
                break
            te_path = dfs_from_both_ends(start, two_end_deadline, best_len)
            te_path = local_search_improve(te_path, max_iters=100)
            if len(te_path) > best_len:
                best_len = len(te_path)
                best_path = list(te_path)

    # Phase 5: Final Polish
    if best_len > 0 and time.time() < deadline:
        improved = try_extend_both_ends(best_path)
        improved = local_search_improve(improved, max_iters=200)
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
