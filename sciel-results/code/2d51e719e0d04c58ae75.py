import time
import random
from collections import deque
import sys
import numpy as np

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    return components

def _greedy_path(adj, start, n, max_steps=200):
    path = [start]
    visited = {start}
    steps = 0
    while steps < max_steps:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        if not cands:
            break
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
        steps += 1
    return path

def _try_extend_path(adj, path, n, deadline):
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        path_set = set(path)
        end = path[-1]
        cands = [nb for nb in adj[end] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.append(cands[0])
            improved = True
            continue
        start = path[0]
        cands = [nb for nb in adj[start] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.insert(0, cands[0])
            improved = True
            continue
    return path

def _insert_vertex(adj, path, vertex, n):
    if vertex in set(path):
        return None
    if vertex in adj[path[0]]:
        return [vertex] + path
    if vertex in adj[path[-1]]:
        return path + [vertex]
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    return None

def _2opt_improve(adj, path, n, deadline):
    best_path = path[:]
    best_len = len(best_path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_path) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_path)):
                if time.time() > deadline:
                    break
                if i > 0:
                    if best_path[j] not in adj[best_path[i-1]]:
                        continue
                if j + 1 < len(best_path):
                    if best_path[i] not in adj[best_path[j+1]]:
                        continue
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                valid = True
                for k in range(len(new_path) - 1):
                    if new_path[k+1] not in adj[new_path[k]]:
                        valid = False
                        break
                if not valid:
                    continue
                new_path = _try_extend_path(adj, new_path, n, deadline)
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(best_path)
                    improved = True
                    break
            if improved:
                break
    return best_path

def _swap_improve(adj, path, n, deadline):
    best_path = path[:]
    best_len = len(best_path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(best_path)
        for i in range(L - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, L):
                if time.time() > deadline:
                    break
                if i > 0:
                    if best_path[j] not in adj[best_path[i-1]]:
                        continue
                if j + 1 < L:
                    if best_path[i] not in adj[best_path[j+1]]:
                        continue
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                valid = True
                if i > 0 and new_path[i-1] not in adj[new_path[i]]:
                    valid = False
                if valid and j + 1 < L and new_path[j] not in adj[new_path[j+1]]:
                    valid = False
                if not valid:
                    continue
                new_path = _try_extend_path(adj, new_path, n, deadline)
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(new_path)
                    improved = True
                    break
            if improved:
                break
    return best_path

def _limited_dfs(adj, start, n, deadline, max_depth=30):
    best_path = [start]
    best_len = 1
    
    def dfs(path, visited):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        if len(path) >= max_depth:
            return
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        for nxt in cands:
            if time.time() > deadline:
                return
            path.append(nxt)
            visited.add(nxt)
            dfs(path, visited)
            path.pop()
            visited.remove(nxt)
    
    dfs([start], {start})
    return best_path

def _multi_start_search(adj, n, deadline):
    best_path = []
    best_len = 0
    rng = random.Random(748)
    
    for start in range(n):
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts[:50]:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    return best_path

def _dfs_enhanced(adj, n, deadline):
    best_path = []
    best_len = 0
    for start in range(n):
        if time.time() > deadline:
            break
        path = _limited_dfs(adj, start, n, deadline, max_depth=25)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    return best_path

def _bfs_diameter_path(adj, comp, n):
    if len(comp) <= 1:
        return list(comp)
    comp_set = set(comp)
    start = comp[0]
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in adj[u]:
            if v in comp_set and v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    dist2 = {farthest: 0}
    parent2 = {farthest: None}
    q = deque([farthest])
    farthest2 = farthest
    while q:
        u = q.popleft()
        if dist2[u] > dist2[farthest2]:
            farthest2 = u
        for v in adj[u]:
            if v in comp_set and v not in dist2:
                dist2[v] = dist2[u] + 1
                parent2[v] = u
                q.append(v)
    path = []
    cur = farthest2
    while cur is not None:
        path.append(cur)
        cur = parent2[cur]
    path.reverse()
    return path

def _branch_and_bound(adj, n, deadline, max_depth=40):
    best_path = []
    best_len = 0
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        visited = {start}
        
        def dfs():
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            if len(path) >= max_depth:
                return
            cur = path[-1]
            cands = [nb for nb in adj[cur] if nb not in visited]
            cands.sort(key=lambda x: len(adj[x] - visited))
            for nxt in cands:
                if time.time() > deadline:
                    return
                path.append(nxt)
                visited.add(nxt)
                dfs()
                path.pop()
                visited.remove(nxt)
        
        dfs()
    return best_path

def _mwpm_path_cover(adj, n, deadline):
    """Use Hungarian algorithm to find a path cover, then extract longest path."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return []
    
    # Build bipartite graph: left side = in-copies, right side = out-copies
    # Edge (u_in, v_out) exists if (u,v) is an edge in original graph
    # Weight: 1 for original edges, 0 for non-edges
    # Also add self-loops (u_in, u_out) with weight 0 to allow skipping
    
    # Size: n x n matrix
    weight = np.zeros((n, n), dtype=float)
    for u in range(n):
        for v in adj[u]:
            weight[u][v] = 1.0
    
    # We want to maximize weight, so negate for linear_sum_assignment (which minimizes)
    neg_weight = -weight
    
    # Solve
    row_ind, col_ind = linear_sum_assignment(neg_weight)
    
    # Reconstruct paths from the matching
    # Each row u matched to col v means u -> v in the path
    next_node = [-1] * n
    prev_node = [-1] * n
    for u, v in zip(row_ind, col_ind):
        next_node[u] = v
        prev_node[v] = u
    
    # Find all paths
    visited = [False] * n
    paths = []
    for i in range(n):
        if not visited[i] and prev_node[i] == -1:
            # Start of a path
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1:
                cur = next_node[cur]
                if visited[cur]:
                    break
                visited[cur] = True
                path.append(cur)
            paths.append(path)
    
    # Also handle cycles
    for i in range(n):
        if not visited[i]:
            path = [i]
            visited[i] = True
            cur = i
            while next_node[cur] != -1 and not visited[next_node[cur]]:
                cur = next_node[cur]
                visited[cur] = True
                path.append(cur)
            paths.append(path)
    
    # Return the longest path
    if not paths:
        return []
    longest = max(paths, key=len)
    return longest

def _exact_longest_path_small(adj, comp, n, deadline):
    """Exact DP for longest path in a component of size <= 18."""
    k = len(comp)
    if k <= 1:
        return list(comp)
    if k > 18:
        return None
    
    vmap = {v: i for i, v in enumerate(comp)}
    imap = {i: v for i, v in enumerate(comp)}
    
    local_adj = [set() for _ in range(k)]
    for u in comp:
        for v in adj[u]:
            if v in vmap:
                local_adj[vmap[u]].add(vmap[v])
    
    dp = [set() for _ in range(1 << k)]
    for i in range(k):
        dp[1 << i].add(i)
    
    best_len = 1
    best_mask = 0
    best_end = 0
    
    for mask in range(1, 1 << k):
        if time.time() > deadline:
            return None
        endpoints = dp[mask]
        if not endpoints:
            continue
        popcount = bin(mask).count('1')
        if popcount > best_len:
            best_len = popcount
            best_mask = mask
            best_end = list(endpoints)[0] if endpoints else 0
        for last in endpoints:
            for nxt in local_adj[last]:
                if not (mask & (1 << nxt)):
                    new_mask = mask | (1 << nxt)
                    dp[new_mask].add(nxt)
    
    path = []
    mask = best_mask
    end = best_end
    while mask:
        path.append(end)
        if mask == (1 << end):
            break
        prev_mask = mask ^ (1 << end)
        found = False
        for prev in dp[prev_mask]:
            if end in local_adj[prev]:
                path.append(prev)
                mask = prev_mask
                end = prev
                found = True
                break
        if not found:
            break
    
    path.reverse()
    result = [imap[v] for v in path]
    return result

def _path_merge(adj, path1, path2, n, deadline):
    """Try to merge two paths into one longer path."""
    if not path1 or not path2:
        return path1 if path1 else path2
    if len(path1) < len(path2):
        path1, path2 = path2, path1
    
    set1 = set(path1)
    set2 = set(path2)
    if set1 & set2:
        return path1
    
    # Try to connect end of path1 to start of path2
    if path1[-1] in adj[path2[0]]:
        return path1 + path2
    # Try to connect end of path1 to end of path2 (reverse path2)
    if path1[-1] in adj[path2[-1]]:
        return path1 + path2[::-1]
    # Try to connect start of path1 to start of path2
    if path1[0] in adj[path2[0]]:
        return path2 + path1
    # Try to connect start of path1 to end of path2
    if path1[0] in adj[path2[-1]]:
        return path2[::-1] + path1
    
    return path1

def construct():
    n = P['n']
    adj = _build_adj()
    
    deadline = time.time() + 85.0
    
    components = _find_components(adj, n)
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in components:
        if time.time() > deadline - 5.0:
            break
        comp_size = len(comp)
        if comp_size <= len(best_path) and comp_size <= 18:
            continue
        if comp_size <= 18:
            exact_path = _exact_longest_path_small(adj, comp, n, deadline)
            if exact_path is not None and len(exact_path) > len(best_path):
                best_path = exact_path
        else:
            edge_count = sum(len(adj[u]) for u in comp) // 2
            if edge_count == comp_size - 1:
                path = _bfs_diameter_path(adj, comp, n)
                if len(path) > len(best_path):
                    best_path = path
            else:
                ms_best = _multi_start_search(adj, n, deadline)
                if len(ms_best) > len(best_path):
                    best_path = ms_best
    
    if len(best_path) < 20:
        bb_best = _branch_and_bound(adj, n, deadline)
        if len(bb_best) > len(best_path):
            best_path = bb_best
    
    # Try MWPM-based path cover
    if time.time() < deadline - 10.0:
        mwpm_path = _mwpm_path_cover(adj, n, deadline)
        if mwpm_path:
            mwpm_path = _try_extend_path(adj, mwpm_path, n, deadline)
            if len(mwpm_path) > len(best_path):
                best_path = mwpm_path
    
    # Optimize
    best_path = _try_extend_path(adj, best_path, n, deadline)
    best_path = _2opt_improve(adj, best_path, n, deadline)
    best_path = _swap_improve(adj, best_path, n, deadline)
    
    # Iterative improvement
    iteration = 0
    while time.time() < deadline - 3.0:
        iteration += 1
        path_set = set(best_path)
        inserted = False
        for v in range(n):
            if v not in path_set:
                new_path = _insert_vertex(adj, best_path, v, n)
                if new_path is not None:
                    best_path = new_path
                    path_set = set(best_path)
                    inserted = True
                    break
        if not inserted:
            new_best = _2opt_improve(adj, best_path, n, deadline)
            if len(new_best) > len(best_path):
                best_path = new_best
            else:
                new_best2 = _swap_improve(adj, best_path, n, deadline)
                if len(new_best2) > len(best_path):
                    best_path = new_best2
                else:
                    if iteration < 8:
                        fresh = _multi_start_search(adj, n, deadline)
                        fresh = _try_extend_path(adj, fresh, n, deadline)
                        if len(fresh) > len(best_path):
                            best_path = fresh
                        else:
                            dfs_best = _dfs_enhanced(adj, n, deadline)
                            dfs_best = _try_extend_path(adj, dfs_best, n, deadline)
                            if len(dfs_best) > len(best_path):
                                best_path = dfs_best
                            else:
                                bb_best = _branch_and_bound(adj, n, deadline)
                                bb_best = _try_extend_path(adj, bb_best, n, deadline)
                                if len(bb_best) > len(best_path):
                                    best_path = bb_best
                                else:
                                    break
        else:
            best_path = _try_extend_path(adj, best_path, n, deadline)
            best_path = _2opt_improve(adj, best_path, n, deadline)
    
    if len(best_path) == 0:
        best_path = [0]
    
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in adj[best_path[i]], f"Invalid edge at position {i}"
    assert len(set(best_path)) == len(best_path), "Path has repeated vertices"
    
    return best_path
# EVOLVE-BLOCK-END
