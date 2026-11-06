"""Direct approach: find longest simple path in the pinned graph using
depth-first search with strong pruning, plus multi-start greedy and local search."""

import time
import sys
from collections import deque

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

# Precompute sorted neighbors (Warnsdorff order)
NEIGHBORS_SORTED = [sorted(adj[i], key=lambda w: (len(adj[w]), w)) for i in range(N)]

# Find connected components
def _find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

COMPONENTS = _find_components()
# Largest component
LARGEST_COMP = max(COMPONENTS, key=len)
LARGEST_COMP_SET = set(LARGEST_COMP)

def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in adj[path[i]]:
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
        v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def greedy_grow_bidirectional(start):
    """Grow a path from both ends using Warnsdorff heuristic."""
    path = [start]
    seen = {start}
    changed = True
    while changed:
        changed = False
        cur_l = path[0]
        cands_l = adj[cur_l] - seen
        cur_r = path[-1]
        cands_r = adj[cur_r] - seen
        if cands_l and cands_r:
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
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
            # Pick the end with fewer candidates (more constrained)
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def local_search_improve(path, deadline):
    """Improve a path via 2-opt style moves: try reversing segments, swapping
    internal edges, and re-extending. Returns the best path found."""
    path = list(path)
    best = list(path)
    best_len = len(path)
    n = len(path)
    
    # Try 2-opt: reverse a segment [i..j] and check if the new connections are valid
    for i in range(n - 1):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            # Check if reversing path[i..j] creates valid edges
            left_ok = (i == 0) or (path[j] in adj[path[i-1]])
            right_ok = (j == n - 1) or (path[i] in adj[path[j+1]])
            if left_ok and right_ok:
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                # Check all edges in new_path
                valid = True
                for k in range(len(new_path) - 1):
                    if new_path[k+1] not in adj[new_path[k]]:
                        valid = False
                        break
                if valid:
                    # Try extending
                    ext = try_extend_both_ends(new_path)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best = ext
                        # Update path for further 2-opt iterations
                        path = list(ext)
                        n = len(path)
                        i = 0  # restart
                        break
    return best

def dfs_longest_path(start, deadline, best_len):
    """
    Depth-first search for longest simple path starting at `start`.
    Uses time-limited search with pruning.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = 1
    
    path = [start]
    seen = [False] * N
    seen[start] = True
    
    # Use iterative DFS with explicit stack to avoid recursion limit issues
    # and to allow better time management
    stack = [(start, 0, iter(NEIGHBORS_SORTED[start]))]
    
    while stack:
        if time.time() > deadline:
            break
        cur, depth, it = stack[-1]
        
        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)
        
        advanced = False
        for nxt in it:
            if not seen[nxt]:
                seen[nxt] = True
                path.append(nxt)
                stack[-1] = (cur, depth, it)  # keep iterator state
                stack.append((nxt, depth + 1, iter(NEIGHBORS_SORTED[nxt])))
                advanced = True
                break
        
        if not advanced:
            # Backtrack
            stack.pop()
            seen[cur] = False
            if path:
                path.pop()
    
    return best_path

def beam_search_bidirectional(start, beam_width, max_depth, deadline):
    """Bidirectional beam search: grow from both ends simultaneously."""
    # Start with single vertex
    beam = [([start], {start})]
    best_path = [start]
    
    for depth in range(max_depth):
        if time.time() > deadline:
            break
        if not beam:
            break
        new_beam = []
        for path, seen in beam:
            # Try extending right end
            cur_r = path[-1]
            cands_r = adj[cur_r] - seen
            for v in cands_r:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
            
            # Try extending left end
            cur_l = path[0]
            cands_l = adj[cur_l] - seen
            for v in cands_l:
                new_path = [v] + path
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        
        if not new_beam:
            break
        # Sort by potential: prefer paths where both ends have fewer unvisited neighbors
        new_beam.sort(key=lambda x: (len(adj[x[0][0]] - x[1]) + len(adj[x[0][-1]] - x[1]), -len(x[0])))
        beam = new_beam[:beam_width]
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 87  # Safety margin before 90s limit
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from all starts in largest component
    for start in LARGEST_COMP:
        if time.time() > deadline:
            break
        path = greedy_grow_bidirectional(start)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: Beam search from all starts in largest component (limited)
    if time.time() < deadline - 5:
        for start in LARGEST_COMP[:30]:  # Limit to first 30 for time
            if time.time() > deadline - 5:
                break
            path = beam_search_bidirectional(start, beam_width=30, max_depth=N, deadline=deadline - 5)
            path = try_extend_both_ends(path)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 3: DFS from the best starting point (with time limit)
    if best_len > 0 and time.time() < deadline - 10:
        dfs_start = best_path[0]
        dfs_path = dfs_longest_path(dfs_start, min(time.time() + 5, deadline - 10), best_len)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = list(dfs_path)
        if time.time() < deadline - 8:
            dfs_start2 = best_path[-1]
            dfs_path2 = dfs_longest_path(dfs_start2, min(time.time() + 4, deadline - 8), best_len)
            if len(dfs_path2) > best_len:
                best_len = len(dfs_path2)
                best_path = list(dfs_path2)
    
    # Phase 4: Local search (2-opt) on the best path, then final extension
    if best_len > 0 and time.time() < deadline - 2:
        improved = local_search_improve(best_path, deadline - 2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 5: Final end-extension
    if best_len > 0 and time.time() < deadline:
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 6: One more round of local search if we have time
    if best_len > 0 and time.time() < deadline - 1:
        improved = local_search_improve(best_path, deadline - 1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        improved = try_extend_both_ends(best_path)
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
