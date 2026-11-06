import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}


def construct():
    t0 = time.time()
    deadline = t0 + 80.0   # explicit safety margin well under the 90s kill

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(s) for s in adj]

    def time_up():
        return time.time() >= deadline

    # ---- connected components ----
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
                for nei in adj_list[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)

    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)

    if comp_size == 0:
        return [0]
    if comp_size == 1:
        return largest_comp[:]

    # degree restricted to the component
    deg = {u: len(adj[u] & comp_set) for u in largest_comp}

    e_count = sum(deg.values()) // 2
    is_tree = (e_count == comp_size - 1)

    def valid_path(path):
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        for a, b in zip(path, path[1:]):
            if b not in adj[a]:
                return False
        return True

    # ---- tree case: diameter is the longest path ----
    if is_tree:
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj_list[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, parent

        u, _ = bfs_farthest(largest_comp[0])
        v, parent = bfs_farthest(u)
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path

    # ---------------- general (cyclic) case ----------------
    best = []
    best_len = 0

    def record(path):
        nonlocal best, best_len
        if len(path) > best_len:
            best_len = len(path)
            best = path[:]

    def onward_count(v, seen):
        c = 0
        for w in adj_list[v]:
            if w in comp_set and w not in seen:
                c += 1
        return c

    def greedy_extend(start, mode):
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [w for w in adj_list[cur] if w in comp_set and w not in seen]
            if not cands:
                break
            if mode == 'min':
                v = min(cands, key=lambda w: (onward_count(w, seen), deg[w], w))
            elif mode == 'max':
                v = max(cands, key=lambda w: (onward_count(w, seen), deg[w], -w))
            else:
                v = cands[0]
            path.append(v)
            seen.add(v)
        return path

    def extend_both_ends(path):
        p = path[:]
        seen = set(p)
        # extend front
        while True:
            cur = p[0]
            cands = [w for w in adj_list[cur] if w in comp_set and w not in seen]
            if not cands:
                break
            v = min(cands, key=lambda w: (onward_count(w, seen), deg[w], w))
            p.insert(0, v)
            seen.add(v)
        # extend back
        while True:
            cur = p[-1]
            cands = [w for w in adj_list[cur] if w in comp_set and w not in seen]
            if not cands:
                break
            v = min(cands, key=lambda w: (onward_count(w, seen), deg[w], w))
            p.append(v)
            seen.add(v)
        return p

    def two_opt(path):
        """Try to find a longer valid path via a 2-opt-style segment flip."""
        L = len(path)
        if L < 4:
            return None
        pos = {v: i for i, v in enumerate(path)}
        for i in range(L - 1):
            a = path[i]
            for j in range(i + 2, L):
                b = path[j]
                if a in adj[b]:
                    # segment path[i+1..j] is valid forward; reversed is valid
                    # only if every consecutive pair is an edge (undirected: yes).
                    new = path[:i + 1] + path[i + 1:j + 1][::-1] + path[j + 1:]
                    if valid_path(new) and len(new) > L:
                        return new
        return None

    # ---- phase 1: greedy from every vertex with several heuristics ----
    for start in largest_comp:
        if time_up():
            break
        for mode in ('min', 'max', 'first'):
            if time_up():
                break
            p = greedy_extend(start, mode)
            record(p)
        if best_len == comp_size:
            break

    # ---- phase 2: extend the best path from both ends, then 2-opt ----
    if best and best_len < comp_size and not time_up():
        ext = extend_both_ends(best)
        if len(ext) > best_len:
            record(ext)
        if best_len < comp_size:
            for _ in range(3):
                if time_up():
                    break
                if best_len == comp_size:
                    break
                cand = two_opt(best)
                if cand is None:
                    break
                ext = extend_both_ends(cand)
                if len(ext) > best_len:
                    record(ext)
                else:
                    break

    # ---- phase 3: bounded DFS with a Warnsdorff heuristic and a tight bound ----
    if best_len < comp_size and not time_up():
        def dfs(node, path, seen):
            nonlocal best, best_len
            if time_up():
                return
            if len(path) > best_len:
                best_len = len(path)
                best = path[:]
            if best_len == comp_size:
                return
            remaining = comp_size - len(seen)
            if len(path) + remaining <= best_len:
                return
            cands = [w for w in adj_list[node] if w in comp_set and w not in seen]
            if not cands:
                return
            cands.sort(key=lambda w: (onward_count(w, seen), deg[w], w))
            for v in cands:
                path.append(v)
                seen.add(v)
                dfs(v, path, seen)
                path.pop()
                seen.remove(v)
                if time_up():
                    return
                if best_len == comp_size:
                    return

        # start from low-degree (leaves) first, then a few high-degree nodes
        starts = sorted(largest_comp, key=lambda x: (deg[x], x))
        starts += sorted(largest_comp, key=lambda x: (-deg[x], x))
        for start in starts:
            if time_up():
                break
            if best_len == comp_size:
                break
            dfs(start, [start], {start})

    # ---- final safety: extend the best path once more ----
    if best and best_len < comp_size and not time_up():
        ext = extend_both_ends(best)
        if len(ext) > best_len:
            record(ext)

    if not best:
        best = [largest_comp[0]]
    if not valid_path(best):
        # absolute fallback: a single valid vertex
        best = [largest_comp[0]]
    return best
