import time

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Greedy lower bound (seed) + try to extend both ends
    def greedy_path():
        best = []
        for s in range(n):
            path = [s]
            seen = {s}
            while adj[path[-1]] - seen:
                v = min(adj[path[-1]] - seen, key=lambda w: (len(adj[w] - seen), w))
                path.append(v); seen.add(v)
            # extend other end
            while adj[path[0]] - seen:
                v = min(adj[path[0]] - seen, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v); seen.add(v)
            if len(path) > len(best):
                best = path
        return best

    def component_size(used, cur):
        # size of connected component of (V \ used) containing cur
        seen = {cur}
        stack = [cur]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in used and w not in seen:
                    seen.add(w); stack.append(w)
        return len(seen)

    def dead_end_count(used):
        c = 0
        for v in range(n):
            if v not in used and len(adj[v] - used) <= 1:
                c += 1
        return c

    # Hamiltonian-path feasibility (necessary conditions for a path using all n vertices)
    def hamiltonian_feasible():
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) < 1:
            return False
        c1 = sum(1 for d in degs if d == 1)
        if c1 > 2:
            return False
        c2 = sum(1 for d in degs if d == 2)
        if c2 > n - 2:
            return False
        return True

    # Exact branch-and-bound with strong pruning
    best_path = greedy_path()
    best_len = len(best_path)
    start_time = time.time()
    DEADLINE = start_time + 80.0   # safety margin within 90s
    node_count = [0]
    order = list(range(n))
    # start-order heuristic: prefer low-degree vertices as endpoints
    order.sort(key=lambda v: (len(adj[v]), v))

    def dfs(path, used, cur):
        if time.time() > DEADLINE:
            return
        node_count[0] += 1
        L = len(path)
        if L > best_len:
            best_len = L
            best_path = path[:]
            if best_len == n:   # Hamiltonian path found: optimal
                return
        # upper bound via connected component of unused containing cur
        ub = L + component_size(used, cur)
        if ub <= best_len:
            return
        # dead-end pruning: unused vertices with <=1 unused neighbor are forced;
        # if more than 1 such vertex, impossible to complete
        c = dead_end_count(used)
        if c > 1:
            return
        # candidate neighbours
        cands = [w for w in adj[cur] if w not in used]
        if not cands:
            return
        # forced-move rule
        forced = None
        if c == 1:
            for v in range(n):
                if v not in used and len(adj[v] - used) == 1:
                    forced = v
                    break
            if forced is not None and forced not in cands:
                return
            if forced is not None:
                cands = [forced]
        else:
            # order: prefer candidates that are "forced" (low unused-degree) first
            cands.sort(key=lambda w: (len(adj[w] - used), w))
        for w in cands:
            used.add(w)
            path.append(w)
            dfs(path, used, w)
            path.pop()
            used.discard(w)

    # If Hamiltonian path is feasible, try to find one (it is optimal if found)
    if hamiltonian_feasible():
        for s in order:
            if time.time() > DEADLINE:
                break
            used = {s}
            path = [s]
            dfs(path, used, s)
            if best_len == n:
                break

    return best_path
# EVOLVE-BLOCK-END
