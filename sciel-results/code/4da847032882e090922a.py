import time
from random import Random
from collections import defaultdict

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start = time.time()
    deadline = start + 105  # safety margin below the 120s limit
    trees = instance_trees(20, 611)
    N = 20
    base_order = list(range(19, -1, -1))  # large first: keeps graph dense for hard trees

    def embed_csp(tree_edges, k, used_edges, rng, node_limit=500000):
        if k == 1:
            return [0]
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for a, b in tree_edges:
            deg[a] += 1; deg[b] += 1
            adj[a].append(b); adj[b].append(a)
        var_order = sorted(range(k), key=lambda v: -deg[v])  # most constrained first
        host_deg = [0] * N
        for (x, y) in used_edges:
            host_deg[x] += 1; host_deg[y] += 1
        host_rem = [N - 1 - host_deg[h] for h in range(N)]
        mapping = [-1] * k
        used_hosts = set()
        nodes = [0]

        def get_candidates(v):
            """Get valid host candidates for tree vertex v given current partial assignment."""
            cands = []
            for h in range(N):
                if h in used_hosts:
                    continue
                ok = True
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1 and (min(h, mw), max(h, mw)) in used_edges:
                        ok = False
                        break
                if ok:
                    cands.append(h)
            return cands

        def backtrack(idx):
            if nodes[0] > node_limit:
                return False
            if idx == k:
                return True
            
            # Dynamic variable ordering: pick unassigned vertex with fewest candidates
            best_v = -1
            best_cands = None
            best_len = N + 1
            for i in range(idx, k):
                v = var_order[i]
                if mapping[v] == -1:
                    cands = get_candidates(v)
                    if len(cands) == 0:
                        return False  # dead end
                    if len(cands) < best_len:
                        best_len = len(cands)
                        best_v = v
                        best_cands = cands
                        if best_len == 1:
                            break  # can't do better than 1
            if best_v == -1:
                # All assigned? Shouldn't happen if idx < k, but handle edge case
                return True
            
            # Swap best_v into position idx for consistent backtracking
            pos_in_order = var_order.index(best_v)
            if pos_in_order != idx:
                var_order[idx], var_order[pos_in_order] = var_order[pos_in_order], var_order[idx]
            
            v = best_v
            cands = best_cands
            
            if deg[v] >= 3:
                cands.sort(key=lambda h: -host_rem[h])
            else:
                rng.shuffle(cands)
            
            for h in cands:
                nodes[0] += 1
                if nodes[0] > node_limit:
                    return False
                mapping[v] = h
                used_hosts.add(h)
                added = []
                for w in adj[v]:
                    mw = mapping[w]
                    if mw != -1:
                        e = (min(h, mw), max(h, mw))
                        used_edges.add(e)
                        added.append(e)
                if backtrack(idx + 1):
                    return True
                for e in added:
                    used_edges.discard(e)
                used_hosts.discard(h)
                mapping[v] = -1
            return False

        if backtrack(0):
            return mapping
        return None

    def evaluate(order, rng):
        used_edges = set()
        result = [None] * N
        total = 0
        for idx in order:
            if time.time() > deadline:
                break
            k = idx + 1
            m = embed_csp(trees[idx], k, used_edges, rng)
            if m is not None:
                result[idx] = [int(x) for x in m]
                total += len(trees[idx])
        return result, total

    master_rng = Random(42)
    orders = [base_order]
    for _ in range(300):  # large-first variants, T20 anchored first
        o = list(range(19, -1, -1))
        master_rng.shuffle(o[1:])
        orders.append(o)
    for _ in range(5000):  # fully random restarts
        o = list(range(20))
        master_rng.shuffle(o)
        orders.append(o)

    best_result = None
    best_val = -1
    attempt = 0
    for o in orders:
        if time.time() > deadline - 3:
            break
        attempt += 1
        r, v = evaluate(o, Random(attempt))
        if v > best_val:
            best_val = v
            best_result = r
        if best_val == 190:
            break

    if best_result is not None and best_val > 0:
        return best_result
    return [None] * 19 + [list(range(20))]
