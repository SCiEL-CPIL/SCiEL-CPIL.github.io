"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s limit

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)

    deg = [len(adj[u]) for u in range(n)]

    best = []
    best_mask = 0

    def update_best(path):
        nonlocal best, best_mask
        if len(path) > len(best):
            best = path[:]
            best_mask = 0
            for v in path:
                best_mask |= (1 << v)

    def popcount(x):
        # Fast popcount using int.bit_count() available in Python 3.8+
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # Strategy 1: Greedy walks from every start with various heuristics
    for start in range(n):
        if time.time() > deadline:
            break
        for heuristic in ['min_rem', 'max_rem', 'min_deg', 'max_deg']:
            path = [start]
            seen = 1 << start
            while True:
                avail = adj_mask[path[-1]] & ~seen
                if avail == 0:
                    break
                candidates = [v for v in range(n) if (avail >> v) & 1]
                if heuristic == 'min_rem':
                    candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), deg[w]))
                elif heuristic == 'max_rem':
                    candidates.sort(key=lambda w: (-popcount(adj_mask[w] & ~seen), -deg[w]))
                elif heuristic == 'min_deg':
                    candidates.sort(key=lambda w: (deg[w], popcount(adj_mask[w] & ~seen)))
                else:
                    candidates.sort(key=lambda w: (-deg[w], -popcount(adj_mask[w] & ~seen)))
                path.append(candidates[0])
                seen |= (1 << candidates[0])
            update_best(path)

    # Strategy 2: Randomized greedy with many trials
    rng = random.Random(12345)
    for trial in range(5000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = 1 << start
        while True:
            avail = adj_mask[path[-1]] & ~seen
            if avail == 0:
                break
            candidates = [v for v in range(n) if (avail >> v) & 1]
            # Randomized greedy: sort by remaining degree with noise
            candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen) + rng.gauss(0, 0.5), rng.random()))
            v = candidates[0]
            path.append(v)
            seen |= (1 << v)
        update_best(path)

    # Strategy 3: Deep backtracking with pruning
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = 1 << start

        def backtrack(path, seen):
            nonlocal best, best_mask
            if time.time() > deadline:
                return
            if len(path) > len(best):
                best = path[:]
                best_mask = seen
            cur = path[-1]
            avail = adj_mask[cur] & ~seen
            if avail == 0:
                return
            candidates = [v for v in range(n) if (avail >> v) & 1]
            # Sort by remaining degree (prefer low to avoid dead-ends)
            candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), deg[w]))
            # Limit branching factor based on time remaining
            time_left = deadline - time.time()
            if time_left > 10:
                limit = min(12, len(candidates))
            elif time_left > 3:
                limit = min(8, len(candidates))
            else:
                limit = min(5, len(candidates))
            for v in candidates[:limit]:
                path.append(v)
                seen |= (1 << v)
                backtrack(path, seen)
                seen &= ~(1 << v)
                path.pop()

        backtrack(path, seen)

    # Strategy 4: Iterative deepening DFS with time limit
    for target_len in range(len(best) + 1, n + 1):
        if time.time() > deadline:
            break
        found = False
        for start in range(n):
            if time.time() > deadline:
                break
            path = [start]
            seen = 1 << start

            def iddfs(path, seen, depth):
                nonlocal found
                if time.time() > deadline:
                    return
                if len(path) >= target_len:
                    update_best(path)
                    found = True
                    return
                cur = path[-1]
                avail = adj_mask[cur] & ~seen
                if avail == 0:
                    return
                candidates = [v for v in range(n) if (avail >> v) & 1]
                candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), deg[w]))
                for v in candidates[:6]:
                    path.append(v)
                    seen |= (1 << v)
                    iddfs(path, seen, depth + 1)
                    seen &= ~(1 << v)
                    path.pop()
                    if found:
                        return

            iddfs(path, seen, 0)
            if found:
                break

    # Strategy 5: Path extension from both ends
    if best:
        if time.time() < deadline:
            # Extend from end
            cur = best[-1]
            seen = best_mask
            while True:
                avail = adj_mask[cur] & ~seen
                if avail == 0:
                    break
                candidates = [v for v in range(n) if (avail >> v) & 1]
                candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), w))
                v = candidates[0]
                best.append(v)
                seen |= (1 << v)
                cur = v
            best_mask = seen

            # Extend from beginning
            if time.time() < deadline:
                cur = best[0]
                seen = best_mask
                while True:
                    avail = adj_mask[cur] & ~seen
                    if avail == 0:
                        break
                    candidates = [v for v in range(n) if (avail >> v) & 1]
                    candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), w))
                    v = candidates[0]
                    best.insert(0, v)
                    seen |= (1 << v)
                    cur = v
                best_mask = seen

    # Strategy 6: Insert unvisited vertices into gaps
    if time.time() < deadline and len(best) < n:
        visited = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_best = best[:]
            best_mask_local = 0
            for v in new_best:
                best_mask_local |= (1 << v)
            for i in range(len(new_best) - 1):
                u = new_best[i]
                v = new_best[i + 1]
                common = adj_mask[u] & adj_mask[v] & ~best_mask_local
                if common:
                    candidates = [w for w in range(n) if (common >> w) & 1]
                    candidates.sort(key=lambda w: -popcount(adj_mask[w] & ~best_mask_local))
                    w = candidates[0]
                    new_best.insert(i + 1, w)
                    best_mask_local |= (1 << w)
                    improved = True
                    break
            if improved:
                best = new_best
                best_mask = best_mask_local

    # Strategy 7: 2-opt style local improvement
    if time.time() < deadline and len(best) >= 4:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            path = best[:]
            m = len(path)
            for i in range(m - 2):
                if time.time() > deadline:
                    break
                for j in range(i + 2, m):
                    a = path[i]
                    b = path[i + 1]
                    c = path[j]
                    d = path[j + 1] if j + 1 < m else None
                    if (adj_mask[a] >> c) & 1:
                        if d is not None:
                            if (adj_mask[b] >> d) & 1:
                                new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                                valid = all(((adj_mask[new_path[k]] >> new_path[k+1]) & 1) for k in range(len(new_path) - 1))
                                if valid:
                                    best = new_path
                                    best_mask = 0
                                    for v in best:
                                        best_mask |= (1 << v)
                                    improved = True
                                    break
                        else:
                            new_path = path[:i+1] + path[i+1:j+1][::-1]
                            valid = all(((adj_mask[new_path[k]] >> new_path[k+1]) & 1) for k in range(len(new_path) - 1))
                            if valid:
                                best = new_path
                                best_mask = 0
                                for v in best:
                                    best_mask |= (1 << v)
                                improved = True
                                break
                if improved:
                    break

    # Strategy 8: Bottleneck-first greedy
    if time.time() < deadline:
        sorted_vertices = sorted(range(n), key=lambda v: deg[v])
        for start in sorted_vertices[:30]:
            if time.time() > deadline:
                break
            path = [start]
            seen = 1 << start
            while True:
                avail = adj_mask[path[-1]] & ~seen
                if avail == 0:
                    break
                candidates = [v for v in range(n) if (avail >> v) & 1]
                candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), deg[w]))
                v = candidates[0]
                path.append(v)
                seen |= (1 << v)
            update_best(path)

    # Final extension pass
    if best:
        cur = best[-1]
        seen = best_mask
        while True:
            avail = adj_mask[cur] & ~seen
            if avail == 0:
                break
            candidates = [v for v in range(n) if (avail >> v) & 1]
            candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), w))
            v = candidates[0]
            best.append(v)
            seen |= (1 << v)
            cur = v
        best_mask = seen

        cur = best[0]
        seen = best_mask
        while True:
            avail = adj_mask[cur] & ~seen
            if avail == 0:
                break
            candidates = [v for v in range(n) if (avail >> v) & 1]
            candidates.sort(key=lambda w: (popcount(adj_mask[w] & ~seen), w))
            v = candidates[0]
            best.insert(0, v)
            seen |= (1 << v)
            cur = v
        best_mask = seen

    return best if best else [0]
# EVOLVE-BLOCK-END
