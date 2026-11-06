import time
import random
import sys
import numpy as np
from itertools import combinations

def construct():
    n = 16
    target = 65  # floor(16^2/4) + 1 = 64 + 1
    deadline = time.time() + 112.0
    rng = random.Random(42)

    # Precompute all possible triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)

    def _score_from_adj(adj, deg):
        best = 0
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best

    def _score_from_edges(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return _score_from_adj(adj, deg)

    def _valid(edges, n, target):
        if len(edges) != target:
            return False
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True

    def _seed_edges(n):
        # K_{8,8} + one intra edge -> 65 edges, score 24
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges

    best_edges = _seed_edges(n)
    best_score = _score_from_edges(best_edges)

    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False

    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def _adj_from_edges(edges):
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    # --- Fast local search using only edge swaps ---
    def local_search(adj, deg, score):
        nonlocal best_score, best_edges
        try_improve(_edges_from_adj(adj), score)

        max_iter = 2000
        for _ in range(max_iter):
            if time.time() >= deadline:
                break

            # Find the worst triangle
            worst_tri = None
            worst_s = 0
            for (u, v, w) in triangles:
                if adj[u][v] and adj[v][w] and adj[u][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > worst_s:
                        worst_s = s
                        worst_tri = (u, v, w)

            if worst_tri is None:
                break

            u, v, w = worst_tri
            improved = False

            # Try to break one of the three edges in the worst triangle
            for a, b in [(u, v), (v, w), (u, w)]:
                best_new_score = score
                best_swap = None

                # Collect candidate non-edges
                candidates = []
                for c in range(n):
                    if c == a or c == b:
                        continue
                    for d in range(c + 1, n):
                        if d == a or d == b:
                            continue
                        if adj[c][d]:
                            continue
                        # Quick degree check
                        nd_a = deg[a] - 1
                        nd_b = deg[b] - 1
                        nd_c = deg[c] + 1
                        nd_d = deg[d] + 1
                        # Skip if new degrees are too high
                        if nd_c + nd_d > 2 * worst_s - 2 * min(nd_a, nd_b):
                            continue
                        candidates.append((c, d))

                if not candidates:
                    continue

                # Shuffle and try a limited number of candidates
                rng.shuffle(candidates)
                for c, d in candidates[:8]:
                    nd = deg[:]
                    nd[a] -= 1
                    nd[b] -= 1
                    nd[c] += 1
                    nd[d] += 1

                    na = [row[:] for row in adj]
                    na[a][b] = na[b][a] = False
                    na[c][d] = na[d][c] = True

                    ns = _score_from_adj(na, nd)
                    if ns < best_new_score:
                        best_new_score = ns
                        best_swap = (a, b, c, d, nd, na)

                if best_swap:
                    a2, b2, c2, d2, nd2, na2 = best_swap
                    adj = na2
                    deg = nd2
                    score = best_new_score
                    improved = True
                    try_improve(_edges_from_adj(adj), score)
                    break

            if not improved:
                # Try random edge swaps to escape local minima
                found = False
                for _ in range(50):
                    a = rng.randint(0, n - 1)
                    b = rng.randint(0, n - 1)
                    if a == b or not adj[a][b]:
                        continue
                    c = rng.randint(0, n - 1)
                    d = rng.randint(0, n - 1)
                    if c == d or adj[c][d] or c in (a, b) or d in (a, b):
                        continue

                    nd = deg[:]
                    nd[a] -= 1
                    nd[b] -= 1
                    nd[c] += 1
                    nd[d] += 1

                    na = [row[:] for row in adj]
                    na[a][b] = na[b][a] = False
                    na[c][d] = na[d][c] = True

                    ns = _score_from_adj(na, nd)
                    if ns < score:
                        adj = na
                        deg = nd
                        score = ns
                        try_improve(_edges_from_adj(adj), score)
                        found = True
                        break

                if not found:
                    break

        return adj, deg, score

    # --- Erdős–Gallai check ---
    def is_graphical(deg_seq):
        d = sorted(deg_seq, reverse=True)
        if sum(d) != 2 * target:
            return False
        if any(x < 0 or x > n - 1 for x in d):
            return False
        for k in range(1, n + 1):
            lhs = sum(d[:k])
            rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
            if lhs > rhs:
                return False
        return True

    # --- Random Havel–Hakimi ---
    def realize_degree_sequence(deg_seq, rng):
        """Randomly realize a degree sequence via Havel–Hakimi."""
        remaining = list(enumerate(deg_seq))
        remaining.sort(key=lambda x: -x[1])

        adj = [[False] * n for _ in range(n)]

        def hh(remaining):
            if not remaining:
                return True
            if all(d == 0 for _, d in remaining):
                return True
            max_idx = max(range(len(remaining)), key=lambda i: remaining[i][1])
            v, dv = remaining.pop(max_idx)
            if dv < 0 or dv > len(remaining):
                return False
            if dv == 0:
                return hh(remaining)

            candidates = list(range(len(remaining)))
            candidates.sort(key=lambda i: -remaining[i][1])

            if dv <= len(candidates):
                chosen = candidates[:dv]
            else:
                return False

            rng.shuffle(chosen)

            for idx in chosen:
                u, du = remaining[idx]
                if du <= 0:
                    return False
                adj[v][u] = adj[u][v] = True
                remaining[idx] = (u, du - 1)

            return hh(remaining)

        rem_copy = list(remaining)
        if not hh(rem_copy):
            return None
        return adj

    # --- Main search loop ---
    start = time.time()

    # First, try the seed
    adj, deg = _adj_from_edges(best_edges)
    score = _score_from_adj(adj, deg)
    adj, deg, score = local_search(adj, deg, score)

    # Generate diverse degree sequences to try
    avg_deg = 2 * target / n  # 8.125

    # Try various degree distributions
    deg_patterns = []

    # Pattern 1: 14 of 8, 2 of 9
    base = [8] * 16
    for i in range(2):
        base[i] = 9
    deg_patterns.append(base[:])

    # Pattern 2: 15 of 8, 1 of 10
    deg_patterns.append([8] * 15 + [10])

    # Pattern 3: 2 of 7, 12 of 8, 2 of 10
    deg_patterns.append([7] * 2 + [8] * 12 + [10] * 2)

    # Pattern 4: 1 of 7, 14 of 8, 1 of 11
    deg_patterns.append([7] + [8] * 14 + [11])

    # Add some random patterns
    for _ in range(20):
        deg_seq = [8] * 16
        remaining = 130 - sum(deg_seq)
        while remaining != 0:
            i = rng.randint(0, 15)
            if remaining > 0 and deg_seq[i] < 15:
                deg_seq[i] += 1
                remaining -= 1
            elif remaining < 0 and deg_seq[i] > 0:
                deg_seq[i] -= 1
                remaining += 1
            else:
                break
        if sum(deg_seq) == 130 and is_graphical(deg_seq):
            deg_patterns.append(deg_seq)

    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in deg_patterns:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    # Try each pattern
    for deg_seq in unique_patterns:
        if time.time() >= deadline - 5:
            break
        if not is_graphical(deg_seq):
            continue

        # Try multiple realizations
        for attempt in range(3):
            if time.time() >= deadline - 5:
                break
            adj = realize_degree_sequence(deg_seq, rng)
            if adj is None:
                continue
            edges = _edges_from_adj(adj)
            if len(edges) != target:
                continue
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            score = _score_from_adj(adj, deg)
            adj, deg, score = local_search(adj, deg, score)

    # Random restarts with random edge sets
    while time.time() < deadline - 5:
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        if not _valid(edges, n, target):
            continue
        adj, deg = _adj_from_edges(edges)
        score = _score_from_adj(adj, deg)
        adj, deg, score = local_search(adj, deg, score)

    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = _score_from_edges(best_edges)

    return {'n': n, 'edges': best_edges}
