"""Packing trees T1..T20 into K20 via backtracking with leaf-first ordering."""
from random import Random
import time
import sys
sys.setrecursionlimit(10000)


def _generate_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _build_adjacency(tree_edges, k):
    """Build adjacency list for a tree with k vertices."""
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _compute_1_factorization():
    """
    Compute a 1-factorization of K20 using the standard cyclic method.
    For n=20 (even), we can use the round-robin tournament scheduling.
    Returns a list of 19 perfect matchings, each a list of 10 edges.
    """
    n = 20
    # Fix vertex 0, rotate the rest
    vertices = list(range(1, n))
    matchings = []
    for round_num in range(n - 1):
        matching = [(0, vertices[0])]
        for i in range(1, n // 2):
            u = vertices[i]
            v = vertices[(n - 1 - i) % (n - 1)]
            matching.append((u, v))
        matchings.append(matching)
        # Rotate
        vertices = [vertices[-1]] + vertices[:-1]
    return matchings


def _find_embedding(tree_edges, k, used_edges, deadline, rng):
    """
    Find an embedding of the tree into K20 such that no edge conflicts with used_edges.
    Returns a list of k integers (the image of each vertex) or None.
    Uses backtracking with leaf-first BFS ordering and randomized restarts.
    """
    if k == 1:
        return [0]
    if k == 2:
        # Need an edge (u,v) not in used_edges
        candidates = []
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    candidates.append((u, v))
        if not candidates:
            return None
        # Shuffle for diversity
        rng.shuffle(candidates)
        u, v = candidates[0]
        return [u, v]

    adj = _build_adjacency(tree_edges, k)
    leaves = [v for v in range(k) if len(adj[v]) == 1]
    if not leaves:
        return None

    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None

        # Pick a random leaf as root
        root = rng.choice(leaves)

        # BFS from root to get ordering
        from collections import deque
        order = [root]
        parent = [-1] * k
        parent[root] = -2
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    order.append(v)
                    queue.append(v)

        # Backtracking with candidate ordering
        mapping = [-1] * k
        host_used = [False] * 20
        emb_edges = set()

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True

            v = order[idx]
            if idx == 0:
                # Root: try all host vertices
                for h in range(20):
                    mapping[root] = h
                    host_used[h] = True
                    if backtrack(idx + 1):
                        return True
                    mapping[root] = -1
                    host_used[h] = False
                return False

            p = parent[v]
            ph = mapping[p]

            # Build candidate list with scoring
            candidates = []
            for h in range(20):
                if host_used[h]:
                    continue
                if (ph, h) in used_edges or (h, ph) in used_edges:
                    continue
                if (ph, h) in emb_edges or (h, ph) in emb_edges:
                    continue
                # Score: number of available neighbors for future children
                # (heuristic: prefer vertices with more available edges)
                score = 0
                for ch in adj[v]:
                    if parent[ch] == v:  # ch is a child of v
                        for h2 in range(20):
                            if not host_used[h2] and h2 != h:
                                if (h, h2) not in used_edges and (h2, h) not in used_edges:
                                    if (h, h2) not in emb_edges and (h2, h) not in emb_edges:
                                        score += 1
                candidates.append((score, h))

            # Sort by score descending (most constraining first), with random tie-breaking
            candidates.sort(key=lambda x: (-x[0], rng.random()))

            for score, h in candidates:
                mapping[v] = h
                host_used[h] = True
                emb_edges.add((ph, h))

                if backtrack(idx + 1):
                    return True

                mapping[v] = -1
                host_used[h] = False
                emb_edges.discard((ph, h))

            return False

        if backtrack(0):
            return mapping

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s to be safe before 150s kill

    trees = _generate_trees(20, 611)
    rng = Random(42)

    # Sort trees by size (largest first) for better constraint propagation
    order = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)

    used_edges = set()
    results = [None] * 20
    total_edges = 0

    for idx in order:
        k = idx + 1
        tree_edges = trees[idx]
        if len(tree_edges) == 0:
            results[idx] = None
            continue

        emb = _find_embedding(tree_edges, k, used_edges, deadline, rng)
        if emb is None:
            # Try with different random seed
            rng2 = Random(rng.random() * 1000000)
            emb = _find_embedding(tree_edges, k, used_edges, deadline, rng2)
            if emb is None:
                break

        results[idx] = emb
        for u, v in tree_edges:
            hu, hv = emb[u], emb[v]
            used_edges.add((hu, hv))
            used_edges.add((hv, hu))
        total_edges += len(tree_edges)

    for i in range(20):
        if results[i] is not None:
            results[i] = [int(x) for x in results[i]]

    return results
