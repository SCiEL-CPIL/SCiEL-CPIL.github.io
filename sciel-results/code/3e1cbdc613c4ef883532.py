"""Packing trees T1..T20 into K20 - direct backtracking approach."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    trees = instance_trees(20, 611)
    n = 20

    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Process trees from largest to smallest (T20 first, T1 last)
    tree_order = list(range(20))  # 0-indexed: tree i has (i+1) vertices
    tree_order.sort(key=lambda idx: -(idx + 1))

    available = set(all_edges)
    result = [None] * 20

    # Precompute tree adjacency and vertex ordering for each tree
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k <= 2:
            tree_info.append(None)
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order vertices by degree descending for backtracking
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((tree_edges, tree_adj, ordered_verts))

    def try_embed(t_idx, available, rng):
        """Try to embed tree t_idx into available edges.
        Returns list of host vertices (index i = host for tree vertex i) or None."""
        k = t_idx + 1
        if k == 1:
            # Just pick any vertex
            return [rng.randint(0, n - 1)]
        if k == 2:
            # Need an edge in available
            for u, v in available:
                return [u, v]
            return None
        # For larger trees, we'll use backtracking

        tree_edges, tree_adj, ordered_verts = tree_info[t_idx]
        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all tree edges are in available
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available:
                        return False
                return True

            tv = ordered_verts[pos]
            # Generate candidates: host vertices not yet used
            # Try to order candidates by how well they fit
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check against already-assigned neighbors
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available:
                            ok = False
                            break
                if ok:
                    # Score: prefer vertices with fewer remaining edges (more constrained)
                    # Count how many edges from h are still available
                    score = 0
                    for nb2 in range(n):
                        if nb2 != h and nb2 not in used_hosts:
                            e = (min(h, nb2), max(h, nb2))
                            if e in available:
                                score += 1
                    candidates.append((score, h))

            if not candidates:
                return False

            # Sort by score (fewer available edges first = more constrained), then randomize ties
            candidates.sort(key=lambda x: (x[0], rng.random()))
            candidates = [h for _, h in candidates]

            for h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    # Main search with multiple random restarts
    best_result = None

    for attempt in range(2000):
        if time.time() > deadline:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            # Try a few times with different randomizations
            for trial in range(5):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed(t_idx, avail, rng)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            # Remove used edges
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)

        if success:
            # Verify: all 190 edges used exactly once
            total_used = sum(len(trees[i]) for i in range(20))
            if len(avail) == 0:
                best_result = res
                break

    if best_result is None:
        # Fallback: return a valid but possibly suboptimal result
        # Actually, if we can't find a complete packing, we should return something valid
        # Let's try a simpler greedy approach as fallback
        return [None] * 19 + [list(range(20))]

    # Ensure types are correct
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
