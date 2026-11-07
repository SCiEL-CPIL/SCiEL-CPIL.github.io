import time
from random import Random
from math import comb

# Fixed seed for reproducibility of the instance trees (must match the contract)
_SEED = 611

def _build_trees(n=20, seed=611):
    """Reproduce the instance trees exactly as in the contract."""
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

_TREES = _build_trees()

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin under 150s wall-clock

    # Precompute edge sets per tree for quick lookup
    # trees[k-1] is a list of (u,v) edges in the tree's own vertex labels 0..k-1
    n_trees = len(_TREES)

    best_value = 0
    best_result = [None] * n_trees

    # Greedy order: largest tree first (index 19 down to 0)
    order = list(range(n_trees - 1, -1, -1))

    def try_embed(edges, used_edges, rng):
        """Try to embed a tree with given edges (in its own labels) into K20.
        Returns a list of 20 host vertices (image of vertex i) or None."""
        k = len(edges) + 1 if edges else (2 if edges is not None and len(edges) == 0 and k > 1 else 1)
        # Determine k from the tree: vertices are 0..k-1
        # For T1: k=1, no edges. For T2: k=2, one edge.
        # Infer k: max vertex label + 1, but T1 has no edges so k=1.
        if not edges:
            # T1: single vertex
            # We need to know k. T1 is index 0.
            return None  # handled specially
        max_label = max(max(u, v) for u, v in edges)
        k = max_label + 1
        # For T2, edges=[(0,1)], max_label=1, k=2. Good.

        # Random permutation of 20 vertices
        perm = list(range(20))
        rng.shuffle(perm)
        # Map tree vertex i -> perm[i]
        mapping = perm[:k]

        # Check edge-disjointness
        for (u, v) in edges:
            host_u = mapping[u]
            host_v = mapping[v]
            edge_key = (min(host_u, host_v), max(host_u, host_v))
            if edge_key in used_edges:
                return None
        return mapping

    def try_embed_special(k, edges, used_edges, rng):
        """Embed tree with k vertices and given edges."""
        if k == 1:
            # Single vertex, no edges to check
            v = rng.randint(0, 19)
            return [v]
        if k == 2:
            # One edge (0,1)
            perm = list(range(20))
            rng.shuffle(perm)
            mapping = perm[:2]
            edge_key = (min(mapping[0], mapping[1]), max(mapping[0], mapping[1]))
            if edge_key in used_edges:
                return None
            return mapping

        max_label = max(max(u, v) for u, v in edges)
        k = max_label + 1

        perm = list(range(20))
        rng.shuffle(perm)
        mapping = perm[:k]

        for (u, v) in edges:
            host_u = mapping[u]
            host_v = mapping[v]
            edge_key = (min(host_u, host_v), max(host_u, host_v))
            if edge_key in used_edges:
                return None
        return mapping

    rng = Random(42)

    while time.time() < deadline:
        used_edges = set()
        result = [None] * n_trees
        success = True

        for idx in order:
            k = idx + 1
            edges = _TREES[idx]

            if k == 1:
                # Single vertex, always embeddable
                result[idx] = [rng.randint(0, 19)]
                continue

            # Try multiple random mappings for this tree
            embedded = None
            for _attempt in range(200):
                mapping = try_embed_special(k, edges, used_edges, rng)
                if mapping is not None:
                    embedded = mapping
                    break

            if embedded is None:
                success = False
                break

            result[idx] = list(embedded)
            # Add edges to used set
            for (u, v) in edges:
                host_u = embedded[u]
                host_v = embedded[v]
                edge_key = (min(host_u, host_v), max(host_u, host_v))
                used_edges.add(edge_key)

        if success:
            # Compute total edges
            total = sum(len(_TREES[i]) for i in range(n_trees) if result[i] is not None)
            if total > best_value:
                best_value = total
                best_result = [list(r) if r is not None else None for r in result]

        # If we achieved the maximum possible, stop
        max_possible = sum(len(t) for t in _TREES)
        if best_value >= max_possible:
            break

    return best_result
