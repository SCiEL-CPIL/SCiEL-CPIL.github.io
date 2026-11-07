"""Packing trees T1..T20 into K20: maximize embedded edge-disjoint edges."""
import time
from random import Random
from collections import defaultdict

_DEADLINE = 150.0
_START = None


def instance_trees(n=20, seed=611):
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


def _key(e):
    a, b = e
    return (a, b) if a < b else (b, a)


def _embed_tree(tree_edges, used_edges, rng, n=20, max_tries=20000):
    """Try to injectively map tree vertices 0..k-1 to host vertices 0..n-1
    so that no mapped edge is in used_edges. Return list of host vertices
    (image of tree vertex i) or None."""
    k = len(tree_edges) + (1 if tree_edges else 1)
    # Determine vertex set of the tree
    verts = set()
    for a, b in tree_edges:
        verts.add(a)
        verts.add(b)
    if not tree_edges:
        verts = {0}
    k = len(verts)
    # Build adjacency of tree
    adj = defaultdict(list)
    for a, b in tree_edges:
        adj[a].append(b)
        adj[b].append(a)
    # Find a root, order vertices for sequential assignment
    root = min(verts)
    order = []
    parent = {root: None}
    stack = [root]
    while stack:
        v = stack.pop()
        order.append(v)
        for w in adj[v]:
            if w not in parent:
                parent[w] = v
                stack.append(w)
    # order is a valid BFS/DFS order where parent assigned before child
    for _ in range(max_tries):
        mapping = {}
        ok = True
        for v in order:
            p = parent[v]
            forbidden = set()
            # host vertex must not already be used
            # and if p is mapped, edge (map[p], candidate) must not be in used_edges
            if p is not None:
                hp = mapping[p]
                for cand in range(n):
                    if cand in mapping.values():
                        continue
                    ek = (hp, cand) if hp < cand else (cand, hp)
                    if ek in used_edges:
                        forbidden.add(cand)
            else:
                forbidden = set(mapping.values())
            avail = [c for c in range(n) if c not in forbidden]
            if not avail:
                ok = False
                break
            mapping[v] = rng.choice(avail)
        if not ok:
            continue
        # verify all edges disjoint
        good = True
        for a, b in tree_edges:
            ha, hb = mapping[a], mapping[b]
            ek = (ha, hb) if ha < hb else (hb, ha)
            if ek in used_edges:
                good = False
                break
        if good:
            # convert to list indexed by tree vertex i in 0..k-1
            # verts may not be 0..k-1 contiguous; but tree vertices are 0..k-1 by construction
            result = [0] * k
            for tv, hv in mapping.items():
                result[tv] = hv
            return result
    return None


def construct():
    global _START
    if _START is None:
        _START = time.time()
    trees = instance_trees(20, 611)
    # Order: embed T20 first (biggest), then descending
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)
    used_edges = set()
    result = [None] * 20
    rng = Random(12345)
    for idx in order:
        tree = trees[idx]
        if not tree:
            # T1: single vertex, pick any free vertex
            for v in range(20):
                result[idx] = [v]
                break
            continue
        mapping = _embed_tree(tree, used_edges, rng, n=20, max_tries=50000)
        if mapping is None:
            # try more with fresh rng seed
            for seed2 in range(20):
                rng2 = Random(1000 + seed2)
                mapping = _embed_tree(tree, used_edges, rng2, n=20, max_tries=50000)
                if mapping is not None:
                    break
        if mapping is not None:
            result[idx] = [int(x) for x in mapping]
            for a, b in tree:
                ha, hb = mapping[a], mapping[b]
                ek = (ha, hb) if ha < hb else (hb, ha)
                used_edges.add(ek)
        # deadline guard
        if time.time() - _START > _DEADLINE - 10:
            break
    # Ensure types are plain ints and lists
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    return result


if __name__ == "__main__":
    r = construct()
    total = sum(len(trees[i]) for i in range(20) if r[i] is not None)
    print("embedded edges:", total)
