import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 135
    trees = instance_trees(20, 611)
    n = 20

    # Pre-process: build adjacency and degrees for each tree
    tree_adj = []
    tree_degs = []
    tree_sizes = []
    for i, edges in enumerate(trees):
        k = i + 1
        if k == 1:
            tree_adj.append([[]])
            tree_degs.append([0])
            tree_sizes.append(1)
            continue
        adj = [[] for _ in range(k)]
        degs = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degs[u] += 1
            degs[v] += 1
        tree_adj.append(adj)
        tree_degs.append(degs)
        tree_sizes.append(k)

    # Embed largest trees first; tie-break by max degree
    tree_order = list(range(20))
    tree_order.sort(key=lambda i: (-tree_sizes[i], -max(tree_degs[i]) if tree_degs[i] else 0))

    # Initialize available edges as a set of frozensets for O(1) lookup
    avail = set()
    for i in range(n):
        for j in range(i + 1, n):
            avail.add((i, j))

    # Track degree of each host in the available graph
    host_deg = [n - 1] * n

    mappings = [None] * 20

    def remove_edge(e):
        u, v = e
        avail.discard(e)
        host_deg[u] -= 1
        host_deg[v] -= 1

    def add_edge(e):
        u, v = e
        avail.add(e)
        host_deg[u] += 1
        host_deg[v] += 1

    def embed_tree(t_idx, host_map, tv_order, pos):
        """Recursively assign host vertices to tree vertices in tv_order."""
        if time.time() > deadline:
            return False
        if pos == tree_sizes[t_idx]:
            # All vertices assigned; verify all edges are available
            for u, v in trees[t_idx]:
                hu, hv = host_map[u], host_map[v]
                e = (min(hu, hv), max(hu, hv))
                if e not in avail:
                    return False
            # Commit: remove edges from avail
            for u, v in trees[t_idx]:
                hu, hv = host_map[u], host_map[v]
                remove_edge((min(hu, hv), max(hu, hv)))
            mappings[t_idx] = [host_map[i] for i in range(tree_sizes[t_idx])]
            return True

        tv = tv_order[pos]
        tdeg = tree_degs[t_idx][tv]
        used_hosts = set(host_map.values())

        # Candidates: hosts not yet used, with sufficient remaining degree
        # Sort by remaining degree ascending (most constrained first)
        cands = [h for h in range(n) if h not in used_hosts and host_deg[h] >= tdeg]
        cands.sort(key=lambda h: host_deg[h])

        for h in cands:
            # Forward check: edges to already-assigned neighbors must be available
            ok = True
            for nb in tree_adj[t_idx][tv]:
                if nb in host_map:
                    hn = host_map[nb]
                    e = (min(h, hn), max(h, hn))
                    if e not in avail:
                        ok = False
                        break
            if not ok:
                continue
            host_map[tv] = h
            if embed_tree(t_idx, host_map, tv_order, pos + 1):
                return True
            del host_map[tv]
        return False

    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == 20:
            return len(avail) == 0

        t_idx = tree_order[pos]
        k = tree_sizes[t_idx]

        if k == 1:
            mappings[t_idx] = [0]
            if backtrack(pos + 1):
                return True
            mappings[t_idx] = None
            return False

        # Order tree vertices by degree descending (most constrained first)
        tv_order = sorted(range(k), key=lambda v: -tree_degs[t_idx][v])
        host_map = {}

        # Symmetry breaking: for the first embedded tree, fix highest-degree vertex to host 0
        if pos == 0:
            tv0 = tv_order[0]
            host_map[tv0] = 0
            if embed_tree(t_idx, host_map, tv_order, 1):
                return True
            del host_map[tv0]
        else:
            if embed_tree(t_idx, host_map, tv_order, 0):
                return True

        return False

    if backtrack(0):
        result = [None] * 20
        for i in range(20):
            if mappings[i] is not None:
                result[i] = [int(x) for x in mappings[i]]
        return result

    # Fallback: return what we have (shouldn't happen for valid instances)
    result = [None] * 20
    for i in range(20):
        if mappings[i] is not None:
            result[i] = [int(x) for x in mappings[i]]
    return result

if __name__ == "__main__":
    r = construct()
    total_edges = sum(len(instance_trees(20, 611)[i]) for i in range(20) if r[i] is not None and instance_trees(20, 611)[i])
    print("embedded edges:", total_edges)
