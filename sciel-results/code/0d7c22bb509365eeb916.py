"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random


def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
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


def _embed_tree(tree_edges, k, free_adj, rng, max_nodes=500000):
    """Try to embed a tree with k vertices using only free edges.
    Returns a list of host vertices (length k) or None."""
    if k == 1:
        # Any free vertex works
        for v in range(20):
            if any(free_adj[v][u] for u in range(20) if u != v):
                return [v]
        return [0]  # fallback

    # Build adjacency for the tree
    tree_adj = [[] for _ in range(k)]
    for a, b in tree_edges:
        tree_adj[a].append(b)
        tree_adj[b].append(a)

    # Find a good root: highest degree vertex
    root = max(range(k), key=lambda v: len(tree_adj[v]))

    # Build parent/children structure via BFS from root
    parent = [-1] * k
    order = [root]
    parent[root] = -2
    for i in range(len(order)):
        u = order[i]
        for v in tree_adj[u]:
            if parent[v] == -1:
                parent[v] = u
                order.append(v)

    # order is BFS order; embed in that order
    # For each vertex, choose a host from free neighbors of its host-parent
    assignment = [None] * k
    assignment[root] = None  # will be assigned

    def dfs(idx, host_for_root=None):
        """Recursively assign hosts. idx is index in order."""
        if idx == k:
            return True
        u = order[idx]
        if idx == 0:
            # Assign root to any vertex with enough free degree
            candidates = list(range(20))
            rng.shuffle(candidates)
            for hv in candidates:
                # Check if hv has enough free neighbors for its tree-neighbors
                tree_neighbors = tree_adj[u]
                free_count = sum(1 for x in range(20) if x != hv and free_adj[hv][x])
                if free_count >= len(tree_neighbors):
                    assignment[u] = hv
                    if dfs(idx + 1, hv):
                        return True
            return False
        else:
            # u's parent in tree is parent[u], whose host is assignment[parent[u]]
            ph = assignment[parent[u]]
            # Choose a free neighbor of ph
            candidates = [x for x in range(20) if x != ph and free_adj[ph][x]]
            rng.shuffle(candidates)
            for hv in candidates:
                # Check hv has enough remaining free neighbors for its unassigned children
                # Count how many tree-neighbors of u are not yet assigned
                unassigned_children = 0
                for v in tree_adj[u]:
                    if v != parent[u] and assignment[v] is None:
                        unassigned_children += 1
                free_count = sum(1 for x in range(20) if x != hv and x != ph and free_adj[hv][x])
                if free_count >= unassigned_children:
                    assignment[u] = hv
                    if dfs(idx + 1, hv):
                        return True
                    assignment[u] = None
            return False

    # Use iterative approach with node limit to avoid deep recursion issues
    # Actually let's use a cleaner recursive approach with counter
    node_count = [0]

    def dfs2(idx):
        if node_count[0] > max_nodes:
            return False
        node_count[0] += 1
        if idx == k:
            return True
        u = order[idx]
        if idx == 0:
            candidates = list(range(20))
            rng.shuffle(candidates)
            for hv in candidates:
                tree_neighbors = tree_adj[u]
                free_count = sum(1 for x in range(20) if x != hv and free_adj[hv][x])
                if free_count >= len(tree_neighbors):
                    assignment[u] = hv
                    if dfs2(idx + 1):
                        return True
            assignment[u] = None
            return False
        else:
            ph = assignment[parent[u]]
            candidates = [x for x in range(20) if x != ph and free_adj[ph][x]]
            rng.shuffle(candidates)
            for hv in candidates:
                unassigned_children = 0
                for v in tree_adj[u]:
                    if v != parent[u] and assignment[v] is None:
                        unassigned_children += 1
                free_count = sum(1 for x in range(20) if x != hv and x != ph and free_adj[hv][x])
                if free_count >= unassigned_children:
                    assignment[u] = hv
                    if dfs2(idx + 1):
                        return True
                    assignment[u] = None
            return False

    if dfs2(0):
        return assignment
    return None


def _try_packing(trees, rng, max_nodes_per_tree=200000):
    """Try to pack all trees. Returns list of assignments or None."""
    n = 20
    # free_adj[v][u] = True if edge (v,u) is still available
    free_adj = [[True] * n for _ in range(n)]
    # Mark self-loops as unavailable
    for v in range(n):
        free_adj[v][v] = False

    # Sort trees by size descending for packing
    indexed = list(enumerate(trees))
    indexed.sort(key=lambda x: -len(x[1]))

    assignments = [None] * n  # indexed by original tree index
    embedded_count = 0

    for orig_idx, tree_edges in indexed:
        k = orig_idx + 1  # tree T_k has k vertices (0-indexed: tree k-1 has k vertices)
        # Actually trees list: trees[i] corresponds to T_{i+1} with i+1 vertices
        # tree_edges has k-1 edges for k vertices
        if k == 1:
            # Single vertex, no edges needed
            assignments[orig_idx] = [0]
            embedded_count += 0
            continue
        if k == 2:
            # Two vertices, one edge
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if free_adj[a][b]:
                        assignments[orig_idx] = [a, b]
                        free_adj[a][b] = False
                        free_adj[b][a] = False
                        embedded_count += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                return None  # Can't pack this tree, abort
            continue

        # General case: use backtracking embedder
        result = _embed_tree(tree_edges, k, free_adj, rng, max_nodes=max_nodes_per_tree)
        if result is None:
            return None
        # Mark edges as used
        for a, b in tree_edges:
            ha, hb = result[a], result[b]
            if not free_adj[ha][hb]:
                return None  # Conflict
            free_adj[ha][hb] = False
            free_adj[hb][ha] = False
        assignments[orig_idx] = result
        embedded_count += k - 1

    return assignments, embedded_count


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    trees = instance_trees(20, 611)

    # Order for packing: largest first
    # Tree indices: 0..19, tree i has i+1 vertices
    pack_order = list(range(19, -1, -1))  # T20, T19, ..., T1

    best_result = [None] * 19 + [list(range(20))]  # Seed fallback
    best_edges = 19

    for restart in range(200):
        if time.time() > deadline:
            break

        rng = Random(restart * 1000 + 42)
        result = _try_packing(trees, rng, max_nodes_per_tree=100000)
        if result is None:
            continue

        assignments, edge_count = result
        if edge_count > best_edges:
            best_edges = edge_count
            # Build output: entry k-1 is list of k host vertices or None
            out = [None] * 20
            for i in range(20):
                if assignments[i] is not None:
                    out[i] = list(assignments[i])
                else:
                    out[i] = None
            best_result = out

    return best_result
# EVOLVE-BLOCK-END
