"""Packing trees T1..T20 into K20 using deterministic backtracking with edge-disjointness."""
import time
from random import Random


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


def _build_adjacency(edges, k):
    """Build adjacency list for a tree with k vertices."""
    adj = [[] for _ in range(k)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _embedding_order(adj, k):
    """Return vertex order: root first, then BFS order (parents before children)."""
    if k <= 1:
        return list(range(k))
    # Find a root: use vertex 0 or a leaf
    root = 0
    visited = [False] * k
    order = [root]
    visited[root] = True
    frontier = [root]
    while frontier:
        next_frontier = []
        for node in frontier:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    order.append(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
    # If not all visited (shouldn't happen for a tree), add remaining
    for v in range(k):
        if not visited[v]:
            order.append(v)
    return order


def construct():
    """
    Embed trees T1..T20 into K20 with edge-disjoint embeddings.
    Returns a list of 20 entries; entry k-1 is None or a list of k distinct ints in 0..19.
    Position i is the image of vertex i of T_k.
    """
    deadline = time.time() + 145  # safety margin before 150s limit
    trees = instance_trees(20, 611)
    n = 20
    total_vertices = 20

    # Precompute adjacency for each tree
    adj_list = [_build_adjacency(trees[k-1], k) for k in range(1, n + 1)]
    order_list = [_embedding_order(adj_list[k-1], k) for k in range(1, n + 1)]

    # Embedding order: larger trees first (harder to place)
    embed_order = list(range(n, 0, -1))  # 20, 19, ..., 1

    # used_edges: set of (min, max) tuples of host vertices
    used_edges = set()
    result = [None] * n

    def _backtrack_embed(k, edges, adj, order, seed=0):
        """
        Find a valid embedding of tree T_k into K20 avoiding used_edges.
        Returns list of k distinct ints (mapping from tree vertex to host vertex) or None.
        """
        used_edges_local = used_edges  # reference to outer set
        mapping = [None] * k
        k_val = k
        rng = Random(seed)

        # Precompute children in the BFS tree for forward checking
        # parent[i] is the index in 'order' of the parent of vertex 'order[i]'
        # Actually, we just need to know for each unplaced vertex, which placed neighbors it has.
        
        def _dfs(idx):
            if idx == k_val:
                return True
            v = order[idx]
            
            # Determine which neighbors of v have already been placed
            placed_neighbors = []
            for nb in adj[v]:
                if mapping[nb] is not None:
                    placed_neighbors.append(nb)

            # Candidate hosts: 0..19, not already used by this tree
            used_hosts = [mapping[i] for i in range(k_val) if mapping[i] is not None]
            used_hosts_set = set(used_hosts)

            # For each candidate host, check edge conflicts with placed neighbors
            # and with global used_edges
            candidates = []
            for h in range(total_vertices):
                if h in used_hosts_set:
                    continue
                # Check edges to placed neighbors
                ok = True
                for nb in placed_neighbors:
                    edge_key = (min(h, mapping[nb]), max(h, mapping[nb]))
                    if edge_key in used_edges_local:
                        ok = False
                        break
                if not ok:
                    continue
                candidates.append(h)

            if not candidates:
                return False

            # Shuffle candidates for randomized restarts
            rng.shuffle(candidates)

            # Try each candidate
            for h in candidates:
                mapping[v] = h
                
                # Forward Checking:
                # Check if any unplaced vertex has lost all candidates
                # An unplaced vertex u (order[j] for j > idx) is constrained by its placed neighbors.
                # If u has a placed neighbor, the edge (mapping[u], mapping[placed_nb]) must be valid.
                # We check if there exists ANY host for u that is:
                # 1. Not used by any placed vertex
                # 2. Forms a valid edge with ALL placed neighbors of u
                
                valid_placement = True
                for j in range(idx + 1, k_val):
                    u = order[j]
                    # Find placed neighbors of u
                    placed_nbs_u = []
                    for nb in adj[u]:
                        if mapping[nb] is not None:
                            placed_nbs_u.append(nb)
                    
                    if not placed_nbs_u:
                        continue # No constraints from placed vertices yet, always has candidates (unless all hosts used, but k<20 usually)
                        
                    # Check if there is at least one valid host for u
                    has_candidate = False
                    for hu in range(total_vertices):
                        if hu in used_hosts_set or hu == h: # hu cannot be h (already used by v) or any other used host
                            continue
                        # Check edges to placed neighbors of u
                        ok_u = True
                        for nb in placed_nbs_u:
                            edge_key = (min(hu, mapping[nb]), max(hu, mapping[nb]))
                            if edge_key in used_edges_local:
                                ok_u = False
                                break
                        if ok_u:
                            has_candidate = True
                            break
                    if not has_candidate:
                        valid_placement = False
                        break
                
                if valid_placement:
                    if _dfs(idx + 1):
                        return True
                mapping[v] = None

            return False

        if _dfs(0):
            return list(mapping)
        return None

    # Process trees in order
    for k in embed_order:
        if time.time() > deadline:
            break
        edges = trees[k - 1]
        if k == 1:
            result[k - 1] = [0]
            continue
        if k == 2:
            # Find an unused edge
            found = False
            for a in range(total_vertices):
                if found:
                    break
                for b in range(a + 1, total_vertices):
                    edge_key = (a, b)
                    if edge_key not in used_edges:
                        result[k - 1] = [a, b]
                        used_edges.add(edge_key)
                        found = True
                        break
            if not found:
                result[k - 1] = None
            continue

        # General case: use backtracking with restarts
        adj = adj_list[k - 1]
        order = order_list[k - 1]
        
        # Try multiple restarts with different seeds if the first one fails
        # We limit the time per tree to ensure we finish all trees
        tree_start = time.time()
        max_time_per_tree = 10.0 # 10 seconds per tree is generous for N=20 with FC
        
        embedding = None
        attempt = 0
        while time.time() - tree_start < max_time_per_tree and time.time() < deadline:
            embedding = _backtrack_embed(k, edges, adj, order, seed=attempt)
            if embedding is not None:
                break
            attempt += 1

        if embedding is not None:
            result[k - 1] = embedding
            for (u, v) in edges:
                edge_key = (min(embedding[u], embedding[v]), max(embedding[u], embedding[v]))
                used_edges.add(edge_key)
        else:
            result[k - 1] = None

    # Verify and compute
    return result


if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k - 1] is not None:
            r = res[k - 1]
            if not isinstance(r, list):
                print(f"T{k}: not a list")
                ok = False
                continue
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v!r}")
                    ok = False
            for (u, v) in trees[k - 1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k - 1]) for k in range(1, 21) if res[k - 1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
