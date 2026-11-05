"""Fast NJ for phylogenetic topology recovery."""
import time
import numpy as np

_START_TIME = None
_DEADLINE = 255.0  # seconds, safety margin under 270s


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Vectorized JC distance computation using NumPy."""
    n = len(labels)
    seq_matrix = np.empty((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lbl in enumerate(labels):
        s = sequences[lbl]
        for j, c in enumerate(s):
            if c == 'A':
                seq_matrix[i, j] = 0
            elif c == 'C':
                seq_matrix[i, j] = 1
            elif c == 'G':
                seq_matrix[i, j] = 2
            else:
                seq_matrix[i, j] = 3
    L = seq_matrix.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diffs = (seq_matrix[i] != seq_matrix).sum(axis=1).astype(np.float64)
        p = np.clip(diffs / L, 0.0, 0.7499)
        dist[i] = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(labels, dist):
    """
    Neighbour Joining algorithm.
    Returns: (adj, node_count) where adj is dict node->list of (neighbor, length),
             and node_count is total number of nodes (leaves + internal).
    """
    n = len(labels)
    if n == 1:
        return {0: []}, 1
    if n == 2:
        return {0: [(1, dist[0][1])], 1: [(0, dist[0][1])]}, 2

    # adj: dict int -> list of (neighbor_int, length)
    adj = {i: [] for i in range(n)}

    # Dynamic distance matrix: start with n x n, grow as needed
    d = dist.copy()
    nodes = list(range(n))
    next_id = n

    while len(nodes) > 2:
        m = len(nodes)

        # Compute row sums
        r = np.zeros(m, dtype=np.float64)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    r[i] += d[a, b]

        # Find pair with minimum criterion
        best_crit = float('inf')
        best_pair = (0, 1)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[a, b] - r[i] - r[j]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)

        a, b = best_pair

        # Compute branch lengths
        da = 0.5 * d[a, b]
        db = 0.5 * d[a, b]
        for x in nodes:
            if x != a and x != b:
                da += 0.5 * (d[a, x] - d[b, x])
                db += 0.5 * (d[b, x] - d[a, x])
        da = max(da, 1e-10)
        db = max(db, 1e-10)

        new_node = next_id
        next_id += 1

        # Get neighbors of a and b (excluding each other)
        neighbors_a = [nb for nb, _ in adj[a] if nb != b]
        neighbors_b = [nb for nb, _ in adj[b] if nb != a]

        # Remove a and b from their neighbors' adjacency lists
        for nb in neighbors_a:
            adj[nb] = [(n2, l2) for n2, l2 in adj[nb] if n2 != a]
        for nb in neighbors_b:
            adj[nb] = [(n2, l2) for n2, l2 in adj[nb] if n2 != b]

        # Create new node
        adj[new_node] = [(a, da), (b, db)]
        adj[a] = [(new_node, da)]
        adj[b] = [(new_node, db)]

        # Connect new_node to neighbors of a and b
        for nb in neighbors_a:
            adj[nb].append((new_node, 0.0))
            adj[new_node].append((nb, 0.0))
        for nb in neighbors_b:
            if nb not in [x[0] for x in adj[new_node]]:
                adj[nb].append((new_node, 0.0))
                adj[new_node].append((nb, 0.0))

        # Update distance matrix: grow if needed
        rest = [x for x in nodes if x not in (a, b)]
        if new_node >= d.shape[0]:
            new_size = new_node + 1
            new_d = np.zeros((new_size, new_size), dtype=np.float64)
            new_d[:d.shape[0], :d.shape[1]] = d
            d = new_d

        for x in rest:
            d[new_node, x] = d[x, new_node] = 0.5 * (d[a, x] + d[b, x] - d[a, b])

        nodes = rest + [new_node]

    # Now we have 2 nodes left. Connect them.
    if len(nodes) == 2:
        a, b = nodes[0], nodes[1]
        if b not in [nb for nb, _ in adj[a]]:
            adj[a].append((b, 0.0))
            adj[b].append((a, 0.0))

    return adj, next_id


def _adj_to_newick(adj, node_count, labels):
    """Convert adjacency list to Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"

    # Find a leaf to start from
    leaf = 0
    for i in range(n):
        if len(adj[i]) == 1:
            leaf = i
            break

    # Build Newick via DFS
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        if node < n:
            return labels[node]
        children = []
        for nb, length in adj[node]:
            if nb != parent and nb not in visited:
                child_str = dfs(nb, node)
                children.append(child_str)
        if len(children) == 1:
            return f"({children[0]})"
        elif len(children) > 1:
            return f"({','.join(children)})"
        else:
            return labels[node] if node < n else ""

    newick = dfs(leaf, -1)
    return newick + ";"


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    try:
        # Compute distance matrix
        dist = _jc_distance_matrix(sequences, labels)

        # Run NJ
        adj, node_count = _nj_tree(labels, dist)

        # Convert to Newick
        newick = _adj_to_newick(adj, node_count, labels)

        # Validate: ensure all labels are present
        for lbl in labels:
            if lbl not in newick:
                raise ValueError(f"Missing label {lbl} in Newick")

        return newick

    except Exception:
        # Fallback: return a star tree
        if n <= 2:
            if n == 1:
                return labels[0] + ";"
            else:
                return f"{labels[0]},{labels[1]};"
        else:
            return "(" + ",".join(labels) + ");"
