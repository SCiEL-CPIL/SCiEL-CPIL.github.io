"""Rewritten: connectivity-aware (MST-leaf) constructive heuristic for TSP tour building."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.perf_counter()
_DEADLINE_FRACTION = 0.90  # safety margin vs 120s limit
_DEADLINE_ABS = 120.0 * _DEADLINE_FRACTION


def _remaining_deadline():
    return _DEADLINE_ABS - (time.perf_counter() - _START_TIME)


def _mst_leaf_scores(unvisited, distance_matrix):
    """
    For the subgraph induced by unvisited nodes, compute for each node a
    'leaf-ness' score based on the minimum spanning tree (MST) of that
    subgraph. Nodes that are leaves in the MST (degree 1) are at the
    boundary of the remaining region; connecting to a leaf tends to keep
    the partial tour a single path and avoids isolating clusters.

    Returns an array aligned to unvisited of length len(unvisited):
      score[i] = weight of the MST edge incident to node unvisited[i]
                 (for leaves) or a small penalty for internal nodes.
    We then prefer nodes that are leaves AND have a cheap incident edge,
    because that indicates an easy, low-cost way to extend the tour while
    preserving global connectivity.
    """
    n = len(unvisited)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)

    # Submatrix of distances among unvisited nodes
    sub = distance_matrix[np.ix_(unvisited, unvisited)]

    # Kruskal's MST (n <= 50, O(n^2 log n) is fine)
    m = n * (n - 1) // 2
    edges = np.empty((m, 3), dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges[k, 0] = i
            edges[k, 1] = j
            edges[k, 2] = k
            k += 1
    # sort by weight
    order = np.argsort(sub[np.triu_indices(n, 1)], kind="stable")
    edge_weights = sub[np.triu_indices(n, 1)][order]
    edge_pairs = np.column_stack(np.unravel_index(order, (n, n)))
    # build sorted edge list
    sorted_edges = np.empty((m, 3), dtype=np.int64)
    sorted_edges[:, 0] = edge_pairs[:, 0]
    sorted_edges[:, 1] = edge_pairs[:, 1]
    sorted_edges[:, 2] = np.arange(m)  # index into edge_weights

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = np.zeros(n, dtype=np.int64)
    deg_weight = np.zeros(n, dtype=float)  # sum of MST edge weights incident
    count = 0
    for a, b, widx in sorted_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            deg[a] += 1
            deg[b] += 1
            w = float(edge_weights[widx])
            deg_weight[a] += w
            deg_weight[b] += w
            count += 1
            if count == n - 1:
                break

    # Leaf-ness: degree 1 -> strong preference; degree 2 -> mild; >2 -> penalty.
    # Prefer low incident weight (cheap to attach).
    leaf_bonus = np.where(deg == 1, 1.0, np.where(deg == 2, 0.3, -0.5))
    # Normalise deg_weight by max for scale-independence
    max_dw = deg_weight.max() if deg_weight.max() > 0 else 1.0
    norm_dw = deg_weight / max_dw

    # Combined: high leaf_bonus and low norm_dw is good.
    score = leaf_bonus - 0.5 * norm_dw
    return score


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = len(unvisited_nodes)
    if n == 0:
        # No unvisited nodes; return current as a safe fallback (should not happen)
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # Deadline guard: if we are very close to the limit, fall back to a
    # fast local greedy so we always return a valid node in time.
    if _remaining_deadline() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Local distance signal (cheap, vectorised)
    local = distance_matrix[current_node, unvisited_nodes]

    # Global connectivity signal: MST leaf-ness of the remaining subgraph.
    # This is the structural change: it discourages moves that would strand
    # a cluster of nodes (creating a disconnected subtour) by preferring
    # boundary (leaf) nodes of the remaining MST, which keeps the partial
    # tour a single extendable path.
    leaf_score = _mst_leaf_scores(unvisited_nodes, distance_matrix)

    # Destination-aware term: encourage progress toward destination_node
    # to keep the tour coherent, but keep it small so it does not dominate.
    dest = distance_matrix[unvisited_nodes, destination_node]

    # Normalise each component to [0,1] for stable combination.
    def _norm01(x):
        x = np.asarray(x, dtype=float)
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    local_n = _norm01(local)          # lower is better -> we subtract
    leaf_n = _norm01(leaf_score)      # higher is better -> we add
    dest_n = _norm01(dest)            # lower is better -> we subtract

    # Weighted combination. Leaf/connectivity term is the primary structural
    # signal; local distance is secondary; destination is a small tie-breaker.
    combined = 0.55 * leaf_n - 0.35 * local_n - 0.10 * dest_n

    # Prefer the candidate with the highest combined score.
    idx = int(np.argmax(combined))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
