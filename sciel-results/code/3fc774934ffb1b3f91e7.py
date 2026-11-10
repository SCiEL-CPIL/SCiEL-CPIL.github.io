# EVOLVE-BLOCK-START
import numpy as np
import time

# ---------------------------------------------------------------------------
# Ring structure
# ---------------------------------------------------------------------------
N_NODES = 10
N_EDGES = 10
N_DEMANDS = 15

DEMANDS = [
    (0, 5, 3),
    (1, 7, 2),
    (2, 9, 4),
    (3, 6, 1),
    (0, 4, 5),
    (2, 8, 2),
    (5, 9, 3),
    (1, 4, 2),
    (6, 0, 4),
    (3, 8, 1),
    (7, 2, 3),
    (4, 9, 2),
    (8, 1, 5),
    (0, 6, 1),
    (5, 2, 2),
]

WEIGHTS = np.array([w for s, t, w in DEMANDS], dtype=np.float64)


def _build_edge_masks():
    """For each demand, 0/1 mask of edges used when routed clockwise."""
    cw_masks = []
    for s, t, w in DEMANDS:
        mask = np.zeros(N_EDGES)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, N_NODES):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)
    return np.array(cw_masks, dtype=np.int8)


CW_MASKS = _build_edge_masks()          # shape (15, 10)
CCW_MASKS = (1 - CW_MASKS).astype(np.int8)  # shape (15, 10)


# ---------------------------------------------------------------------------
# Exact max-flow (Dinic) on the bipartite feasibility network
# ---------------------------------------------------------------------------
def _dinic_maxflow(n_nodes, adj, s, t):
    """adj: list of lists of (to, cap). Returns (maxflow, residual list of dicts)."""
    # residual stored as dict per node: neighbour -> remaining capacity
    residual = [dict() for _ in range(n_nodes)]
    for u in range(n_nodes):
        for v, cap in adj[u]:
            if cap <= 0:
                continue
            residual[u][v] = residual[u].get(v, 0) + cap
            residual[v][u] = residual[v].get(u, 0)  # reverse edge (0 initially)

    n = n_nodes
    flow = 0
    INF = float('inf')
    while True:
        # BFS level graph
        level = [-1] * n
        level[s] = 0
        q = [s]
        for u in q:
            for v in residual[u]:
                if residual[u][v] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[t] < 0:
            break
        # DFS blocking flow (iterative)
        it = [0] * n
        stack = [(s, INF, [])]  # (node, bottleneck, path of edges)
        while stack:
            u, f, path = stack[-1]
            if u == t:
                # augment
                for (a, b) in path:
                    residual[a][b] -= f
                    residual[b][a] += f
                flow += f
                stack.pop()
                continue
            # find next admissible edge
            nbrs = list(residual[u].keys())
            advanced = False
            while it[u] < len(nbrs):
                v = nbrs[it[u]]
                it[u] += 1
                if residual[u][v] > 0 and level[v] == level[u] + 1:
                    stack.append((v, min(f, residual[u][v]), path + [(u, v)]))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
    return flow, residual


def _feasible(L):
    """Return (feasible, routing_or_None) for max-load bound L.

    Network:
      node 0 = source
      nodes 1..15 = demands (CW arc for demand d is node 1+d)
      nodes 16..25 = demands (CCW arc for demand d is node 16+d)
      nodes 26..35 = edges
      node 36 = sink
    """
    n_nodes = 37
    adj = [[] for _ in range(n_nodes)]
    src = 0
    sink = 36

    for d in range(N_DEMANDS):
        w = WEIGHTS[d]
        cw_node = 1 + d
        ccw_node = 16 + d
        adj[src].append((cw_node, w))
        adj[src].append((ccw_node, w))
        for e in range(N_EDGES):
            edge_node = 26 + e
            if CW_MASKS[d, e]:
                adj[cw_node].append((edge_node, w))
            else:
                adj[ccw_node].append((edge_node, w))
            adj[edge_node].append((sink, L))

    target = 10.0 * L
    flow, residual = _dinic_maxflow(n_nodes, adj, src, sink)

    if flow + 1e-9 >= target:
        # Reconstruct routing: demand d goes CW iff residual[src][1+d] < w
        routing = np.zeros(N_DEMANDS, dtype=np.float64)
        for d in range(N_DEMANDS):
            w = WEIGHTS[d]
            cw_node = 1 + d
            rem = residual[src].get(cw_node, 0.0)
            if rem < w - 1e-9:
                routing[d] = 1.0  # counter-clockwise
            else:
                routing[d] = 0.0  # clockwise
        return True, routing
    return False, None


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------
def _compute_loads(routing):
    cw_w = (1.0 - routing) * WEIGHTS
    ccw_w = routing * WEIGHTS
    return CW_MASKS.T @ cw_w + CCW_MASKS.T @ ccw_w


def _max_load(routing):
    return float(np.max(_compute_loads(routing)))


def _lower_bound():
    lb = 0.0
    for e in range(N_EDGES):
        s = 0.0
        for d in range(N_DEMANDS):
            if CW_MASKS[d, e] or CCW_MASKS[d, e]:
                s += WEIGHTS[d]
        lb = max(lb, s / 2.0)
    return lb


def _upper_bound():
    # all clockwise
    return _max_load(np.zeros(N_DEMANDS))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 620.0  # safety margin vs 630s wall-clock

    lb = _lower_bound()
    ub = _upper_bound()

    best_routing = np.zeros(N_DEMANDS, dtype=np.float64)
    best_max = ub

    # Binary search on the smallest feasible integer L.
    # Loads are integer-valued, so the optimum is an integer; binary search
    # over integers in [ceil(lb), floor(ub)] finds it exactly.
    lo = int(np.ceil(lb - 1e-9))
    hi = int(np.floor(ub + 1e-9))
    best_L = hi
    best_route = None

    while lo <= hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        ok, route = _feasible(float(mid))
        if ok:
            best_L = mid
            best_route = route
            hi = mid - 1
        else:
            lo = mid + 1

    if best_route is not None:
        best_routing = best_route
        best_max = _max_load(best_routing)
    else:
        best_routing = np.zeros(N_DEMANDS, dtype=np.float64)
        best_max = _max_load(best_routing)

    # Optional polish: a tiny local search to confirm / improve the max load
    # (should not be needed, but guards against any float edge effects).
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            cur = _max_load(best_routing)
            for i in range(N_DEMANDS):
                cand = best_routing.copy()
                cand[i] = 1.0 - cand[i]
                m = _max_load(cand)
                if m < cur - 1e-12:
                    best_routing = cand
                    improved = True
        best_max = _max_load(best_routing)

    return best_routing


# EVOLVE-BLOCK-END
