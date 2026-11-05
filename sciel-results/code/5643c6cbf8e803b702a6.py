import time
import random

_N = 20
_COLORS = 20
_ALL_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]

# Precompute, for each vertex, the list of (other_vertex, edge_index)
def _precompute():
    adj = [[] for _ in range(_N)]
    for idx, (i, j) in enumerate(_ALL_EDGES):
        adj[i].append((j, idx))
        adj[j].append((i, idx))
    return adj

_ADJ = _precompute()

def _find_color(adj_colors, edge, u, v, adj):
    """Find a color for edge (u,v) so that no 4-cycle through it becomes non-rainbow.
    adj_colors: dict edge_idx -> color for currently present edges.
    Returns a valid color in 0..19 or None."""
    # For each color, check validity. A color is invalid if it would create a
    # 4-cycle with a repeated color.
    # Build sets of neighbor colors for u and v.
    # For a 4-cycle u-w-x-v-u: edges (u,w),(w,x),(x,v),(v,u).
    # We need all 4 colors distinct. The new edge color c must differ from
    # colors of (u,w),(w,x),(x,v).
    # So for a candidate c: for every w in N(u), x in N(v) such that (w,x) is an edge
    #   (i.e., x adjacent to w and w!=v, x!=u), we need c != col(u,w), c != col(w,x), c != col(x,v).
    # Equivalently, c must not be in the union of those colors over all such (w,x).
    forbidden = set()
    nu = adj[u]  # list of (other, idx)
    nv = adj[v]
    # map vertex -> edge idx from u
    # For efficiency, for each w in nu (w != v), for each x in nv (x != u and x != w),
    #   check if (w,x) is an edge (present or not). If present, add its color and the
    #   colors of (u,w) and (x,v).
    # But (w,x) must be a present edge for the 4-cycle to exist.
    present = set(adj_colors.keys())
    for w, uwi in nu:
        if w == v:
            continue
        c_uw = adj_colors[uwi]
        for x, vx_i in nv:
            if x == u or x == w:
                continue
            # is (w,x) a present edge?
            # find edge index for (w,x)
            a, b = (w, x) if w < x else (x, w)
            # need edge index; precompute a dict
            ei = _EDGE_IDX.get((a, b))
            if ei is None or ei not in present:
                continue
            c_wx = adj_colors[ei]
            c_xv = adj_colors[vx_i]
            forbidden.add(c_uw)
            forbidden.add(c_wx)
            forbidden.add(c_xv)
    for c in range(_COLORS):
        if c not in forbidden:
            return c
    return None

_EDGE_IDX = {e: i for i, e in enumerate(_ALL_EDGES)}

def _validate(edges):
    """Full validation: every 4-cycle is rainbow. edges: list of (u,v,c)."""
    col = {}
    for (u, v, c) in edges:
        col[(u, v)] = c
        col[(v, u)] = c
    present = set()
    for (u, v, c) in edges:
        present.add((min(u, v), max(u, v)))
    # enumerate 4-cycles
    for i in range(_N):
        for j in range(i + 1, _N):
            if (i, j) not in present:
                continue
            for k in range(j + 1, _N):
                if (j, k) not in present:
                    continue
                for l in range(k + 1, _N):
                    if (k, l) not in present:
                        continue
                    if (l, i) not in present:
                        continue
                    # 4-cycle i-j-k-l-i
                    colors = [col[(i, j)], col[(j, k)], col[(k, l)], col[(l, i)]]
                    if len(set(colors)) != 4:
                        return False
    return True

def _try_order(order, deadline):
    """Greedy add edges in given order. Returns list of (u,v,c) for valid edges."""
    adj_colors = {}
    result = []
    for ei in order:
        if time.time() > deadline:
            break
        u, v = _ALL_EDGES[ei]
        c = _find_color(adj_colors, ei, u, v, _ADJ)
        if c is not None:
            adj_colors[ei] = c
            result.append((u, v, c))
    return result

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    rng = random.Random(12345)
    best = []
    best_len = 0

    # Start with a guaranteed-valid baseline: a 20-cycle (or star) to have something.
    # Star: edges (0, v) all color 0 — valid, 19 edges.
    baseline = [(0, v, 0) for v in range(1, _N)]
    best = baseline
    best_len = len(baseline)

    # Many restarts with random edge orderings.
    max_edges = len(_ALL_EDGES)
    while time.time() < deadline:
        order = list(range(max_edges))
        rng.shuffle(order)
        res = _try_order(order, deadline)
        if len(res) > best_len:
            # full-validate to be safe
            if _validate(res):
                best = res
                best_len = len(res)
            # early stop if we hit a very dense graph
            if best_len >= 170:
                break

    # Convert to required format
    return [[u, v, c] for (u, v, c) in best]
