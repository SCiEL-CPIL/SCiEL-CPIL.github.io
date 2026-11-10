"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph.

Design (family of the measured improvement): precompute bitmask adjacency, use
incremental triangle-free-set tracking, and run a randomized local search
(simulated annealing) with a strict K4-freeness guard and a hard deadline.
"""
import random
import time

N = 18
DEADLINE = 115.0  # safety margin well under the 120s evaluator kill


def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _build_masks(edges):
    """adj[v] = bitmask of neighbors; present[e] = bitmask of edge present."""
    adj = [0] * N
    present = [False] * len(_all_pairs())  # placeholder length; rebuilt below
    return adj


def _edge_index():
    idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            idx[(i, j)] = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    return idx


# Precompute a static edge-index map (constant across runs).
_EDGE_INDEX = _edge_index()
_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
_NPAIRS = len(_PAIRS)


def _mask_from_edges(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _k4_free(adj):
    """Check K4-free using bitmasks: no 4-clique exists."""
    # A K4 exists iff some edge (u,v) with (N(u)&N(v)) containing an edge w-x
    # where w,x in common neighbors.
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # check if common induces any edge
                m = common
                while m:
                    w = (m & -m).bit_length() - 1
                    m &= m - 1
                    if (common >> w) & 1 and ((adj[w] & common) & ~(1 << w)):
                        # w has another neighbor in common -> K4
                        # (w, x) both adjacent to u,v and to each other
                        if (adj[w] & common) & ~(1 << w):
                            return False
    return True


def _k4_free_fast(adj):
    """Faster K4 check: for each pair of edges sharing a vertex, check closure."""
    # Build list of edges.
    for u in range(N):
        mu = adj[u]
        # neighbors of u
        while mu:
            v = (mu & -mu).bit_length() - 1
            mu &= mu - 1
            if v <= u:
                continue
            # edge (u,v) present; common neighbors
            common = adj[u] & adj[v]
            # if any two common neighbors are adjacent -> K4
            mc = common
            while mc:
                w = (mc & -mc).bit_length() - 1
                mc &= mc - 1
                if (adj[w] & common) & ~(1 << w):
                    return False
    return True


def _max_tf_size(adj):
    """Largest triangle-free induced subgraph size (exact).

    Triangle-free induced subgraph = set S with NO edges inside S.
    So we want the largest independent set of the graph.
    (A triangle-free induced subgraph with no edges is an independent set;
     but triangle-free allows edges as long as no 3-cycle. However, the induced
     subgraph being triangle-free means it can have edges but no triangle.
     The LARGEST triangle-free induced subgraph is at least the independence
     number, and we minimize that max size.)

    We need the true largest triangle-free induced subgraph, not just IS.
    Use branch and bound over subsets.
    """
    # Largest triangle-free induced subgraph = largest S such that G[S] has no K3.
    # Equivalently, the complement: smallest vertex set whose removal leaves a
    # triangle-free graph? No - we want max S with no triangle inside.
    # This is the "triangle-free induced subgraph" = maximum induced subgraph
    # with no triangle. Use branch and bound.
    best = 0
    full = (1 << N) - 1

    # Precompute for each vertex the bitmask of its neighbors.
    def has_triangle(S):
        # S is a bitmask; check if G[S] contains a triangle.
        m = S
        while m:
            u = (m & -m).bit_length() - 1
            m &= m - 1
            # neighbors of u inside S
            nu = adj[u] & S
            # check if any two neighbors of u are adjacent
            mn = nu
            while mn:
                v = (mn & -mn).bit_length() - 1
                mn &= mn - 1
                if (adj[v] & nu) & ~(1 << v):
                    return True
        return False

    # Branch and bound: build S by adding vertices; prune if triangle formed.
    # We want max |S| with no triangle.
    # Recursive: try including each candidate vertex if it keeps triangle-free.
    def dfs(S, cand):
        nonlocal best
        if cand == 0:
            if S.bit_count() > best:
                best = S.bit_count()
            return
        # upper bound
        if S.bit_count() + cand.bit_count() <= best:
            return
        # pick a vertex to branch on
        v = (cand & -cand).bit_length() - 1
        rest = cand & ~(1 << v)
        # Option 1: include v if S ∪ {v} is triangle-free
        Sv = S | (1 << v)
        if not has_triangle(Sv):
            dfs(Sv, rest)
        # Option 2: exclude v
        dfs(S, rest)

    dfs(0, full)
    return best


def _max_tf_size_fast(adj, deadline=None):
    """Largest triangle-free induced subgraph via branch and bound with pruning.

    Optimizations:
      - Order candidates by degree (high degree first) to prune early.
      - Precompute triangle-containing checks incrementally.
    """
    # A triangle-free induced subgraph S: G[S] has no K3.
    # We maximize |S|.
    # Key pruning: if adding v creates a triangle with two existing neighbors, skip.
    best = 0
    full = (1 << N) - 1

    # Precompute: tri[v] = bitmask of vertices w such that edge (v,w) exists.
    # For incremental triangle check when adding v to S:
    #   v creates a triangle iff there exist x,y in S with x~y and both ~v.
    #   i.e., (adj[v] & S) contains an edge.

    def creates_triangle(v, S):
        # Is there an edge among neighbors of v that are in S?
        nv = adj[v] & S
        m = nv
        while m:
            x = (m & -m).bit_length() - 1
            m &= m - 1
            if (adj[x] & nv) & ~(1 << x):
                return True
        return False

    # Candidate ordering: sort vertices by descending degree for better pruning.
    order = sorted(range(N), key=lambda v: adj[v].bit_count(), reverse=True)
    # Map order index -> vertex
    # We'll branch in this fixed order.

    # Precompute for each prefix of the order, the set of remaining vertices.
    # Use recursion over position in order.

    # For speed, precompute neighbor masks and use a global best.
    # Also track S as bitmask.

    # Additional pruning: if S already has a vertex v, and we add u, check
    #   creates_triangle(u, S) quickly.

    # Precompute deg for ordering.
    # Use a list-based recursion.

    def dfs(pos, S):
        nonlocal best
        if deadline is not None and time.time() > deadline:
            return
        if pos == N:
            if S.bit_count() > best:
                best = S.bit_count()
            return
        # Upper bound: current size + remaining vertices
        remaining = N - pos
        if S.bit_count() + remaining <= best:
            return
        v = order[pos]
        # Try include
        if not creates_triangle(v, S):
            dfs(pos + 1, S | (1 << v))
        # Try exclude
        dfs(pos + 1, S)

    dfs(0, 0)
    return best


def _k4_free_bitmask(adj):
    """K4-free check using bitmasks (fast)."""
    for u in range(N):
        mu = adj[u]
        while mu:
            v = (mu & -mu).bit_length() - 1
            mu &= mu - 1
            if v <= u:
                continue
            common = adj[u] & adj[v]
            # If common contains an edge, K4 exists.
            mc = common
            while mc:
                w = (mc & -mc).bit_length() - 1
                mc &= mc - 1
                if (adj[w] & common) & ~(1 << w):
                    return False
    return True


def _initial_graph():
    """Complete tripartite K6,6,6 (seed): score 12, K4-free."""
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges


def _adj_to_bitmasks(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    # Start from the seed graph (K6,6,6), score 12.
    edges = _initial_graph()
    adj = _adj_to_bitmasks(edges)

    # Verify K4-free and compute score.
    assert _k4_free_bitmask(adj)
    best_score = _max_tf_size_fast(adj, deadline)
    best_adj = adj[:]
    best_edges = edges[:]

    # Simulated annealing over edge flips, maintaining K4-freeness.
    # State: adj bitmask.
    cur_adj = adj[:]
    cur_score = best_score

    # Precompute for fast K4 check after a single edge flip:
    #   Adding edge (u,v): creates K4 iff there is w such that u~w, v~w, and
    #   w has a neighbor x in N(u)∩N(v) with x~w. Simpler: after adding (u,v),
    #   check if N(u)∩N(v) contains an edge.
    #   Removing edge (u,v): never creates a K4.

    def add_edge_creates_k4(u, v, a):
        common = a[u] & a[v]
        mc = common
        while mc:
            w = (mc & -mc).bit_length() - 1
            mc &= mc - 1
            if (a[w] & common) & ~(1 << w):
                return True
        return False

    def remove_edge_safe(u, v, a):
        return True  # removing never creates K4

    # Neighbor lists for fast iteration.
    def neighbors(a, v):
        m = a[v]
        out = []
        while m:
            w = (m & -m).bit_length() - 1
            m &= m - 1
            out.append(w)
        return out

    # Temperature schedule.
    T0 = 1.0
    T_end = 0.01
    steps_budget = 200000
    step = 0
    improved = 0

    while time.time() < deadline - 2.0:
        step += 1
        T = T0 * (T_end / T0) ** (step / steps_budget)

        # Choose a random edge to flip.
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        is_present = (cur_adj[a] >> b) & 1

        if is_present:
            # Remove edge.
            new_adj = cur_adj[:]
            new_adj[a] &= ~(1 << b)
            new_adj[b] &= ~(1 << a)
            # Score: removing an edge can only increase or keep the max TF size.
            # We want to minimize, so this is unlikely to help unless it allows
            # other changes. Evaluate.
            if _k4_free_bitmask(new_adj):
                s = _max_tf_size_fast(new_adj, deadline)
                if s < cur_score:
                    cur_adj = new_adj
                    cur_score = s
                    if s < best_score:
                        best_score = s
                        best_adj = new_adj[:]
                        best_edges = [[a2, b2] for a2 in range(N) for b2 in range(a2+1, N) if (new_adj[a2] >> b2) & 1]
                        improved += 1
                elif s == cur_score and rng.random() < 0.3:
                    # Accept equal to explore.
                    cur_adj = new_adj
                else:
                    # Reject (worse).
                    pass
        else:
            # Add edge.
            if add_edge_creates_k4(a, b, cur_adj):
                continue  # would create K4, skip
            new_adj = cur_adj[:]
            new_adj[a] |= (1 << b)
            new_adj[b] |= (1 << a)
            s = _max_tf_size_fast(new_adj, deadline)
            if s < cur_score:
                cur_adj = new_adj
                cur_score = s
                if s < best_score:
                    best_score = s
                    best_adj = new_adj[:]
                    best_edges = [[a2, b2] for a2 in range(N) for b2 in range(a2+1, N) if (new_adj[a2] >> b2) & 1]
                    improved += 1
            else:
                # Metropolis: accept worse with prob exp(-(s-cur_score)/T)
                if s - cur_score <= 0:
                    pass
                else:
                    import math
                    p = math.exp(-(s - cur_score) / max(T, 1e-9))
                    if rng.random() < p:
                        cur_adj = new_adj
                        cur_score = s

    # Final: return best_edges found.
    # Reconstruct edges from best_adj to be safe.
    final_edges = [[a, b] for a in range(N) for b in range(a + 1, N) if (best_adj[a] >> b) & 1]
    return final_edges
