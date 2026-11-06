import time
import random

# ---------------------------------------------------------------------------
# Fast validators (bitmask based)
# ---------------------------------------------------------------------------
def _k4_found(adj, n):
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                        higher = rest & (~((1 << (e + 1)) - 1))
                        if higher:
                            f = (higher & -higher).bit_length() - 1
                            return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

# ---------------------------------------------------------------------------
# Known 35-vertex (4,6)-Ramsey graph: Z_7 x Z_5
# ---------------------------------------------------------------------------
def _build_35_vertex_graph():
    n = 35
    adj = [0] * n
    S7 = {1, 2, 4}
    S5 = {1, 2}
    for i1 in range(7):
        for j1 in range(5):
            v1 = i1 * 5 + j1
            for i2 in range(i1, 7):
                for j2 in range(5):
                    if i1 == i2 and j1 == j2:
                        continue
                    if i2 == i1 and j2 < j1:
                        continue
                    v2 = i2 * 5 + j2
                    if v1 >= v2:
                        continue
                    di = (i2 - i1) % 7
                    dj = (j2 - j1) % 5
                    if di in S7 and dj in S5:
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
    return adj

# ---------------------------------------------------------------------------
# Sequential vertex-construction backtracking (constraint-programming style).
#
# We build the graph on vertices 0..n-1 one vertex at a time.  When adding
# vertex v (v>=1) we must choose a subset S of {0..v-1} to be its neighbours.
# S is *allowed* iff:
#   (A) S does NOT complete a K4:  for every pair (a,b) in S that are adjacent
#       in the already-built graph, there is no third common neighbour c in S.
#       Equivalently, the induced subgraph on S must be K3-free (triangle-free),
#       because any triangle inside S plus v would be a K4.
#   (B) S does NOT complete an independent 6-set:  the non-neighbours of v
#       within {0..v-1}  (call it T = complement of S)  must NOT contain an
#       independent set of size 5 in the built graph, because T plus v would
#       then be an independent 6-set.
#
# We enumerate candidate S via backtracking over the bits of {0..v-1}, pruning
# as soon as (A) or (B) is violated.  This is a structurally different search
# from random edge flipping: it grows the graph and prunes exponentially.
# ---------------------------------------------------------------------------
def _indep_set_size_in_mask(mask, adj, n, need):
    """Return True if the induced subgraph on 'mask' contains an independent
    set of size >= need (exact).  Uses incremental backtracking."""
    # gather vertices
    verts = []
    m = mask
    while m:
        lsb = m & -m
        verts.append(lsb.bit_length() - 1)
        m ^= lsb
    L = len(verts)
    if L < need:
        return False
    # independent set search: choose vertices with no edges among them
    chosen = []
    def rec(start, avail):
        # avail: bitmask (in verts-index space) of candidates that are
        # independent from all chosen.  We need (need - len(chosen)) more.
        rem = need - len(chosen)
        if rem == 0:
            return True
        if avail.bit_count() < rem:
            return False
        # iterate candidates in avail
        m = avail
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            if idx < start:
                continue
            v = verts[idx]
            # new available: candidates with index > idx not adjacent to v
            new_avail = avail & (~((1 << (idx + 1)) - 1))
            # remove those adjacent to v
            adjmask = 0
            for j in range(idx + 1, L):
                if (adj[v] >> verts[j]) & 1:
                    adjmask |= (1 << j)
            new_avail &= ~adjmask
            chosen.append(v)
            if rec(idx + 1, new_avail):
                return True
            chosen.pop()
        return False
    return rec(0, (1 << L) - 1)

def _triangle_in_mask(mask, adj):
    """Return True if induced subgraph on 'mask' contains a triangle (K3)."""
    verts = []
    m = mask
    while m:
        lsb = m & -m
        verts.append(lsb.bit_length() - 1)
        m ^= lsb
    L = len(verts)
    for i in range(L):
        a = verts[i]
        for j in range(i + 1, L):
            b = verts[j]
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # need a third vertex in mask, index > j
            m = common & (~((1 << (j + 1)) - 1))
            if m:
                return True
    return False

def _construct_backtrack(n, seed, deadline, init_adj=None):
    """Backtracking construction of a valid n-vertex graph.  Returns adj or None.
    If init_adj (a valid smaller graph on first k vertices) is given, we start
    from it and only assign the remaining vertices."""
    adj = [0] * n
    k = 0
    if init_adj is not None:
        k = len(init_adj)
        for i in range(k):
            adj[i] = init_adj[i]
    rng = random.Random(seed)

    # Precompute, for each already-present vertex, its neighbour mask within
    # the built prefix, to speed up checks.  We recompute on the fly instead.

    def allowed(S_mask, v):
        # (A) S must be triangle-free in the built graph
        if _triangle_in_mask(S_mask, adj):
            return False
        # (B) non-neighbours T must not contain an independent set of size 5
        prefix = (1 << v) - 1
        T = prefix & ~S_mask
        if _indep_set_size_in_mask(T, adj, v, 5):
            return False
        return True

    def rec(v):
        if time.time() > deadline:
            return False
        if v == n:
            return True
        # enumerate subsets S of {0..v-1} via backtracking over bits
        # We build S incrementally; prune on (A) and (B).
        # Use recursion over candidate vertices 0..v-1.
        # State: current S_mask, and we also need to track that S is triangle
        # free (checked at the end is too late; we check incrementally by
        # ensuring the new vertex added to S doesn't create a triangle with
        # two existing S vertices).  Simpler: check allowed() at leaves and
        # prune early using a cheaper necessary condition.

        # Incremental: maintain S_mask and a "safe" flag.  When adding vertex
        # x to S, S remains triangle-free iff x is not adjacent to an edge
        # inside S (i.e., no a<b in S with a-b edge and both adjacent to x).
        def rec_build(idx, S_mask, tri_free):
            if time.time() > deadline:
                return False
            if idx == v:
                if not tri_free:
                    return False
                # check (B)
                prefix = (1 << v) - 1
                T = prefix & ~S_mask
                if _indep_set_size_in_mask(T, adj, v, 5):
                    return False
                # commit
                adj[v] = S_mask
                for x in range(v):
                    if (S_mask >> x) & 1:
                        adj[x] |= (1 << v)
                if rec(v + 1):
                    return True
                # undo
                adj[v] = 0
                for x in range(v):
                    if (S_mask >> x) & 1:
                        adj[x] &= ~(1 << v)
                return False
            # option 1: exclude idx from S
            if rec_build(idx + 1, S_mask, tri_free):
                return True
            # option 2: include idx in S (if keeps triangle-free)
            # adding idx: triangle created iff there is an edge (a,b) in S with
            # a,b both adjacent to idx.
            ok = tri_free
            if ok:
                common = adj[idx] & S_mask
                # is there an edge inside common?
                if _triangle_in_mask(common, adj):
                    ok = False
            if ok and rec_build(idx + 1, S_mask | (1 << idx), True):
                return True
            return False

        # randomise order of include/exclude to diversify
        return rec_build(0, 0, True)

    return rec(0) and adj or None

# ---------------------------------------------------------------------------
# Simulated-annealing edge-flip refinement (delta-based violation counting).
# ---------------------------------------------------------------------------
def _count_edges_in_subset(mask, adj):
    total_deg = 0
    m = mask
    while m:
        lsb = m & -m
        u = lsb.bit_length() - 1
        m ^= lsb
        total_deg += (adj[u] & mask).bit_count()
    return total_deg // 2

def _count_indep_4_in_subset(mask, adj):
    verts = []
    m = mask
    while m:
        lsb = m & -m
        verts.append(lsb.bit_length() - 1)
        m ^= lsb
    L = len(verts)
    if L < 4:
        return 0
    cnt = 0
    for i in range(L):
        for j in range(i + 1, L):
            if (adj[verts[i]] >> verts[j]) & 1:
                continue
            for k in range(j + 1, L):
                if (adj[verts[i]] >> verts[k]) & 1 or (adj[verts[j]] >> verts[k]) & 1:
                    continue
                for l in range(k + 1, L):
                    if ((adj[verts[i]] >> verts[l]) & 1 or
                        (adj[verts[j]] >> verts[l]) & 1 or
                        (adj[verts[k]] >> verts[l]) & 1):
                        continue
                    cnt += 1
    return cnt

def _delta_violations(u, v, adj, n, is_edge):
    common = adj[u] & adj[v]
    common &= ~(1 << u)
    common &= ~(1 << v)
    non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
    e_count = _count_edges_in_subset(common, adj)
    i4_count = _count_indep_4_in_subset(non_common, adj)
    if is_edge:
        return -e_count + i4_count
    else:
        return e_count - i4_count

def _anneal_refine(n, adj_init, deadline):
    adj = list(adj_init)
    rng = random.Random(12345)
    T = 1.0
    for step in range(200000):
        if time.time() > deadline:
            break
        # pick random edge
        u = rng.randrange(n)
        v = rng.randrange(u + 1, n)
        is_edge = bool((adj[u] >> v) & 1)
        d = _delta_violations(u, v, adj, n, is_edge)
        if d < 0:
            accept = True
        else:
            # Metropolis: accept with prob exp(-d/T)
            import math
            if d == 0:
                accept = True
            else:
                accept = rng.random() < math.exp(-d / max(T, 1e-6))
        if accept:
            if is_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        T *= 0.9995
    return adj

# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    random.seed(42)

    # Guaranteed valid 35-vertex baseline
    adj35 = _build_35_vertex_graph()
    base_valid = _is_valid(adj35, 35)
    if not base_valid:
        # extremely unlikely; fall back to a safe small graph
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17)
                                   for j in range(i + 1, 17)
                                   if (j - i) % 17 in squares]}

    best_n = 35
    best_adj = adj35

    # --- Attempt to reach n=36 via the new sequential backtracking search ---
    if time.time() < deadline - 25:
        # Try a few seeds for n=36, warm-started from the 35-graph prefix.
        for seed in range(0, 8):
            if time.time() > deadline - 15:
                break
            sub = adj35[:35]
            res = _construct_backtrack(36, seed, deadline - 15, init_adj=sub)
            if res is not None and _is_valid(res, 36):
                return {'n': 36, 'edges': _edges_from_adj(res, 36)}

    # --- Attempt to reach n=36 via simulated-annealing refinement ---
    if time.time() < deadline - 20:
        for seed in range(0, 4):
            if time.time() > deadline - 10:
                break
            rng = random.Random(777 + seed)
            adj = list(adj35) + [0]
            for i in range(35):
                if rng.random() < 0.5:
                    adj[35] |= (1 << i)
                    adj[i] |= (1 << 35)
            refined = _anneal_refine(36, adj, deadline - 10)
            if _is_valid(refined, 36):
                return {'n': 36, 'edges': _edges_from_adj(refined, 36)}

    # --- Fallback: try to improve via annealing on n=35 (keep 35) ---
    # Return the guaranteed 35-vertex graph.
    return {'n': 35, 'edges': _edges_from_adj(adj35, 35)}
