"""Fresh design: maximum independent set via bitset branch-and-bound."""
import time

_N = 256

def _conflicts():
    """Build conflict adjacency as list of Python ints (bitsets) indexed 0..255 for values 1..256."""
    adj = [0] * (_N + 1)
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            g = _gcd(a, b)
            # conflict iff (a+b)/g divides g
            q = (a + b) // g
            if g % q == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    return adj

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _popcount(x):
    return bin(x).count('1')

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s
    adj = _conflicts()

    # Candidate pool: all 256 numbers.
    # Greedy lower bound via eigenvector-guided / low-conflict greedy.
    def greedy_seed():
        # Start with odds (always independent), then add evens with low conflict.
        chosen = set(range(1, _N + 1, 2))
        chosen_bits = 0
        for v in chosen:
            chosen_bits |= (1 << v)
        # Try to add evens that don't conflict with chosen.
        for e in range(2, _N + 1, 2):
            # check conflicts with all chosen
            if (adj[e] & chosen_bits) == 0:
                chosen.add(e)
                chosen_bits |= (1 << e)
        return chosen, chosen_bits

    best_set, best_bits = greedy_seed()
    best_size = len(best_set)

    # Branch and bound for maximum independent set.
    # We work on the candidate universe = all 256 vertices.
    # Upper bound: size of a maximal independent set via greedy coloring on the
    # subgraph induced by remaining candidates (the classic Tomita bound).

    import sys
    sys.setrecursionlimit(100000)

    # Precompute vertex order heuristic: process vertices in an order.
    # Use a simple ordering by degree (ascending) for the greedy coloring bound.
    degrees = [bin(adj[i]).count('1') for i in range(_N + 1)]

    # We'll do branch and bound over a candidate bitset.
    # State: candidates bitset C (vertices still eligible), current set S (chosen).
    # We want |S| + alpha(G[C]) > best to explore.
    # alpha(G[C]) upper bounded by greedy coloring of G[C].

    # Greedy coloring upper bound on independent set size of induced subgraph on C.
    # (Coloring the graph gives upper bound on independent set = max color class,
    #  but standard bound: alpha <= number of colors in a greedy coloring? No.)
    # Actually: a proper k-coloring of G implies alpha(G) <= k? No, alpha <= k only
    # if... wait. In a proper coloring, each color class is an independent set, so
    # alpha(G) >= max color class size. That's a lower bound. For an UPPER bound on
    # alpha we need: alpha(G) <= sum over... The standard Tomita bound: alpha <=
    # the number of colors used by a greedy coloring of the graph. Because each
    # color class is independent, so the union of all color classes... no.
    # Correct: In a proper coloring with k colors, the graph's vertices are partitioned
    # into k independent sets, so alpha(G) (max independent set) can be at most the
    # total number of vertices, but the bound used in Tomita's algorithm is:
    # alpha(G) <= number of colors. This is because... Actually the bound is that
    # in any coloring, alpha(G) <= number of colors is FALSE in general.
    # The correct Tomita bound: color the graph greedily; alpha(G) <= the number
    # of colors used. Reason: each color class is independent, so the largest
    # independent set has size at most the number of colors? No that's not right either.
    # Let me recall: The bound is alpha(G) <= chi(G) (chromatic number) is FALSE.
    # Actually alpha(G) * chi(G) >= n. The Tomita bound uses: a greedy coloring gives
    # an upper bound on the size of a maximal independent set via the fact that in
    # the coloring, you can pick at most one vertex per color from any independent set?
    # NO - an independent set can contain multiple vertices of the same color? No!
    # In a proper coloring, two vertices of the same color are NOT adjacent, so an
    # independent set CAN contain multiple same-color vertices. So that bound is wrong.
    #
    # The ACTUAL Tomita bound: color the graph; the number of colors is an upper
    # bound on the size of a CLIQUE in the complement... I'm confusing MIS and MCS.
    #
    # For maximum independent set in G = maximum clique in complement of G.
    # Tomita's MCS algorithm: bound = number of colors in greedy coloring of the
    # subgraph. For a clique, each vertex must have a distinct color, so clique size
    # <= number of colors. YES that's correct for cliques.
    #
    # So: MIS in G = MCS in complement(G). Build complement adjacency, use Tomita.
    # That's more work. Alternatively, use a simpler but valid upper bound:
    # For independent set in G: alpha(G) <= n - tau(G) where tau is min vertex cover.
    # Or use: alpha(G) <= n - (matching number)... 
    #
    # Simpler valid upper bound for MIS: greedy coloring of G gives lower bound on
    # chromatic number, and alpha(G) >= n / chi(G). Not an upper bound.
    #
    # Let me just use the complement-graph Tomita approach for a tight bound.
    # Build complement adjacency.

    comp = [0] * (_N + 1)
    full = 0
    for i in range(1, _N + 1):
        full |= (1 << i)
    for i in range(1, _N + 1):
        # complement within [1..256]
        comp[i] = full ^ adj[i] ^ (1 << i)

    # Tomita-style branch and bound for maximum clique in complement graph.
    # A clique in complement = independent set in G.
    # Greedy coloring of a subgraph (in complement) gives upper bound on clique size.

    def color_bound(cand):
        """Greedy color the subgraph induced by cand in complement; return number of colors
        (upper bound on clique size in complement = independent set size in G)."""
        if cand == 0:
            return 0
        colors = 0
        remaining = cand
        # Greedy coloring: repeatedly find a maximal independent set in the
        # complement-subgraph? No. Coloring the complement subgraph: adjacent in
        # complement must get different colors.
        # Greedy: while remaining: pick a color class = maximal independent set in
        # the complement subgraph on remaining. Independent in complement = clique in G.
        # That's expensive. Simpler greedy coloring: order vertices, assign smallest
        # color not used by already-colored neighbors in complement.
        # For an upper bound, any proper coloring works; greedy gives one.
        # Implement: iterate vertices in a fixed order, assign min available color.
        # Number of colors used = upper bound on clique size.
        # To make it tight, order by degree descending in complement.
        verts = []
        c = cand
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        # sort by complement-degree within cand descending
        verts.sort(key=lambda v: bin(comp[v] & cand).count('1'), reverse=True)
        color_of = {}
        next_color = 0
        for v in verts:
            # find smallest color not used by v's complement-neighbors already colored
            used = set()
            nb = comp[v] & cand
            # only consider already colored neighbors
            for u in verts:
                if u == v:
                    break
                if (nb >> u) & 1:
                    if u in color_of:
                        used.add(color_of[u])
            col = 0
            while col in used:
                col += 1
            color_of[v] = col
            if col + 1 > next_color:
                next_color = col + 1
        return next_color

    # This color_bound is O(n^2) per call which is too slow for deep B&B.
    # Given time budget, let's do a limited B&B with a cheaper bound:
    # upper bound = popcount(cand) (trivial) plus greedy independent set lower bound.
    # Actually a decent cheap upper bound: alpha(G[C]) <= |C| - max matching... 
    # Let's use: alpha <= |C| and prune only when |S| + |C| <= best.
    # That's weak. Better: use a fast greedy independent set on C as a LOWER bound
    # to update best, and use |C| as upper bound. Combined with good vertex ordering
    # and the fact that we already have a strong seed (158+), the search may still
    # be too large.
    #
    # Given the 85s budget and that a greedy already gives ~158, and exact MIS on a
    # 256-vertex graph is generally intractable, let me do a bounded B&B that runs
    # until deadline, keeping the best found.

    best = [best_size]
    best_set_final = [set(best_set)]

    def greedy_is(cand):
        """Greedy maximal independent set in G on candidate bitset; return (set, bits)."""
        s = set()
        bits = 0
        c = cand
        # order by degree ascending
        verts = []
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        verts.sort(key=lambda v: bin(adj[v] & cand).count('1'))
        for v in verts:
            if (adj[v] & bits) == 0:
                s.add(v)
                bits |= (1 << v)
        return s, bits

    # Update best with greedy on full set
    gs, gb = greedy_is(full)
    if len(gs) > best[0]:
        best[0] = len(gs)
        best_set_final[0] = set(gs)

    # B&B: branch on a vertex: include it (remove its neighbors) or exclude it.
    # Use the color bound if fast enough; else trivial bound.
    # Let's attempt with a timeout-checked recursion.
    import threading

    # To keep within time, use an iterative deepening / bounded search.
    # Given complexity, do a DFS with the trivial bound but strong ordering,
    # and a node limit tied to deadline.

    def dfs(cand, s_size, s_bits):
        if time.time() > DEADLINE:
            return
        if cand == 0:
            if s_size > best[0]:
                best[0] = s_size
                # reconstruct set from s_bits
                st = set()
                bb = s_bits
                while bb:
                    lsb = bb & (-bb)
                    st.add(lsb.bit_length() - 1)
                    bb ^= lsb
                best_set_final[0] = st
            return
        # trivial upper bound
        if s_size + bin(cand).count('1') <= best[0]:
            return
        # pick a vertex to branch on: highest degree in cand
        c = cand
        verts = []
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        # choose vertex with max degree in cand
        v = max(verts, key=lambda x: bin(adj[x] & cand).count('1'))
        # Branch 1: include v
        dfs(cand & ~adj[v] & ~(1 << v), s_size + 1, s_bits | (1 << v))
        # Branch 2: exclude v
        dfs(cand & ~(1 << v), s_size, s_bits)

    # The trivial-bound DFS on 256 vertices will explode. We need the color bound.
    # Let me implement a faster color bound using the complement and a simple
    # sequential coloring, and only call it at the root of each branch to prune.
    # Given time constraints, I'll run the DFS but with the color bound computed
    # via a cheaper method: alpha(G[C]) <= |C| / (1 + avg_degree_in_C / |C|) ... 
    # Not reliable.
    #
    # Practical decision: The greedy already gives a strong solution (~158, score
    # ~1.445 matching the known island best). Exact B&B likely won't finish in 85s
    # to prove optimality, but can improve the greedy if it explores promising
    # branches. Let me run a time-limited B&B with the color bound (complement
    # Tomita) but with a node cap, and fall back to best greedy.

    # Implement Tomita coloring bound on complement restricted to cand, but FAST:
    # Use the standard: color the complement subgraph; the number of colors is an
    # upper bound on the clique (independent set in G).
    # Fast greedy coloring of complement subgraph on cand:
    def fast_color_bound(cand):
        if cand == 0:
            return 0
        # order vertices by complement degree in cand descending
        c = cand
        verts = []
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        verts.sort(key=lambda v: bin(comp[v] & cand).count('1'), reverse=True)
        # greedy color
        col = {}
        ncolors = 0
        for v in verts:
            used = [False] * (len(verts) + 1)
            nb = comp[v] & cand
            for u in verts:
                if u == v:
                    break
                if (nb >> u) & 1 and u in col:
                    used[col[u]] = True
            k = 0
            while k < len(used) and used[k]:
                k += 1
            col[v] = k
            if k + 1 > ncolors:
                ncolors = k + 1
        return ncolors

    # This is O(n^2) ~ 65k ops per call; with node cap of ~5000 that's ~3e8, too slow.
    # Reduce: only compute color bound at nodes with depth < some level, else trivial.
    # Given the risk, let me just run the trivial-bound DFS with a strict node cap
    # and rely on the strong greedy seed. The B&B may find a slightly better set.

    node_count = [0]
    NODE_CAP = 200000

    def dfs2(cand, s_size, s_bits):
        if time.time() > DEADLINE:
            return
        node_count[0] += 1
        if node_count[0] > NODE_CAP:
            return
        if cand == 0:
            if s_size > best[0]:
                best[0] = s_size
                st = set()
                bb = s_bits
                while bb:
                    lsb = bb & (-bb)
                    st.add(lsb.bit_length() - 1)
                    bb ^= lsb
                best_set_final[0] = st
            return
        if s_size + bin(cand).count('1') <= best[0]:
            return
        c = cand
        verts = []
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            verts.append(v)
            c ^= lsb
        v = max(verts, key=lambda x: bin(adj[x] & cand).count('1'))
        # include v
        dfs2(cand & ~adj[v] & ~(1 << v), s_size + 1, s_bits | (1 << v))
        # exclude v
        dfs2(cand & ~(1 << v), s_size, s_bits)

    dfs2(full, 0, 0)

    result = sorted(best_set_final[0])
    return result
