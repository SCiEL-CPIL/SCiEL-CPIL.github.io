import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (non-backtracking BFS from every vertex) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        # verify degree
        for i in range(n):
            if len(adj[i]) != k:
                return -2
        # check for loops / multi-edges
        seen = set()
        for i, j in edges:
            if i == j:
                return -3
            key = (min(i, j), max(i, j))
            if key in seen:
                return -4
            seen.add(key)
        # non-backtracking BFS from each vertex to find shortest cycle through it
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            # state: (node, prev) meaning we are at node having come from prev
            # distance: number of steps taken
            best = {}
            queue = []
            for nb in adj[s]:
                queue.append((nb, s, 1))
            while queue:
                node, prev, dist = queue.pop()
                if node == s:
                    if dist < min_cycle:
                        min_cycle = dist
                    if min_cycle < g:
                        return min_cycle
                    continue
                state = (node, prev)
                if state in best and best[state] <= dist:
                    continue
                best[state] = dist
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    queue.append((nb, node, dist + 1))
        return min_cycle

    # ---------------- voltage lift of a base graph ----------------
    def voltage_lift(base_adj, base_n, m, seed):
        """Lift base graph by factor m using Z_m voltages. Returns (n, edges) or None."""
        rng = random.Random(seed)
        # assign random voltage to each directed edge (i->j)
        # For undirected edge {i,j}, assign voltage v_ij for i->j and -v_ij for j->i
        volt = {}
        for i in range(base_n):
            for j in base_adj[i]:
                if i < j:
                    volt[(i, j)] = rng.randrange(m)
                    volt[(j, i)] = (-volt[(i, j)]) % m
        # Build lifted graph: vertices are (base_vertex, fiber)
        lifted_n = base_n * m
        edges = []
        for i in range(base_n):
            for j in base_adj[i]:
                if i < j:
                    v = volt[(i, j)]
                    for f in range(m):
                        a = i * m + f
                        b = j * m + ((f + v) % m)
                        edges.append((a, b))
        return lifted_n, edges

    # ---------------- Tutte-Coxeter graph (30 vertices, girth 8, cubic) ----------------
    def build_tutte_coxeter():
        # The Tutte-Coxeter graph is the point-line incidence graph of the
        # generalized quadrangle GQ(2,2) (the "doily").
        # Points: 15, Lines: 15, total 30 vertices.
        # Each point on 3 lines, each line through 3 points.
        # We build it as the bipartite graph of the GQ(2,2).
        #
        # GQ(2,2) can be constructed as follows:
        # Points are (x, y, z) in {0,1,2}^3 with x+y+z = 0 mod 3, plus one extra? 
        # Actually GQ(2,2) has 15 points and 15 lines.
        # A standard construction: points are pairs (a, b) where a in {0,1,2,3,4} and b in {0,1,2}
        # with some constraints... Let me use a known adjacency list.
        #
        # The Tutte-Coxeter graph is the 3-regular bipartite graph on 30 vertices
        # with girth 8. Here is a standard representation:
        # Vertices 0-14 (points), 15-29 (lines).
        # The incidence is given by the 15 lines of the GQ(2,2).
        #
        # I'll use the construction from the "doily" / Clebsch graph relation.
        # The Clebsch graph has 16 vertices. The Tutte-Coxeter is its bipartite double cover? No.
        #
        # Let me just hardcode the adjacency of the Tutte-Coxeter graph.
        # It is the unique cubic graph on 30 vertices with girth 8.
        # One standard labeling:
        # Points: 0..14, Lines: 15..29
        # Each point i is incident to 3 lines.
        # The 15 lines each contain 3 points.
        #
        # Here's a known edge list (from OEIS / House of Graphs):
        adj = [[] for _ in range(30)]
        # I'll construct GQ(2,2) properly.
        # GQ(2,2): points are elements of a 3-dimensional vector space over GF(3) 
        # satisfying a quadratic form? No, that gives different numbers.
        #
        # Standard construction of GQ(2,2):
        # Points: (x, y, z) in Z_3^3 with x + y + z = 0 (mod 3). That's 9 points. Not 15.
        #
        # Alternative: GQ(2,2) is the unique generalized quadrangle of order (2,2).
        # It has (s+1)(st+1) = 3*5 = 15 points and 15 lines.
        #
        # Construction via the "grid" model:
        # Take a 3x3 grid of points. Add 6 more points and lines...
        #
        # Let me use a different approach: I'll just hardcode the Tutte-Coxeter graph
        # from a known source. The graph is 3-regular, bipartite, 30 vertices, girth 8.
        #
        # From the House of Graphs / standard references, one adjacency list:
        # Let me use the construction as the Cayley graph of Z_3 x Z_5 with specific generators.
        # Actually, the Tutte-Coxeter graph is NOT a Cayley graph (it's bipartite but not vertex-transitive in the Cayley sense... actually it IS vertex-transitive).
        #
        # Let me just use a known edge list. I'll generate it via the GQ(2,2) construction.
        #
        # GQ(2,2) construction:
        # Points: all (a, b) where a in {0,1,2,3,4}, b in {0,1,2} with a+b even? No.
        #
        # OK, I'll use the following known construction:
        # The 15 points are indexed 0..14.
        # The 15 lines are indexed 0..14.
        # Point i is on lines L(i,0), L(i,1), L(i,2) where:
        #
        # I'll use the "doily" construction:
        # The doily is the 3x3 grid with 6 extra points/lines.
        #
        # Let me just hardcode a valid Tutte-Coxeter graph.
        # From https://www.graphclasses.org/smallgraphs/Graph30_3_8.html
        # The graph has 30 vertices, 45 edges, girth 8.
        #
        # I'll construct it as follows (bipartite, parts A={0..14}, B={15..29}):
        # Line j contains points p(j,0), p(j,1), p(j,2).
        # Point i is on lines l(i,0), l(i,1), l(i,2).
        #
        # Using the standard GQ(2,2) model:
        # Points: (x, y) in Z_5 x Z_3 with x + y^2 = 0 mod 5? No, that gives 5*3=15 but not all valid.
        #
        # I'll use a concrete known edge list for the Tutte-Coxeter graph:
        edges = [
            (0,15),(0,16),(0,17),
            (1,15),(1,18),(1,19),
            (2,16),(2,18),(2,20),
            (3,17),(3,19),(3,20),
            (4,21),(4,22),(4,23),
            (5,21),(5,24),(5,25),
            (6,22),(6,24),(6,26),
            (7,23),(7,25),(7,26),
            (8,27),(8,28),(8,29),
            (9,27),(9,30),(9,31),  # oops, 30 vertices max index 29
        ]
        # This is getting messy. Let me use a cleaner construction.
        #
        # The Tutte-Coxeter graph can be built as the incidence graph of the
        # projective line over GF(9) with a specific structure... too complex.
        #
        # Let me use a different small cage: the Heawood graph (14 vertices, girth 6).
        # It's the incidence graph of the Fano plane, which is easy to construct.
        return None  # I'll use Heawood instead

    def build_heawood():
        """Heawood graph: incidence graph of Fano plane. 14 vertices, girth 6."""
        # Fano plane: 7 points, 7 lines.
        # Points: 0,1,2,3,4,5,6
        # Lines: {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
        lines = [
            [0,1,2],
            [0,3,4],
            [0,5,6],
            [1,3,5],
            [1,4,6],
            [2,3,6],
            [2,4,5]
        ]
        n = 14
        adj = [[] for _ in range(n)]
        for j, line in enumerate(lines):
            line_vertex = 7 + j
            for p in line:
                adj[p].append(line_vertex)
                adj[line_vertex].append(p)
        # Build edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, adj, edges

    # ---------------- cycle packing construction ----------------
    def cycle_packing(n, k, g, seed):
        """Build a k-regular graph as union of k/2 edge-disjoint Hamiltonian cycles (for even k)
        or (k-1)/2 cycles + 1 perfect matching (for odd k).
        For k=3: one Hamiltonian cycle + one perfect matching.
        This gives girth at least 4 (from matching) but can be higher if cycles are long.
        """
        if k != 3:
            return None
        rng = random.Random(seed)
        # One Hamiltonian cycle: 0-1-2-...-(n-1)-0
        # One perfect matching: pair (0,1), (2,3), ... but that creates 4-cycles with the Hamiltonian cycle.
        # Instead, use a "long" perfect matching: pair i with i + n//2 (if n even).
        # For n even: matching is (i, i+n//2) for i in 0..n//2-1.
        # The girth of this graph: the Hamiltonian cycle has length n, the matching chords
        # create cycles of length n//2 + 2 (going along the cycle from i to i+n//2, then chord back).
        # So girth = min(n, n//2 + 2) = n//2 + 2 (for n >= 4).
        # For girth 16: need n//2 + 2 >= 16, so n >= 28.
        # But we also need no shorter cycles. The only cycles are:
        # - The Hamiltonian cycle: length n
        # - Cycles using one matching edge: length n//2 + 2
        # - Cycles using multiple matching edges: longer
        # So girth = n//2 + 2.
        # For girth >= 16: n >= 28.
        # For girth >= 16 with n even: smallest n is 28.
        # Let's verify: n=28, matching (i, i+14) for i=0..13.
        # Cycle using matching edge (0,14): 0-1-2-...-14-0, length 15. That's < 16!
        # Wait, the cycle is 0->1->2->...->14->0, which is 15 edges. So girth = 15, not 16.
        # We need n//2 + 2 >= 16 AND the actual shortest cycle >= 16.
        # The cycle using matching edge (i, i+n//2) has length n//2 + 1 (going i->i+1->...->i+n//2->i).
        # Wait: from i to i+n//2 along the cycle is n//2 steps, then 1 step back via matching. Total n//2 + 1.
        # For n=28: 14+1 = 15. Still < 16.
        # For n=30: 15+1 = 16. 
        # For n=32: 16+1 = 17.
        # So smallest n with girth >= 16 is n=30.
        # But wait, are there other shorter cycles? 
        # Using two matching edges: (i, i+15) and (j, j+15). 
        # Cycle: i -> ... -> j -> j+15 -> ... -> i+15 -> i.
        # Length = (j-i) + 1 + (i+15 - j - 15) + 1 = (j-i) + 1 + (i-j) + 1 = 2. That's not right.
        # Let me think more carefully.
        # Vertices on cycle: 0,1,...,29 (n=30).
        # Matching: (i, i+15) for i=0..14.
        # Cycle using matching edge (0,15): 0->1->2->...->15->0. Length 16. Good.
        # Cycle using matching edges (0,15) and (1,16): 0->1->16->17->...->15->0? No.
        # 0->1 (cycle), 1->16 (matching), 16->17->...->15 (cycle, going backwards? No, cycle is directed 0->1->...->29->0).
        # From 16, go 16->17->...->29->0? That's 14 steps. Then 0->15 (matching). Total: 1+1+14+1 = 17.
        # Or 16->15 (backwards along cycle)? No, the cycle is undirected, so we can go either way.
        # Shortest path from 16 to 15 along the cycle: |16-15| = 1. So cycle: 0->1->16->15->0, length 4!
        # That's a 4-cycle: 0-1-16-15-0. Edges: (0,1) cycle, (1,16) matching, (16,15) cycle, (15,0) matching.
        # So the girth is 4, not 16!
        #
        # So the simple "half-turn" matching doesn't work. We need a different matching.
        #
        # For a 3-regular graph from a Hamiltonian cycle + perfect matching to have girth >= 16,
        # we need the matching to be "long" in the sense that no two matching edges are close
        # on the cycle. Specifically, if matching edge (i, j) exists, then for any other matching
        # edge (i', j'), the cycle distance between {i,j} and {i',j'} should be large.
        #
        # This is equivalent to finding a perfect matching on the cycle graph C_n such that
        # the girth of C_n + matching is >= 16.
        #
        # For n=30, we need a perfect matching where the closest pair of matching edges
        # (measured by cycle distance) is at least 7 apart (so that the 4-cycle would have
        # length >= 16).
        #
        # Actually, a 4-cycle in C_n + M exists iff there are two matching edges (a,b) and (c,d)
        # such that a and c are adjacent on the cycle and b and d are adjacent on the cycle (or similar).
        # More generally, a cycle of length 2m exists iff there are m matching edges that form
        # a "ladder" on the cycle.
        #
        # For girth >= 16, we need no cycles of length 4, 6, 8, 10, 12, 14.
        #
        # This is a constraint satisfaction problem. Let me try random matchings and check.
        if n % 2 != 0:
            return None
        # Try random perfect matchings
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
            # Create a random perfect matching
            perm = list(range(n))
            rng.shuffle(perm)
            matching = []
            for i in range(n // 2):
                matching.append((perm[2*i], perm[2*i+1]))
            # Build graph
            edges = []
            for i in range(n):
                edges.append((i, (i+1) % n))
            for a, b in matching:
                edges.append((a, b))
            # Check girth
            gc = check_girth(n, edges, g, deadline - 5)
            if gc == -1:
                continue
            if gc >= g:
                return n, edges
        return None

    # ---------------- local repair: start from random cubic graph, flip edges to increase girth ----------------
    def random_cubic_graph(n, seed):
        """Generate a random 3-regular graph on n vertices (n must be even)."""
        if n % 2 != 0:
            return None
        rng = random.Random(seed)
        # Configuration model: 3n half-edges, pair them randomly
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        rng.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        edges = set()
        for i in range(0, 3*n, 2):
            a, b = half_edges[i], half_edges[i+1]
            if a == b:
                return None  # loop, retry
            if (a, b) in edges or (b, a) in edges:
                return None  # multi-edge, retry
            edges.add((a, b))
            edges.add((b, a))
            adj[a].append(b)
            adj[b].append(a)
        return n, adj, list(edges)

    def local_repair(n, adj, edges, g, seed, max_iters=5000):
        """Try to increase girth by 2-switches (edge flips)."""
        rng = random.Random(seed)
        current_edges = list(edges)
        current_girth = check_girth(n, current_edges, g, deadline - 5)
        if current_girth >= g:
            return n, current_edges
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            # Find a short cycle
            # Do a BFS from a random vertex to find a cycle of length < g
            s = rng.randrange(n)
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            short_cycle = None
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < g:
                            # Found a short cycle; try to break it
                            # Reconstruct the cycle
                            cycle = [node]
                            cur = node
                            while cur != nb:
                                cur = parent[cur]
                                cycle.append(cur)
                            cycle.append(nb)
                            short_cycle = cycle
                            break
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                if short_cycle:
                    break
            if short_cycle is None:
                # No short cycle found from this vertex; try another
                continue
            # Try to break the cycle by flipping an edge
            # For each edge in the cycle, try removing it and adding a random non-edge
            cycle_edges = []
            for i in range(len(short_cycle)):
                a = short_cycle[i]
                b = short_cycle[(i+1) % len(short_cycle)]
                cycle_edges.append((a, b))
            # Pick a random edge from the cycle to remove
            a, b = rng.choice(cycle_edges)
            # Find a random non-neighbor of a to connect to
            neighbors_a = set(adj[a])
            candidates = [v for v in range(n) if v not in neighbors_a and v != a]
            if not candidates:
                continue
            c = rng.choice(candidates)
            if c in adj[b] or b in adj[c]:
                # Adding (a,c) would create a multi-edge or loop; skip
                continue
            # Perform the flip: remove (a,b), add (a,c)
            # But we need to maintain 3-regularity, so also remove (b,?) and add (b,?)
            # Actually, a 2-switch: remove edges (a,b) and (c,d), add (a,c) and (b,d)
            # Let's pick d as a neighbor of c
            neighbors_c = set(adj[c])
            d_candidates = [v for v in neighbors_c if v != a and v != b]
            if not d_candidates:
                continue
            d = rng.choice(d_candidates)
            # Check that (a,d) and (b,c) are not edges
            if d in adj[a] or c in adj[b]:
                continue
            # Perform the switch
            # Remove (a,b) and (c,d)
            adj[a].remove(b)
            adj[b].remove(a)
            adj[c].remove(d)
            adj[d].remove(c)
            # Add (a,c) and (b,d)
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            # Rebuild edge list
            current_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        current_edges.append((i, j))
            # Check girth
            gc = check_girth(n, current_edges, g, deadline - 5)
            if gc >= g:
                return n, current_edges
        return None

    # ---------------- main ----------------
    best = None
    best_n = float('inf')

    if k == 3:
        # Method 1: Voltage lift of Heawood graph
        heawood = build_heawood()
        if heawood is not None:
            hn, hadj, hedges = heawood
            # Lift by factor m: n = 14*m
            # Girth of lift: if base girth is 6, lift girth is at least 6 (could be more)
            # For girth 16, need m large enough. Try m=4,6,8,10,...
            for m in [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40]:
                if remaining() < 15:
                    break
                for seed in range(20):
                    if remaining() < 10:
                        break
                    res = voltage_lift(hadj, hn, m, seed)
                    if res is None:
                        continue
                    n, edges = res
                    if n >= best_n:
                        continue
                    gc = check_girth(n, edges, g, deadline - 5)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best, best_n = (n, edges), n
                        break

        # Method 2: Cycle packing
        for n_try in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]:
            if remaining() < 15:
                break
            if n_try >= best_n:
                continue
            for seed in range(50):
                if remaining() < 10:
                    break
                res = cycle_packing(n_try, 3, g, seed)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 5)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n
                    break

        # Method 3: Random cubic graph + local repair
        for n_try in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]:
            if remaining() < 20:
                break
            if n_try >= best_n:
                continue
            for seed in range(10):
                if remaining() < 15:
                    break
                res = random_cubic_graph(n_try, seed)
                if res is None:
                    continue
                n, adj, edges = res
                gc = check_girth(n, edges, g, deadline - 5)
                if gc >= g:
                    best, best_n = (n, edges), n
                    break
                # Try local repair
                repaired = local_repair(n, adj, edges, g, seed + 1000, max_iters=2000)
                if repaired is not None:
                    n2, edges2 = repaired
                    if n2 < best_n:
                        best, best_n = (n2, edges2), n2
                    break

    if best is not None:
        return best

    # Fallback: use the LUW construction from the original (simplified)
    # For k=3, g=16, use a large n to guarantee girth
    # The LUW construction with d = (g-2)//2 = 7 gives a graph with girth >= g
    # n = q * 2^d = 3 * 128 = 384 (for q=3)
    # But this is large. Let's just use a cycle packing with a large n.
    n = 2 * g  # n=32, girth = n//2 + 1 = 17 >= 16
    edges = []
    for i in range(n):
        edges.append((i, (i+1) % n))
    for i in range(n // 2):
        edges.append((i, i + n//2))
    # This has girth 4 as we saw. So this won't work.
    # Use a different fallback: just return a large cycle packing with random matching
    for n_try in [60, 70, 80, 90, 100]:
        for seed in range(100):
            res = cycle_packing(n_try, 3, g, seed)
            if res is not None:
                return res
    # Last resort: return a trivial valid graph (large n)
    # Use the voltage lift with a large m
    heawood = build_heawood()
    if heawood is not None:
        hn, hadj, hedges = heawood
        for m in [40, 50, 60, 80, 100]:
            res = voltage_lift(hadj, hn, m, 42)
            if res is not None:
                n, edges = res
                gc = check_girth(n, edges, g, deadline - 2)
                if gc >= g:
                    return n, edges
    return (0, [])

