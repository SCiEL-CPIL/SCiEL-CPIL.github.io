import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def check_c4_safe(adj, u, v):
        """Check if adding edge (u,v) creates a C4 using bitmasks."""
        mask_u = adj[u]
        mask_v = adj[v]
        m = mask_u
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            m ^= lsb
            if adj[a] & mask_v:
                return False
        return True

    def verify_graph(adj, n, min_deg_req):
        """Verify C4-free and min degree."""
        if min(d for d in [bin(adj[i]).count('1') for i in range(n)]) < min_deg_req:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    if common & (common - 1):  # more than 1 common neighbor
                        return False
        return True

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    def build_pg2_5():
        """Build PG(2,5): 31 points, 31 lines, incidence structure.
        Points and lines are 1-dim subspaces of GF(5)^3.
        Two points determine a unique line (C4-free guarantee in incidence graph).
        """
        q = 5
        # Generate all 1-dim subspaces (points = lines in PG(2,q))
        reps = []
        seen = set()
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    # Find canonical representative: first nonzero coord = 1
                    if a != 0:
                        key = (1, b * pow(a, q-2, q), c * pow(a, q-2, q))
                    elif b != 0:
                        key = (0, 1, c * pow(b, q-2, q))
                    else:
                        key = (0, 0, 1)
                    if key not in seen:
                        seen.add(key)
                        reps.append(key)
        
        # reps has 31 elements (q^2+q+1 = 25+5+1 = 31)
        # Point i is reps[i], Line j is reps[j] (same set)
        # Incidence: point i is on line j if reps[i] is in span of reps[j]
        # Actually in PG(2,q), points and lines are dual. 
        # A point p is on a line l if p is in the 2-dim subspace corresponding to l.
        # Line l (1-dim subspace with rep r) corresponds to the set of points {x : x dot r_perp = 0}
        # More simply: point with rep p is on line with rep l if p is a scalar multiple... no.
        # In PG(2,q), a line is a 2-dim subspace of GF(q)^3. A point (1-dim) is on it if contained.
        # The line corresponding to rep l = (a,b,c) consists of points p = (x,y,z) with ax+by+cz = 0.
        
        n_pts = len(reps)  # 31
        # incidence[i][j] = True if point i is on line j
        incidence = [[False] * n_pts for _ in range(n_pts)]
        for i, p in enumerate(reps):
            for j, l in enumerate(reps):
                # p is on line l if p dot l = 0 in GF(5)
                val = (p[0] * l[0] + p[1] * l[1] + p[2] * l[2]) % q
                if val == 0:
                    incidence[i][j] = True
        
        return reps, incidence

    # Build PG(2,5) structure
    reps, incidence = build_pg2_5()
    n_pts = len(reps)  # 31
    
    # Bipartite incidence graph: 62 vertices (31 points + 31 lines)
    # Vertex i (0..30) = point i, vertex 31+i (31..61) = line i
    # Edge between point i and line j if incidence[i][j]
    # This graph is C4-free because any two points share at most one line (PG property)
    # and any two lines share at most one point.
    
    # For n=46: select 46 out of 62 vertices
    # min_deg_req = 46 - 39 = 7
    # Each point is on 6 lines (q+1 = 6), each line has 6 points.
    # So in the full bipartite graph, every vertex has degree 6.
    # We need degree >= 7, so we can't just use the incidence graph directly.
    # 
    # Wait: in PG(2,5), each point lies on q+1=6 lines, each line contains q+1=6 points.
    # So the bipartite incidence graph has all vertices of degree 6.
    # For n=46, we need min degree >= 7. This won't work directly.
    #
    # Let me reconsider: maybe we can add some extra edges or use a different construction.
    # Actually, let me just try the greedy approach but with better initialization.
    
    # Try from high n down using greedy with bitmask
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break

        min_deg_req = n - 39
        if min_deg_req <= 0:
            return {'n': n, 'edges': []}

        found = False
        best_edges = []

        # Multiple strategies with different random seeds
        for strategy in range(80):
            if time.time() > deadline:
                break

            random.seed(strategy * 7919 + n)  # Deterministic per strategy

            adj = [0] * n
            deg = [0] * n

            # Strategy: demand-driven greedy with random tie-breaking
            stuck = False
            max_steps = 300  # Max edges to add
            steps = 0

            while steps < max_steps:
                # Find vertices below min degree
                low_verts = [i for i in range(n) if deg[i] < min_deg_req]
                if not low_verts:
                    break

                # Pick a low-degree vertex (random among tied)
                u = random.choice(low_verts)

                # Find safe neighbors
                safe_vs = []
                for v in range(n):
                    if v == u or (adj[u] >> v) & 1:
                        continue
                    if check_c4_safe(adj, u, v):
                        safe_vs.append(v)

                if not safe_vs:
                    # Try repair: remove a random edge incident to u to free up options
                    if deg[u] > 0:
                        m = adj[u]
                        neighbors = []
                        while m:
                            lsb = m & (-m)
                            v = lsb.bit_length() - 1
                            m ^= lsb
                            neighbors.append(v)
                        if neighbors:
                            v_remove = random.choice(neighbors)
                            adj[u] &= ~(1 << v_remove)
                            adj[v_remove] &= ~(1 << u)
                            deg[u] -= 1
                            deg[v_remove] -= 1
                            steps += 1
                            continue
                    stuck = True
                    break

                # Pick v: prefer low degree to balance
                if random.random() < 0.7:
                    v = min(safe_vs, key=lambda x: deg[x])
                else:
                    v = random.choice(safe_vs)

                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                steps += 1

            if not stuck and verify_graph(adj, n, min_deg_req):
                found = True
                best_edges = extract_edges(adj, n)
                break

        if found:
            return {'n': n, 'edges': best_edges}

    # Fallback: return a valid small graph
    return {'n': 39, 'edges': []}
