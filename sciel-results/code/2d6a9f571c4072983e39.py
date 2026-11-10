import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def extract_triangles(n, adj, deadline):
        """
        Greedily extract edge-disjoint triangles (3-cliques).
        Priority: triangles whose edges have fewer alternative coverings.
        Returns list of triangles (each is a 3-vertex list) and updated adj.
        """
        rem_adj = adj[:]
        triangles = []
        
        while time.time() < deadline:
            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find best triangle to extract
            # Score each potential triangle by how "critical" its edges are
            best_score = -1
            best_tri = None
            
            for u in range(n):
                if not rem_adj[u]:
                    continue
                # For each neighbor v of u
                v_mask = rem_adj[u]
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if v <= u:
                        continue
                    # For each common neighbor w of u and v
                    common = rem_adj[u] & rem_adj[v]
                    w_mask = common
                    while w_mask:
                        w_bit = w_mask & (-w_mask)
                        w = w_bit.bit_length() - 1
                        w_mask ^= w_bit
                        if w <= v:
                            continue
                        # Triangle (u, v, w) is valid
                        # Score: sum of (n - common_neighbors) for each edge
                        # Edges with fewer common neighbors are more critical
                        cu = rem_adj[u] & rem_adj[v]
                        cv = rem_adj[v] & rem_adj[w]
                        cw = rem_adj[u] & rem_adj[w]
                        score = (n - popcount(cu)) + (n - popcount(cv)) + (n - popcount(cw))
                        # Also add a small bonus for larger triangles being available
                        if score > best_score:
                            best_score = score
                            best_tri = (u, v, w)
            
            if best_tri is None:
                break
            
            u, v, w = best_tri
            # Remove edges of this triangle
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            rem_adj[u] &= ~(1 << w)
            rem_adj[w] &= ~(1 << u)
            rem_adj[v] &= ~(1 << w)
            rem_adj[w] &= ~(1 << v)
            
            triangles.append([u, v, w])
        
        return triangles, rem_adj

    def max_clique_in_induced(n, adj, mask, rng):
        """
        Find max clique in G[mask].
        adj is the global adjacency list (remaining edges).
        mask is a bitmask of vertices to consider.
        """
        best_clique = []

        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return

            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return

            # Pivot selection: find vertex in P with max degree in P
            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)

            if not candidates:
                return

            u = rng.choice(candidates)
            Q = P & ~adj[u]

            while Q:
                # Pick v from Q with max degree in P
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)

                if not v_candidates:
                    break

                v = rng.choice(v_candidates)

                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)

                P &= ~(1 << v)
                Q = Q & P

        expand([], mask)
        return best_clique

    def greedy_partition_remaining(n, rem_adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MAX common neighbors,
        extract max clique containing it, remove those edges.
        """
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MAX common neighbors in remaining graph
            max_common = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u, best_v = u, v

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def greedy_partition_min_remaining(n, rem_adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MIN common neighbors,
        extract max clique containing it, remove those edges.
        """
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find edge (u,v) with MIN common neighbors in remaining graph
            min_common = n + 1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Iterate neighbors v > u
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt < min_common:
                        min_common = cnt
                        best_u, best_v = u, v
                        if min_common == 0:
                            break
                if min_common == 0:
                    break

            if best_u == -1:
                break

            # Find max clique containing best_u, best_v
            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove edges of this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)
            
            # Phase 1: Extract triangles
            tri_deadline = min(DEADLINE, start + time_limit * 0.3)
            triangles, rem_adj = extract_triangles(n, adj, tri_deadline)
            
            if time.time() > DEADLINE:
                break
            
            # Phase 2: Extract larger cliques from remaining edges
            # Alternate between max and min heuristics
            if attempt % 2 == 0:
                result = greedy_partition_remaining(n, rem_adj, rng, DEADLINE)
            else:
                result = greedy_partition_min_remaining(n, rem_adj, rng, DEADLINE)
            
            if result is not None:
                # Combine triangles and remaining cliques
                all_cliques = triangles + result
                # Filter: only keep cliques with at least 2 vertices
                all_cliques = [c for c in all_cliques if len(c) >= 2]
                count = len(all_cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = all_cliques
            attempt += 1

        return best_cliques

    # Time budget allocation - give more time to the larger graph
    t1 = 10.0
    t2 = 95.0

    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]
