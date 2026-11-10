import time
import random

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

    def max_clique_in_induced(n, adj, mask, rng, max_size=None):
        """
        Find max clique in G[mask].
        adj is the global adjacency list.
        mask is a bitmask of vertices to consider.
        max_size: if set, stop early if clique of this size found
        """
        best_clique = []
        found_large = [False]

        def expand(R, P):
            nonlocal best_clique
            if max_size and len(R) >= max_size:
                found_large[0] = True
                return
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
                if found_large[0]:
                    return

                P &= ~(1 << v)
                Q = Q & P

        expand([], mask)
        return best_clique

    def greedy_partition_fast(n, adj, rng, deadline, use_density_heuristic=True):
        """
        Fast greedy: repeatedly find edge, extract max clique containing it, remove those edges.
        Uses bitmask adjacency for O(1) edge removal.
        """
        rem_adj = adj[:]
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

            # Find best edge to expand
            best_u, best_v = -1, -1
            best_score = -1

            if use_density_heuristic:
                # Score edges by potential: common neighbors / (degree_u + degree_v - common)
                # This favors edges that can form larger cliques
                for u in range(n):
                    if not rem_adj[u]:
                        continue
                    deg_u = popcount(rem_adj[u])
                    v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                    while v_mask:
                        v_bit = v_mask & (-v_mask)
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit

                        common = rem_adj[u] & rem_adj[v]
                        cnt = popcount(common)
                        deg_v = popcount(rem_adj[v])
                        
                        # Score: prefer edges with many common neighbors relative to total degree
                        # Higher score = better candidate for clique expansion
                        score = cnt / (deg_u + deg_v - cnt + 1)
                        
                        if score > best_score:
                            best_score = score
                            best_u, best_v = u, v
            else:
                # Original: min common neighbors
                min_common = n + 1
                for u in range(n):
                    if not rem_adj[u]:
                        continue
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

    def local_improve(cliques, n, adj, deadline):
        """
        Try to merge adjacent cliques or move edges between cliques to reduce count.
        Returns improved clique list or None if no improvement.
        """
        if time.time() > deadline:
            return None

        improved = True
        current_cliques = [c[:] for c in cliques]
        best_count = len(current_cliques)

        while improved and time.time() < deadline:
            improved = False
            new_cliques = []

            for i in range(len(current_cliques)):
                if time.time() > deadline:
                    return None
                ci = current_cliques[i]

                # Try to extend this clique by adding a vertex that's adjacent to all in ci
                # and whose edges to ci are all in this clique (i.e., not in other cliques)
                # Actually, simpler: try to find a vertex v not in ci such that
                # v is adjacent to all vertices in ci, and all edges (v, u) for u in ci
                # are currently only covered by ci (not by other cliques)

                # Alternative: try to merge two cliques that form a larger clique
                # This is complex, so let's try a simpler local move:
                # For each clique, try removing a vertex and seeing if we can add it to another clique

                new_cliques.append(ci)

            # Try pairwise merging
            for i in range(len(current_cliques)):
                if time.time() > deadline:
                    return None
                ci = current_cliques[i]
                for j in range(i + 1, len(current_cliques)):
                    if time.time() > deadline:
                        return None
                    cj = current_cliques[j]

                    # Check if ci ∪ cj is a clique
                    merged = ci + cj
                    is_clique = True
                    for a in range(len(merged)):
                        for b in range(a + 1, len(merged)):
                            u, v = merged[a], merged[b]
                            if not (adj[u] & (1 << v)):
                                is_clique = False
                                break
                        if not is_clique:
                            break

                    if is_clique and len(merged) >= 2:
                        # Check that all edges in merged are covered by ci or cj
                        # Since they were separate cliques, edges within ci are in ci,
                        # edges within cj are in cj, but cross edges (u in ci, v in cj)
                        # must be in exactly one of them. Since they're separate cliques,
                        # cross edges are NOT in either, so merging would be invalid unless
                        # those cross edges don't exist or are handled elsewhere.
                        # Actually, if ci and cj are both cliques and merged is a clique,
                        # then all cross edges exist. But those cross edges must be covered
                        # by some clique. If they're not in ci or cj, then they're in other
                        # cliques, so we can't merge.
                        # So only merge if there are no cross edges between ci and cj.
                        has_cross_edge = False
                        for u in ci:
                            for v in cj:
                                if adj[u] & (1 << v):
                                    has_cross_edge = True
                                    break
                            if has_cross_edge:
                                break

                        if not has_cross_edge:
                            # No cross edges, so we can merge
                            new_cliques = [c for k, c in enumerate(current_cliques) if k != i and k != j]
                            new_cliques.append(merged)
                            if len(new_cliques) < best_count:
                                best_count = len(new_cliques)
                                current_cliques = new_cliques
                                improved = True
                            break
                if improved:
                    break

        return current_cliques

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
            use_density = (attempt % 3 != 0)  # Use density heuristic 2/3 of the time
            result = greedy_partition_fast(n, adj, rng, DEADLINE, use_density_heuristic=use_density)
            if result is not None:
                # Try local improvement
                improved = local_improve(result, n, adj, DEADLINE)
                if improved is not None:
                    count = len(improved)
                    if count < best_count:
                        best_count = count
                        best_cliques = improved
                else:
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 15.0
    t2 = 90.0

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
