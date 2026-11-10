import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

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

    def max_clique_branch_and_bound(n, adj, mask, rng):
        """Find maximum clique in G[mask] using branch and bound with pivot."""
        best_clique = []
        best_size = 0

        def expand(R, P):
            nonlocal best_clique, best_size
            if not P:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return

            # Upper bound
            if len(R) + P.bit_count() <= best_size:
                return

            # Pivot: choose vertex with max degree in P
            max_deg = -1
            pivot = -1
            P_tmp = P
            while P_tmp:
                b = P_tmp & (-P_tmp)
                v = b.bit_length() - 1
                P_tmp ^= b
                d = (P & adj[v]).bit_count()
                if d > max_deg:
                    max_deg = d
                    pivot = v

            # Only consider vertices NOT adjacent to pivot
            Q = P & ~adj[pivot]
            # Also include pivot itself if in P (it's not adjacent to itself)
            if P & (1 << pivot):
                Q |= (1 << pivot)

            while Q:
                b = Q & (-Q)
                v = b.bit_length() - 1
                Q ^= b

                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)

                P &= ~(1 << v)
                Q &= P

        expand([], mask)
        return best_clique

    def greedy_partition(n, adj_orig, rng, deadline):
        """Greedy edge clique partition: pick hardest edge, extract max clique containing it."""
        rem = adj_orig[:]
        cliques = []

        while True:
            if time.time() > deadline:
                return None

            # Check if any edges remain
            has_edge = False
            for i in range(n):
                if rem[i]:
                    has_edge = True
                    break
            if not has_edge:
                break

            # Find edge (u,v) with minimum common neighbors
            min_common = n + 1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem[u]:
                    continue
                # Only consider v > u to avoid duplicates
                v_mask = rem[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    b = v_mask & (-v_mask)
                    v = b.bit_length() - 1
                    v_mask ^= b
                    common = (rem[u] & rem[v]).bit_count()
                    if common < min_common:
                        min_common = common
                        best_u, best_v = u, v
                        if min_common == 0:
                            break
                if min_common == 0:
                    break

            if best_u == -1:
                break

            # Find max clique in common neighbors of best_u, best_v
            common_mask = rem[best_u] & rem[best_v]
            clique_core = max_clique_branch_and_bound(n, rem, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            # Remove all edges within this clique from rem
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def local_improve(cliques, n, adj_orig, rng, deadline):
        """Try to merge adjacent cliques or reassign edges to reduce count."""
        # This is a simple improvement: try to find pairs of cliques that can be merged
        # if their union is also a clique
        if len(cliques) < 2:
            return cliques

        # Check if merging two cliques into one is possible
        # For simplicity, try a few random pairs
        improved = cliques[:]
        attempts = 0
        while time.time() < deadline and attempts < 50:
            if len(improved) < 2:
                break
            i = rng.randrange(len(improved))
            j = rng.randrange(len(improved) - 1)
            if j >= i:
                j += 1
            c1 = set(improved[i])
            c2 = set(improved[j])
            union = c1 | c2
            # Check if union is a clique
            is_clique = True
            verts = list(union)
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u, v = verts[a], verts[b]
                    if not (adj_orig[u] >> v) & 1:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                improved.pop(j)
                improved.pop(i)
                improved.append(list(union))
            attempts += 1
        return improved

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        total_edges = len(edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit and time.time() < DEADLINE:
            rng = random.Random(seed * 7919 + attempt * 104729 + 13)
            result = greedy_partition(n, adj, rng, DEADLINE)
            if result is not None:
                # Verify validity
                valid = True
                edge_count = 0
                for c in result:
                    if len(c) < 2:
                        valid = False
                        break
                    # Check clique
                    for i in range(len(c)):
                        for j in range(i + 1, len(c)):
                            u, v = c[i], c[j]
                            if not (adj[u] >> v) & 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    edge_count += len(c) * (len(c) - 1) // 2
                if valid and edge_count == total_edges:
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
                        # Try local improvement
                        if time.time() < DEADLINE:
                            improved = local_improve(best_cliques, n, adj, rng, DEADLINE)
                            if len(improved) < best_count:
                                best_count = len(improved)
                                best_cliques = improved
            attempt += 1

        return best_cliques

    # Time budget: allocate more to the larger graph
    t1 = 12.0
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
