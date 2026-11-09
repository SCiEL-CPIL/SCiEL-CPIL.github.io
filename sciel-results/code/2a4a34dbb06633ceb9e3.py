"""Clique partition of two fixed dense random graphs via bitmask + Bron-Kerbosch."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def adj_mask_from_edges(n, edges):
        m = [0] * n
        for (u, v) in edges:
            m[u] |= 1 << v
            m[v] |= 1 << u
        return m

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def mask_to_list(mask, n):
        return [v for v in range(n) if (mask >> v) & 1]

    def bron_kerbosch_maximal_cliques(adj_m, n, min_size=3, cap=200000, deadline_ok=None):
        """Enumerate maximal cliques via Bron-Kerbosch with pivot. Returns list of bitmasks."""
        result = []
        aborted = [False]

        def bronk(R, P, X):
            if aborted[0]:
                return
            if deadline_ok is not None and not deadline_ok():
                aborted[0] = True
                return
            if not P and not X:
                if R.bit_count() >= min_size:
                    result.append(R)
                    if len(result) >= cap:
                        aborted[0] = True
                return
            if not P:
                return
            # Pivot selection: pick u from P|X that maximizes |P ∩ N(u)|
            ux = P | X
            best_u = -1
            best_c = -1
            for u in iter_bits(ux):
                c = (P & adj_m[u]).bit_count()
                if c > best_c:
                    best_c = c
                    best_u = u
            # Branch on P \ N(best_u)
            for v in iter_bits(P & ~adj_m[best_u]):
                bit = 1 << v
                bronk(R | bit, P & adj_m[v], X & adj_m[v])
                if aborted[0]:
                    return
                P &= ~bit
                X |= bit

        bronk(0, (1 << n) - 1, 0)
        return result

    def greedy_clique_from_seed(adj_m, n, seed_edge, remaining_mask):
        """Build a large clique containing seed_edge using greedy expansion in remaining subgraph."""
        u, v = seed_edge
        clique = {u, v}
        # Candidates: common neighbors of u and v that are in remaining subgraph
        candidates = adj_m[u] & adj_m[v] & remaining_mask
        # Remove u and v from candidates
        candidates &= ~(1 << u) & ~(1 << v)

        # Greedily add vertices that connect to all in clique and are in remaining
        changed = True
        while changed:
            changed = False
            # Find best candidate: one with most connections to other candidates
            best_w = -1
            best_conn = -1
            for w in iter_bits(candidates):
                # Check w connects to all in clique (within remaining)
                can_add = True
                for x in clique:
                    if not (adj_m[w] & (1 << x)) and not (adj_m[x] & (1 << w)):
                        can_add = False
                        break
                    # Also check edge (w,x) is in remaining
                    e = edge_key(w, x)
                    # We'll verify by checking if both are in remaining_mask's edge set
                    # Actually, remaining_mask is an adjacency mask, so check:
                    if not (remaining_mask[w] & (1 << x)):
                        can_add = False
                        break
                if not can_add:
                    continue
                # Count connections to other candidates
                conn = (candidates & adj_m[w]).bit_count()
                if conn > best_conn:
                    best_conn = conn
                    best_w = w
            if best_w >= 0:
                clique.add(best_w)
                candidates &= ~(1 << best_w)
                # Filter candidates: must connect to best_w in remaining
                candidates &= remaining_mask[best_w]
                changed = True

        return sorted(clique)

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Build initial adjacency mask and edge set
        adj_m = adj_mask_from_edges(n, edges)
        edge_set = set(edge_key(u, v) for u, v in edges)
        remaining = set(edge_set)

        cliques = []

        while remaining:
            if time.time() > DEADLINE:
                break

            # Find the "hardest" edge: one with fewest common neighbors in remaining subgraph
            # This targets sparse regions where large cliques are harder to form
            # but when found, they cover many edges
            best_edge = None
            best_common = float('inf')

            # Build remaining adjacency mask for fast common neighbor counting
            rem_adj = [0] * n
            for (u, v) in remaining:
                rem_adj[u] |= 1 << v
                rem_adj[v] |= 1 << u

            for e in remaining:
                u, v = e
                # Common neighbors in remaining subgraph
                common = (rem_adj[u] & rem_adj[v]).bit_count()
                if common < best_common:
                    best_common = common
                    best_edge = e
                    if common <= 0:
                        break

            if best_edge is None:
                break

            u, v = best_edge

            # Try to find a large clique using Bron-Kerbosch on remaining subgraph
            # that contains this edge
            rem_mask = [0] * n
            for (a, b) in remaining:
                rem_mask[a] |= 1 << b
                rem_mask[b] |= 1 << a

            # Find maximal cliques in remaining subgraph
            max_cliques = bron_kerbosch_maximal_cliques(rem_mask, n, min_size=3, cap=50000,
                                                         deadline_ok=lambda: time.time() < DEADLINE)

            # Among maximal cliques, find one that contains edge (u,v) and is largest
            best_clique = None
            best_size = 2  # at least the edge itself
            for cm in max_cliques:
                if (cm & (1 << u)) and (cm & (1 << v)):
                    size = cm.bit_count()
                    if size > best_size:
                        best_size = size
                        best_clique = cm

            if best_clique is not None and best_size >= 3:
                clique_list = mask_to_list(best_clique, n)
            else:
                # Fallback: greedy expansion from the seed edge
                clique_list = greedy_clique_from_seed(adj_m, n, best_edge, rem_mask)
                if len(clique_list) < 2:
                    clique_list = [u, v]

            cliques.append(clique_list)

            # Remove all edges within the clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = edge_key(a, b)
                    remaining.discard(e)

        # Handle any remaining edges as size-2 cliques
        for e in remaining:
            cliques.append(list(e))

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > DEADLINE:
            edges = random_graph(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
# EVOLVE-BLOCK-END
