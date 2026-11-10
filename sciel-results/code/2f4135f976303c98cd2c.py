import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(all_edges)

    best_edges = []
    best_count = 0

    # Precompute all Q3 embeddings as tuples of 8 vertex indices
    # Each embedding is a tuple (u0, u1, u2, u3, u4, u5, u6, u7)
    # where the edges of Q3 are:
    # (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)
    # We'll generate all possible assignments of 8 distinct vertices to the 8 positions of Q3

    # The 12 edges of Q3 in terms of position indices
    q3_edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]

    # For each edge in Q3, we need to check if the corresponding graph edge exists
    # We'll precompute for each possible graph edge (u,v), which Q3 edges map to it
    # Actually, let's think differently:
    # For a given graph edge (u,v), we want to check if there exists a Q3 embedding
    # where (u,v) is one of the 12 edges.
    # We can iterate over all 12 positions, assign u,v to that position,
    # and check if the remaining 6 vertices can be assigned to complete a Q3.

    # But this is complex. Let's use a different approach:
    # Precompute all 12-edge subsets of the complete graph K16 that form a Q3.
    # There are C(16,8) * 8! / (aut) possible embeddings, but many are isomorphic.
    # Actually, the number of distinct Q3 subgraphs in K16 is:
    # Choose 8 vertices: C(16,8) = 12870
    # For each set of 8 vertices, the number of Q3 embeddings is 8! / 48 = 280
    # (48 is the automorphism group size of Q3)
    # Total: 12870 * 280 = 3,603,600 possible Q3 subgraphs
    # This is too many to precompute.

    # Let's use the incremental check approach but optimize it.

    def creates_q3(u, v, adj):
        # Check if adding edge (u,v) creates a Q3 containing (u,v)
        # We iterate over all ways to assign u,v to an edge of Q3
        # and check if the remaining vertices can complete the Q3

        # The 12 edges of Q3, each as a pair of position indices
        # For each edge position (p, q) in q3_edges, we try:
        #   adj[p] = u, adj[q] = v  and  adj[p] = v, adj[q] = u
        # Then we need to assign the remaining 6 positions to 6 distinct vertices
        # from the remaining 14 vertices such that all 12 edges of Q3 exist in the graph.

        # This is still expensive. Let's use the bitmask approach from the current program
        # but optimize the inner loops.

        # Use the structure of Q3:
        # If (u,v) is an edge, in Q3 it can be:
        # - An edge between two vertices at Hamming distance 1
        # Let's fix u as one endpoint and v as the other.
        # In Q3, each vertex has degree 3.
        # So u has 2 other neighbors in Q3, and v has 2 other neighbors in Q3.

        # Let a, b be the other neighbors of u in Q3
        # Let c, d be the other neighbors of v in Q3
        # Then the remaining two vertices are:
        # e = neighbor of a and b (other than u)
        # f = neighbor of c and d (other than v)
        # And (e,f) must be an edge.

        # So we need:
        # a, b in N(u) \ {v}
        # c, d in N(v) \ {u}
        # (a,c) in E, (b,d) in E  [or (a,d) and (b,c)]
        # e in N(a) ∩ N(b) \ {u, c, d, f}
        # f in N(c) ∩ N(d) \ {v, a, b, e}
        # (e,f) in E
        # All vertices distinct

        # There are two cases for the "matching" between {a,b} and {c,d}:
        # Case 1: (a,c) and (b,d) are edges
        # Case 2: (a,d) and (b,c) are edges

        mask_u = adj[u]
        mask_v = adj[v]

        # Candidates for a, b: neighbors of u excluding v
        # Candidates for c, d: neighbors of v excluding u

        # To reduce symmetry, we can iterate over a in N(u)\{v}, b in N(u)\{v}, a < b
        # But we also need to handle the two matching cases.

        # Let's iterate over all pairs (a,b) from N(u)\{v} with a < b
        # For each, iterate over (c,d) from N(v)\{u} with c < d
        # Check both matchings.

        # Optimize: precompute lists
        nu = [x for x in range(n) if x != u and x != v and (mask_u >> x) & 1]
        nv = [x for x in range(n) if x != u and x != v and (mask_v >> x) & 1]

        for i in range(len(nu)):
            a = nu[i]
            for j in range(i + 1, len(nu)):
                b = nu[j]
                mask_a = adj[a]
                mask_b = adj[b]

                for k in range(len(nv)):
                    c = nv[k]
                    for l in range(k + 1, len(nv)):
                        d = nv[l]

                        # Check distinctness
                        if a == c or a == d or b == c or b == d:
                            continue

                        mask_c = adj[c]
                        mask_d = adj[d]

                        # Case 1: (a,c) and (b,d) are edges
                        if (mask_a >> c) & 1 and (mask_b >> d) & 1:
                            # e in N(a) ∩ N(b) \ {u, v, a, b, c, d}
                            # f in N(c) ∩ N(d) \ {u, v, a, b, c, d}
                            # (e,f) in E
                            # e != f

                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used

                            # Check if there exists e in e_cand and f in f_cand with (e,f) in E
                            # and e != f
                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                # f must be in f_cand \ {e}
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return True

                        # Case 2: (a,d) and (b,c) are edges
                        if (mask_a >> d) & 1 and (mask_b >> c) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & mask_d & ~used

                            e_mask = e_cand
                            while e_mask:
                                lsb = e_mask & -e_mask
                                e = lsb.bit_length() - 1
                                e_mask ^= lsb
                                f_mask = f_cand & ~(1 << e)
                                if adj[e] & f_mask:
                                    return True

        return False

    # Greedy with random restarts and kicks
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []

        # Greedy phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))

        # Local search: try kicks
        for _ in range(5):
            if time.time() > deadline:
                break
            if not current_edges:
                break

            # Remove a random edge
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)

            # Try to add edges in random order
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append((u, v))

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()

    return [[u, v] for u, v in best_edges]
