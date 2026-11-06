import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 115.0

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def solve_instance(n, p, seed):
        adj, edges = generate_graph(n, p, seed)
        # Use bitmask per vertex for remaining edges
        # rem_adj[i] = bitmask of j such that (i,j) is in remaining (symmetric)
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        remaining_count = len(edges)
        cliques = []

        while remaining_count > 0:
            if time.time() - start_time > timeout:
                # Fallback: collect all remaining edges as 2-cliques
                for u in range(n):
                    mask = rem_adj[u]
                    while mask:
                        v_bit = mask & (-mask)
                        v = v_bit.bit_length() - 1
                        if v > u:
                            cliques.append([u, v])
                        mask ^= v_bit
                break

            # Find the most constrained edge: minimum number of common neighbors in remaining graph
            best_edge = None
            best_common = n + 1

            # Sample edges for efficiency
            all_edges = []
            for u in range(n):
                mask = rem_adj[u]
                while mask:
                    v_bit = mask & (-mask)
                    v = v_bit.bit_length() - 1
                    if v > u:
                        all_edges.append((u, v))
                    mask ^= v_bit

            if len(all_edges) > 80:
                sample = random.sample(all_edges, 80)
            else:
                sample = all_edges

            for u, v in sample:
                # Common neighbors of (u,v) in remaining graph
                common_mask = rem_adj[u] & rem_adj[v]
                common_mask &= ~(1 << u)
                common_mask &= ~(1 << v)
                common = common_mask.bit_count()
                if common < best_common:
                    best_common = common
                    best_edge = (u, v)

            if best_edge is None:
                break

            u, v = best_edge
            # Find candidates: vertices w connected to both u and v in remaining
            cand_mask = rem_adj[u] & rem_adj[v]
            cand_mask &= ~(1 << u)
            cand_mask &= ~(1 << v)

            # Greedy clique growth from {u, v}
            C = [u, v]
            C_mask = (1 << u) | (1 << v)
            cand_set_mask = cand_mask

            while cand_set_mask:
                best_c = -1
                best_score = -1
                m = cand_set_mask
                while m:
                    c_bit = m & (-m)
                    c = c_bit.bit_length() - 1
                    m ^= c_bit

                    # Check c connects to all in C via remaining
                    if (rem_adj[c] & C_mask) != C_mask:
                        continue

                    # Score: degree within remaining candidate set
                    score = (rem_adj[c] & cand_set_mask).bit_count()
                    if score > best_score:
                        best_score = score
                        best_c = c

                if best_c < 0:
                    break

                C.append(best_c)
                C_mask |= (1 << best_c)
                # Update candidate set: only keep candidates connected to best_c
                cand_set_mask &= rem_adj[best_c]

            # Remove all edges within C from remaining
            for i in range(len(C)):
                a = C[i]
                for j in range(i + 1, len(C)):
                    b = C[j]
                    rem_adj[a] &= ~(1 << b)
                    rem_adj[b] &= ~(1 << a)
                    remaining_count -= 1

            cliques.append(C)

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
