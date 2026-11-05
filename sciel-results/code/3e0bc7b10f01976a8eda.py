"""Combined: 3-partite Turán structure + bitmask B&B + targeted inter-part edge removal."""
import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    N = 18
    full_mask = (1 << N) - 1

    # --- Fast helpers ---
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    # --- Exact B&B for largest triangle-free induced subgraph ---
    def largest_triangle_free(adj, time_limit=None):
        best = [0]

        # Multiple greedy passes for good initial lower bound
        for _ in range(15):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = S.bit_count()
            if sz > best[0]:
                best[0] = sz

        if best[0] == N:
            return N

        # Precompute: for each pair (u,v) that is an edge, the set of common neighbors
        # A vertex v can be added to S iff N(v) ∩ S is independent
        # Equivalently: no edge (a,b) with a,b in N(v) ∩ S

        def bnb(S_mask, C_mask, sz):
            if time_limit and time.time() > time_limit:
                return
            if sz + C_mask.bit_count() <= best[0]:
                return

            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            # Filter C_mask: only keep vertices that can be added to S_mask
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb

            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return

            C_mask = new_C
            if sz + C_mask.bit_count() <= best[0]:
                return

            # Branch on vertex with max degree in C_mask (best pruning)
            best_v = -1
            best_deg = -1
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = (adj[v] & C_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                c ^= lsb

            if best_v < 0:
                lsb = C_mask & -C_mask
                best_v = lsb.bit_length() - 1

            v_bit = 1 << best_v
            C_mask ^= v_bit

            # Include
            bnb(S_mask | v_bit, C_mask, sz + 1)
            # Exclude
            bnb(S_mask, C_mask, sz)

        bnb(0, full_mask, 0)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    if adj[k] & (common ^ lsb):
                        return False
                    c ^= lsb
        return True

    # --- Build 3-partite Turán graph ---
    # Parts are independent sets; all edges between different parts.
    # Inherently K4-free (any 4 vertices include 2 from same part = non-edge).
    # In complete 3-partite: triangle-free induced subgraph = union of at most 2 parts.
    # So score = 18 - min(part_sizes).
    # We then remove inter-part edges to break large 2-part unions.

    def make_turan3(part_sizes):
        """part_sizes: 3 ints summing to 18. Returns (adj, parts) where parts is list of 3 vertex lists."""
        adj = [0] * N
        parts = []
        v = 0
        for size in part_sizes:
            parts.append(list(range(v, v + size)))
            v += size
        # All edges between different parts
        for a in range(3):
            for b in range(a + 1, 3):
                for u in parts[a]:
                    for w in parts[b]:
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
        return adj, parts

    def part_of_vertex(parts, v):
        for i, p in enumerate(parts):
            if v in p:
                return i
        return -1

    # Precompute part membership for fast lookup
    def make_part_lookup(parts):
        lookup = [0] * N
        for i, p in enumerate(parts):
            for v in p:
                lookup[v] = i
        return lookup

    random.seed(42)

    # --- Phase 1: Try all 3-partite configurations ---
    best_score = 999
    best_adj = None
    best_parts = None
    best_part_lookup = None

    part_configs = []
    for a in range(1, 19):
        for b in range(1, 19 - a):
            c = 18 - a - b
            if c >= 1:
                part_configs.append([a, b, c])

    # Deduplicate (permutations of same multiset)
    seen = set()
    unique_configs = []
    for cfg in part_configs:
        key = tuple(sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)

    for config in unique_configs:
        adj, parts = make_turan3(config)
        # For complete 3-partite, score is exactly 18 - min(config)
        # But verify with B&B (should match)
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_parts = [p[:] for p in parts]
            best_part_lookup = make_part_lookup(parts)

    # --- Phase 2: Local search on 3-partite structure ---
    # Strategy: remove inter-part edges to break the "take all of part A + all of part B" triangle-free set.
    # After removing edge (u,v) where u in part i, v in part j:
    # The subgraph induced by (part_i ∪ part_j) minus edge (u,v) may still be triangle-free
    # but we need to check if removing edges from the complete bipartite between two parts
    # reduces the max triangle-free set.

    def local_search_3partite(adj, parts, part_lookup, best_score, deadline):
        """Remove inter-part edges strategically to reduce largest triangle-free induced subgraph."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Try removing each inter-part edge
            for i in range(N):
                if time.time() > deadline:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return best_score
                    pi = part_lookup[i]
                    pj = part_lookup[j]
                    if pi == pj:
                        continue  # Intra-part: no edge exists in Turán
                    if not ((adj[i] >> j) & 1):
                        continue  # Edge already removed

                    # Remove edge (i, j)
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)

                    score = largest_triangle_free(adj, deadline - 5.0)

                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

        return best_score

    best_score = local_search_3partite(best_adj, best_parts, best_part_lookup, best_score, deadline)

    # --- Phase 3: Try adding some intra-part edges (making parts into cliques) ---
    # This changes the structure: now within a part, vertices are connected.
    # A 3-partite graph with cliques within parts is still K4-free as long as
    # we don't create a K4. Actually, if parts are cliques, a K4 could form with
    # 2 vertices from part A (connected) + 1 from B + 1 from C if all cross edges exist.
    # So we need to be careful. Let's stick with independent parts.

    # --- Phase 4: More aggressive: try different starting configurations with perturbation ---
    # Start from the best found, try removing more edges or trying other configs

    attempts = 0
    while time.time() < deadline - 15.0:
        attempts += 1

        # Try a random 3-partite config with some random inter-part edges removed
        config = random.choice(unique_configs)
        adj, parts = make_turan3(config)
        part_lookup = make_part_lookup(parts)

        # Randomly remove some inter-part edges
        num_removals = random.randint(3, 20)
        inter_part_edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if part_lookup[i] != part_lookup[j] and ((adj[i] >> j) & 1):
                    inter_part_edges.append((i, j))

        random.shuffle(inter_part_edges)
        for u, v in inter_part_edges[:num_removals]:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

        if not is_k4_free(adj):
            continue

        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_parts = [p[:] for p in parts]
            best_part_lookup = part_lookup
            # Local search from here
            best_score = local_search_3partite(best_adj, best_parts, best_part_lookup, best_score, deadline)

        # Also: take current best and try removing a few more random inter-part edges
        if attempts % 3 == 0 and time.time() < deadline - 20.0:
            adj2 = best_adj[:]
            part_lookup2 = best_part_lookup
            current_inter_edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if part_lookup2[i] != part_lookup2[j] and ((adj2[i] >> j) & 1):
                        current_inter_edges.append((i, j))

            if current_inter_edges:
                num_rem = random.randint(1, 4)
                for u, v in random.sample(current_inter_edges, min(num_rem, len(current_inter_edges))):
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)

                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]

        # Try simulated annealing on the 3-partite structure
        if attempts % 5 == 0 and time.time() < deadline - 25.0:
            adj_sa = best_adj[:]
            current_score = best_score
            T = 1.0
            cooling = 0.997

            while time.time() < deadline - 15.0 and T > 0.05:
                # Pick a random inter-part edge to flip
                i, j = random.sample(range(N), 2)
                pi = best_part_lookup[i]
                pj = best_part_lookup[j]
                if pi == pj:
                    continue

                old_val = (adj_sa[i] >> j) & 1

                if old_val == 0:
                    # Try adding: must maintain K4-free
                    if not can_add_edge(adj_sa, i, j):
                        continue
                    adj_sa[i] |= (1 << j)
                    adj_sa[j] |= (1 << i)
                else:
                    adj_sa[i] &= ~(1 << j)
                    adj_sa[j] &= ~(1 << i)

                new_score = largest_triangle_free(adj_sa, deadline - 5.0)
                delta = new_score - current_score

                if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj_sa[:]
                else:
                    # Revert
                    if old_val == 0:
                        adj_sa[i] &= ~(1 << j)
                        adj_sa[j] &= ~(1 << i)
                    else:
                        adj_sa[i] |= (1 << j)
                        adj_sa[j] |= (1 << i)

                T *= cooling

    # Final verification
    if not is_k4_free(best_adj):
        # Fallback: return a known valid C9 union
        adj_fallback = [0] * N
        for i in range(9):
            a, b = i, (i + 1) % 9
            adj_fallback[a] |= (1 << b)
            adj_fallback[b] |= (1 << a)
        for i in range(9):
            a, b = 9 + i, 9 + (i + 1) % 9
            adj_fallback[a] |= (1 << b)
            adj_fallback[b] |= (1 << a)
        best_adj = adj_fallback

    return get_edge_list(best_adj)
