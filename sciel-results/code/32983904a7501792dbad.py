import time
import random
import itertools

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the size of the
    largest triangle-free induced subgraph."""
    n = 18
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # --- Fast K4-free check using bitmasks ---
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                common_ab = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not (common_ab & (1 << c)):
                        continue
                    # a-b-c is a triangle; check if any d is adjacent to all three
                    if common_ab & adj[c]:
                        return False
        return True

    # --- Largest triangle-free induced subgraph via branch and bound ---
    def largest_triangle_free(adj):
        # Precompute triangles as list of bitmasks
        tri_masks = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                common_ab = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if common_ab & (1 << c):
                        tri_masks.append((1 << a) | (1 << b) | (1 << c))

        if not tri_masks:
            return n

        # For each vertex, precompute which triangles it participates in
        # tri_by_vertex[v] = list of triangle masks containing v
        tri_by_vertex = [[] for _ in range(n)]
        for tm in tri_masks:
            for v in range(n):
                if tm & (1 << v):
                    tri_by_vertex[v].append(tm)

        best = [0]
        full_mask = (1 << n) - 1

        def dfs(cands, chosen_mask, chosen_count):
            # Upper bound: can't exceed chosen_count + number of candidates
            if chosen_count + cands.bit_count() <= best[0]:
                return
            if not cands:
                if chosen_count > best[0]:
                    best[0] = chosen_count
                return

            # Pick vertex with most triangle constraints among candidates
            # for better pruning
            best_v = -1
            best_score = -1
            temp = cands
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                score = len(tri_by_vertex[v])
                if score > best_score:
                    best_score = score
                    best_v = v
                    if best_score > 20:  # good enough
                        break
                temp ^= lsb

            lsb = 1 << best_v

            # Branch 1: include v
            # Check if adding v creates any complete triangle with chosen vertices
            can_add = True
            for tm in tri_by_vertex[best_v]:
                other = tm ^ lsb
                if other & chosen_mask == other:
                    can_add = False
                    break

            if can_add:
                dfs(cands ^ lsb, chosen_mask | lsb, chosen_count + 1)

            # Branch 2: exclude v
            dfs(cands ^ lsb, chosen_mask, chosen_count)

        dfs(full_mask, 0, 0)
        return best[0]

    # --- Build edge list from adjacency masks ---
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # --- Initial candidate generators ---
    def make_turan_3():
        """K6,6,6 complete tripartite"""
        adj = [0] * n
        parts = [set(range(0, 6)), set(range(6, 12)), set(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def make_4_partite():
        """K(4,4,5,5) complete 4-partite, then repair to K4-free"""
        adj = [0] * n
        parts = [list(range(0, 4)), list(range(4, 8)), list(range(8, 13)), list(range(13, 18))]
        for i in range(4):
            for j in range(i + 1, 4):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return repair_k4_free(adj)

    def repair_k4_free(adj):
        """Remove edges to eliminate K4s"""
        adj = adj[:]
        for _ in range(200):
            if is_k4_free(adj):
                break
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] & (1 << b)):
                        continue
                    for c in range(b + 1, n):
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                            continue
                        common = adj[a] & adj[b] & adj[c]
                        if common:
                            d = (common & -common).bit_length() - 1
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def make_random_graph(p=0.5):
        """Random graph with edge probability p, repaired to K4-free"""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return repair_k4_free(adj)

    # --- Simulated annealing local search ---
    def simulated_annealing(adj, current_l, max_steps=3000, T_init=2.0, T_final=0.01):
        adj = adj[:]
        current_l = current_l
        best_adj = adj[:]
        best_l = current_l

        cooling = (T_final / T_init) ** (1.0 / max(1, max_steps))

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break

            T = T_init * (cooling ** step)

            # Pick random edge to flip
            i, j = random.sample(range(n), 2)
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)

            if not is_k4_free(adj):
                # Revert
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                continue

            new_l = largest_triangle_free(adj)

            if new_l < current_l:
                current_l = new_l
                if new_l < best_l:
                    best_l = new_l
                    best_adj = adj[:]
            else:
                # Accept with probability exp(-(new_l - current_l) / T)
                delta = new_l - current_l
                if random.random() < (2.718281828 ** (-delta / max(T, 1e-10))):
                    current_l = new_l
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)

        return best_adj, best_l

    # --- Main search ---
    random.seed(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(make_turan_3())
    candidates.append(make_4_partite())
    for p in [0.4, 0.5, 0.6, 0.7]:
        candidates.append(make_random_graph(p))

    best_adj = None
    best_l = n + 1

    # Evaluate each candidate and run simulated annealing
    for idx, adj in enumerate(candidates):
        if time.time() > deadline - 5:
            break

        l = largest_triangle_free(adj)
        if l < best_l:
            best_l = l
            best_adj = adj[:]

        # Run simulated annealing from this candidate
        if time.time() > deadline - 10:
            break

        new_adj, new_l = simulated_annealing(adj, l, max_steps=2000)
        if new_l < best_l:
            best_l = new_l
            best_adj = new_adj

    # Final: one more round of deterministic local search (try all single-edge flips)
    if best_adj is not None and time.time() < deadline - 5:
        current_adj = best_adj[:]
        current_l = best_l
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for u in range(n):
                if time.time() > deadline - 3:
                    break
                for v in range(u + 1, n):
                    if time.time() > deadline - 3:
                        break
                    # Try flipping edge (u,v)
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)

                    if not is_k4_free(current_adj):
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)
                        continue

                    new_l = largest_triangle_free(current_adj)
                    if new_l < current_l:
                        current_l = new_l
                        improved = True
                        if current_l < best_l:
                            best_l = current_l
                            best_adj = current_adj[:]
                    else:
                        # Revert
                        current_adj[u] ^= (1 << v)
                        current_adj[v] ^= (1 << u)

    # Return edges of best graph found
    if best_adj is None:
        # Fallback: return empty graph (all independent, L=18 but K4-free)
        return []

    return adj_to_edges(best_adj)
