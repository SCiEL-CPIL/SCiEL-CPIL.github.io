import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    n = 18
    full_mask = (1 << n) - 1

    # Precompute all size-12 subsets as bitmasks
    subsets_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        subsets_12.append(mask)

    # Precompute bit counts for speed
    bitcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        bitcount[i] = bitcount[i >> 1] + (i & 1)

    def is_triangle_free_mask(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        # For each vertex in mask, its neighbors in mask must form an independent set
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            # Check if nu is independent: no two vertices in nu are adjacent
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[v] & nu) != 0:
                    return False
        return True

    def max_triangle_free_size(adj):
        """Find largest triangle-free induced subgraph size via branch and bound."""
        # First check: is the whole graph triangle-free?
        if is_triangle_free_mask(full_mask, adj):
            return n

        # Get all triangles
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                k_mask = common & (sum(1 << x for x in range(j + 1, n)))
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    triangles.append((1 << i) | (1 << j) | (1 << k))
                    k_mask ^= lsb

        if not triangles:
            return n

        best = [0]
        deadline_local = deadline - 3.0

        # Sort triangles by size (all size 3) but order for better pruning
        tri_masks = triangles

        def branch(remaining_mask):
            if time.time() > deadline_local:
                return
            cnt = bitcount[remaining_mask]
            if cnt <= best[0]:
                return
            # Check if triangle-free
            found_tri = None
            for tm in tri_masks:
                if (remaining_mask & tm) == tm:
                    found_tri = tm
                    break
            if found_tri is None:
                if cnt > best[0]:
                    best[0] = cnt
                return
            # Branch: remove one vertex from this triangle
            # Order: try removing vertex that appears in most triangles in remaining
            verts = []
            for v in range(n):
                if found_tri & (1 << v):
                    verts.append(v)
            # Heuristic: remove vertex with highest "triangle participation"
            counts = [0] * 3
            for idx, v in enumerate(verts):
                c = 0
                for tm in tri_masks:
                    if (remaining_mask & tm) == tm and (tm & (1 << v)):
                        c += 1
                counts[idx] = c
            order = sorted(range(3), key=lambda i: -counts[i])
            for idx in order:
                v = verts[idx]
                new_mask = remaining_mask & ~(1 << v)
                branch(new_mask)

        branch(full_mask)
        return best[0]

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True

    def generate_random_max_k4_free():
        """Generate a maximal K4-free graph by random edge ordering."""
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def count_tf_12(adj):
        """Count how many size-12 subsets are triangle-free."""
        count = 0
        for mask in subsets_12:
            if is_triangle_free_mask(mask, adj):
                count += 1
        return count

    def local_search_add_edges(adj):
        """Try to eliminate all triangle-free 12-subsets by adding edges."""
        adj = list(adj)
        max_iter = 300
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Find a random triangle-free 12-subset
            tf_found = None
            for mask in subsets_12:
                if is_triangle_free_mask(mask, adj):
                    tf_found = mask
                    break
            if tf_found is None:
                return adj
            # Find non-edges within this subset
            non_edges = []
            for u in range(n):
                if not (tf_found & (1 << u)):
                    continue
                for v in range(u + 1, n):
                    if (tf_found & (1 << v)) and not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
            if not non_edges:
                break
            random.shuffle(non_edges)
            added = False
            for u, v in non_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if not added:
                break
        return adj

    def simulated_annealing(adj, max_iters=500):
        """SA-style edge flip to reduce max triangle-free size."""
        adj = list(adj)
        current_score = max_triangle_free_size(adj)
        best_adj = adj[:]
        best_score = current_score
        T = 5.0
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            new_adj = adj[:]
            if adj[u] & (1 << v):
                # Remove edge
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                # Add edge (check K4-free)
                if not can_add_edge(adj, u, v):
                    continue
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)

            new_score = max_triangle_free_size(new_adj)
            delta = new_score - current_score

            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
            T *= cooling

        return best_adj, best_score

    # Seed: 3 disjoint K6's (complement of K6,6,6)
    seed_adj = [0] * n
    for i in range(6):
        for j in range(i + 1, 6):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(i + 1, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(12, 18):
        for j in range(i + 1, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)

    best_edges = adj_to_edges(seed_adj)
    best_score = 12

    # Phase 1: Random maximal K4-free graphs with local search
    while time.time() < deadline - 30:
        adj = generate_random_max_k4_free()

        # Quick check: count triangle-free 12-subsets
        tf_count = 0
        for mask in subsets_12:
            if is_triangle_free_mask(mask, adj):
                tf_count += 1
                if tf_count > 100:
                    break

        if tf_count == 0:
            return adj_to_edges(adj)

        if tf_count <= 50:
            # Promising: run local search to add edges
            adj_ls = local_search_add_edges(adj)
            tf_final = count_tf_12(adj_ls)
            if tf_final == 0:
                return adj_to_edges(adj_ls)

            # Check actual max triangle-free size
            s = max_triangle_free_size(adj_ls)
            if s < best_score:
                best_score = s
                best_edges = adj_to_edges(adj_ls)

        if best_score <= 11:
            break

    # Phase 2: Simulated annealing from best candidate
    if time.time() < deadline - 15:
        adj_cur = [0] * n
        for e in best_edges:
            u, v = e
            adj_cur[u] |= (1 << v)
            adj_cur[v] |= (1 << u)
        refined_adj, refined_score = simulated_annealing(adj_cur, max_iters=400)
        if refined_score < best_score:
            best_score = refined_score
            best_edges = adj_to_edges(refined_adj)

    # Phase 3: More random restarts with SA
    while time.time() < deadline - 5:
        adj = generate_random_max_k4_free()
        refined_adj, refined_score = simulated_annealing(adj, max_iters=200)
        if refined_score < best_score:
            best_score = refined_score
            best_edges = adj_to_edges(refined_adj)

    return best_edges

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
