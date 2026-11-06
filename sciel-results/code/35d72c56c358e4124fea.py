import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n = 18

    # Precompute all 4-sets for K4 checking
    all_quads = list(itertools.combinations(range(n), 4))

    def is_k4_free(adj):
        for quad in all_quads:
            i, j, k, l = quad
            if (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[i] & (1 << l)) and \
               (adj[j] & (1 << k)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                return False
        return True

    def max_triangle_free_size(adj):
        # Find all triangles
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    k_mask = common
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        if k > j:
                            triangles.append((i, j, k))
                        k_mask ^= lsb

        if not triangles:
            return n

        # Exact branch and bound for minimum vertex cover of triangle hypergraph
        # Score = n - min_cover
        best_cover = n

        # Sort triangles to improve pruning: put triangles with high-degree vertices first
        tri_freq = [0] * n
        for t in triangles:
            for v in t:
                tri_freq[v] += 1

        # Order triangles by sum of frequencies of their vertices (descending)
        triangles_sorted = sorted(triangles, key=lambda t: -(tri_freq[t[0]] + tri_freq[t[1]] + tri_freq[t[2]]))

        def solve(tri_idx, covered_mask, current_size):
            nonlocal best_cover
            if current_size >= best_cover:
                return

            # Find first uncovered triangle
            while tri_idx < len(triangles_sorted):
                t = triangles_sorted[tri_idx]
                if not (covered_mask & (1 << t[0])) and not (covered_mask & (1 << t[1])) and not (covered_mask & (1 << t[2])):
                    break
                tri_idx += 1

            if tri_idx >= len(triangles_sorted):
                best_cover = current_size
                return

            t = triangles_sorted[tri_idx]

            # Lower bound: count disjoint uncovered triangles greedily
            # Simple bound: if we have m uncovered triangles and they're all disjoint, need m
            # Quick greedy packing:
            remaining = tri_idx
            packed = 0
            used_mask = 0
            while remaining < len(triangles_sorted):
                tt = triangles_sorted[remaining]
                if not (used_mask & (1 << tt[0])) and not (used_mask & (1 << tt[1])) and not (used_mask & (1 << tt[2])):
                    if not (covered_mask & (1 << tt[0])) and not (covered_mask & (1 << tt[1])) and not (covered_mask & (1 << tt[2])):
                        packed += 1
                        used_mask |= (1 << tt[0]) | (1 << tt[1]) | (1 << tt[2])
                remaining += 1

            if current_size + packed >= best_cover:
                return

            # Branch on each vertex of t
            # Try vertices in order of frequency (most constrained first)
            verts = sorted(t, key=lambda v: -tri_freq[v])
            for v in verts:
                new_mask = covered_mask | (1 << v)
                solve(tri_idx + 1, new_mask, current_size + 1)

        solve(0, 0, 0)
        return n - best_cover

    def generate_random_k4_free():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            # Check if adding (u, v) creates a K4
            common = adj[u] & adj[v]
            c_mask = common
            has_k4 = False
            verts = []
            while c_mask:
                lsb = c_mask & -c_mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c_mask ^= lsb

            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    if adj[verts[i]] & (1 << verts[j]):
                        has_k4 = True
                        break
                if has_k4:
                    break

            if not has_k4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Generate multiple random initial graphs and pick the best
    best_adj = None
    best_score = n + 1

    for _ in range(20):
        if time.time() > deadline:
            break
        adj = generate_random_k4_free()
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    # Simulated Annealing
    T = 2.0
    T_min = 0.05
    alpha = 0.9995

    step = 0
    while time.time() < deadline:
        step += 1

        # Generate neighbor
        adj_copy = best_adj[:]
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u

        if adj_copy[u] & (1 << v):
            # Remove edge - always valid
            adj_copy[u] ^= (1 << v)
            adj_copy[v] ^= (1 << u)
            new_adj = adj_copy
        else:
            # Add edge - check K4
            common = adj_copy[u] & adj_copy[v]
            c_mask = common
            has_k4 = False
            verts = []
            while c_mask:
                lsb = c_mask & -c_mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c_mask ^= lsb

            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    if adj_copy[verts[i]] & (1 << verts[j]):
                        has_k4 = True
                        break
                if has_k4:
                    break

            if has_k4:
                continue

            adj_copy[u] |= (1 << v)
            adj_copy[v] |= (1 << u)
            new_adj = adj_copy

        new_score = max_triangle_free_size(new_adj)

        # We want to MINIMIZE new_score
        delta = new_score - best_score

        if delta < 0 or (T > T_min and random.random() < pow(2.718, -delta / T)):
            best_adj = new_adj
            best_score = new_score

        T *= alpha
        if T < T_min:
            T = T_min

        # Occasional restart to escape local minima
        if step % 5000 == 0 and random.random() < 0.1:
            adj = generate_random_k4_free()
            score = max_triangle_free_size(adj)
            if score < best_score:
                best_adj = adj[:]
                best_score = score
                T = 1.0  # Reset temperature

    # Final polish: try all single-edge flips to see if we can improve
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline - 5:
                        break
                    adj_copy = best_adj[:]
                    if adj_copy[u] & (1 << v):
                        adj_copy[u] ^= (1 << v)
                        adj_copy[v] ^= (1 << u)
                    else:
                        common = adj_copy[u] & adj_copy[v]
                        c_mask = common
                        has_k4 = False
                        verts = []
                        while c_mask:
                            lsb = c_mask & -c_mask
                            idx = lsb.bit_length() - 1
                            verts.append(idx)
                            c_mask ^= lsb
                        for i in range(len(verts)):
                            for j in range(i + 1, len(verts)):
                                if adj_copy[verts[i]] & (1 << verts[j]):
                                    has_k4 = True
                                    break
                            if has_k4:
                                break
                        if has_k4:
                            continue
                        adj_copy[u] |= (1 << v)
                        adj_copy[v] |= (1 << u)

                    new_score = max_triangle_free_size(adj_copy)
                    if new_score < best_score:
                        best_adj = adj_copy
                        best_score = new_score
                        improved = True
                if time.time() > deadline - 5:
                    break

    # Return edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
