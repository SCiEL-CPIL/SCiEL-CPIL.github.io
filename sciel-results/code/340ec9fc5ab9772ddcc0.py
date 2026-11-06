import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1

        # Greedy lower bound: try to find a large independent set quickly
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v

            if best_v == -1:
                break

            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)

        if size >= 10:
            return True

        # Exact branch and bound
        sys.setrecursionlimit(10000)

        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False

            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False

            # Pick vertex with max degree in candidates (removes most)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            # Branch 1: include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            # Branch 2: exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False

        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_andrasfai(k):
        """Create Andrásfai graph And(k) with 3k vertices. Triangle-free, alpha=k."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    def mycielski(adj, n):
        """Mycielski construction: triangle-free, alpha increases by 1."""
        new_n = 2 * n + 1
        new_adj = [0] * new_n
        # Original vertices 0..n-1 keep their edges
        for i in range(n):
            new_adj[i] = adj[i]
        # Copy vertices n..2n-1
        for i in range(n):
            new_adj[n + i] = adj[i]
        # Add edges from copy to original neighbors
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                new_adj[n + i] |= (1 << j)
                new_adj[j] |= (1 << (n + i))
                temp &= temp - 1
        # Add apex vertex 2n
        for i in range(n):
            new_adj[2 * n] |= (1 << (n + i))
            new_adj[n + i] |= (1 << (2 * n))
        return new_adj, new_n

    def blow_up(adj, n, sizes):
        """Blow up each vertex i into sizes[i] independent vertices."""
        total = sum(sizes)
        new_adj = [0] * total
        # Map: vertex i in original maps to positions start[i]..start[i]+sizes[i]-1
        start = [0] * n
        for i in range(n):
            start[i] = sum(sizes[:i])

        for i in range(n):
            for j in range(n):
                if i != j and (adj[i] >> j) & 1:
                    for a in range(sizes[i]):
                        for b in range(sizes[j]):
                            new_adj[start[i] + a] |= (1 << (start[j] + b))
                            new_adj[start[j] + b] |= (1 << (start[i] + a))
        return new_adj, total

    # Try known constructions in order of expected n
    candidates = []

    # 1. Andrásfai And(9): n=27, alpha=9
    adj9, n9 = make_andrasfai(9)
    candidates.append((adj9, n9, "And(9)"))

    # 2. Mycielski of And(9): n=55, too large (>41)
    # But Mycielski of smaller graphs might work
    # Mycielski of And(3) (C5): n=11, alpha=2 -> Mycielski: n=23, alpha=3
    adj5, n5 = make_andrasfai(1)  # This is C5? No, And(1) has 3 vertices
    # Actually And(1) is just a 3-cycle... wait, And(k) for k=1: 3 vertices, edges (0,1),(1,2),(2,0) mod 3
    # That's C3, not triangle-free! Let me reconsider.
    # And(k) is triangle-free for k>=2. For k=1, it's C3 which has a triangle.
    # So use k>=2.

    # Mycielski of C5: n=11, alpha=2
    # C5: vertices 0-4, edges (0,1),(1,2),(2,3),(3,4),(4,0)
    c5_adj = [0] * 5
    for i in range(5):
        c5_adj[i] |= (1 << ((i + 1) % 5))
        c5_adj[i] |= (1 << ((i + 4) % 5))
    myc_c5, myc_n = mycielski(c5_adj, 5)
    candidates.append((myc_c5, myc_n, "Mycielski(C5)"))

    # Mycielski of Mycielski(C5): n=23, alpha=3
    myc2_c5, myc2_n = mycielski(myc_c5, myc_n)
    candidates.append((myc2_c5, myc2_n, "Mycielski^2(C5)"))

    # Mycielski of Mycielski^2(C5): n=47, too large
    # But maybe blow up some vertices of smaller constructions

    # 3. Try to extend And(9) by adding vertices
    # Strategy: add vertices one at a time, choosing adjacency to maintain triangle-free and alpha<10
    def try_extend(adj_base, n_base, target_n):
        """Try to extend base graph to target_n vertices."""
        if target_n <= n_base:
            return None

        adj = adj_base[:] + [0] * (target_n - n_base)

        # For each new vertex, try different adjacency patterns
        # A new vertex v can connect to a subset S of existing vertices
        # such that S is independent (to maintain triangle-free)
        # and the remaining vertices still can't form an independent set of size 10

        for new_v in range(n_base, target_n):
            if time.time() > deadline:
                return None

            # Try all independent sets of existing vertices as neighbor sets
            # For efficiency, try random independent sets
            found_valid = False
            for attempt in range(200):
                if time.time() > deadline:
                    return None

                # Pick a random independent set from existing vertices
                candidates_mask = (1 << n_base) - 1
                chosen = 0
                while candidates_mask:
                    v = (candidates_mask & -candidates_mask).bit_length() - 1
                    if random.random() < 0.5:  # 50% chance to include
                        chosen |= (1 << v)
                        candidates_mask &= ~adj[v] & ~(1 << v)
                    else:
                        candidates_mask &= ~(1 << v)

                # Check if chosen is independent
                is_indep = True
                temp = chosen
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    if adj[v] & chosen:
                        is_indep = False
                        break
                if not is_indep:
                    continue

                # Tentatively add vertex with neighbors = chosen
                adj[new_v] = chosen
                for i in range(n_base):
                    if (chosen >> i) & 1:
                        adj[i] |= (1 << new_v)

                # Check triangle-free
                if not is_triangle_free(adj, target_n):
                    # Revert
                    adj[new_v] = 0
                    for i in range(n_base):
                        if (chosen >> i) & 1:
                            adj[i] &= ~(1 << new_v)
                    continue

                # Check independent set
                if not has_independent_set_10(adj, target_n):
                    found_valid = True
                    break

                # Revert
                adj[new_v] = 0
                for i in range(n_base):
                    if (chosen >> i) & 1:
                        adj[i] &= ~(1 << new_v)

            if not found_valid:
                return None

        return adj

    # Try extending And(9) to larger n
    for target_n in range(40, 27, -1):
        if time.time() > deadline:
            break
        extended = try_extend(adj9, n9, target_n)
        if extended is not None:
            return {'n': target_n, 'edges': build_edge_list(extended, target_n)}

    # 4. Try random search with smarter initialization
    for target_n in range(40, 27, -1):
        if time.time() > deadline:
            break

        for attempt in range(500):
            if time.time() > deadline:
                break

            adj = [0] * target_n

            # Start with a sparse random triangle-free graph
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)

            # Add edges greedily, maintaining triangle-free
            for u, v in edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Try to add more edges to reduce independence number
            # Randomly try adding edges
            for _ in range(100):
                if time.time() > deadline:
                    break
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u != v and not (adj[u] >> v) & 1 and (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not has_independent_set_10(adj, target_n):
                return {'n': target_n, 'edges': build_edge_list(adj, target_n)}

    # 5. Fallback: return the best known valid graph
    if not has_independent_set_10(adj9, n9) and is_triangle_free(adj9, n9):
        return {'n': n9, 'edges': build_edge_list(adj9, n9)}

    # Last resort
    return {'n': 1, 'edges': []}
