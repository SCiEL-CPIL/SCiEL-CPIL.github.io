import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s

    def time_exceeded():
        return time.time() - start_time > deadline

    # Precompute for q=29 (Paley coloring)
    q = 29
    squares_29 = {1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}

    def build_paley_colors(n):
        """Build Paley coloring for prime n ≡ 1 (mod 4)."""
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def edge_index(n, i, j):
        """Get the index of edge (i,j) in the colors list, where i < j."""
        if i > j:
            i, j = j, i
        # Number of edges before vertex i: i*(n-1) - i*(i-1)//2
        # Then j-i-1 edges after that
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        """Build bitmask adjacency for each color."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def find_k222(n, colors):
        """
        Find a K_{2,2,2} in either color class using bitmasks.
        Returns (color, vertices) if found, else (None, None).
        """
        adj = build_adj(n, colors)

        def has_k222(adj_color):
            """
            Check if adj_color contains K_{2,2,2}.
            Returns (vertices) if found, else None.
            """
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_color[i] & (1 << j):
                        continue  # skip pairs with an edge

                    # Common neighbors of i and j
                    cn_mask = (adj_color[i] & adj_color[j]) & ~(1 << i) & ~(1 << j)

                    # Get list of common neighbors
                    cn_list = []
                    mask = cn_mask
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        cn_list.append(v)
                        mask ^= lsb

                    if len(cn_list) < 4:
                        continue  # need at least 4 more vertices for B and C groups

                    # Find pairs (k,l) in cn_list with no edge
                    no_edge_pairs = []
                    for a_idx in range(len(cn_list)):
                        k = cn_list[a_idx]
                        for b_idx in range(a_idx + 1, len(cn_list)):
                            l = cn_list[b_idx]
                            if not (adj_color[k] & (1 << l)):
                                no_edge_pairs.append((k, l))

                    for (k, l) in no_edge_pairs:
                        # Common neighbors of i, j, k, l
                        cn2_mask = (adj_color[i] & adj_color[j] & adj_color[k] & adj_color[l])
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)

                        # Check if cn2_mask contains a pair with no edge
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2

                        if len(cn2_list) < 2:
                            continue

                        for c_idx in range(len(cn2_list)):
                            m = cn2_list[c_idx]
                            for d_idx in range(c_idx + 1, len(cn2_list)):
                                o = cn2_list[d_idx]
                                if not (adj_color[m] & (1 << o)):
                                    return (i, j, k, l, m, o)

            return None

        # Check color 0
        result = has_k222(adj[0])
        if result is not None:
            return (0, result)

        # Check color 1
        result = has_k222(adj[1])
        if result is not None:
            return (1, result)

        return (None, None)

    # Verify Paley(29) is valid
    color, vertices = find_k222(q, colors_29)

    if color is None:
        # Paley(29) is valid! Try to extend to n=30 via local search.
        best_n = q
        best_colors = colors_29

        if not time_exceeded():
            random.seed(42)

            # Strategy: Try to extend to n=30 with smart initialization
            # and targeted local search

            max_attempts = 50
            for attempt in range(max_attempts):
                if time_exceeded():
                    break

                # Create initial coloring for n=30
                colors_30 = colors_29[:]

                # Smart initialization: try to assign colors that minimize conflicts
                # For each vertex i < 29, count how many times each color would create
                # potential K_{2,2,2} patterns, then choose the less problematic color
                for i in range(29):
                    # Count conflicts for each color choice
                    conflict_0 = 0
                    conflict_1 = 0

                    # For vertex 30 connected to vertex i with color c,
                    # check if this creates any new K_{2,2,2} opportunities
                    # This is complex, so use a simpler heuristic:
                    # Try both colors and see which creates fewer immediate issues

                    # Simple approach: alternate or use based on some pattern
                    # Try deterministic patterns first, then random
                    if attempt < 5:
                        # Deterministic patterns
                        if attempt == 0:
                            c = 0
                        elif attempt == 1:
                            c = 1
                        elif attempt == 2:
                            c = i % 2
                        elif attempt == 3:
                            c = (i // 3) % 2
                        else:
                            c = (i * 7) % 2
                    else:
                        c = random.randint(0, 1)

                    colors_30.append(c)

                # Local search with targeted fixes
                max_iter = 500
                for iteration in range(max_iter):
                    if time_exceeded():
                        break

                    color, vertices = find_k222(30, colors_30)
                    if color is None:
                        # Found valid n=30 coloring!
                        return {'n': 30, 'colors': colors_30}

                    # We have a K_{2,2,2} in `color` with vertices
                    # The K_{2,2,2} has 6 vertices in 3 pairs
                    # To break it, we need to change one of the cross-pair edges
                    # or add an intra-pair edge (but intra-pair edges are the "no edge" pairs)

                    # The vertices are (a, b, c, d, e, f) where:
                    # (a,b) is pair 1, (c,d) is pair 2, (e,f) is pair 3
                    # All cross-pair edges have color `color`
                    # To break: change one cross-pair edge to the other color

                    # Pick a random cross-pair edge to flip
                    # Cross-pair edges are between different pairs
                    pairs = [(vertices[0], vertices[1]),
                             (vertices[2], vertices[3]),
                             (vertices[4], vertices[5])]

                    # All edges between different pairs should have color `color`
                    # Flip one of them
                    cross_edges = []
                    for p1 in range(3):
                        for p2 in range(p1 + 1, 3):
                            for v1 in pairs[p1]:
                                for v2 in pairs[p2]:
                                    cross_edges.append((v1, v2))

                    # Randomly pick one to flip
                    v1, v2 = random.choice(cross_edges)
                    idx = edge_index(30, v1, v2)
                    colors_30[idx] = 1 - colors_30[idx]

                # After local search, check if we succeeded
                color, vertices = find_k222(30, colors_30)
                if color is None:
                    return {'n': 30, 'colors': colors_30}

            # If we couldn't find n=30, return the best valid coloring
            return {'n': best_n, 'colors': best_colors}
        else:
            return {'n': q, 'colors': colors_29}
    else:
        # Paley(29) is NOT valid - need to find another construction
        # Try local search on n=29
        random.seed(42)

        # Start with Paley(29) and do local search
        colors = colors_29[:]
        max_iter = 5000
        for iteration in range(max_iter):
            if time_exceeded():
                break

            color, vertices = find_k222(29, colors)
            if color is None:
                return {'n': 29, 'colors': colors}

            # Fix the violation by flipping a cross-pair edge
            pairs = [(vertices[0], vertices[1]),
                     (vertices[2], vertices[3]),
                     (vertices[4], vertices[5])]

            cross_edges = []
            for p1 in range(3):
                for p2 in range(p1 + 1, 3):
                    for v1 in pairs[p1]:
                        for v2 in pairs[p2]:
                            cross_edges.append((v1, v2))

            v1, v2 = random.choice(cross_edges)
            idx = edge_index(29, v1, v2)
            colors[idx] = 1 - colors[idx]

        # If we couldn't find valid coloring, return best effort
        return {'n': 29, 'colors': colors}
