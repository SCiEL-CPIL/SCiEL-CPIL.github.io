"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill

    # --- Greenwood-Gleason / Clebsch construction for K_16, doubled to K_32 ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    # Build K_32 coloring: two copies of K_16 (Clebsch), color 3 between copies
    def build_k32_colors():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    base_colors = build_k32_colors()

    # Try to extend to K_33, K_34, ... by backtracking
    # The new vertex v connects to existing vertices 0..n-1.
    # Constraint: for any edge (i,j) with color c, the edges (v,i) and (v,j) cannot both be color c.
    # Equivalently: for each color c, the set of neighbors of v that get color c must be an independent set in the c-colored subgraph.

    def try_extend(colors, n, max_extra=5):
        """Try to extend coloring of K_n to K_{n+k} for k up to max_extra.
        Returns (new_n, new_colors) for the largest extension found, or (n, colors) if no extension."""
        cur_colors = list(colors)
        cur_n = n
        for target_n in range(n + 1, n + max_extra + 1):
            if time.time() > deadline:
                break
            if not _try_add_one_vertex(cur_colors, cur_n):
                break
            cur_n += 1
        return cur_n, cur_colors

    def _try_add_one_vertex(colors, n):
        """Try to add vertex n to coloring of K_n. Returns True on success and updates colors in place."""
        # For each existing vertex i, the new vertex gets some color c_i in {0,1,2,3}.
        # Constraint: for any pair (i,j) where edge (i,j) has color c, we cannot have c_i = c AND c_j = c.
        # This means: for each color c, the set S_c = {i : c_i = c} must be an independent set in the c-graph.
        #
        # We can precompute for each vertex i and color c whether assigning color c to (n,i) is "safe"
        # given the current assignments. But since all c_i are unknown, we use a different approach:
        #
        # For a fixed color c, S_c must be an independent set in the c-colored subgraph.
        # The c-colored subgraph on vertices 0..n-1 is known.
        # So we need to partition {0,...,n-1} into 4 independent sets S_0, S_1, S_2, S_3
        # where S_c is independent in the c-graph.
        #
        # This is a 4-coloring problem on a constraint graph. Let's use backtracking.
        #
        # Optimization: For each vertex i and color c, check if it's "conflict-free" with already-assigned vertices.
        # Process vertices in order of most-constrained first.

        # Precompute: for each color c, the adjacency in the c-graph (as bitmasks for speed)
        # adj[c][i] = bitmask of vertices j > i such that edge (i,j) has color c
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1

        # Assignment: assign[i] = color for edge (n, i), or -1 if unassigned
        assign = [-1] * n

        # For backtracking: for each vertex i, the allowed colors are those c such that
        # no already-assigned vertex j with assign[j] == c is adjacent to i in the c-graph.
        # We can check: (adj[c][i] & assigned_mask[c]) == 0, where assigned_mask[c] is bitmask of vertices assigned color c.

        # Order vertices by degree (most constrained first) - vertices with high degree in the "conflict" sense
        # Actually, let's just try in a heuristic order.

        # Use iterative deepening / simple backtracking with forward checking
        # Since n can be up to ~37, this might be slow. Let's use a smarter approach.

        # Key insight: For the Clebsch graph (K_16 part), each color class is a perfect matching (12 edges, but actually
        # the Clebsch graph has each color class as a 2-regular graph... let me reconsider).
        # Actually in the GF(16) construction, color c = log(x) % 3 where x = i^j.
        # The color classes are not matchings; they form the Clebsch graph structure.

        # Let's just do backtracking with good heuristics.
        # For speed, represent assigned masks as integers.

        def backtrack(pos, masks):
            """pos: next vertex to assign (in the order), masks[c]: bitmask of vertices assigned color c"""
            if pos == n:
                return True
            i = order[pos]
            # Try colors in order of fewest conflicts
            for c in range(4):
                if (adj[c][i] & masks[c]) == 0:
                    # Check: can we still complete? (simple forward check)
                    # Assign
                    masks[c] |= (1 << i)
                    assign[i] = c
                    if backtrack(pos + 1, masks):
                        return True
                    masks[c] &= ~(1 << i)
                    assign[i] = -1
            return False

        # Order: try vertices with highest total degree across all color graphs first
        # (these are most constrained)
        deg = [0] * n
        for c in range(4):
            for i in range(n):
                deg[i] += bin(adj[c][i]).count('1')
        order = sorted(range(n), key=lambda i: -deg[i])

        # Try multiple random orderings if the first fails
        for attempt in range(50):
            if time.time() > deadline:
                return False
            if attempt == 0:
                order = sorted(range(n), key=lambda i: -deg[i])
            else:
                order = list(range(n))
                random.shuffle(order)
            # Re-sort with some randomness for diversity
            if attempt > 0:
                # Perturb: swap some elements
                for _ in range(min(5, n // 4)):
                    a, b = random.randrange(n), random.randrange(n)
                    order[a], order[b] = order[b], order[a]

            masks = [0, 0, 0, 0]
            assign = [-1] * n
            if backtrack(0, masks):
                # Success: append the new colors
                for i in range(n):
                    colors.append(assign[i])
                return True
        return False

    # Try to extend from K_32
    best_n = 32
    best_colors = base_colors

    # Try sequential extension
    cur_n = 32
    cur_colors = list(base_colors)
    for _ in range(10):  # Try up to 10 more vertices
        if time.time() > deadline:
            break
        if _try_add_one_vertex(cur_colors, cur_n):
            cur_n += 1
            best_n = cur_n
            best_colors = list(cur_colors)
        else:
            # Try with different random seeds for the backtracking
            # (already handled inside _try_add_one_vertex)
            break

    # If we couldn't extend sequentially, try a fresh approach:
    # Use a randomized local search to find a valid coloring for larger n
    # This is more complex; for now, return what we have.

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
