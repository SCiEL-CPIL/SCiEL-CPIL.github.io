import time
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit

    n = 256

    # Build graph: edge (a,b) if (a*b) % (a+b) == 0
    adj_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << (b - 1))
                adj_mask[b] |= (1 << (a - 1))

    # Complement graph for Max Clique: comp_adj_mask[v] has bit (u-1) set if u!=v and (u,v) NOT an edge in G
    full_mask = (1 << n) - 1
    comp_adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        comp_adj_mask[v] = full_mask ^ adj_mask[v] ^ (1 << (v - 1))

    def popcount(x):
        return bin(x).count('1')

    # ---------- Initial solution: greedy + local search ----------
    def greedy_initial():
        conflict_deg = [0] * (n + 1)
        for v in range(1, n + 1):
            conflict_deg[v] = popcount(adj_mask[v])
        init_set = []
        init_mask = 0
        for v in sorted(range(1, n + 1), key=lambda x: (conflict_deg[x], x)):
            if (adj_mask[v] & init_mask) == 0:
                init_set.append(v)
                init_mask |= (1 << (v - 1))
        return init_set

    best_set = greedy_initial()
    best_size = len(best_set)
    best_mask = 0
    for v in best_set:
        best_mask |= (1 << (v - 1))

    # Local search: try to add any excluded vertex that is compatible with all in best_set
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if (best_mask >> (v - 1)) & 1:
                continue
            if (adj_mask[v] & best_mask) == 0:
                best_set.append(v)
                best_mask |= (1 << (v - 1))
                best_size += 1
                improved = True

    # Also try: all odds as a starting point
    odds = [v for v in range(1, n + 1) if v % 2 == 1]
    odds_valid = True
    for i in range(len(odds)):
        for j in range(i + 1, len(odds)):
            a, b = odds[i], odds[j]
            if (a * b) % (a + b) == 0:
                odds_valid = False
                break
        if not odds_valid:
            break
    if odds_valid and len(odds) > best_size:
        best_set = odds[:]
        best_size = len(odds)
        best_mask = 0
        for v in best_set:
            best_mask |= (1 << (v - 1))
        improved = True
        while improved:
            improved = False
            for v in range(1, n + 1):
                if (best_mask >> (v - 1)) & 1:
                    continue
                if (adj_mask[v] & best_mask) == 0:
                    best_set.append(v)
                    best_mask |= (1 << (v - 1))
                    best_size += 1
                    improved = True

    # ---------- Tomita MAXCLQ on complement graph ----------
    # We want to find the maximum clique in comp_adj_mask, which equals the MIS in adj_mask.

    def get_vertices_by_degree(mask):
        """Extract vertices from mask, sorted by degree in induced subgraph descending."""
        verts = []
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            verts.append(v)
            temp ^= lsb
        verts.sort(key=lambda v: popcount(comp_adj_mask[v] & mask), reverse=True)
        return verts

    # Global state for the search
    best_clique = best_set[:]
    best_clique_size = best_size

    # Tomita MAXCLQ with greedy coloring bound
    # P_list: list of candidate vertices, already sorted by degree descending.
    # clique: current clique (list of vertices).

    def expand(P_list, clique_size, clique):
        nonlocal best_clique, best_clique_size

        if time.time() - start_time > timeout:
            return

        if not P_list:
            if clique_size > best_clique_size:
                best_clique_size = clique_size
                best_clique = clique[:]
            return

        # Greedy coloring of P_list
        # We color in the order of P_list (which is sorted by degree descending).
        # color_of[v] = color assigned to v (1-indexed)
        color_of = {}
        colored_mask = 0
        max_color = 0
        v_colors = []  # list of (v, color)

        for v in P_list:
            # Find colors used by neighbors of v in the complement graph that are already colored
            neighbor_mask = comp_adj_mask[v] & colored_mask
            used_colors = set()
            temp = neighbor_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length()
                if u in color_of:
                    used_colors.add(color_of[u])
                temp ^= lsb

            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
            v_colors.append((v, c))
            colored_mask |= (1 << (v - 1))

        # Sort by color ascending, then by degree descending (for tie-breaking)
        # This is crucial for the Tomita bound to work effectively.
        v_colors.sort(key=lambda x: (x[1], -popcount(comp_adj_mask[x[0]] & colored_mask)))

        m = len(v_colors)
        # Precompute suffix masks: suffix_masks[i] = bitmask of vertices in v_colors[i:]
        suffix_masks = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            v_i = v_colors[i][0]
            suffix_masks[i] = suffix_masks[i + 1] | (1 << (v_i - 1))

        # Branch: iterate through v_colors in order
        for idx in range(m):
            if time.time() - start_time > timeout:
                return

            v, c = v_colors[idx]

            # Tomita bound:
            # A clique can contain at most one vertex from each color class.
            # The remaining vertices (idx..m-1) have colors >= c (since sorted by color).
            # Thus, the maximum number of additional vertices we can add from the remaining
            # set is at most (max_color - c + 1).
            # However, the standard Tomita bound used in many implementations is:
            # if clique_size + c <= best_clique_size: break
            # Why is this valid?
            # In the sorted list, vertex at index idx has color c.
            # The vertices from idx to m-1 have colors in the range [c, max_color].
            # The number of distinct colors in this range is max_color - c + 1.
            # But wait, the standard bound is often cited as clique_size + color_of[v] <= best.
            # Let's verify.
            # If we are at vertex v with color c, we are considering adding v.
            # The subsequent vertices have colors >= c.
            # The maximum size of a clique in the subgraph induced by v_colors[idx:] is
            # bounded by the number of colors used in that subgraph, which is max_color - c + 1.
            # So the bound should be: clique_size + (max_color - c + 1) <= best_clique_size => break.
            #
            # Let's check the inspiration code:
            # if clique_size + 1 + (max_color - c) <= best_size: break
            # This is equivalent to clique_size + (max_color - c + 1) <= best_size.
            # This is the correct tighter bound.
            #
            # The current program uses: if clique_size + c <= best_clique_size: break
            # This is a looser bound (since c <= max_color - c + 1 is not always true, but usually c is small).
            # Actually, c is the color of the *current* vertex. The remaining vertices have colors >= c.
            # The maximum clique size from the remaining vertices is at most the number of distinct colors
            # present in the remaining vertices, which is max_color - c + 1.
            # So the tighter bound is: clique_size + (max_color - c + 1) <= best_clique_size.
            #
            # Let's use the tighter bound from the inspiration.
            if clique_size + (max_color - c + 1) <= best_clique_size:
                break

            # Branch: include v
            new_P_mask = comp_adj_mask[v] & suffix_masks[idx + 1]
            if new_P_mask:
                new_P_list = get_vertices_by_degree(new_P_mask)
                expand(new_P_list, clique_size + 1, clique + [v])
            else:
                if clique_size + 1 > best_clique_size:
                    best_clique_size = clique_size + 1
                    best_clique = clique + [v]

    # Initial P: all vertices sorted by degree in complement graph
    initial_P_list = get_vertices_by_degree(full_mask)
    expand(initial_P_list, 0, [])

    if not best_clique:
        best_clique = [1]

    return list(best_clique)
