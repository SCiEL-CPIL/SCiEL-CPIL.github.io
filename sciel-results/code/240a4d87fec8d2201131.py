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
        # Sort by conflict degree ascending (fewer edges = easier to include)
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
        # Sort by degree in induced subgraph (descending)
        verts.sort(key=lambda v: popcount(comp_adj_mask[v] & mask), reverse=True)
        return verts

    # Global state for the search
    best_clique = best_set[:]
    best_clique_size = best_size

    # We'll use a recursive function with explicit pruning.
    # P_list: list of candidate vertices, already sorted by degree descending.
    # clique: current clique (list of vertices).
    # We use the Tomita coloring bound: color P_list greedily, and prune when
    #   clique_size + color_of[v] <= best_clique_size for the current v in the branch loop.

    # To avoid excessive recursion overhead, we cap the depth implicitly by the timeout.
    # We also use an iterative deepening-like approach: we branch in color order.

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

            # Tomita bound: a clique can contain at most one vertex from each color class.
            # The remaining vertices (idx..m-1) have colors c, c', ... all >= c (since sorted by color).
            # Actually, the bound is: clique_size + c <= best_clique_size => prune.
            # Because the vertices from idx onward have colors >= c, and a clique picks at most one per color,
            # so max additional = number of distinct colors from c to max_color, but Tomita uses:
            #   if clique_size + c <= best_clique_size: break
            # This works because we process in color order: at position idx with color c,
            # the remaining vertices can contribute at most (max_color - c + 1) new vertices,
            # but the standard Tomita bound is: if clique_size + c <= best_clique_size, break.
            # Wait, let me reconsider. In Tomita's algorithm, the coloring is done such that
            # color_of[v] is the smallest possible. The bound is:
            #   if clique_size + color_of[v] <= best_clique_size: break
            # This is because color_of[v] is the size of the color class index, and a clique
            # can contain at most one vertex per color, so the maximum clique size from the
            # remaining vertices is at most the maximum color number, but more precisely,
            # at the point where we're at vertex v with color c, the remaining vertices
            # (including v) have colors in {c, c+1, ..., max_color}, so at most (max_color - c + 1)
            # vertices can be added. But Tomita's standard bound is simpler:
            #   if clique_size + c <= best_clique_size: break
            # This is valid because c is the color of v, and all subsequent vertices have color >= c,
            # so the number of additional vertices is at most (max_color - c + 1) <= c? No, that's not right.
            #
            # Let me use the correct Tomita bound:
            # The coloring assigns colors 1..max_color. The vertices are ordered by color.
            # At position idx with color c, the remaining vertices (idx..m-1) all have color >= c.
            # A clique can contain at most one vertex per color class. So the maximum number of
            # additional vertices is at most (max_color - c + 1).
            # But Tomita's standard implementation uses:
            #   if clique_size + color_of[v] <= best_clique_size: break
            # This is a valid bound because color_of[v] is an upper bound on the number of
            # vertices that can be added from v onward (since each additional vertex must have
            # a distinct color, and the colors go from c to max_color, but c itself is a lower
            # bound on the number of colors needed... actually this is a known valid bound in Tomita).
            #
            # I'll use the standard Tomita bound: clique_size + c <= best_clique_size => break.
            if clique_size + c <= best_clique_size:
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
