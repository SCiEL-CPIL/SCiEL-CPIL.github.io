import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # GF(4) arithmetic: represent as 2-bit integers
    # GF(4) = {0, 1, 2, 3} with 2=alpha, 3=alpha+1
    gf4_mult = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):2,
        (3,0):0, (3,1):3, (3,2):2, (3,3):1,
    }
    gf4_inv = {1: 1, 2: 3, 3: 2}

    def gf4_add(a, b):
        return a ^ b

    def gf4_mul(a, b):
        return gf4_mult[(a, b)]

    def gf4_div(a, b):
        return gf4_mul(a, gf4_inv[b])

    # Construction for n=16: vertices are elements of GF(4)^2
    n = 16
    colors = []
    for i in range(n):
        ai, bi = i // 4, i % 4
        for j in range(i + 1, n):
            aj, bj = j // 4, j % 4
            d1 = gf4_add(ai, aj)
            d2 = gf4_add(bi, bj)

            if d1 == 0 and d2 == 0:
                c = 0
            elif d1 == 0:
                c = 0  # horizontal
            elif d2 == 0:
                c = 1  # vertical
            else:
                slope = gf4_div(d2, d1)
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                else:  # slope == 3
                    c = 4
            colors.append(c)

    # Verify C4-freeness
    def has_c4(color_val, n, colors):
        """Check if any color class contains a C4."""
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color_val:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1

        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return True
        return False

    def verify_valid(n, colors):
        for c in range(5):
            if has_c4(c, n, colors):
                return False
        return True

    # Fallback construction for n=11
    def construct_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return n, colors

    # First, verify the GF(4) construction
    if verify_valid(n, colors):
        return {'n': n, 'colors': colors}

    # Greedy randomized construction with forbidden-pair tracking
    def greedy_forbidden_pair(target_n, max_restarts=200):
        """
        Greedy construction that tracks 'forbidden pairs' per color.
        A forbidden pair (u,v) in color c means u and v have exactly 1 common neighbor in color c.
        Adding edge (u,v) in color c is forbidden if (u,v) is already a forbidden pair in color c.
        Otherwise, adding it creates new forbidden pairs: for each w adjacent to u in color c,
        (v,w) becomes a forbidden pair (and vice versa).
        """
        import random

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            random.seed(42 + restart * 7)
            n = target_n
            color_of = {}  # (i,j) -> color, i<j

            # adj[c][v] = set of neighbors of v in color c
            adj = [[set() for _ in range(n)] for _ in range(5)]

            # forbidden[c][(u,v)] with u<v = True if u,v have exactly 1 common neighbor in color c
            # We track count of common neighbors per pair per color
            # For efficiency, track common_count[c][u][v] implicitly via sets
            # Actually, let's track: for each color c, for each vertex v, 
            #   adj[c][v] = set of neighbors
            # When adding (u,v) in color c:
            #   For each w in adj[c][u]: pair (v,w) gains a common neighbor (u) in color c
            #   For each w in adj[c][j]: pair (u,w) gains a common neighbor (v) in color c
            #   Pair (u,v) itself: common neighbors = adj[c][u] & adj[c][j]
            #   If |adj[c][u] & adj[c][j]| >= 1, then (u,v) would have >= 1 common neighbors
            #   We need |adj[c][u] & adj[c][j]| == 0 to add (u,v) in color c
            #   After adding, for each w in adj[c][u], (v,w) gets one more common neighbor
            #   For each w in adj[c][j], (u,w) gets one more common neighbor

            # Better approach: track common_neighbor_count[c][u][v] = number of common neighbors
            # But that's O(n^2) per color = O(5*n^2) space. For n=29, that's 5*841=4205 entries, fine.
            # But updating is O(deg) per edge addition.

            # Let's use: cn[c][u] = dict mapping v -> count of common neighbors of u and v in color c
            # Actually simpler: just use sets and compute intersection on the fly for the check,
            # but track which pairs are "at risk" (have exactly 1 common neighbor).

            # Most efficient: maintain for each color c:
            #   adj[c][v] = set of neighbors
            #   When checking if we can add (u,v) in color c:
            #     common = adj[c][u] & adj[c][v]
            #     if len(common) >= 1: forbidden (would create C4 or already has common neighbor)
            #     Wait, no. If they have 0 common neighbors, adding (u,v) in color c is fine.
            #     If they have 1 common neighbor, adding (u,v) in color c creates a C4.
            #     If they have >= 2, already invalid.
            #   So we need: len(adj[c][u] & adj[c][v]) == 0

            # After adding (u,v) in color c:
            #   adj[c][u].add(v), adj[c][v].add(u)
            #   For each w in adj[c][u] (before adding v): 
            #     (w, v) now has one more common neighbor (u) in color c
            #   For each w in adj[c][v] (before adding u):
            #     (w, u) now has one more common neighbor (v) in color c

            # The key insight for the greedy: we want to minimize the number of "new" 
            # pairs that will have exactly 1 common neighbor (forbidden pairs) after the addition.
            # When we add (u,v) in color c:
            #   - (u,v) itself: common neighbors = adj[c][u] & adj[c][v] (must be 0)
            #   - For each w in adj[c][u] \ {v}: pair (w,v) gains common neighbor u
            #     If (w,v) previously had 0 common neighbors, it now has 1 -> becomes forbidden
            #     If (w,v) previously had 1 common neighbor, it now has 2 -> C4! (but we checked (u,v) has 0 common, so this shouldn't happen for w in adj[c][u] because w and v sharing u as common neighbor means... wait)
            
            # Actually let me reconsider. The constraint is: no pair can have >= 2 common neighbors in any color.
            # So when adding (u,v) in color c, we need:
            #   1. adj[c][u] & adj[c][v] must be empty (otherwise u,v would have >= 1 common neighbor, 
            #      and after adding the edge, they'd have exactly |intersection| common neighbors, which must be 0)
            #      Wait, the common neighbors of u and v in color c are vertices w such that (u,w) and (v,w) are both color c.
            #      Adding edge (u,v) doesn't change the common neighbors of u and v; it just adds an edge between them.
            #      So the check is: len(adj[c][u] & adj[c][v]) must be 0.
            
            #   2. For each w in adj[c][u] (w != v): after adding (u,v), the pair (w,v) now has u as a common neighbor.
            #      Previously, (w,v) had some number of common neighbors. If it had 0, now it has 1 (ok, becomes "at risk").
            #      If it had 1, now it has 2 (C4! invalid).
            #      So we need: for all w in adj[c][u], the pair (w,v) must have had 0 common neighbors in color c.
            #      But wait, (w,v) having u as a common neighbor means (w,u) and (v,u) are both color c.
            #      (w,u) is color c because w in adj[c][u]. (v,u) is color c because we're adding it.
            #      So u IS a common neighbor of w and v in color c.
            #      For this not to create a C4, (w,v) must not have any OTHER common neighbor in color c.
            #      i.e., (adj[c][w] & adj[c][v]) \ {u} must be empty, i.e., adj[c][w] & adj[c][v] ⊆ {u}.
            #      Since w in adj[c][u], we have (w,u) is color c, so u in adj[c][w].
            #      So adj[c][w] & adj[c][v] can contain u and possibly others.
            #      We need adj[c][w] & adj[c][v] = {u} at most, i.e., |adj[c][w] & adj[c][v]| <= 1.
            #      But actually, we need it to be exactly {u} or empty before adding, and after adding (u,v),
            #      it becomes {u} (since u is now also adjacent to v).
            #      Hmm, this is getting complex. Let me just track common neighbor counts.

            # Let me use a cleaner approach with numpy for the common neighbor counts.
            # cn[c] is an n x n matrix where cn[c][u][v] = number of common neighbors of u,v in color c.
            # Symmetric, cn[c][u][u] = 0.
            # When adding edge (u,v) in color c:
            #   - Check cn[c][u][v] == 0 (no existing common neighbors)
            #   - For all w: cn[c][u][w] += 1 if v is... no wait.
            #   Adding (u,v) in color c means:
            #     For any w, w is a common neighbor of u and v iff (u,w) and (v,w) are both color c.
            #     Adding (u,v) doesn't change common neighbors of (u,v) with others directly.
            #     But it DOES change: for any w, the pair (w,v) now has u as a potential common neighbor
            #     (if (w,u) is color c). Similarly (w,u) now has v as potential common neighbor.
            #     
            #     Specifically:
            #     - cn[c][u][v]: was 0 (checked), stays 0 (adding edge between u,v doesn't add common neighbors to u,v)
            #       Wait, actually cn[c][u][v] counts vertices w such that (u,w) and (v,w) are color c.
            #       Adding (u,v) doesn't change this count. So cn[c][u][v] remains 0. Good.
            #     - For w != u,v: 
            #       cn[c][w][u]: counts vertices x such that (w,x) and (u,x) are color c.
            #       Adding (u,v): if (w,v) is color c, then v becomes a new common neighbor of w and u.
            #       So cn[c][w][u] += 1 if (w,v) is color c, i.e., if v in adj[c][w].
            #       Similarly cn[c][w][v] += 1 if (w,u) is color c, i.e., if u in adj[c][w].
            #       Also cn[c][u][v] += 1 if... no, (u,v) edge doesn't affect cn[c][u][v] count.
            #       
            #     So the updates are:
            #       For each w in adj[c][u] (w != v): cn[c][w][v] += 1  (because u is now a common neighbor of w and v via (w,u) and (v,u))
            #         Wait, (w,u) is color c (w in adj[c][u]), and (v,u) is now color c (we just added it).
            #         So u is a common neighbor of w and v. Thus cn[c][w][v] += 1.
            #       For each w in adj[c][v] (w != u): cn[c][u][w] += 1  (because v is now a common neighbor of u and w)
            #       cn[c][u][v] stays the same (0).
            #       
            #     And the validity check: after updates, all cn[c][i][j] must be <= 1.
            #     So before adding, we need:
            #       - cn[c][u][v] == 0
            #       - For all w in adj[c][u] (w != v): cn[c][w][v] == 0 (so after +1, it becomes 1, ok)
            #       - For all w in adj[c][v] (w != u): cn[c][u][w] == 0 (so after +1, it becomes 1, ok)
            #       - Also need: for w in adj[c][u] ∩ adj[c][v] (w != u,v): 
            #         cn[c][w][v] would get +1 (from w in adj[c][u]) AND cn[c][u][w] would get +1 (from w in adj[c][v])
            #         But these are different pairs: (w,v) and (u,w).
            #         For pair (w,v): cn[c][w][v] must be 0 before, becomes 1.
            #         For pair (u,w): cn[c][u][w] must be 0 before, becomes 1.
            #         These are independent constraints.
            #       - Also, what about pair (u,v) itself? cn[c][u][v] must be 0, stays 0.
            #       - What about the case where w in adj[c][u] and w in adj[c][v]? 
            #         Then u and v both have w as neighbor. But that's fine as long as cn constraints hold.

            # So the check for adding (u,v) in color c:
            #   1. cn[c][u][v] == 0
            #   2. For all w in adj[c][u] (w != v): cn[c][w][v] == 0
            #   3. For all w in adj[c][v] (w != u): cn[c][u][w] == 0
            #   
            #   And the "risk" (number of new forbidden pairs created) is:
            #   |adj[c][u] \ {v}| + |adj[c][v] \ {u}| = deg_c(u) + deg_c(v) - 2*(1 if u in adj[c][v] else 0)
            #   Wait, u is not in adj[c][v] before adding (since we're adding it). So:
            #   risk = deg_c(u) + deg_c(v)
            #   Each of these creates one new "at-risk" pair (going from 0 to 1 common neighbors).

            # Let me implement this.

            # cn[c] as numpy array for speed
            cn = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]
            adj = [[set() for _ in range(n)] for _ in range(5)]

            valid = True
            # Process edges in a random order
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edge_order = list(range(len(edges)))
            random.shuffle(edge_order)

            for idx in edge_order:
                if time.time() > deadline - 5:
                    valid = False
                    break

                u, v = edges[idx]
                best_color = -1
                best_risk = float('inf')

                for c in range(5):
                    # Check 1: cn[c][u][v] == 0
                    if cn[c][u][v] != 0:
                        continue

                    # Check 2: for all w in adj[c][u] (w != v): cn[c][w][v] == 0
                    # Since v is not yet in adj[c][u], all w in adj[c][u] are != v
                    ok = True
                    for w in adj[c][u]:
                        if cn[c][w][v] != 0:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Check 3: for all w in adj[c][v] (w != u): cn[c][u][w] == 0
                    # Since u is not yet in adj[c][v], all w in adj[c][v] are != u
                    for w in adj[c][v]:
                        if cn[c][u][w] != 0:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Valid color. Compute risk.
                    risk = len(adj[c][u]) + len(adj[c][v])
                    if risk < best_risk:
                        best_risk = risk
                        best_color = c

                if best_color == -1:
                    valid = False
                    break

                c = best_color
                color_of[(u, v)] = c

                # Update cn and adj
                # For each w in adj[c][u]: cn[c][w][v] += 1 and cn[c][v][w] += 1
                for w in adj[c][u]:
                    cn[c][w][v] += 1
                    cn[c][v][w] += 1
                # For each w in adj[c][v]: cn[c][u][w] += 1 and cn[c][w][u] += 1
                for w in adj[c][v]:
                    cn[c][u][w] += 1
                    cn[c][w][u] += 1

                adj[c][u].add(v)
                adj[c][v].add(u)

            if not valid:
                continue

            # Build the colors list in lexicographic order
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors_list.append(color_of[(i, j)])

            # Verify
            if verify_valid(n, colors_list):
                return n, colors_list

        return None

    # Try greedy forbidden-pair construction for n=29 down to 17
    for target_n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
        # Use more restarts for smaller n (easier), fewer for larger n
        max_r = 500 if target_n <= 20 else 200
        result = greedy_forbidden_pair(target_n, max_restarts=max_r)
        if result is not None:
            n_found, colors_found = result
            return {'n': n_found, 'colors': colors_found}

    # Final fallback
    n_fb, colors_fb = construct_fallback()
    return {'n': n_fb, 'colors': colors_fb}

# EVOLVE-BLOCK-END
