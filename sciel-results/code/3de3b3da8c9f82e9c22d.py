import random
import time
import itertools

def construct():
    start = time.time()
    deadline = start + 112
    rng = random.Random(42)

    def build_color_edges(n, colors):
        """Build list of edges per color from flat color list."""
        edges = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges[colors[idx]].add((i, j))
                idx += 1
        return edges

    def build_adj_bitsets(n, color_edges):
        """Build adjacency bitsets: adj[c][v] = bitmask of neighbors of v in color c."""
        adj = [[0] * n for _ in range(4)]
        for c in range(4):
            for (u, v) in color_edges[c]:
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
        return adj

    def has_c6_fast(adj, n):
        """
        Fast C6 detection using bitset intersection.
        A C6: v0-v1-v2-v3-v4-v5-v0 means
        (adj[v0] & adj[v2]) contains v1,
        (adj[v1] & adj[v3]) contains v2, etc.
        
        We enumerate pairs (v0, v2) that have at least 2 common neighbors,
        then for each pair of common neighbors (v1, v1'), check if a path
        v1-v2-v3-v4-v5-v0 exists... 
        
        Actually, simpler: enumerate 6-tuples via common neighbors.
        For a C6 (a,b,c,d,e,f): adj[a]&adj[c] contains b, adj[b]&adj[d] contains c, etc.
        
        Efficient approach: for each pair (a,c) with |N(a)&N(c)|>=2,
        for each pair (b,b2) in N(a)&N(c) with b<b2,
        check if there exists d,e such that b-c-d-e-b2-a forms the cycle.
        That is: N(b)&N(d) contains c (i.e., c in N(b) which is true since b in N(c)), 
        and d in N(c), e in N(d), b2 in N(e), and a in N(b2).
        
        Let me use a cleaner enumeration:
        C6: 0-1-2-3-4-5-0
        Fix 0. For each 1 in N(0), for each 2 in N(1)&N(0) ... no.
        
        Let me just do: for each ordered pair (a,c) with common neighbors,
        for each b in common(a,c): check if there's a path b-c-d-e-f-a of length 4
        from b to a avoiding... this is getting complex.
        
        Simpler: enumerate all 6-subsets and check for Hamiltonian cycle.
        But use bitsets for speed.
        """
        # Precompute: for each vertex v, the bitmask of its neighbors
        # Enumerate C6 via: pick a, then 1 in N(a), 2 in N(1)\{a}, 3 in N(2)\{a,1}, 4 in N(3)\{a,1,2}, 5 in N(4)\{a,1,2,3} and 5 in N(a)
        # This is 6-step path + closing edge.
        # To avoid duplicates, enforce a < b < ... but that's hard.
        # Just check existence: any C6.
        
        # Use: for each pair (a,c) with at least 2 common neighbors
        for a in range(n):
            for c in range(a + 1, n):
                common = adj[a] & adj[c]
                if common.bit_count() < 2:
                    continue
                # For each pair of common neighbors (b, b2)
                common_list = [v for v in range(n) if (common >> v) & 1]
                for i_b in range(len(common_list)):
                    b = common_list[i_b]
                    for i_b2 in range(i_b + 1, len(common_list)):
                        b2 = common_list[i_b2]
                        # Now check if there's a path b-c-d-e-b2 of length 3 (4 vertices: b,c,d,e,b2 with edges b-c, c-d, d-e, e-b2)
                        # b-c is guaranteed (b in N(c)).
                        # Need: d in N(c), e in N(d), b2 in N(e)
                        # So: for each d in N(c)\{a,b,b2}:
                        nc_mask = adj[c]
                        for d in range(n):
                            if d == a or d == b or d == b2:
                                continue
                            if not ((nc_mask >> d) & 1):
                                continue
                            # d in N(c), and c in N(d) (symmetric)
                            # Need e in N(d) & N(b2), e != a, b, c, d
                            nd_nb2 = adj[d] & adj[b2]
                            # Remove a, b, c, d from consideration
                            nd_nb2 &= ~(1 << a)
                            nd_nb2 &= ~(1 << b)
                            nd_nb2 &= ~(1 << c)
                            nd_nb2 &= ~(1 << d)
                            if nd_nb2:
                                # Found: a-b-c-d-e-b2-a ... wait, we need a-b2 edge too
                                # a-b2: b2 in N(a)? Yes, b2 is in common(a,c) so b2 in N(a).
                                # So the cycle is: a - b - c - d - e - b2 - a. 
                                # But wait: is a connected to b? b is in N(a) (b in common(a,c) means b in N(a) and b in N(c)). Yes.
                                # So we have a 6-cycle: a,b,c,d,e,b2 with edges a-b, b-c, c-d, d-e, e-b2, b2-a. All present.
                                return True
        return False

    def find_c6_instance(adj, n):
        """Find one C6 if it exists, return (a,b,c,d,e,f) or None."""
        for a in range(n):
            for c in range(a + 1, n):
                common = adj[a] & adj[c]
                if common.bit_count() < 2:
                    continue
                common_list = [v for v in range(n) if (common >> v) & 1]
                for i_b in range(len(common_list)):
                    b = common_list[i_b]
                    for i_b2 in range(i_b + 1, len(common_list)):
                        b2 = common_list[i_b2]
                        nc_mask = adj[c]
                        for d in range(n):
                            if d == a or d == b or d == b2:
                                continue
                            if not ((nc_mask >> d) & 1):
                                continue
                            nd_nb2 = adj[d] & adj[b2]
                            nd_nb2 &= ~(1 << a)
                            nd_nb2 &= ~(1 << b)
                            nd_nb2 &= ~(1 << c)
                            nd_nb2 &= ~(1 << d)
                            if nd_nb2:
                                e = nd_nb2.bit_length() - 1
                                # Verify
                                if ((adj[a] >> b) & 1) and ((adj[b] >> c) & 1) and \
                                   ((adj[c] >> d) & 1) and ((adj[d] >> e) & 1) and \
                                   ((adj[e] >> b2) & 1) and ((adj[b2] >> a) & 1):
                                    return (a, b, c, d, e, b2)
        return None

    def edge_idx(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)

    def try_fix_c6(colors, n, c6, max_iters=2000):
        """Try to break a C6 by flipping one edge in the cycle to a different color."""
        colors = colors[:]
        a, b, c, d, e, f = c6
        cycle_edges = [(a, b), (b, c), (c, d), (d, e), (e, f), (f, a)]
        # Try flipping each cycle edge to a random different color
        for _ in range(max_iters):
            # Pick a random cycle edge
            u, v = rng.choice(cycle_edges)
            idx = edge_idx(n, u, v)
            old_c = colors[idx]
            # Try a random new color
            new_c = rng.choice([x for x in range(4) if x != old_c])
            colors[idx] = new_c
            # Rebuild and check if the original C6 is broken
            # Just check: is this specific C6 still present?
            # We need to check all 6 edges
            edges_to_check = [(a, b), (b, c), (c, d), (d, e), (e, f), (f, a)]
            broken = False
            for (eu, ev) in edges_to_check:
                ci = edge_idx(n, eu, ev)
                if colors[ci] != old_c:  # only the flipped one might be different
                    pass
            # Actually, let me just rebuild adj and check
            edges_by_color = [set() for _ in range(4)]
            idx2 = 0
            for i2 in range(n):
                for j2 in range(i2 + 1, n):
                    edges_by_color[colors[idx2]].add((i2, j2))
                    idx2 += 1
            adj_new = [[0] * n for _ in range(4)]
            for cc in range(4):
                for (uu, vv) in edges_by_color[cc]:
                    adj_new[cc][uu] |= (1 << vv)
                    adj_new[cc][vv] |= (1 << uu)
            # Check if the original C6 is still in the old color
            still_there = True
            for (eu, ev) in edges_to_check:
                if not ((adj_new[old_c][eu] >> ev) & 1):
                    still_there = False
                    break
            if not still_there:
                # Check if any new C6 was introduced... do a full check
                if not has_c6_fast(adj_new, n):
                    return colors
        return None

    def verify_full(n, colors):
        """Full verification: no C6 in any color class."""
        edges_by_color = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges_by_color[colors[idx]].add((i, j))
                idx += 1
        adj = [[0] * n for _ in range(4)]
        for c in range(4):
            for (u, v) in edges_by_color[c]:
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
        for c in range(4):
            if has_c6_fast(adj, n):
                return False
        return True

    # --- Main search ---
    best_n = 0
    best_colors = None

    for n_target in [18, 17, 16]:
        if time.time() > deadline - 15:
            break
        num_pairs = n_target * (n_target - 1) // 2

        # Try random colorings
        attempts = 0
        while time.time() < deadline - 15 and attempts < 500:
            attempts += 1
            colors = [rng.randint(0, 3) for _ in range(num_pairs)]
            
            # Quick check: build adj and look for C6
            edges_by_color = [set() for _ in range(4)]
            idx = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    edges_by_color[colors[idx]].add((i, j))
                    idx += 1
            adj = [[0] * n_target for _ in range(4)]
            for c in range(4):
                for (u, v) in edges_by_color[c]:
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)

            # Check each color for C6
            has_issue = False
            problem_c = -1
            c6_found = None
            for c in range(4):
                if has_c6_fast(adj, n_target):
                    has_issue = True
                    problem_c = c
                    c6_found = find_c6_instance(adj, n_target)
                    break

            if not has_issue:
                # Valid!
                best_n = n_target
                best_colors = colors
                break

            # Try to fix
            if c6_found is not None:
                fixed = try_fix_c6(colors, n_target, c6_found, max_iters=300)
                if fixed is not None:
                    if verify_full(n_target, fixed):
                        best_n = n_target
                        best_colors = fixed
                        break

        if best_n == n_target:
            break

    # Fallback: deterministic construction for smaller n
    if best_n == 0:
        # n=9 construction: Z_9 with distances
        n_fb = 9
        colors = []
        for i in range(n_fb):
            for j in range(i + 1, n_fb):
                d = (j - i) % n_fb
                d = min(d, n_fb - d)
                # Map distances 1,2,3,4 to colors 0,1,2,3
                colors.append(d - 1)
        # Verify
        if verify_full(n_fb, colors):
            best_n = n_fb
            best_colors = colors
        else:
            # Try n=8
            n_fb = 8
            colors = []
            for i in range(n_fb):
                for j in range(i + 1, n_fb):
                    d = (j - i) % n_fb
                    d = min(d, n_fb - d)
                    colors.append(min(d - 1, 3))
            best_n = n_fb
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}
