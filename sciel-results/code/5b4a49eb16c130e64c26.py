"""
Structural change: exploit the known cyclic (Z_n) structure for n=17.
The best-known 4-coloring of K_17 without monochromatic C6 uses a cyclic
construction where each color class is a union of cyclic distances.
This reduces search from 4^136 to ~C(8,2)^4 ≈ 1.2M combinations.
Combined with fast C6 detection via adjacency-matrix squaring.
"""
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = 138.0

    def has_c6_fast(adj, n):
        """Check if adjacency matrix (list of sets) contains a C6."""
        for a0 in range(n):
            nbrs0 = adj[a0]
            for a1 in nbrs0:
                if a1 <= a0:
                    continue
                nbrs1 = adj[a1]
                for a2 in nbrs1:
                    if a2 <= a0 or a2 == a1:
                        continue
                    nbrs2 = adj[a2]
                    for a3 in nbrs2:
                        if a3 <= a0 or a3 in (a1, a2):
                            continue
                        nbrs3 = adj[a3]
                        for a4 in nbrs3:
                            if a4 <= a0 or a4 in (a1, a2, a3):
                                continue
                            nbrs4 = adj[a4]
                            for a5 in nbrs4:
                                if a5 <= a0 or a5 in (a1, a2, a3, a4):
                                    continue
                                if a0 in adj[a5]:
                                    return True
        return False

    def verify_coloring(colors, n):
        """Verify no color class contains a C6."""
        for c in range(4):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            if has_c6_fast(adj, n):
                return False
        return True

    def cyclic_colors(n, partition):
        """Build coloring from cyclic distance partition.
        partition: list of 4 sets of distances (1..m where m=(n-1)//2).
        Returns list of colors in lexicographic order, or None if invalid partition."""
        m = (n - 1) // 2
        # Verify partition covers all distances 1..m exactly once
        all_dists = set()
        for s in partition:
            for d in s:
                if d in all_dists:
                    return None
                all_dists.add(d)
        if all_dists != set(range(1, m + 1)):
            return None

        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                d = min(d, n - d)
                for c in range(4):
                    if d in partition[c]:
                        colors.append(c)
                        break
                else:
                    return None
        return colors

    def search_cyclic_17(time_limit):
        """Search all cyclic distance partitions for n=17."""
        n = 17
        m = 8  # distances 1..8
        distances = list(range(1, m + 1))

        # Generate all ways to partition {1..8} into 4 (possibly empty) subsets
        # Use stars-and-bars style: assign each distance to one of 4 colors
        # Total: 4^8 = 65536 combinations
        count = 0
        for assign in itertools.product(range(4), repeat=8):
            if time.time() > time_limit:
                return None

            count += 1
            # Build partition
            partition = [set() for _ in range(4)]
            for d_idx, c in enumerate(assign):
                partition[c].add(d_idx + 1)

            # Quick check: each color class should have reasonable size
            # For C6-free, each color class on 17 vertices should be sparse enough
            # A C6 requires at least 6 vertices with edges. If a color has too few
            # edges, it can't have C6. But we need to check all.

            colors = cyclic_colors(n, partition)
            if colors is None:
                continue

            # Verify no C6 in any color class
            valid = True
            for c in range(4):
                adj = [set() for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == c:
                            adj[i].add(j)
                            adj[j].add(i)
                        idx += 1
                if has_c6_fast(adj, n):
                    valid = False
                    break

            if valid:
                return colors

        return None

    def search_cyclic_18(time_limit):
        """Try cyclic coloring for n=18 (even, so use Z_18 with distances 1..8)."""
        n = 18
        m = 9  # distances 1..9
        # For even n, distance n/2=9 is special (each vertex has exactly one vertex at that distance)
        # 4^9 = 262144 combinations
        count = 0
        for assign in itertools.product(range(4), repeat=9):
            if time.time() > time_limit:
                return None

            count += 1
            partition = [set() for _ in range(4)]
            for d_idx, c in enumerate(assign):
                partition[c].add(d_idx + 1)

            colors = cyclic_colors(n, partition)
            if colors is None:
                continue

            valid = True
            for c in range(4):
                adj = [set() for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == c:
                            adj[i].add(j)
                            adj[j].add(i)
                        idx += 1
                if has_c6_fast(adj, n):
                    valid = False
                    break

            if valid:
                return colors

        return None

    def greedy_extend(n_target, time_limit):
        """Greedy/backtracking hybrid: build coloring edge by edge with forward checking."""
        num_edges = n_target * (n_target - 1) // 2
        edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        assignment = [-1] * num_edges
        adj_c = [[set() for _ in range(n_target)] for _ in range(4)]

        def would_create_c6(c, u, v):
            """Check if adding edge (u,v) to color c creates a C6."""
            # Need path of length 4 from u to v in color c (excluding the new edge)
            # u - a - b - d - e - v, all distinct
            visited = {u, v}
            # Level 1: neighbors of u in color c (not v)
            level1 = [a for a in adj_c[c][u] if a != v]
            if not level1:
                return False
            visited.update(level1)

            # Level 2
            level2 = set()
            for a in level1:
                for b in adj_c[c][a]:
                    if b not in visited:
                        level2.add(b)
            if not level2:
                return False
            visited.update(level2)

            # Level 3
            level3 = set()
            for b in level2:
                for d in adj_c[c][b]:
                    if d not in visited:
                        level3.add(d)
            if not level3:
                return False
            visited.update(level3)

            # Level 4: check if any neighbor of level3 is v
            for d in level3:
                for e in adj_c[c][d]:
                    if e == v:
                        return True
            return False

        # Recursive backtracking with symmetry breaking
        import sys
        sys.setrecursionlimit(10000)

        def solve(pos):
            if time.time() > time_limit:
                return False
            if pos == num_edges:
                return True

            u, v = edges[pos]
            for c in range(4):
                if would_create_c6(c, u, v):
                    continue
                assignment[pos] = c
                adj_c[c][u].add(v)
                adj_c[c][v].add(u)
                if solve(pos + 1):
                    return True
                assignment[pos] = -1
                adj_c[c][u].discard(v)
                adj_c[c][v].discard(u)
            return False

        # Symmetry breaking: first edge (0,1) = color 0
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)

        if solve(1):
            return assignment[:]
        return None

    # Strategy:
    # 1. Try cyclic search for n=18 (262K combos, should be fast)
    # 2. Try cyclic search for n=17 (65K combos)
    # 3. Fallback to backtracking for n=17
    # 4. Fallback to known constructions

    # Try n=18 cyclic
    if time.time() < deadline - 60:
        colors = search_cyclic_18(deadline - 60)
        if colors is not None and verify_coloring(colors, 18):
            return {'n': 18, 'colors': colors}

    # Try n=17 cyclic
    if time.time() < deadline - 30:
        colors = search_cyclic_17(deadline - 30)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}

    # Try n=17 backtracking (limited time)
    if time.time() < deadline - 15:
        colors = greedy_extend(17, deadline - 10)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}

    # Try n=16 cyclic (distances 1..7, 4^7=16K)
    if time.time() < deadline - 10:
        n = 16
        m = 7
        for assign in itertools.product(range(4), repeat=7):
            if time.time() > deadline - 5:
                break
            partition = [set() for _ in range(4)]
            for d_idx, c in enumerate(assign):
                partition[c].add(d_idx + 1)
            colors = cyclic_colors(n, partition)
            if colors is None:
                continue
            if verify_coloring(colors, n):
                return {'n': n, 'colors': colors}

    # Fallback: K9 construction
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
