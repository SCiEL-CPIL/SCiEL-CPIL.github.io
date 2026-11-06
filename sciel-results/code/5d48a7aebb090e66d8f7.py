import time
import math
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * (2 * n - 1 - u) // 2 + (v - u - 1)

    def build_edge_map(n):
        em = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                em[(i, j)] = idx
                idx += 1
        return em

    def count_c6(colors, adj, n):
        """Count monochromatic 6-cycles (each cycle counted 12 times)."""
        total = 0
        for c in range(4):
            for a in range(n):
                for b in range(n):
                    if not adj[c][a][b]:
                        continue
                    for cc in range(n):
                        if cc == a or cc == b or not adj[c][b][cc]:
                            continue
                        for d in range(n):
                            if d in (a, b, cc) or not adj[c][cc][d]:
                                continue
                            for e in range(n):
                                if e in (a, b, cc, d) or not adj[c][d][e]:
                                    continue
                                if adj[c][e][a] and e != b:
                                    total += 1
        return total

    def spectral_embedding(n, iters=30):
        """Embed K_n vertices via Laplacian eigenvectors (spectral clustering)."""
        rng = random.Random(12345)
        # Start with a random 2D embedding, then run a few Laplacian diffusion steps
        # to spread vertices. We use a simple iterative "repulsion + attraction"
        # on the complete graph to get a reasonable 2D layout.
        pts = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
        # Run Laplacian-like diffusion: each vertex moves toward mean of all others
        # but is repelled to maintain spread (simulating eigenvector spread).
        for _ in range(iters):
            new_pts = []
            for i in range(n):
                # mean of all other points
                mx = sum(pts[j][0] for j in range(n) if j != i) / (n - 1)
                my = sum(pts[j][1] for j in range(n) if j != i) / (n - 1)
                # repulsion: push away from centroid
                cx = sum(p[0] for p in pts) / n
                cy = sum(p[1] for p in pts) / n
                dx = pts[i][0] - cx
                dy = pts[i][1] - cy
                new_pts.append([mx * 0.5 + dx * 0.5, my * 0.5 + dy * 0.5])
            pts = new_pts
        # Normalize
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        max_r = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in pts) or 1.0
        pts = [[(p[0] - cx) / max_r, (p[1] - cy) / max_r] for p in pts]
        return pts

    def quadrant_color(pts, n):
        """Assign colours by quadrant of the spectral embedding."""
        colors = [0] * (n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                # Use the angle of the vector from i to j relative to i's position
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                angle = math.atan2(dy, dx)
                # Map angle to 4 quadrants
                if angle < 0:
                    angle += 2 * math.pi
                colors[edge_index(i, j)] = int(angle / (2 * math.pi) * 4) % 4
        return colors

    def local_search(colors, n, time_budget):
        """Edge-flip local search that only accepts flips that never create a C6."""
        m = n * (n - 1) // 2
        # Build adjacency
        adj = [[[[False] * n for _ in range(n)] for _ in range(4)]]
        for c in range(4):
            adj[c] = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index(i, j)]
                adj[c][i][j] = True
                adj[c][j][i] = True

        def creates_c6(c, i, j, adj_c):
            """Check if adding edge (i,j) to colour c creates a C6."""
            # For each path of length 5 between i and j in adj_c (excluding edge (i,j) if present)
            # Actually simpler: check if there's a path of length 5 from i to j in adj_c
            # that doesn't use (i,j). If yes, adding (i,j) creates a C6.
            # BFS from i, find shortest path to j. If shortest path length is 5 and
            # (i,j) not already in adj_c, then adding creates C6.
            # But we need ANY path of length exactly 5 (not just shortest).
            # For n<=20, we can do a depth-limited DFS.
            # Simpler: count paths of length 5 from i to j in adj_c.
            # If count > 0 and (i,j) not in adj_c, then adding creates C6.
            # Actually if (i,j) IS in adj_c, we're removing it, so no C6 created.
            if adj_c[i][j]:
                return False  # removing edge, can't create C6
            # Count paths of length 5 from i to j
            # Use dynamic programming: dp[k][v] = number of paths of length k from i to v
            dp = [[0] * n for _ in range(6)]
            dp[0][i] = 1
            for k in range(5):
                for v in range(n):
                    if dp[k][v] == 0:
                        continue
                    for w in range(n):
                        if adj_c[v][w] and w != v:
                            dp[k + 1][w] += dp[k][v]
            return dp[5][j] > 0

        def count_bad(c, adj_c):
            """Count C6s in colour class c."""
            total = 0
            for a in range(n):
                for b in range(n):
                    if not adj_c[a][b]:
                        continue
                    for cc in range(n):
                        if cc == a or cc == b or not adj_c[b][cc]:
                            continue
                        for d in range(n):
                            if d in (a, b, cc) or not adj_c[cc][d]:
                                continue
                            for e in range(n):
                                if e in (a, b, cc, d) or not adj_c[d][e]:
                                    continue
                                if adj_c[e][a] and e != b:
                                    total += 1
            return total

        best_colors = colors[:]
        best_bad = sum(count_bad(c, adj[c]) for c in range(4))
        T = 1.0
        cool_rate = 0.9995
        start_ls = time.time()

        for step in range(200000):
            if time.time() > start_ls + time_budget:
                break
            if time.time() > deadline - 10:
                break

            # Pick random edge
            e = random.randint(0, m - 1)
            # Find (i,j) for edge e
            # Inverse of edge_index
            # e = i*(2n-1-i)//2 + (j-i-1)
            # Solve for i: i*(2n-1-i)//2 <= e < (i+1)*(2n-1-i-1)//2
            # Simpler: precompute
            # Actually let's just iterate (slow but n<=20, m<=190)
            # Precompute edge list
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
            i, j = edge_list[e]

            c_old = colors[e]
            c_new = random.choice([c for c in range(4) if c != c_old])

            # Check if flip creates a C6 in c_new
            if creates_c6(c_new, i, j, adj[c_new]):
                continue

            # Remove edge from c_old, add to c_new
            adj[c_old][i][j] = False
            adj[c_old][j][i] = False
            adj[c_new][i][j] = True
            adj[c_new][j][i] = True
            colors[e] = c_new

            # Recount bad (expensive, but n<=20)
            new_bad = sum(count_bad(c, adj[c]) for c in range(4))
            delta = new_bad - best_bad

            if delta < 0 or (delta == 0 and random.random() < 0.5):
                best_bad = new_bad
                best_colors = colors[:]
                if best_bad == 0:
                    break
            else:
                # Revert
                adj[c_new][i][j] = False
                adj[c_new][j][i] = False
                adj[c_old][i][j] = True
                adj[c_old][j][i] = True
                colors[e] = c_old

            T *= cool_rate

        return best_colors, best_bad

    def verify_no_c6(colors, n):
        """Exact check: no colour class contains a C6."""
        adj = [[False] * n for _ in range(n)]
        for c in range(4):
            adj_c = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[edge_index(i, j)] == c:
                        adj_c[i][j] = True
                        adj_c[j][i] = True
            for a in range(n):
                for b in range(n):
                    if not adj_c[a][b]:
                        continue
                    for cc in range(n):
                        if cc == a or cc == b or not adj_c[b][cc]:
                            continue
                        for d in range(n):
                            if d in (a, b, cc) or not adj_c[cc][d]:
                                continue
                            for e in range(n):
                                if e in (a, b, cc, d) or not adj_c[d][e]:
                                    continue
                                if adj_c[e][a] and e != b:
                                    return False  # Found C6
        return True

    def has_c6_in_cayley(D_set, p):
        """Check if Cayley graph on Z_p with difference set D_set contains a C6."""
        D = set(D_set)
        negD = {(p - d) % p for d in D}
        for v1 in D:
            for v2 in [(v1 + d) % p for d in D]:
                if v2 == 0 or v2 == v1:
                    continue
                for v3 in [(v2 + d) % p for d in D]:
                    if v3 in (0, v1, v2):
                        continue
                    for v4 in [(v3 + d) % p for d in D]:
                        if v4 in (0, v1, v2, v3):
                            continue
                        for v5 in [(v4 + d) % p for d in D]:
                            if v5 in (0, v1, v2, v3, v4):
                                continue
                            if (-v5) % p in D:
                                return True
        return False

    def cyclic_search(n, time_budget):
        """Search for a cyclic coloring of K_n with no monochromatic C6."""
        if n < 2:
            return None
        p = n
        all_diffs = list(range(1, p))
        colors = [set() for _ in range(4)]

        def backtrack(idx):
            if time.time() > deadline - 10:
                return None
            if idx == len(all_diffs):
                for c in range(4):
                    if has_c6_in_cayley(colors[c], p):
                        return None
                return [list(colors[c]) for c in range(4)]
            d = all_diffs[idx]
            for c in range(4):
                if len(colors[c]) < len(all_diffs) // 4 + 1:
                    colors[c].add(d)
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colors[c].remove(d)
            return None

        result = backtrack(0)
        if result is not None:
            color_map = {}
            for c, D in enumerate(result):
                for d in D:
                    color_map[d] = c
                    color_map[(p - d) % p] = c
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % p
                    colors_list.append(color_map[diff])
            return colors_list
        return None

    # Fallback: valid small colouring
    def fallback(n):
        m = n * (n - 1) // 2
        colors = [0] * m
        return {'n': n, 'colors': colors}

    best_result = fallback(5)

    # Try n from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() > deadline - 15:
            break

        # Route 1: Spectral embedding + local search
        if time.time() < deadline - 20:
            pts = spectral_embedding(n, iters=50)
            colors = quadrant_color(pts, n)
            colors, bad = local_search(colors, n, time_budget=30)
            if verify_no_c6(colors, n):
                return {'n': n, 'colors': colors}

        # Route 2: Cyclic search (for n where cyclic structure might work)
        if time.time() < deadline - 15:
            cyc = cyclic_search(n, time_budget=20)
            if cyc is not None and verify_no_c6(cyc, n):
                return {'n': n, 'colors': cyc}

    return best_result
