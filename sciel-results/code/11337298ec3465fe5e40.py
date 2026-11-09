"""Program for R_5(C4) maximization:
1) Fast algebraic-style direct construction for n=26 (Z_5 x Z_5 + 1) with a
   deterministic colour rule, verified in O(n^3 / word).
2) If that fails or if we want to push to 27/28, run a vectorised local search
   that minimises the number of C4-excess pairs using NumPy bitsets.
3) Fallback to smaller n with a strong local search."""
import time
import random
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def build_adj(n, colors):
        """Return list of 5 numpy bool arrays of shape (n,) per vertex? 
        Actually return adj[c] as a list of sets or a numpy matrix.
        For speed, use numpy uint64 bitsets if n <= 64."""
        adj = [np.zeros((n, n), dtype=bool) for _ in range(5)]
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c][i, j] = True
            adj[c][j, i] = True
        return adj

    def count_c4_fast(n, colors):
        """Count total C4-excess: for each colour, sum over pairs (u,v) of max(0, |N_u & N_v| - 1)."""
        adj = build_adj(n, colors)
        total = 0
        for c in range(5):
            A = adj[c]
            # For each pair u<v, common = np.sum(A[u] & A[v])
            # We can do this with matrix multiplication? 
            # Actually, (A @ A.T)[u,v] = number of common neighbours.
            # But that includes u and v themselves? No, A[u,v] is 1 if edge exists.
            # A @ A.T gives for each u,v the number of w such that A[u,w]=1 and A[w,v]=1.
            # That is exactly the number of common neighbours (paths of length 2).
            # But note: if u and v are adjacent, A[u,v]=1, so w=u gives A[u,u]=0 (no self-loop), 
            # and w=v gives A[u,v]=1 and A[v,v]=0. So it's fine.
            # However, A @ A.T is O(n^3) but with numpy it's fast for n<=30.
            # Let's use it.
            try:
                # Use matrix multiplication: M = A @ A.T
                # M[u,v] = number of common neighbours of u and v in colour c.
                # We need to sum max(0, M[u,v] - 1) for u < v.
                # But note: M[u,u] = degree of u, which we don't care about.
                # And M[u,v] for u != v is the number of length-2 paths.
                # A C4 exists if M[u,v] >= 2 for some u != v.
                # The "excess" is max(0, M[u,v] - 1).
                M = A.astype(np.int32) @ A.T.astype(np.int32)
                # Set diagonal to 0
                np.fill_diagonal(M, 0)
                # Sum over upper triangle
                total += int(np.sum(np.maximum(M - 1, 0)))
            except:
                # Fallback to loop
                for u in range(n):
                    for v in range(u + 1, n):
                        common = np.sum(A[u] & A[v])
                        if common >= 2:
                            total += common - 1
        return total

    def is_valid(n, colors):
        return count_c4_fast(n, colors) == 0

    # ---------- Direct construction for n=26 ----------
    # Vertices 0..24: (a, b) in Z_5 x Z_5, vertex = a*5 + b
    # Vertex 25: the "infinity" vertex
    # Colour rule for edge between (a,b) and (a',b'):
    #   Let da = (a' - a) mod 5, db = (b' - b) mod 5
    #   If (da, db) != (0,0):
    #       colour = (da * 2 + db) mod 5
    #   This is a linear map from Z_5^2 \ {(0,0)} to Z_5.
    #   We need to verify no C4.
    # For the infinity vertex, connect to each (a,b) with colour = a mod 5.
    # Let's try this and verify.

    def construct_26():
        n = 26
        m = n * (n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            if i < 25 and j < 25:
                a1, b1 = i // 5, i % 5
                a2, b2 = j // 5, j % 5
                da = (a2 - a1) % 5
                db = (b2 - b1) % 5
                # Ensure we use the "positive" direction? The map should be symmetric.
                # If we swap, da -> -da, db -> -db. (da*2+db) mod 5 vs (-da*2-db) mod 5.
                # These are not the same. So we need a symmetric rule.
                # Use: colour = (da * 2 + db) mod 5 if (da, db) is in a canonical half, else the other.
                # Better: use a quadratic form or a symmetric function.
                # Let's use: colour = (da * da + db * db) mod 5? No, that's only 0,1,4.
                # Let's use: colour = (da + 2*db) mod 5, but we need symmetry.
                # Actually, for a complete graph, the colour of (u,v) must equal colour of (v,u).
                # So the rule must be symmetric in (i,j).
                # Let's define: for u != v, let d = (u - v) mod 25? No, that's for cyclic.
                # Let's use the difference in Z_5^2 and take the "smaller" representative.
                # Or simply: colour = (da * 2 + db) mod 5, but if we swap, we get (-da*2 - db) mod 5.
                # To make it symmetric, we can use: colour = min((da*2+db)%5, (-da*2-db)%5, 5-((da*2+db)%5), 5-((-da*2-db)%5))
                # That's messy. Let's use a different rule.
                # How about: colour = (da * db) mod 5? That's symmetric (da*db = (-da)*(-db)).
                # But da*db mod 5 can be 0,1,4. Only 3 colours.
                # Let's use: colour = (da * 3 + db) mod 5, but again not symmetric.
                # 
                # Alternative: Use the fact that the graph is vertex-transitive.
                # Let's just use a random symmetric assignment and verify.
                # Or better: use the known Lazebnik-Woldar construction if I can recall it.
                # 
                # Let's try: colour = (da + db) mod 5. Not symmetric.
                # 
                # Okay, let's use a different approach: 
                # Assign colour based on the slope in Z_5.
                # If da != 0: slope = db * da^{-1} mod 5. Colour = slope.
                # If da == 0: colour = 4 (for vertical edges).
                # This is not symmetric either, because slope from u to v is -slope from v to u? 
                # No, slope = db/da. From v to u: (-db)/(-da) = db/da. So it IS symmetric!
                # Great. So:
                if da != 0:
                    # da^{-1} mod 5
                    inv_da = pow(da, -1, 5) if hasattr(pow, '__call__') else None
                    # Python 3.8+ has pow(a, -1, m)
                    try:
                        inv_da = pow(da, -1, 5)
                    except:
                        # Fallback
                        for x in range(5):
                            if (da * x) % 5 == 1:
                                inv_da = x
                                break
                    slope = (db * inv_da) % 5
                    colors[idx] = slope
                else:
                    # da == 0, so db != 0
                    colors[idx] = 4
            else:
                # One of i or j is 25
                if i == 25:
                    a = j // 5
                    b = j % 5
                    colors[idx] = a % 5
                else:
                    a = i // 5
                    b = i % 5
                    colors[idx] = a % 5
        return colors

    # Try the direct construction
    if time.time() < deadline:
        colors26 = construct_26()
        if is_valid(26, colors26):
            return {'n': 26, 'colors': colors26}
        # If not valid, we could try to repair, but let's just fall through to local search.

    # ---------- Local search for n=26, 27, 28 ----------
    def local_search(n, time_limit):
        m = n * (n - 1) // 2
        # Start with the structured initialisation
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            if n == 26 and i < 25 and j < 25:
                a1, b1 = i // 5, i % 5
                a2, b2 = j // 5, j % 5
                da = (a2 - a1) % 5
                db = (b2 - b1) % 5
                if da != 0:
                    try:
                        inv_da = pow(da, -1, 5)
                    except:
                        inv_da = 1
                    colors[idx] = (db * inv_da) % 5
                else:
                    colors[idx] = 4
            else:
                colors[idx] = (i + j) % 5

        current = count_c4_fast(n, colors)
        if current == 0:
            return colors

        t_start = time.time()
        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        # Local search: random restarts
        for restart in range(5):
            if time.time() > t_start + time_limit:
                break
            if current == 0:
                break
            # Perturb
            for _ in range(50):
                eidx = random.randint(0, m - 1)
                colors[eidx] = random.randint(0, 4)
            current = count_c4_fast(n, colors)
            # Hill climb
            no_improve = 0
            while time.time() < t_start + time_limit and current > 0:
                if no_improve > 1000:
                    break
                eidx = random.randint(0, m - 1)
                old_c = colors[eidx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                colors[eidx] = new_c
                new_val = count_c4_fast(n, colors)
                if new_val < current:
                    current = new_val
                    no_improve = 0
                else:
                    colors[eidx] = old_c
                    no_improve += 1
        return colors

    # Try local search for n=28, 27, 26
    for target_n in [28, 27, 26]:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 5
        if remaining < 5:
            break
        colors = local_search(target_n, time_limit=remaining)
        if is_valid(target_n, colors):
            return {'n': target_n, 'colors': colors}

    # Fallback: smaller n
    for fallback_n in [25, 24, 23, 22, 21, 20, 15, 10, 5, 1]:
        if time.time() > deadline:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        if is_valid(fallback_n, colors):
            return {'n': fallback_n, 'colors': colors}
        # Try local search for a bit
        remaining = deadline - time.time() - 2
        if remaining > 2:
            colors = local_search(fallback_n, time_limit=remaining)
            if is_valid(fallback_n, colors):
                return {'n': fallback_n, 'colors': colors}

    return {'n': 1, 'colors': []}
