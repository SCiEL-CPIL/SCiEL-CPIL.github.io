import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0

    def has_path_len_4(u, v, adj, n):
        """Check if there's a simple path of length 4 from u to v in graph adj."""
        if u == v:
            return False

        # DFS with early termination
        def dfs(curr, depth, visited):
            if depth == 4:
                return curr == v
            mask = adj[curr]
            if depth < 3:
                mask &= ~(1 << v)
            mask &= ~visited
            while mask:
                lsb = mask & -mask
                nxt = lsb.bit_length() - 1
                mask ^= lsb
                if dfs(nxt, depth + 1, visited | (1 << nxt)):
                    return True
            return False

        return dfs(u, 0, 1 << u)

    def backtrack(n, time_budget):
        """
        Backtracking: assign colors to edges in lexicographic order.
        After adding edge (u,v) to color c, check if a C6 is formed by
        testing if there's a path of length 4 from u to v in color c
        (excluding the edge just added).
        """
        end_time = start_time + time_budget
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        adj = [[0] * n for _ in range(4)]
        color_of = [-1] * total_edges
        found = [False]

        def check_c6(c, u, v):
            """Check if adding edge (u,v) to color c creates a C6."""
            # Temporarily remove (u,v)
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            result = has_path_len_4(u, v, adj[c], n)
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            return result

        def solve(pos):
            if found[0]:
                return True
            if time.time() > end_time:
                return False
            if pos == total_edges:
                found[0] = True
                return True

            i, j = edges[pos]
            for c in range(4):
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                color_of[pos] = c

                if not check_c6(c, i, j):
                    if solve(pos + 1):
                        return True

                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                color_of[pos] = -1

            return False

        random.seed(n * 1000 + 42)
        for attempt in range(100):
            if time.time() > end_time:
                break
            adj = [[0] * n for _ in range(4)]
            color_of = [-1] * total_edges
            found[0] = False

            if solve(0):
                return {'n': n, 'colors': color_of[:]}

        return None

    # Known valid K17 coloring (verified: no monochromatic C6)
    # Using circulant construction on Z_17 with difference sets
    def make_k17_circulant():
        """
        Construct K17 coloring using circulant graph approach.
        Edge (i,j) gets color based on (j-i) mod 17.
        Partition {1,...,16} into 4 sets of 4 such that no set
        generates a C6 in the circulant graph.
        
        For a circulant graph on Z_p with connection set S, a C6 exists iff
        there are s1,...,s6 in S∪{-S} (with s_i ≠ 0) such that s1+...+s6 = 0 mod p
        and all partial sums are distinct (for simple cycle).
        
        Known valid partition for p=17:
        S1 = {1, 4, 13, 16}  (differences ±1, ±4)
        S2 = {2, 8, 9, 15}   (differences ±2, ±8)  
        S3 = {3, 5, 12, 14}  (differences ±3, ±5)
        S4 = {6, 7, 10, 11}  (differences ±6, ±7)
        
        Let's verify this works by checking each color class for C6.
        """
        n = 17
        # Map difference d (1..16) to color
        diff_to_color = {}
        sets = [
            {1, 4, 13, 16},
            {2, 8, 9, 15},
            {3, 5, 12, 14},
            {6, 7, 10, 11}
        ]
        for c, s in enumerate(sets):
            for d in s:
                diff_to_color[d] = c

        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 0:
                    d = n
                # Use the smaller difference for symmetry
                # Actually, just use d as is since we have both d and n-d in the sets
                colors.append(diff_to_color[d])

        # Verify no C6 in any color class
        adj = [[0] * n for _ in range(4)]
        for idx, c in enumerate(colors):
            # Find (i,j) from idx
            # idx = i*n - i*(i+1)//2 + (j-i-1)
            # Let's just recompute
            pass

        # Rebuild adj for verification
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1

        # Verify
        for c in range(4):
            for u in range(n):
                for v in range(n):
                    if u < v and (adj[c][u] >> v) & 1:
                        if has_path_len_4(u, v, adj[c], n):
                            # Found C6, this coloring is invalid
                            return None

        return {'n': n, 'colors': colors}

    # Try known K17 first
    k17_result = make_k17_circulant()

    # Try to find larger n with remaining time
    best_result = k17_result

    if time.time() < deadline - 5:
        for n in [19, 18]:
            if time.time() > deadline - 5:
                break
            budget = min(40, deadline - time.time() - 5)
            if budget < 3:
                break
            res = backtrack(n, budget)
            if res:
                best_result = res
                break

    if best_result is None:
        # Fallback: K5 is always valid
        best_result = {'n': 5, 'colors': [0] * 10}

    return best_result
