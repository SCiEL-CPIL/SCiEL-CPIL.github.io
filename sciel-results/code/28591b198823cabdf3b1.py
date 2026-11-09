import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_cycles(n):
        cycles = []
        # Generate all unique C6 cycles in K_n
        # A cycle is a sequence of 6 vertices (v0, v1, v2, v3, v4, v5)
        # We fix v0 as the smallest vertex to reduce duplicates?
        # No, we just generate all distinct cyclic orderings of 6 distinct vertices.
        # For a set of 6 vertices, there are 60 distinct cycles.
        for verts in itertools.combinations(range(n), 6):
            v0 = verts[0]
            rest = verts[1:]
            # Generate permutations of the remaining 5 vertices
            # To avoid generating both (p) and reverse(p), we enforce p[0] < p[-1]
            for perm in itertools.permutations(rest):
                if perm[0] < perm[-1]:
                    cycles.append((v0, perm[0], perm[1], perm[2], perm[3], perm[4], perm[5]))
        return cycles

    def count_violations(n, colors, cycles):
        violations = 0
        for c in cycles:
            v0, v1, v2, v3, v4, v5 = c
            # Check if all 6 edges have the same color
            c0 = colors[v0][v1]
            c1 = colors[v1][v2]
            c2 = colors[v2][v3]
            c3 = colors[v3][v4]
            c4 = colors[v4][v5]
            c5 = colors[v5][v0]
            if c0 == c1 == c2 == c3 == c4 == c5:
                violations += 1
        return violations

    def try_solve(n, time_limit):
        # Check if we have enough time
        if time.time() > deadline:
            return None
            
        cycles = get_cycles(n)
        
        # Initialize random coloring
        colors = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                col = random.randint(0, 3)
                colors[i][j] = col
                colors[j][i] = col
        
        current_violations = count_violations(n, colors, cycles)
        
        # Local search
        while time.time() < deadline - 5.0:
            if current_violations == 0:
                return colors
            
            # Pick a random cycle
            c = random.choice(cycles)
            
            # Check if it's violated
            v0, v1, v2, v3, v4, v5 = c
            c0 = colors[v0][v1]
            c1 = colors[v1][v2]
            c2 = colors[v2][v3]
            c3 = colors[v3][v4]
            c4 = colors[v4][v5]
            c5 = colors[v5][v0]
            
            if not (c0 == c1 == c2 == c3 == c4 == c5):
                continue
                
            # It is violated. Pick a random edge in this cycle and change its color
            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            u, v = random.choice(edges)
            old_color = colors[u][v]
            new_color = random.choice([x for x in range(4) if x != old_color])
            colors[u][v] = new_color
            colors[v][u] = new_color
            
            # Recompute violations
            current_violations = count_violations(n, colors, cycles)
            
        return None

    # Try to solve for n=13 with a generous time limit
    if time.time() < deadline - 30.0:
        sol = try_solve(13, 30.0)
        if sol:
            n = 13
            colors_list = [sol[i][j] for i in range(n) for j in range(i + 1, n)]
            return {'n': n, 'colors': colors_list}

    # If n=13 failed, try n=12
    if time.time() < deadline - 15.0:
        sol = try_solve(12, 15.0)
        if sol:
            n = 12
            colors_list = [sol[i][j] for i in range(n) for j in range(i + 1, n)]
            return {'n': n, 'colors': colors_list}

    # Fallback to the known valid n=9 solution
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
