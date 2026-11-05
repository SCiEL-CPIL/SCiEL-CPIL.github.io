import time
import random
import numpy as np

DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Build the known 32-vertex solution (Greenwood-Gleason doubled)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table, x = {}, 1
    for e in range(15):
        log_table[x] = e
        x = mul(x, 2)

    # Build 32-vertex base
    base_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
            else:
                base_colors.append(3)

    best_n = 32
    best_colors = base_colors[:]

    # Convert to matrix for easy manipulation
    def colors_to_matrix(colors, n):
        mat = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mat[i, j] = c
                mat[j, i] = c
                idx += 1
        return mat

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(mat[i, j]))
        return colors

    def count_triangles(mat, n):
        """Count total monochromatic triangles using numpy."""
        total = 0
        for c in range(4):
            A = (mat == c).astype(np.int32)
            np.fill_diagonal(A, 0)
            # Number of triangles in color c = trace(A^3) / 6
            total += int(np.trace(A @ A @ A)) // 6
        return total

    def try_extend_with_backtracking(mat, n, max_add=25, time_budget=30.0):
        """Try to extend by adding vertices using constrained backtracking."""
        current_n = n
        current_mat = mat.copy()
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining <= 0:
            return current_n, current_mat

        for _ in range(max_add):
            if time.time() > deadline - 5:
                break
            if current_n >= 62:
                break

            # Try to add vertex at position current_n
            # We need to assign colors c[0..current_n-1] for edges (current_n, u)
            # Constraint: for each color c, {u : c[u] == c} must be independent in color-c subgraph

            # Precompute for each color c, the adjacency as a set of edges
            # For backtracking, we need: for vertex u to be assigned color c,
            # no other vertex v with c[v] == c can have edge (u,v) in color c

            found = False
            # Try multiple random orderings and backtracking
            for trial in range(50):
                if time.time() > deadline - 5:
                    break

                # Random vertex ordering for assignment
                order = list(range(current_n))
                random.shuffle(order)

                assign = [-1] * current_n
                # For each color, track which vertices are assigned to it
                color_sets = [[], [], [], []]

                valid = True
                for u in order:
                    # Try colors in random order
                    colors_to_try = [0, 1, 2, 3]
                    random.shuffle(colors_to_try)
                    placed = False
                    for c in colors_to_try:
                        # Check if u can be assigned color c
                        # For all v in color_sets[c], edge (u,v) must NOT be color c
                        ok = True
                        for v in color_sets[c]:
                            if current_mat[u, v] == c:
                                ok = False
                                break
                        if ok:
                            assign[u] = c
                            color_sets[c].append(u)
                            placed = True
                            break
                    if not placed:
                        valid = False
                        break

                if valid:
                    # Found a valid extension!
                    for u in range(current_n):
                        current_mat[current_n, u] = assign[u]
                        current_mat[u, current_n] = assign[u]
                    current_n += 1
                    found = True
                    break

            if not found:
                break

        return current_n, current_mat

    def simulated_annealing(mat, n, time_budget=30.0):
        """Apply simulated annealing to eliminate monochromatic triangles."""
        current_n = n
        current_mat = mat.copy()
        cost = count_triangles(current_mat, current_n)

        if cost == 0:
            return current_n, current_mat

        T = 500.0
        cooling = 0.9995
        best_cost = cost
        best_mat = current_mat.copy()

        while time.time() < deadline - 3:
            if cost == 0:
                return current_n, current_mat

            # Pick random edge
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue

            c_old = current_mat[i, j]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue

            # Calculate delta in triangle count
            # A triangle (i,j,k) is monochromatic in color c if all three edges have color c
            # Changing edge (i,j) from c_old to c_new:
            #   - Removes triangles where i-k and j-k were c_old
            #   - Adds triangles where i-k and j-k are c_new
            delta = 0
            for k in range(current_n):
                if k == i or k == j:
                    continue
                if current_mat[i, k] == c_old and current_mat[j, k] == c_old:
                    delta -= 1
                if current_mat[i, k] == c_new and current_mat[j, k] == c_new:
                    delta += 1

            if delta < 0 or (delta == 0 and random.random() < 0.5) or random.random() < np.exp(-delta / T):
                current_mat[i, j] = c_new
                current_mat[j, i] = c_new
                cost += delta
                if cost < best_cost:
                    best_cost = cost
                    best_mat = current_mat.copy()
                    if cost == 0:
                        return current_n, current_mat

            T *= cooling

        # Return best found
        if best_cost < cost:
            return current_n, best_mat
        return current_n, current_mat

    # Phase 1: Extend from 32 using backtracking
    mat = colors_to_matrix(base_colors, 32)
    n, mat = try_extend_with_backtracking(mat, 32, max_add=25, time_budget=25.0)

    if n > best_n:
        best_n = n
        best_colors = matrix_to_colors(mat, n)

    # Phase 2: Apply simulated annealing to try to fix any issues and potentially allow more extension
    if time.time() < deadline - 10:
        n2, mat2 = simulated_annealing(mat, n, time_budget=20.0)
        if n2 > best_n:
            best_n = n2
            best_colors = matrix_to_colors(mat2, n2)
        elif n2 == best_n:
            # Check if this version has fewer triangles (might help with further extension)
            cost2 = count_triangles(mat2, n2)
            cost1 = count_triangles(colors_to_matrix(best_colors, best_n), best_n)
            if cost2 < cost1:
                best_colors = matrix_to_colors(mat2, n2)

    # Phase 3: Try to extend further from the annealed result
    if time.time() < deadline - 10 and best_n < 62:
        mat_ext = colors_to_matrix(best_colors, best_n)
        n3, mat3 = try_extend_with_backtracking(mat_ext, best_n, max_add=20, time_budget=15.0)
        if n3 > best_n:
            best_n = n3
            best_colors = matrix_to_colors(mat3, n3)

    # Phase 4: Final annealing pass
    if time.time() < deadline - 5 and best_n < 62:
        mat_final = colors_to_matrix(best_colors, best_n)
        n4, mat4 = simulated_annealing(mat_final, best_n, time_budget=10.0)
        if n4 > best_n:
            best_n = n4
            best_colors = matrix_to_colors(mat4, n4)

    # Phase 5: Multiple restarts with different seeds if time remains
    if time.time() < deadline - 15:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            random.seed(42 + restart * 1000)
            mat_r = colors_to_matrix(base_colors, 32)
            nr, matr = try_extend_with_backtracking(mat_r, 32, max_add=25, time_budget=10.0)
            if nr > best_n:
                best_n = nr
                best_colors = matrix_to_colors(matr, nr)
            # Quick anneal
            if time.time() < deadline - 5 and nr < 62:
                na, mata = simulated_annealing(matr, nr, time_budget=5.0)
                if na > best_n:
                    best_n = na
                    best_colors = matrix_to_colors(mata, na)

    # Final verification: ensure no monochromatic triangles
    # For safety, do a quick check and if invalid, fall back to base
    final_mat = colors_to_matrix(best_colors, best_n)
    final_cost = count_triangles(final_mat, best_n)
    if final_cost > 0:
        # Try one more anneal pass to fix
        if time.time() < deadline - 2:
            nf, matf = simulated_annealing(final_mat, best_n, time_budget=3.0)
            if count_triangles(matf, nf) == 0:
                best_n = nf
                best_colors = matrix_to_colors(matf, nf)
            else:
                # Fall back to known valid n=32
                best_n = 32
                best_colors = base_colors[:]
        else:
            best_n = 32
            best_colors = base_colors[:]

    return {'n': best_n, 'colors': best_colors}
