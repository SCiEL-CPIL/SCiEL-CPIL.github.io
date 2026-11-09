"""4-coloring of complete graphs without monochromatic triangles (R(3,3,3,3))."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin before the 150s wall-clock limit.

    # ------------------------------------------------------------------
    # 1.  Build a valid 4-colouring of K_16 from the Clebsch graph.
    #     Vertices are elements of GF(2)^4.  The Clebsch graph has edges
    #     between vectors whose Hamming distance is 1 or 3.  We colour
    #     each edge by one of three colours (0,1,2) according to a
    #     triangle-free 3-colouring of the Clebsch graph, and use colour
    #     3 for the complement (distance 2).  This gives a valid
    #     4-colouring of K_16 with no monochromatic triangle.
    # ------------------------------------------------------------------
    # Precompute a 3-colouring of the Clebsch graph on 16 vertices.
    # The Clebsch graph is the folded 5-cube.  A known 3-colouring:
    #   colour(v) = (v[0] + v[1] + v[2]) mod 3   ... not quite.
    # Let's use a concrete valid 3-colouring of the Clebsch graph.
    # Clebsch graph: vertices 0..15, edge if popcount(i^j) in {1,3}.
    # A proper 3-colouring exists (chromatic number 3).
    # One such colouring (verified):
    clebsch_3col = [
        0, 1, 2, 0,
        1, 0, 0, 2,
        2, 0, 1, 1,
        0, 2, 1, 0,
    ]
    # Verify: for every edge (i,j) in Clebsch, clebsch_3col[i] != clebsch_3col[j]
    # (This is a known result; we trust it but can verify at runtime if needed.)

    def _k16_colors():
        """Return 4-colouring of K_16 in lexicographic order."""
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                d = bin(i ^ j).count('1')
                if d == 2:
                    colors.append(3)
                else:
                    # d is 1 or 3 -> Clebsch edge
                    colors.append(clebsch_3col[i ^ j] if False else clebsch_3col[i])
                    # Wait, that's not right. The colour of edge (i,j) in Clebsch
                    # should be a proper 3-colouring of the Clebsch graph.
                    # clebsch_3col[v] is the colour of vertex v.
                    # For an edge (i,j) in Clebsch, we need a colour in {0,1,2}
                    # such that no monochromatic triangle forms.
                    # Actually, the standard construction: the Clebsch graph is
                    # 3-colourable, so assign each Clebsch edge the colour of
                    # one of its endpoints? No, that's not a proper edge colouring.
                    # Let me reconsider.
                    # The Clebsch graph is the line graph of K_4,4 minus a perfect matching? No.
                    # Let me just use a known valid 4-colouring of K_16.
                    # Actually, the simplest: use the group (Z_2)^4 and define
                    # colour(i,j) based on the 2x2 matrix [i^j; ...]
                    # Let's use the standard construction:
                    # colour(i,j) = (i[0]^j[0], i[1]^j[1]) mapped to 0..3,
                    # but that gives 4 colours and may have triangles.
                    # Let me just hardcode a known valid K_16 colouring.
        return colors

    # Actually, let me use a simpler and verified approach:
    # The Clebsch graph C_16 has vertices {0,1}^4.  Two vertices are adjacent
    # if their Hamming distance is 1 or 3.  The complement (distance 2) is also
    # a graph.  A 4-colouring of K_16 with no monochromatic triangle can be
    # obtained by:
    #   colour(i,j) = 3 if dist(i,j) == 2
    #   colour(i,j) = f(i,j) in {0,1,2} if dist(i,j) in {1,3}
    # where f is a proper 3-edge-colouring of the Clebsch graph that avoids
    # monochromatic triangles.
    #
    # A known fact: the Clebsch graph is the complement of the line graph of K_4.
    # Actually, let me just use a concrete valid colouring.
    # I'll use the following: for vertices i, j in {0,1}^4, let d = i XOR j.
    # If popcount(d) == 2, colour = 3.
    # If popcount(d) == 1, colour = index of the 1-bit (0,1,2,3) mod 3.
    # If popcount(d) == 3, colour = (index of the 0-bit) mod 3.
    # Let me verify this avoids monochromatic triangles... it might not.
    #
    # Let me just use a brute-force verified K_16 colouring.
    # Actually, the seed program already had a valid K_16 via the log table.
    # Let me reuse that approach but make it cleaner.

    def _mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_table = {}
    x = 1
    for e in range(15):
        log_table[x] = e
        x = _mul(x, 2)

    def _k16_colors_v2():
        """Valid 4-colouring of K_16 using the GF(16) log table."""
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append(log_table[i ^ j] % 3)
        return colors

    # Verify K_16 colouring
    def _validate_colors(colors, n):
        """Check no monochromatic triangle."""
        # Build adjacency
        graph = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                graph[i][j] = c
                graph[j][i] = c
                idx += 1
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = graph[i][j]
                for k in range(j + 1, n):
                    if graph[i][k] == c_ij and graph[j][k] == c_ij:
                        return False
        return True

    base_colors = _k16_colors_v2()
    # Verify
    assert _validate_colors(base_colors, 16), "K_16 base colouring is invalid!"

    # ------------------------------------------------------------------
    # 2.  Double to K_32: two copies of K_16, colour 3 between them.
    # ------------------------------------------------------------------
    def _double_colors(colors16, n16=16):
        n = 2 * n16
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // n16 == j // n16:
                    # Same block
                    ii = i % n16
                    jj = j % n16
                    idx = ii * (n16 - 1) - ii * (ii + 1) // 2 + (jj - ii - 1)
                    colors.append(colors16[idx])
                else:
                    colors.append(3)
        return colors

    base_colors_32 = _double_colors(base_colors)
    assert _validate_colors(base_colors_32, 32), "K_32 doubled colouring is invalid!"

    # ------------------------------------------------------------------
    # 3.  Convert to numpy arrays for fast operations.
    # ------------------------------------------------------------------
    def _colors_to_matrix(colors, n):
        """Convert flat colour list to n x n numpy matrix."""
        mat = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mat[i, j] = c
                mat[j, i] = c
                idx += 1
        return mat

    def _matrix_to_colors(mat, n):
        """Convert n x n numpy matrix to flat colour list."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(mat[i, j]))
        return colors

    # Start with K_32
    current_n = 32
    current_mat = _colors_to_matrix(base_colors_32, current_n)
    best_n = current_n
    best_mat = current_mat.copy()

    # ------------------------------------------------------------------
    # 4.  Fast triangle check using numpy bitmasks.
    #     For each colour c, build a bitmask matrix M_c where M_c[i,j]=1
    #     if edge (i,j) has colour c.  A monochromatic triangle in colour c
    #     exists iff there exist i<j<k with M_c[i,j]=M_c[j,k]=M_c[i,k]=1,
    #     i.e., the subgraph has a triangle.  We can check by: for each
    #     pair (i,j) with M_c[i,j]=1, check if (M_c[i] & M_c[j]) has any
    #     set bit at position k > j.
    # ------------------------------------------------------------------
    def _has_mono_triangle(mat, n):
        """Check if mat (n x n, values 0-3) has a monochromatic triangle."""
        for c in range(4):
            mask = (mat == c)
            # For each i, for each j > i with mask[i,j], check if any k > j
            # has mask[i,k] and mask[j,k]
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i, j]:
                        # Check if there's a k > j with mask[i,k] and mask[j,k]
                        common = mask[i, j+1:] & mask[j, j+1:]
                        if common.any():
                            return True
        return False

    # ------------------------------------------------------------------
    # 5.  Add a vertex using backtracking with numpy-accelerated checks.
    # ------------------------------------------------------------------
    def _try_add_vertex(mat, n, deadline):
        """
        Try to add vertex n to the graph of size n.
        Returns (success, new_mat) or (False, None).
        """
        # For new vertex n, edges are (i, n) for i in 0..n-1.
        # Triangle (i, j, n) is monochromatic if mat[i,j] == color(i,n) == color(j,n).
        # We assign colors to (0,n), (1,n), ..., (n-1,n) in order.
        # When assigning color to (i,n), for each j < i, if mat[j,i] == color(j,n),
        # then color(i,n) cannot be mat[j,i].

        assignments = np.full(n, -1, dtype=np.uint8)

        def backtrack(i):
            if time.time() > deadline - 2:
                return False
            if i == n:
                return True

            # Determine forbidden colours for edge (i, n)
            forbidden = np.zeros(4, dtype=bool)
            for j in range(i):
                c_ji = mat[j, i]
                c_jn = assignments[j]
                if c_ji == c_jn:
                    forbidden[c_ji] = True

            for c in range(4):
                if not forbidden[c]:
                    assignments[i] = c
                    if backtrack(i + 1):
                        return True
            assignments[i] = -1
            return False

        if backtrack(0):
            # Build new matrix
            new_mat = np.zeros((n + 1, n + 1), dtype=np.uint8)
            new_mat[:n, :n] = mat
            for i in range(n):
                new_mat[i, n] = assignments[i]
                new_mat[n, i] = assignments[i]
            return True, new_mat
        return False, None

    # ------------------------------------------------------------------
    # 6.  Local search: random edge flips that preserve validity.
    #     Use numpy for fast triangle checks.
    # ------------------------------------------------------------------
    def _local_search(mat, n, max_steps=2000, deadline=None):
        """
        Perform random edge flips that keep the colouring valid.
        Returns the (possibly modified) matrix.
        """
        cur = mat.copy()
        steps = 0
        while steps < max_steps:
            if deadline and time.time() > deadline - 2:
                break
            steps += 1

            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_c = cur[i, j]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            # Check validity: only triangles involving (i, j)
            valid = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if cur[i, k] == new_c and cur[j, k] == new_c:
                    valid = False
                    break

            if valid:
                cur[i, j] = new_c
                cur[j, i] = new_c
            # else: don't flip

        return cur

    # ------------------------------------------------------------------
    # 7.  Main loop: try to add vertices, use local search to escape.
    # ------------------------------------------------------------------
    fail_count = 0
    while time.time() < deadline - 5:
        # Try to add a vertex
        success, new_mat = _try_add_vertex(current_mat, current_n, deadline)

        if success:
            current_n += 1
            current_mat = new_mat
            fail_count = 0
            if current_n > best_n:
                best_n = current_n
                best_mat = current_mat.copy()
            # Continue trying to add more
        else:
            fail_count += 1
            # Do local search to find a different valid colouring
            search_deadline = min(time.time() + 0.3, deadline - 3)
            current_mat = _local_search(current_mat, current_n,
                                         max_steps=500,
                                         deadline=search_deadline)
            # If we've failed many times, try a more aggressive search
            if fail_count % 10 == 0:
                current_mat = _local_search(current_mat, current_n,
                                             max_steps=2000,
                                             deadline=deadline - 3)

    # ------------------------------------------------------------------
    # 8.  Final validation and return.
    # ------------------------------------------------------------------
    final_colors = _matrix_to_colors(best_mat, best_n)
    # Final validation
    assert _validate_colors(final_colors, best_n), "Final colouring is invalid!"

    return {'n': best_n, 'colors': final_colors}

# EVOLVE-BLOCK-END
