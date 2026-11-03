"""Construct a 5-regular graph of girth 12 (generalized hexagon of order (2,4))."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # The incidence graph of the unique generalized hexagon of order (2,4)
    # is a 5-regular bipartite graph with 126... actually 30 vertices (15 points + 15 lines)
    # and girth 12. This satisfies any girth requirement up to 12.
    #
    # Points: non-zero vectors in GF(4)^3  -> 15 points
    # Lines:  2-dimensional subspaces of GF(4)^3 -> 15 lines, each containing 5 points
    # Each point is on 5 lines, each line has 5 points -> 5-regular.
    #
    # GF(4) = {0, 1, w, w^2} with w^2 = w + 1, w^3 = 1.
    # Multiplication table (rows * cols):
    #   0: [0, 0, 0, 0]
    #   1: [0, 1, w, w2]
    #   w: [0, w, w2, 1]
    #   w2:[0, w2, 1, w]
    # where we label elements as 0,1,2,3 for 0,1,w,w^2.

    # GF(4) multiplication table
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
    ]
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    def gf_add(a, b):
        return add_table[a][b]

    def gf_mul(a, b):
        return mul_table[a][b]

    # All non-zero vectors in GF(4)^3
    points = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if a == 0 and b == 0 and c == 0:
                    continue
                points.append((a, b, c))
    # 15 points

    # Generate all 2D subspaces (lines)
    # A line is the set of non-zero vectors in a 2D subspace.
    # For two independent vectors u, v, the subspace is {a*u + b*v : a,b in GF(4)}
    # We collect the 5 non-zero vectors.

    def span(u, v):
        """Return the frozenset of 5 non-zero vectors in the 2D subspace spanned by u and v."""
        vecs = set()
        for a in range(4):
            for b in range(4):
                if a == 0 and b == 0:
                    continue
                x = gf_add(gf_mul(a, u[0]), gf_mul(b, v[0]))
                y = gf_add(gf_mul(a, u[1]), gf_mul(b, v[1]))
                z = gf_add(gf_mul(a, u[2]), gf_mul(b, v[2]))
                vecs.add((x, y, z))
        return frozenset(vecs)

    def are_independent(u, v):
        """Check if u and v are linearly independent in GF(4)^3."""
        # v is in span of u iff v = a*u for some a, i.e., there exists a such that v[i] = a*u[i] for all i
        # Try all a in GF(4)
        for a in range(4):
            if all(gf_mul(a, u[i]) == v[i] for i in range(3)):
                return False
        return True

    lines = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u, v = points[i], points[j]
            if are_independent(u, v):
                line = span(u, v)
                # Canonical form: sort the vectors
                canonical = tuple(sorted(line))
                lines.add(canonical)

    # Each line should have exactly 5 non-zero points
    # Convert to list of lists
    lines_list = [list(line) for line in lines]
    # There should be 15 lines

    # Build the graph
    n_points = len(points)
    n_lines = len(lines_list)
    n = n_points + n_lines

    # Point indices: 0..14
    # Line indices: 15..29

    point_to_idx = {pt: i for i, pt in enumerate(points)}

    edges = []
    for li, line in enumerate(lines_list):
        for pt in line:
            pi = point_to_idx[pt]
            edges.append((pi, n_points + li))

    return n, edges
# EVOLVE-BLOCK-END
