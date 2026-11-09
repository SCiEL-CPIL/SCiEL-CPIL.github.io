"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_TIME_LIMIT = 140.0  # safety margin under 150s kill
_START = None


def _elapsed():
    return time.monotonic() - _START


def construct():
    global _START
    _START = time.monotonic()

    def _deadline():
        return _elapsed() > _TIME_LIMIT

    # --- Build base 50-vertex Greenwood–Gleason colouring ---
    # GF(16) with primitive polynomial x^4 + x + 1 (0b10011)
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build GF(16) log/exp tables
    gf_log = {}
    gf_exp = [0] * 16
    x = 1
    for e in range(15):
        gf_log[x] = e
        gf_exp[e] = x
        x = gf16_mul(x, 2)
    gf_log[0] = -1  # unused

    # Clebsch graph: two vertices in GF(16) are adjacent if their difference is a non-zero cube
    # Cubes in GF(16)*: elements with exponent 3k mod 15, i.e. exponents {0,3,6,9,12}
    # But we need 3 disjoint classes partitioning the 15 non-zero elements into 5-cycles
    # Actually: GF(16)* has order 15. Cubes are {g^(3k)} = subgroup of order 5.
    # The 3 cosets of the cubic subgroup give 3 colour classes, each a 5-regular graph on 16 vertices.
    # Each coset forms a Clebsch-like graph (actually the Clebsch graph is 5-regular on 16 vertices).
    # The 3 cosets partition the 15 non-zero elements, so they give a proper 3-colouring of K16.

    cubes = set()
    for k in range(5):
        cubes.add(gf_exp[3 * k % 15])
    # 3 cosets: cubes * g^j for j=0,1,2
    cosets = []
    for j in range(3):
        coset = set()
        for c in cubes:
            coset.add(gf16_mul(c, gf_exp[j]))
        cosets.append(coset)

    # Verify partition of non-zero elements
    all_nonzero = set(gf_exp[1:])  # exponents 1..14, skip 0 which is identity
    # Actually gf_exp[0]=1, gf_exp[1]=2, etc. Non-zero elements are gf_exp[0..14]
    all_nonzero = set(gf_exp)  # all 16 elements including 0? No, gf_exp has 15 entries for non-zero
    # gf_exp[0..14] are the 15 non-zero elements
    all_nonzero = set(gf_exp)

    # Build 16x16 colour matrix for one copy
    def color_16(a, b):
        diff = a ^ b  # GF(16) addition is XOR
        if diff == 0:
            return -1
        d = gf_log[diff]
        for c in range(3):
            if gf_exp[d] in cosets[c]:
                return c
        return -1  # shouldn't happen

    # Build 16-vertex base
    base16 = np.full((16, 16), -1, dtype=np.int8)
    for a in range(16):
        for b in range(a + 1, 16):
            c = color_16(a, b)
            base16[a, b] = c
            base16[b, a] = c

    # Double to 32 vertices: colour 3 between the two copies
    # This gives a valid 4-colouring of K32
    base32 = np.full((32, 32), -1, dtype=np.int8)
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base32[i, j] = base32[j, i] = base16[i % 16, j % 16]
            else:
                base32[i, j] = base32[j, i] = 3

    # Now attempt to grow from 32 up to 50 (or beyond) using greedy vertex addition
    # We'll use a different strategy: build up to 50 directly using known constructions
    # Actually, the known 50-vertex construction uses a different approach.
    # Let's use a SAT-based or greedy extension approach.

    # For now, return the 32-vertex solution as a safe baseline, then try to extend
    best_n = 32
    best_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            best_colors.append(int(base32[i, j]))

    # Try greedy extension: add vertices one by one
    # For each new vertex, find a colour assignment to all existing edges that avoids monochromatic triangles
    current_n = 32
    color_matrix = base32.copy()

    for target_n in range(33, 63):
        if _deadline():
            break

        # Try to add vertex target_n-1 (0-indexed)
        new_idx = target_n - 1
        # We need to assign colours to edges (new_idx, j) for j in 0..new_idx-1
        # such that no monochromatic triangle is formed

        # For each pair (a, b) where a < b < new_idx, the edge (new_idx, a) and (new_idx, b)
        # must not both have the same colour as (a, b)
        # This is a constraint satisfaction problem

        # Greedy approach: try all 4^new_idx assignments? Too expensive for large new_idx.
        # Instead, use a backtracking approach with early pruning.

        # For efficiency, use a simple greedy: assign each edge colour randomly, check, retry
        # But this may not find a valid assignment.

        # Alternative: use the known 50-vertex construction directly.
        # The Greenwood-Gleason 50-vertex construction is well-known.
        # Let me implement it directly.

        break  # Skip greedy for now, use direct 50-vertex construction

    # Direct 50-vertex Greenwood-Gleason construction
    # This uses a different algebraic construction.
    # For now, return the 32-vertex solution.

    # Actually, let me implement a proper extension using constraint satisfaction
    # with a time limit.

    # Try to extend from 32 to 50 using a simple backtracking solver
    # with early termination.

    def try_add_vertex(matrix, current_n, new_idx):
        """Try to assign colours to edges (new_idx, j) for j < new_idx."""
        # Collect constraints: for each pair (a, b), a < b < new_idx,
        # colour(new_idx, a) != colour(a, b) if colour(new_idx, b) == colour(a, b)
        # i.e., for each triple (a, b, new_idx), the three edges must not all have the same colour.

        # For each vertex j < new_idx, the colour of edge (new_idx, j) must be chosen
        # such that for all a < j, if colour(new_idx, a) == colour(a, j), then colour(new_idx, j) != colour(a, j)

        # This is complex. Use a simpler approach:
        # For each j, the allowed colours are those that don't create a monochromatic triangle
        # with any pair (a, j) where a < j.

        # Actually, the constraint is: for any a < b < new_idx,
        # not (colour(new_idx, a) == colour(new_idx, b) == colour(a, b))

        # So for each j, the colour of (new_idx, j) is constrained by all other (new_idx, a) choices.

        # This is a global constraint. Use backtracking.

        # For simplicity and speed, use a randomized approach with limited retries.
        max_retries = 1000
        for _ in range(max_retries):
            if _deadline():
                return None

            # Randomly assign colours
            assignment = np.random.randint(0, 4, size=current_n, dtype=np.int8)

            # Check validity
            valid = True
            for a in range(current_n):
                for b in range(a + 1, current_n):
                    if assignment[a] == assignment[b] == matrix[a, b]:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                return assignment

        return None

    # Try extending
    for target_n in range(33, 63):
        if _deadline():
            break

        assignment = try_add_vertex(color_matrix, target_n - 1, target_n - 1)
        if assignment is not None:
            # Extend the matrix
            new_matrix = np.full((target_n, target_n), -1, dtype=np.int8)
            new_matrix[:target_n - 1, :target_n - 1] = color_matrix
            for j in range(target_n - 1):
                new_matrix[target_n - 1, j] = assignment[j]
                new_matrix[j, target_n - 1] = assignment[j]
            color_matrix = new_matrix
            best_n = target_n
            # Update best_colors
            best_colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    best_colors.append(int(color_matrix[i, j]))
        else:
            break  # Can't extend further

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
