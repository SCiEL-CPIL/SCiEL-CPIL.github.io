"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _clebsch_coloring():
    """Return 3-coloring of K16 with no monochromatic triangle (Clebsch graph per color)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(16):
        for j in range(i + 1, 16):
            colors.append(log[i ^ j] % 3)
    return colors

def _check_triangle_free(colors, n):
    """colors: list for pairs (i,j) i<j in lex order. Return True if no monochromatic triangle."""
    import numpy as np
    M = np.zeros((n, n), dtype=np.int8)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = colors[k]
            k += 1
    for c in range(4):
        A = (M == c)
        # triangle exists iff A has a 3-clique; check via A @ A @ A diagonal or A^3
        # A[i,j]=1 if edge of color c. Triangle on (i,j,k) iff A[i,j]&A[j,k]&A[k,i]
        # Use: for each i, neighbors of i in color c, check if any pair among them is connected in color c
        for i in range(n):
            nbrs = [j for j in range(n) if j != i and A[i, j]]
            # check pairs in nbrs
            for a in range(len(nbrs)):
                for b in range(a + 1, len(nbrs)):
                    if A[nbrs[a], nbrs[b]]:
                        return False
    return True

def _edge_index(n, i, j):
    if i > j:
        i, j = j, i
    # number of pairs before (i, j): pairs with first < i, plus within i with second < j
    # pairs with first vertex < i: sum_{v=0}^{i-1} (n-1-v) = i*(n-1) - i*(i-1)//2
    cnt = i * (n - 1) - i * (i - 1) // 2
    # within vertex i, pairs (i, k) for k < j, k>i : (j - i - 1)
    cnt += (j - i - 1)
    return cnt

def _add_vertex(colors, n, c_new=None):
    """Try to add a new vertex n with colors c_new[i] for edge (i,n). Return new colors or None."""
    # check for monochromatic triangle involving new vertex
    for i in range(n):
        for j in range(i):
            ci = c_new[i]
            cj = c_new[j]
            cij = colors[_edge_index(n, i, j)]
            if ci == cj == cij:
                return None
    # also check within existing: none added, so fine
    new_colors = list(colors)
    for i in range(n):
        new_colors.append(c_new[i])
    return new_colors

def _try_grow(colors, n, deadline):
    """Greedy + backtracking-ish: try to add vertices until deadline."""
    import random
    rng = random.Random(12345)
    best_n = n
    best_colors = list(colors)
    for step in range(200000):
        if time.time() > deadline:
            break
        # try to add one vertex
        added = False
        for attempt in range(50):
            c_new = [rng.randint(0, 3) for _ in range(n)]
            nc = _add_vertex(best_colors, n, c_new)
            if nc is not None:
                best_colors = nc
                n += 1
                best_n = n
                added = True
                break
        if not added:
            # try random restart of last vertex color via local search
            # do a few local improvements: pick a random vertex, reassign its colors to try to keep
            # Actually just break to avoid infinite loop; try a different random seed
            # We'll do a light local search: try flipping one color of new vertex
            # Since we can't add, stop growing
            break
    return best_n, best_colors

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    # Start from Clebsch-based n=32 coloring
    c16 = _clebsch_coloring()
    # double: two copies of 16, cross edges color 3
    n = 32
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(c16[_edge_index(16, i % 16, j % 16)])
            else:
                colors.append(3)
    # Verify
    if not _check_triangle_free(colors, n):
        # fallback to n=16
        n = 16
        colors = c16
    # Try to grow
    n, colors = _try_grow(colors, n, deadline)
    # Final verify
    if not _check_triangle_free(colors, n):
        # fall back to the known-good n=32
        n = 32
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(c16[_edge_index(16, i % 16, j % 16)])
                else:
                    colors.append(3)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
