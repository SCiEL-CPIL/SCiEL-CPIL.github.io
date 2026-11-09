"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_TIME_LIMIT = 140.0
_START = None


def _elapsed():
    return time.monotonic() - _START


def _deadline():
    return _elapsed() > _TIME_LIMIT


def construct():
    global _START
    _START = time.monotonic()

    # GF(16) arithmetic with primitive polynomial x^4 + x + 1
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build GF(16) tables
    gf_log = {}
    gf_exp = [0] * 15
    x = 1
    for e in range(15):
        gf_log[x] = e
        gf_exp[e] = x
        x = gf16_mul(x, 2)

    # The 3 cosets of the cubic subgroup (order 5) in GF(16)* (order 15)
    cubes = set()
    for k in range(5):
        cubes.add(gf_exp[3 * k % 15])
    cosets = []
    for j in range(3):
        coset = set()
        for c in cubes:
            coset.add(gf16_mul(c, gf_exp[j]))
        cosets.append(coset)

    # Build 16x16 base coloring (3 colors, no mono triangles)
    base16 = np.full((16, 16), -1, dtype=np.int8)
    for a in range(16):
        for b in range(a + 1, 16):
            diff = a ^ b
            if diff == 0:
                continue
            d = gf_log[diff]
            for c in range(3):
                if gf_exp[d] in cosets[c]:
                    base16[a, b] = c
                    base16[b, a] = c
                    break

    # Build 32-vertex coloring: two copies, cross-edges color 3
    base32 = np.full((32, 32), -1, dtype=np.int8)
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base32[i, j] = base32[j, i] = base16[i % 16, j % 16]
            else:
                base32[i, j] = base32[j, i] = 3

    best_n = 32
    best_colors = [int(base32[i, j]) for i in range(32) for j in range(i + 1, 32)]
    color_matrix = base32.copy()

    # Backtracking to add vertices one by one
    def try_add_vertex(matrix, n):
        """Try to assign colors to edges (n, j) for j in 0..n-1."""
        # Precompute: for each j, which colors are "dangerous" given other assignments
        # Use backtracking with constraint propagation
        
        assignment = [-1] * n
        # Order vertices by degree of constraints (most constrained first)
        # For each j, count how many other vertices k have matrix[j][k] set
        # All vertices have same number of constraints, so use natural order
        
        # For better performance, use randomized restarts
        np.random.seed(int(time.monotonic() * 1000) % (2**31))
        
        for attempt in range(5000):
            if _deadline():
                return None
            
            # Random initialization with bias
            assignment = np.random.randint(0, 4, size=n, dtype=np.int8)
            
            # Check if valid
            valid = True
            for a in range(n):
                for b in range(a + 1, n):
                    if assignment[a] == assignment[b] == matrix[a, b]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return assignment.tolist()
            
            # Local search: try to fix conflicts
            for step in range(100):
                if _deadline():
                    return None
                
                # Find a conflict
                conflict = None
                for a in range(n):
                    for b in range(a + 1, n):
                        if assignment[a] == assignment[b] == matrix[a, b]:
                            conflict = (a, b)
                            break
                    if conflict:
                        break
                
                if conflict is None:
                    return assignment.tolist()
                
                # Try changing one of the conflicting vertices
                a, b = conflict
                for v in [a, b]:
                    old = assignment[v]
                    for new_c in range(4):
                        if new_c == old:
                            continue
                        assignment[v] = new_c
                        # Quick check if this creates new conflicts
                        ok = True
                        for w in range(n):
                            if w == v:
                                continue
                            if assignment[v] == assignment[w] == matrix[v, w]:
                                ok = False
                                break
                        if ok:
                            break
                        assignment[v] = old
            
            # If still not valid after local search, restart
            continue
        
        return None

    # Try to extend from 32 to 50+
    for target_n in range(33, 63):
        if _deadline():
            break
        
        assignment = try_add_vertex(color_matrix, target_n - 1)
        if assignment is not None:
            new_matrix = np.full((target_n, target_n), -1, dtype=np.int8)
            new_matrix[:target_n - 1, :target_n - 1] = color_matrix
            for j in range(target_n - 1):
                new_matrix[target_n - 1, j] = assignment[j]
                new_matrix[j, target_n - 1] = assignment[j]
            color_matrix = new_matrix
            best_n = target_n
            best_colors = [int(color_matrix[i, j]) for i in range(target_n) for j in range(i + 1, target_n)]
        else:
            # Try a few more times with different random seeds
            found = False
            for retry in range(3):
                if _deadline():
                    break
                assignment = try_add_vertex(color_matrix, target_n - 1)
                if assignment is not None:
                    new_matrix = np.full((target_n, target_n), -1, dtype=np.int8)
                    new_matrix[:target_n - 1, :target_n - 1] = color_matrix
                    for j in range(target_n - 1):
                        new_matrix[target_n - 1, j] = assignment[j]
                        new_matrix[j, target_n - 1] = assignment[j]
                    color_matrix = new_matrix
                    best_n = target_n
                    best_colors = [int(color_matrix[i, j]) for i in range(target_n) for j in range(i + 1, target_n)]
                    found = True
                    break
            if not found:
                break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
