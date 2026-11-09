import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Try Paley tournaments on primes p ≡ 3 (mod 4), smallest first
    # p must be ≡ 3 (mod 4) so that -1 is a non-square, ensuring antisymmetry
    candidates = [31, 43, 47, 59]

    def build_paley(p):
        """Build orientation list for Paley tournament on Z_p."""
        squares = set((x * x) % p for x in range(1, p))
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                orientation.append(0 if diff in squares else 1)
        return orientation

    def is_valid_paley(p):
        """Check if every 4-set has a common dominator in the Paley tournament on Z_p."""
        squares = set((x * x) % p for x in range(1, p))
        is_square = np.zeros(p, dtype=bool)
        for s in squares:
            is_square[s] = True

        # out[v, u] = True if v -> u (i.e., (u-v) % p is a quadratic residue)
        # out_mask: for each vertex v, a set of vertices it dominates
        # Use numpy for the check
        # out[v, u] = is_square[(u - v) % p] for v != u
        out = np.zeros((p, p), dtype=bool)
        for v in range(p):
            for u in range(p):
                if v == u:
                    continue
                diff = (u - v) % p
                out[v, u] = is_square[diff]

        # For each 4-set {a,b,c,d}, check if there exists v such that out[v,a] and out[v,b] and out[v,c] and out[v,d]
        # Vectorize: for each combination of 4 vertices, check OR over v of AND over 4
        from itertools import combinations
        verts = range(p)
        for a, b, c, d in combinations(verts, 4):
            # dominators of a, b, c, d
            # v dominates a iff out[v, a]
            dom_a = out[:, a]
            dom_b = out[:, b]
            dom_c = out[:, c]
            dom_d = out[:, d]
            common = dom_a & dom_b & dom_c & dom_d
            if not common.any():
                return False
        return True

    best_n = 67
    best_orientation = build_paley(67)

    for p in candidates:
        if time.time() > deadline:
            break
        if is_valid_paley(p):
            best_n = p
            best_orientation = build_paley(p)
            break

    return {'n': best_n, 'orientation': best_orientation}
