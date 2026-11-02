import numpy as np
from itertools import product, combinations
from collections import defaultdict
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}

    entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def get_pattern_coverage(su, sv, sw):
        covered = set()
        for a in su:
            i, j = divmod(a, m)
            for b in sv:
                j2, k = divmod(b, p)
                if j == j2:
                    for c in sw:
                        k2, i2 = divmod(c, n)
                        if k == k2 and i == i2:
                            covered.add((i, j, k))
        return covered
    
    def try_pair(e1, e2):
        i1, j1, k1 = e1
        i2, j2, k2 = e2
        su = [i1 * m + j1, i2 * m + j2]
        sv = [j1 * p + k1, j2 * p + k2]
        sw = [k1 * n + i1, k2 * n + i2]
        covered = get_pattern_coverage(su, sv, sw)
        if covered == {e1, e2}:
            return (tuple(su), tuple(sv), tuple(sw))
        return None
    
    # Generate more pairings to try
    # Try all possible ways to pick 3 disjoint pairs from the 18 entries
    # But that's too many, so let's be strategic
    
    # First, let's find all valid pairs (pairs that can be covered by a clean pattern)
    valid_pairs = []
    for idx1 in range(len(entries)):
        for idx2 in range(idx1 + 1, len(entries)):
            e1, e2 = entries[idx1], entries[idx2]
            pat = try_pair(e1, e2)
            if pat is not None:
                valid_pairs.append((e1, e2, pat))
    
    # Now find 3 disjoint valid pairs
    # Use backtracking
    def find_three_disjoint_pairs(pairs, used, result, depth):
        if depth == 3:
            return result[:]
        for e1, e2, pat in pairs:
            if e1 not in used and e2 not in used:
                new_used = used | {e1, e2}
                result.append((e1, e2, pat))
                sol = find_three_disjoint_pairs(pairs, new_used, result, depth + 1)
                if sol is not None:
                    return sol
                result.pop()
        return None
    
    solution = find_three_disjoint_pairs(valid_pairs, set(), [], 0)
    
    if solution is not None:
        rank = 15
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        used_entries = set()
        for e1, e2, (su, sv, sw) in solution:
            for a in su:
                U[a][col] = 1
            for b in sv:
                V[b][col] = 1
            for c in sw:
                W[c][col] = 1
            col += 1
            used_entries.add(e1)
            used_entries.add(e2)
        
        for e in entries:
            if e not in used_entries:
                i, j, k = e
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Verify
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            su_idx = [a for a in range(n * m) if U[a][c] != 0]
            sv_idx = [b for b in range(m * p) if V[b][c] != 0]
            sw_idx = [c_idx for c_idx in range(p * n) if W[c_idx][c] != 0]
            covered = get_pattern_coverage(su_idx, sv_idx, sw_idx)
            for e in covered:
                tensor[e] += 1.0
        
        if np.allclose(tensor, 1.0):
            return {"U": U, "V": V, "W": W}
    
    return fallback
# EVOLVE-BLOCK-END
