"""
Strategy:
1. Base: 44-term decomposition (4 Strassen blocks for 2x2x8 + schoolbook for 2x1x8).
2. Apply global linear algebraic compression: check if the 44 rank-1 tensors are
   linearly independent by building the tensor and performing Gaussian elimination.
3. If dependencies are found, eliminate them to reduce rank.
4. The key insight: while the Strassen blocks operate on disjoint output columns,
   there might be subtle linear dependencies in the full tensor space that allow
   compression beyond the obvious block structure.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    # Attempt global linear algebraic compression
    # The idea: check if the 44 rank-1 tensors are linearly independent.
    # We can do this by checking if the tensor T = sum_r U_r ⊗ V_r ⊗ W_r
    # has a decomposition of lower rank.
    # 
    # A practical approach: try to eliminate one term at a time by checking
    # if it can be expressed as a linear combination of the others.
    # This is expensive, so we'll try a heuristic: check small subsets.
    
    # Alternative approach: use the fact that the tensor lives in a space
    # of dimension nm * mp * pn = 6 * 24 * 16 = 2304.
    # The 44 rank-1 tensors span a subspace of dimension at most 44.
    # If they are linearly independent, the rank is exactly 44.
    
    # Let's try a direct approach: for each term, check if removing it
    # still allows us to represent the tensor. This is equivalent to
    # checking if the term is in the span of the others.
    
    # Due to time constraints, we'll try a limited search:
    # Check if any single term can be removed (i.e., is linearly dependent).
    
    def _try_remove_term(U_mat, V_mat, W_mat, term_idx, rank):
        """Check if removing term_idx still gives a valid decomposition."""
        if not _elapsed_ok():
            return None, None, None
        
        # Create new matrices without term_idx
        new_rank = rank - 1
        U_new = [[0] * new_rank for _ in range(nm)]
        V_new = [[0] * new_rank for _ in range(mp)]
        W_new = [[0] * new_rank for _ in range(pn)]
        
        new_col = 0
        for c in range(rank):
            if c == term_idx:
                continue
            for i in range(nm):
                U_new[i][new_col] = U_mat[i][c]
            for a in range(mp):
                V_new[a][new_col] = V_mat[a][c]
            for k in range(pn):
                W_new[k][new_col] = W_mat[k][c]
            new_col += 1
        
        candidate = {"U": U_new, "V": V_new, "W": W_new}
        if _validate(candidate):
            return U_new, V_new, W_new
        return None, None, None
    
    # Try to remove terms one at a time
    U_cur = [row[:] for row in U_44]
    V_cur = [row[:] for row in V_44]
    W_cur = [row[:] for row in W_44]
    rank_cur = 44
    
    # Try removing each term
    for term_idx in range(43, -1, -1):  # Try from last to first
        if not _elapsed_ok():
            break
        U_new, V_new, W_new = _try_remove_term(U_cur, V_cur, W_cur, term_idx, rank_cur)
        if U_new is not None:
            U_cur = U_new
            V_cur = V_new
            W_cur = W_new
            rank_cur -= 1
    
    if rank_cur < 44:
        candidate_compressed = {"U": U_cur, "V": V_cur, "W": W_cur}
        if _validate(candidate_compressed):
            return candidate_compressed
    
    return candidate_44

# EVOLVE-BLOCK-END
