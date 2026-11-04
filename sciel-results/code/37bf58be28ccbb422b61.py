"""
Strategy: Start with 44-term decomposition (Strassen 2x2x8 + schoolbook 2x1x8),
then attempt to find linear dependencies to reduce rank, or use a rank-8 decomposition
for 2x3x2 sub-problems to achieve rank 32.
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

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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

    # Build 28-term Strassen basis for 2x2x8
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]
    V_s = [[0] * rank_s for _ in range(m * p)]
    W_s = [[0] * rank_s for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Try to find a lower-rank decomposition by checking for linear dependencies
    # among the 44 terms, or by trying a more efficient construction
    
    def _try_find_lower_rank(U44, V44, W44, rank_44):
        """Try to find a lower-rank decomposition by identifying redundant terms."""
        # For each term, check if it can be expressed as a linear combination of others
        # This is a complex search, so let's try a few heuristics
        
        # Check if any two terms are identical (up to scaling)
        # If term r2 = c * term r1, we can eliminate one term
        for r1 in range(rank_44):
            for r2 in range(r1 + 1, rank_44):
                # Check if term r2 = c * term r1 for some rational c
                # This means U[r2] = c*U[r1], V[r2] = c*V[r1], W[r2] = c*W[r1]
                # But since we need the product to be the same, this is tricky
                
                pass
        
        # Try a different approach: use the structure of the problem
        # The 2x3x8 matrix multiplication can be decomposed as:
        # - 4 blocks of 2x2x2 (using Strassen, rank 7 each) = 28 terms
        # - Plus the a=2 part: 2x1x8 = 16 terms
        # Total: 44 terms
        
        # Can we do better? Let's try to combine the a=2 part more efficiently
        # The a=2 part contributes A[i,2]*B[2,k] to C[i,k]
        # This is a rank-1 tensor in the (i,k) space for each a=2
        # But we need to sum over a, so it's not straightforward
        
        # Alternative: use a different decomposition for the 2x3x2 subproblem
        # Let's try to find a rank-8 decomposition for 2x3x2
        
        return None
    
    # Try a more efficient decomposition: use rank-8 for 2x3x2 subproblems
    # If we can find a rank-8 decomposition for 2x3x2, we can use it 4 times
    # for a total rank of 32
    
    def _build_rank8_2x3x2():
        """Try to construct a rank-8 decomposition for 2x3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        r = 8
        
        # U: 6 x r (indices: (i,a) where i in {0,1}, a in {0,1,2})
        # V: 6 x r (indices: (a,k) where a in {0,1,2}, k in {0,1})
        # W: 4 x r (indices: (k,i) where k in {0,1}, i in {0,1})
        
        # We need: sum_r U[i,a,r] * V[a,k,r] * W[k,i,r] = 1 for all i,a,k
        
        # Try a structured approach:
        # Use 4 terms for the 2x2x2 part (a=0,1) with a simple decomposition
        # Plus 4 terms for a=2 and cross terms
        
        # For the 2x2x2 part (a=0,1), use:
        # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> contributes to C[0,0]
        # Term 1: U[0,1]=1, V[1,0]=1, W[0,0]=1 -> contributes to C[0,0]
        # Term 2: U[1,0]=1, V[0,1]=1, W[1,0]=1 -> contributes to C[1,0]
        # Term 3: U[1,1]=1, V[1,1]=1, W[1,1]=1 -> contributes to C[1,1]
        # But this doesn't give us C[0,1] and C[1,0] correctly
        
        # Let's use a different approach: Strassen for the 2x2x2 part
        # But Strassen has rank 7, so we need to be more efficient
        
        # Actually, let's try to solve this as a system of equations
        # This is complex, so let's just return the 44-term solution for now
        
        return None
    
    # Build the 44-term solution as a fallback
    rank_44 = 44
    U44 = [[0] * rank_44 for _ in range(n * m)]
    V44 = [[0] * rank_44 for _ in range(m * p)]
    W44 = [[0] * rank_44 for _ in range(p * n)]
    
    for r in range(rank_s):
        for i in range(n):
            for a in range(2):
                U44[i * m + a][r] = U_s[i * m + a][r]
        for a in range(2):
            for k in range(p):
                V44[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W44[k * n + i][r] = W_s[k * n + i][r]
    
    col = rank_s
    for i in range(n):
        for k in range(p):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()

# EVOLVE-BLOCK-END
