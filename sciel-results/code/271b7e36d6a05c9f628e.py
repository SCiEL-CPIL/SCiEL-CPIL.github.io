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

    # Try rank-8 decomposition for 2x3x2, then use 4 times for rank 32
    # For 2x3x2: U is 6xr, V is 6xr, W is 4xr
    # We need sum_r U[i,a,r]*V[a,k,r]*W[k,i,r] = 1 for all i,a,k
    
    # Try to construct a rank-8 decomposition
    # U indices: (i,a) in {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)} -> 6 rows
    # V indices: (a,k) in {(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)} -> 6 rows  
    # W indices: (k,i) in {(0,0),(0,1),(1,0),(1,1)} -> 4 rows
    
    # Try a systematic construction
    # Let's use the fact that we can decompose as:
    # C[i,k] = sum_a A[i,a]*B[a,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Use 2 Strassen terms for (i in {0,1}, a in {0,1}, k in {0,1})
    # Plus 6 more terms for a=2
    
    # Actually, let's try a different approach: use rank 9 for 2x3x2
    # Total rank would be 36
    
    # Let me try to find a rank-8 decomposition by solving a system
    
    def _try_rank8_2x3x2():
        """Try to find a rank-8 decomposition for 2x3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        r = 8
        
        # U: 6 x r, V: 6 x r, W: 4 x r
        # We need integer/half-integer entries
        
        # Try a parameterized construction
        # For each term r, we have vectors u_r (6-dim), v_r (6-dim), w_r (4-dim)
        
        # Let's try to use the structure:
        # Terms 0-1: handle a=0,1 with Strassen-like structure for k=0,1
        # Terms 2-7: handle a=2 and cross terms
        
        # Actually, let me just try to solve this numerically and round
        from scipy.optimize import minimize
        
        # Flatten parameters
        # U: 6*8 = 48, V: 6*8 = 48, W: 4*8 = 32, total 128 params
        # This is too many for direct optimization
        
        # Instead, let's try a structured approach
        # Use 4 terms for the 2x2x2 part (a=0,1; k=0,1) with rank 4 instead of 7
        # Plus 4 terms for a=2
        
        # For 2x2x2 with rank 4, we can use:
        # M1 = A00*B00, M2 = A01*B01, M3 = A10*B10, M4 = A11*B11
        # But this gives C00 = M1, C01 = M2, C10 = M3, C11 = M4
        # Which is correct but doesn't help for the a=2 part
        
        # Let me try a different structured approach
        # Use 7 Strassen terms for the 2x2x2 part, plus 1 term for a=2
        # Total 8 terms
        
        # But the a=2 part needs to contribute to all 4 output entries
        # C[i,k] includes A[i,2]*B[2,k]
        # So we need U[i,2,r]*V[2,k,r]*W[k,i,r] to sum to the right values
        
        # This is getting complex. Let me just try the 44-term solution and see
        # if I can find any linear dependencies.
        
        return None
    
    # Try to find linear dependencies in the 44-term decomposition
    def _find_dependencies(U44, V44, W44, rank_44):
        """Check if any subset of terms sums to zero, allowing rank reduction."""
        # For each pair of terms, check if one is a scalar multiple of the other
        # in a way that allows cancellation
        # This is a complex search, so let's just try a few heuristics
        
        # Check if any two terms are identical (up to scaling)
        for r1 in range(rank_44):
            for r2 in range(r1 + 1, rank_44):
                # Check if term r2 = c * term r1 for some c
                # This would mean we can eliminate one term
                # But this is unlikely to help directly
                
                pass
        
        return None  # No dependencies found
    
    # Build the 44-term solution
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
