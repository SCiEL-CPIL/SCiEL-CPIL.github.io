import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_2x2x2():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        This gives rank 7 instead of rank 8.
        """
        # Strassen's 7 multiplication formulas
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # U matrix: 4x7 (rows: a11, a12, a21, a22)
        U_strassen = [
            [1, 0, 1, 0, 1, -1, 0],  # a11
            [0, 0, 0, 0, 1, 0, -1],  # a12
            [0, 1, 0, 0, 0, 1, 0],   # a21
            [1, 1, 0, 1, 0, 0, -1]   # a22
        ]
        
        # V matrix: 4x7 (rows: b11, b12, b21, b22)
        V_strassen = [
            [1, 1, 0, -1, 0, 1, 0],  # b11
            [0, 0, 1, 0, 0, 1, 0],   # b12
            [0, 0, 0, 1, 0, 0, 1],   # b21
            [1, 0, -1, 0, 1, 0, 1]   # b22
        ]
        
        # W matrix: 4x7 (rows: c11, c12, c21, c22)
        # c11 = m1 - m4 + m6 - m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m3 + m5 + m7
        W_strassen = [
            [1, 0, 0, -1, 0, 1, -1],  # c11
            [0, 0, 1, 0, 1, 0, 0],    # c12
            [0, 1, 0, 1, 0, 0, 0],    # c21
            [1, 0, -1, 0, 1, 0, 1]    # c22
        ]
        
        return U_strassen, V_strassen, W_strassen

    def build_rank_29_decomposition():
        """
        Build a rank-29 decomposition by using Strassen for the 2x2x2 sub-block
        and schoolbook for the remaining terms.
        
        The 2x3x5 multiplication can be viewed as:
        - 2x2x2 block (rows 0-1, cols 0-1) using Strassen (rank 7)
        - Remaining terms using schoolbook
        
        Total: 7 + (2*3*5 - 2*2*2) = 7 + (30-8) = 29
        """
        rank = 29
        U = [[0] * rank for _ in range(n * m)]  # 6x29
        V = [[0] * rank for _ in range(m * p)]  # 15x29
        W = [[0] * rank for _ in range(p * n)]  # 10x29
        
        # First 7 columns: Strassen for 2x2x2 sub-block
        U_strassen, V_strassen, W_strassen = get_strassen_2x2x2()
        
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = U_strassen[i * 2 + j][r]
                    V[j * p + r][r] = V_strassen[j * 2 + r][r]  # Wait, this is wrong
                    # Actually V should be indexed by (j, k) where j is 0-1 for the 2x2 block
                    # Let me fix this
                    break
                break
            break
        
        # Let me redo this more carefully
        # For the 2x2x2 sub-block:
        # U indices: (i, j) where i in {0,1}, j in {0,1} -> rows 0,1,3,4 in U (since U is 6x, indexed by i*m+j = i*3+j)
        # V indices: (j, k) where j in {0,1}, k in {0,1} -> rows 0,1,3,4 in V (since V is 15x, indexed by j*p+k = j*5+k)
        # W indices: (k, i) where k in {0,1}, i in {0,1} -> rows 0,1,5,6 in W (since W is 10x, indexed by k*n+i = k*2+i)
        
        # Reset
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen part (first 7 columns)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = U_strassen[i * 2 + j][r]
            for j in range(2):
                for k in range(2):
                    V[j * p + k][r] = V_strassen[j * 2 + k][r]
            for k in range(2):
                for i in range(2):
                    W[k * n + i][r] = W_strassen[k * 2 + i][r]
        
        # Schoolbook part for remaining terms (columns 7-28)
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if this is part of the 2x2x2 sub-block
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify we used exactly 29 columns
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return {"U": U, "V": V, "W": W}

    def validate_exact(U, V, W):
        """
        Validate using exact Fraction arithmetic.
        """
        try:
            rank = len(U[0])
            
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                        if s != 1:
                            return False
            return True
        except Exception:
            return False

    def validate_float(U, V, W, tol=1e-9):
        """
        Fast float validation for candidate generation.
        """
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T_recon, 1.0, atol=tol):
            return False
        return True

    # Try the structured rank-29 decomposition
    result = build_rank_29_decomposition()
    
    # Validate with float first (fast)
    if validate_float(result["U"], result["V"], result["W"]):
        # Convert to Fractions for exact validation
        U_frac = [[Fraction(x) for x in row] for row in result["U"]]
        V_frac = [[Fraction(x) for x in row] for row in result["V"]]
        W_frac = [[Fraction(x) for x in row] for row in result["W"]]
        
        # Exact validation
        if validate_exact(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
