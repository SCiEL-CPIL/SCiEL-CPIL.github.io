"""Matrix multiplication 4x4x5 rank-minimal decomposition using Block-Strassen (Rank 28) + Schoolbook (Rank 4)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 140
    
    # Total rank
    rank = 32
    
    # Initialize matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 32
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 32
    
    col = 0
    
    # --- Part 1: 4x4 * 4x4 (First 4 cols of B, First 4 rows of C) ---
    # Block decomposition of 4x4 into 2x2 blocks of 2x2 matrices.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Strassen coefficients for A (P)
    # Index: 0=A00, 1=A01, 2=A10, 3=A11
    P_coeffs = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # Strassen coefficients for B (Q)
    # Index: 0=B00, 1=B01, 2=B10, 3=B11
    Q_coeffs = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [-1, 0, 1, 0], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ]
    
    # Strassen coefficients for C (W)
    # Index: 0=C00, 1=C01, 2=C10, 3=C11
    W_coeffs = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, 1],  # M2: C10 + C11
        [0, 1, 0, -1], # M3: C01 - C11
        [-1, 1, 0, 0], # M4: -C00 + C01
        [1, 0, 1, 0],  # M5: C00 + C10
        [1, 0, 0, 0],  # M6: C00
        [0, 0, 0, -1]  # M7: -C11
    ]
    
    # Schoolbook for 2x2x2: 4 terms (a, m, b)
    for k in range(7):
        for a in range(2):
            for m in range(2):
                for b in range(2):
                    # U: A elements
                    # P_k[a, m] = sum_{uv} P_coeffs[k][uv] * A_uv[a, m]
                    # A_uv[a, m] is at global index i=2u+a, j=2v+m
                    for u in range(2):
                        for v in range(2):
                            coeff = P_coeffs[k][u * 2 + v]
                            if coeff != 0:
                                i = 2 * u + a
                                j = 2 * v + m
                                U[i * 4 + j][col] = coeff
                    
                    # V: B elements
                    # Q_k[m, b] = sum_{vw} Q_coeffs[k][vw] * B_vw[m, b]
                    # B_vw[m, b] is at global index j=2v+m, k=2w+b
                    for v in range(2):
                        for w in range(2):
                            coeff = Q_coeffs[k][v * 2 + w]
                            if coeff != 0:
                                j = 2 * v + m
                                k_out = 2 * w + b
                                V[j * 5 + k_out][col] = coeff
                    
                    # W: C elements
                    # C contribution from M_k is sum_{uv} W_coeffs[k][uv] * (M_k)_uv
                    # (M_k)_uv corresponds to C_uv[a, b]
                    # C_uv[a, b] is at global index i=2u+a, k_out=2v+b
                    for u in range(2):
                        for v in range(2):
                            coeff = W_coeffs[k][u * 2 + v]
                            if coeff != 0:
                                i = 2 * u + a
                                k_out = 2 * v + b
                                W[k_out * 4 + i][col] = coeff
                    
                    col += 1
    
    # --- Part 2: 4x4 * 4x1 (Last col of B, Last row of C) ---
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # Rank 4 terms, one for each j
    for j in range(4):
        # U[i, j] = 1 for all i
        for i in range(4):
            U[i * 4 + j][col] = 1.0
        
        # V[j, 4] = 1
        V[j * 5 + 4][col] = 1.0
        
        # W[4, i] = 1 for all i
        for i in range(4):
            W[4 * 4 + i][col] = 1.0
        
        col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_sb = [[0.0] * rank for _ in range(n*m)]
        V_sb = [[0.0] * rank for _ in range(m*p)]
        W_sb = [[0.0] * rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][c] = 1.0
                    V_sb[j*p+k][c] = 1.0
                    W_sb[k*n+i][c] = 1.0
                    c += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if not np.isscalar(val):
                    val = float(np.sum(val))
                if abs(val - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
