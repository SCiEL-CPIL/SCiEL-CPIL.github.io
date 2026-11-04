"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen helper
    def add_strassen(i0, j0, k0):
        nonlocal col
        # U indices
        u_00 = i0 * m + j0
        u_01 = i0 * m + j0 + 1
        u_10 = (i0 + 1) * m + j0
        u_11 = (i0 + 1) * m + j0 + 1
        
        # V indices
        v_00 = j0 * p + k0
        v_01 = j0 * p + k0 + 1
        v_10 = (j0 + 1) * p + k0
        v_11 = (j0 + 1) * p + k0 + 1
        
        # W indices (k*n + i)
        w_00 = k0 * n + i0       # C00
        w_10 = k0 * n + i0 + 1   # C10
        w_01 = (k0 + 1) * n + i0 # C01
        w_11 = (k0 + 1) * n + i0 + 1 # C11
        
        # M1: (A00+A11)(B00+B11)
        U[u_00][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_10][col] += 1
        U[u_11][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1
        W[w_11][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_00][col] += 1
        V[v_01][col] += 1
        V[v_11][col] -= 1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_11][col] += 1
        V[v_10][col] += 1
        V[v_00][col] -= 1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_00][col] += 1
        U[u_01][col] += 1
        V[v_11][col] += 1
        W[w_00][col] -= 1
        W[w_01][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_10][col] += 1
        U[u_00][col] -= 1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_01][col] += 1
        U[u_11][col] -= 1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        col += 1

    # 3 Strassen blocks for A_L (2x2) and B_T (2x6)
    # A_L uses A cols 0,1. B_T uses B rows 0,1.
    # B_T is split into 3 2x2 blocks: cols 0-1, 2-3, 4-5.
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 0, 4)
    
    # A_R (2x1) and B_B (1x6)
    # A_R is A col 2. B_B is B row 2.
    # C_ik = A_i2 * B_2k
    for k in range(6):
        # i=0
        u_idx = 0 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 0
        U[u_idx][col] += 1
        V[v_idx][col] += 1
        W[w_idx][col] += 1
        col += 1
        
        # i=1
        u_idx = 1 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 1
        U[u_idx][col] += 1
        V[v_idx][col] += 1
        W[w_idx][col] += 1
        col += 1

    # Verify decomposition correctness
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r] so W index is (k,i) not (i,k)
    # Let me check: output factor follows W[k*n+i, r]
    # So the tensor element for (i,j,k) should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # every (i,j,k) triple contributes exactly once
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-10:
                    errors += 1
    
    if errors > 0:
        # Fallback: return schoolbook decomposition (rank 36)
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col_sb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col_sb] = 1
                    V_sb[j*p+k][col_sb] = 1
                    W_sb[k*n+i][col_sb] = 1
                    col_sb += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
