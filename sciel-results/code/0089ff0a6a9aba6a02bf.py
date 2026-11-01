import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Rank 32 decomposition using optimal block structure
    # M_{2,2,9} split as 4+5: M_{2,2,4} + M_{2,2,5} = 14 + 18 = 32
    # M_{2,2,4} = 2 * M_{2,2,2} = 2*7 = 14
    # M_{2,2,5} = M_{2,2,3} + M_{2,2,2} = 11 + 7 = 18
    #   where M_{2,2,3} = M_{2,2,2} + M_{2,2,1} = 7 + 4 = 11
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    def add_strassen(k0, k1):
        """Add Strassen decomposition for 2x2 x 2x2 block (cols k0, k1)."""
        nonlocal col
        
        # M1: (a00+a11)(b0k0+b1k1)
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M2: (a10+a11)b0k0
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        
        # M3: a00(b0k1-b1k1)
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M4: a11(b1k0-b0k0)
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        
        # M5: (a00+a01)b1k1
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        
        # M6: (a10-a00)(b0k0+b0k1)
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7: (a01-a11)(b1k0+b1k1)
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
    
    def add_trivial(k):
        """Add trivial 4-term decomposition for 2x2 x 2x1 block (col k)."""
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
    
    # Block 1: cols 0-3 (M_{2,2,4} = 2*Strassen)
    add_strassen(0, 1)
    add_strassen(2, 3)
    
    # Block 2: cols 4-8 (M_{2,2,5} = M_{2,2,3} + M_{2,2,2})
    # M_{2,2,3} for cols 4,5,6: Strassen(4,5) + Trivial(6)
    add_strassen(4, 5)
    add_trivial(6)
    # M_{2,2,2} for cols 7,8
    add_strassen(7, 8)
    
    # Verify: should have used exactly 32 columns
    assert col == 32, f"Expected 32 columns, got {col}"
    
    # Quick sanity check with numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    # Fallback to trivial if verification fails
                    rank_t = n * m * p
                    U_t = [[0] * rank_t for _ in range(n * m)]
                    V_t = [[0] * rank_t for _ in range(m * p)]
                    W_t = [[0] * rank_t for _ in range(p * n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_t[ii*m+jj][c] = 1
                                V_t[jj*p+kk][c] = 1
                                W_t[kk*n+ii][c] = 1
                                c += 1
                    return {"U": U_t, "V": V_t, "W": W_t}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
