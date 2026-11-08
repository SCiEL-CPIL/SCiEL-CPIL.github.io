import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Define Strassen terms
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(i0, jA0, jB0, k0):
        nonlocal col
        # Mapping functions
        def u_idx(r, c): return (i0 + r) * m + (jA0 + c)
        def v_idx(r, c): return (jB0 + r) * p + (k0 + c)
        def w_idx(r, c): return (k0 + c) * n + (i0 + r)
        
        # M1
        U[u_idx(0,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1
        
        # M2
        U[u_idx(1,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        W[w_idx(1,0)][col] += 1
        col += 1
        
        # M3
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] -= 1
        W[w_idx(0,1)][col] += 1
        col += 1
        
        # M4
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,0)][col] += 1
        V[v_idx(0,0)][col] -= 1
        W[w_idx(0,0)][col] += 1
        W[w_idx(1,0)][col] += 1
        col += 1
        
        # M5
        U[u_idx(0,0)][col] += 1
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] -= 1
        W[w_idx(0,1)][col] += 1
        col += 1
        
        # M6
        U[u_idx(1,0)][col] += 1
        U[u_idx(0,0)][col] -= 1
        V[v_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1
        
        # M7
        U[u_idx(0,1)][col] += 1
        U[u_idx(1,1)][col] -= 1
        V[v_idx(1,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1

    # 8 blocks
    # 1: A[0:2,0:2] B[0:2,0:2] C[0:2,0:2]
    add_strassen(0, 0, 0, 0)
    # 2: A[0:2,2:4] B[2:4,0:2] C[0:2,0:2]
    add_strassen(0, 2, 2, 0)
    # 3: A[0:2,0:2] B[0:2,2:4] C[0:2,2:4]
    add_strassen(0, 0, 0, 2)
    # 4: A[0:2,2:4] B[2:4,2:4] C[0:2,2:4]
    add_strassen(0, 2, 2, 2)
    # 5: A[0:2,0:2] B[0:2,4:6] C[0:2,4:6]
    add_strassen(0, 0, 0, 4)
    # 6: A[0:2,2:4] B[2:4,4:6] C[0:2,4:6]
    add_strassen(0, 2, 2, 4)
    # 7: A[0:2,0:2] B[0:2,6:8] C[0:2,6:8]
    add_strassen(0, 0, 0, 6)
    # 8: A[0:2,2:4] B[2:4,6:8] C[0:2,6:8]
    add_strassen(0, 2, 2, 6)

    return {"U": U, "V": V, "W": W}
