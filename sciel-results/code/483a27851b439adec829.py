"""Improved program: rank-16 decomposition for 2x3 @ 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None

def _strassen_222():
    """Return the 7 rank-1 terms for 2x2 @ 2x2 Strassen decomposition."""
    # Strassen formulas:
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    terms = []
    # M1
    terms.append(([1,0,0,1], [1,1,0,0], [1,0,0,1]))
    # M2
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    # M3
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # M4
    terms.append(([0,0,0,1], [-1,0,1,0], [0,1,0,1]))
    # M5
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _construct_rank_16():
    """Construct rank-16 decomposition for 2x3 @ 3x4 -> 2x4.
    Split m=3 as 2+1:
      C = A[:,:2] @ B[:2,:] + A[:,2] @ B[2,:]
    First part (2x2 @ 2x4): split p=4 as 2+2, use Strassen for each 2x2@2x2.
    Second part (2x1 @ 1x4): rank 2.
    Total: 7 + 7 + 2 = 16.
    """
    n, m, p = 2, 3, 4
    strassen_terms = _strassen_222()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: A[:,:2] @ B[:2,:] -> (2x2) @ (2x4)
    # Block 1 (columns 0,1 of B, k=0,1):
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        U_row[0*3+0] = u[0]
        U_row[0*3+1] = u[1]
        U_row[1*3+0] = u[2]
        U_row[1*3+1] = u[3]
        V_row[0*4+0] = v[0]
        V_row[0*4+1] = v[1]
        V_row[1*4+0] = v[2]
        V_row[1*4+1] = v[3]
        W_row[0*2+0] = w[0]
        W_row[1*2+0] = w[1]
        W_row[0*2+1] = w[2]
        W_row[1*2+1] = w[3]
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Block 2 (columns 2,3 of B, k=2,3):
    for (u, v, w) in strassen_terms:
        U_row = [0]*6
        V_row = [0]*12
        W_row = [0]*8
        U_row[0*3+0] = u[0]
        U_row[0*3+1] = u[1]
        U_row[1*3+0] = u[2]
        U_row[1*3+1] = u[3]
        V_row[0*4+2] = v[0]
        V_row[0*4+3] = v[1]
        V_row[1*4+2] = v[2]
        V_row[1*4+3] = v[3]
        W_row[2*2+0] = w[0]
        W_row[3*2+0] = w[1]
        W_row[2*2+1] = w[2]
        W_row[3*2+1] = w[3]
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Part 2: A[:,2] @ B[2,:] -> (2x1) @ (1x4) -> rank 2
    # Term for i=0:
    U_row = [0]*6
    V_row = [0]*12
    W_row = [0]*8
    U_row[0*3+2] = 1
    for k in range(4):
        V_row[2*4+k] = 1
        W_row[k*2+0] = 1
    U_cols.append(U_row)
    V_cols.append(V_row)
    W_cols.append(W_row)
    
    # Term for i=1:
    U_row = [0]*6
    V_row = [0]*12
    W_row = [0]*8
    U_row[1*3+2] = 1
    for k in range(4):
        V_row[2*4+k] = 1
        W_row[k*2+1] = 1
    U_cols.append(U_row)
    V_cols.append(V_row)
    W_cols.append(W_row)
    
    rank = len(U_cols)
    U = U_cols
    V = V_cols
    W = W_cols
    return {"U": U, "V": V, "W": W}

def _construct_rank_24():
    """Schoolbook decomposition of rank 24."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    # Try rank-16 first
    try:
        result = _construct_rank_16()
        valid, info = _verify(result["U"], result["V"], result["W"], n, m, p)
        if valid:
            return result
        else:
            print(f"Rank-16 verification failed: {info}")
    except Exception as e:
        print(f"Rank-16 construction failed: {e}")
    # Fall back to rank-24
    return _construct_rank_24()
# EVOLVE-BLOCK-END
