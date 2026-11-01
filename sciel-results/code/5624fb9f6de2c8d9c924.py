"""Rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication."""
DIMENSIONS = [2, 2, 3]

import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return max absolute error of the tensor reconstruction."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _snap_half_int(x):
    """Snap a float to the nearest half-integer."""
    return int(round(2 * x)) / 2.0

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---- Strassen decomposition for 2x2 * 2x2 (rank 7) ----
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Products:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(f-h)
    #   M4 = d(g-e)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    # Output C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6

    # U rows indexed by (i,j), V rows by (j,k), W rows by (k,i)
    # For 2x2*2x2: U[i*2+j], V[j*2+k], W[k*2+i]

    # Strassen columns 0..6
    strassen_U = [
        # i=0,j=0 (a): M1->1, M3->1, M5->1, M6->1
        [1, 0, 1, 0, 1, 1, 0],
        # i=0,j=1 (b): M5->1, M7->1
        [0, 0, 0, 0, 1, 0, 1],
        # i=1,j=0 (c): M2->1, M4->1, M6->-1
        [0, 1, 0, 1, 0, -1, 0],
        # i=1,j=1 (d): M1->1, M2->1, M4->1, M7->-1
        [1, 1, 0, 1, 0, 0, -1],
    ]
    strassen_V = [
        # j=0,k=0 (e): M1->1, M2->1, M4->-1, M6->1
        [1, 1, 0, -1, 0, 1, 0],
        # j=0,k=1 (f): M3->1, M5->-1, M6->1
        [0, 0, 1, 0, -1, 1, 0],
        # j=1,k=0 (g): M2->0, M4->1, M7->1
        [0, 0, 0, 1, 0, 0, 1],
        # j=1,k=1 (h): M1->1, M5->1, M7->1
        [1, 0, 0, 0, 1, 0, 1],
    ]
    strassen_W = [
        # k=0,i=0: C[0,0]=M1+M4-M5+M7, C[1,0]=M2+M4
        [1, 1, 0, 1, -1, 0, 1],
        # k=0,i=1: C[1,0]=M2+M4, C[0,0]=M1+M4-M5+M7
        [1, 1, 0, 1, -1, 0, 1],
        # k=1,i=0: C[0,1]=M3+M5, C[1,1]=M1-M2+M3+M6
        [1, -1, 1, 0, 1, 1, 0],
        # k=1,i=1: C[1,1]=M1-M2+M3+M6, C[0,1]=M3+M5
        [1, -1, 1, 0, 1, 1, 0],
    ]

    # ---- Schoolbook decomposition for 2x2 * 2x1 (rank 4) ----
    # A = [[a, b], [c, d]], B = [[e], [g]]
    # C[0,0] = a*e + b*g, C[1,0] = c*e + d*g
    # U rows: (i,j) -> a,b,c,d ; V rows: (j,k) with k=0 -> e,g ; W rows: (k,i) with k=0
    # Columns 7,8,9,10 correspond to (a*e), (b*g), (c*e), (d*g)
    schoolbook_U = [
        # i=0,j=0 (a): col7
        [1, 0, 0],
        # i=0,j=1 (b): col8
        [0, 1, 0],
        # i=1,j=0 (c): col9
        [0, 0, 1],
        # i=1,j=1 (d): col10
        [0, 0, 0],
    ]
    # Wait, I need 4 columns for schoolbook. Let me redo.
    # Columns 7,8,9,10:
    # col7: a*e -> U[0,0]=1, V[0,0]=1, W[0,0]=1  (i=0,j=0,k=0)
    # col8: b*g -> U[0,1]=1, V[1,0]=1, W[0,0]=1  (i=0,j=1,k=0)
    # col9: c*e -> U[1,0]=1, V[0,0]=1, W[0,1]=1  (i=1,j=0,k=0)
    # col10: d*g -> U[1,1]=1, V[1,0]=1, W[0,1]=1 (i=1,j=1,k=0)

    schoolbook_U = [
        # i=0,j=0 (a): col7
        [1, 0, 0, 0],
        # i=0,j=1 (b): col8
        [0, 1, 0, 0],
        # i=1,j=0 (c): col9
        [0, 0, 1, 0],
        # i=1,j=1 (d): col10
        [0, 0, 0, 1],
    ]
    schoolbook_V = [
        # j=0,k=0 (e): col7, col9
        [1, 0, 1, 0],
        # j=1,k=0 (g): col8, col10
        [0, 1, 0, 1],
    ]
    schoolbook_W = [
        # k=0,i=0: C[0,0] = col7 + col8
        [1, 1, 0, 0],
        # k=0,i=1: C[1,0] = col9 + col10
        [0, 0, 1, 1],
    ]

    # ---- Assemble full matrices ----
    # U has n*m = 4 rows, rank 11 columns
    # V has m*p = 6 rows, rank 11 columns
    # W has p*n = 6 rows, rank 11 columns
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen: columns 0-6
    # U indices: i*m+j. m=2.
    # Rows: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    for row in range(4):
        for c in range(7):
            U[row][c] = strassen_U[row][c]
            
    # V indices: j*p+k. p=3.
    # strassen_V rows: 0(j=0,k=0), 1(j=0,k=1), 2(j=1,k=0), 3(j=1,k=1).
    # Target indices: 0, 1, 3, 4.
    v_indices = [0, 1, 3, 4]
    for i, row in enumerate(v_indices):
        for c in range(7):
            V[row][c] = strassen_V[i][c]
            
    # W indices: k*n+i. n=2.
    # strassen_W rows: 0(k=0,i=0), 1(k=0,i=1), 2(k=1,i=0), 3(k=1,i=1).
    # Target indices: 0, 1, 2, 3.
    for row in range(4):
        for c in range(7):
            W[row][c] = strassen_W[row][c]

    # Schoolbook: columns 7-10 for k=2
    # Term 1: i=0, j=0, k=2 -> U[0,7]=1, V[2,7]=1, W[4,7]=1
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 2: i=0, j=1, k=2 -> U[1,8]=1, V[5,8]=1, W[4,8]=1
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 3: i=1, j=0, k=2 -> U[2,9]=1, V[2,9]=1, W[5,9]=1
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 4: i=1, j=1, k=2 -> U[3,10]=1, V[5,10]=1, W[5,10]=1
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    # Verify
    err = _verify(U, V, W, n, m, p)
    if err < 1e-9:
        # Snap to half-integers
        U_out = [[_snap_half_int(x) for x in row] for row in U]
        V_out = [[_snap_half_int(x) for x in row] for row in V]
        W_out = [[_snap_half_int(x) for x in row] for row in W]
        # Re-verify after snapping
        err2 = _verify(U_out, V_out, W_out, n, m, p)
        if err2 < 1e-9:
            return {"U": U_out, "V": V_out, "W": W_out}
        else:
            return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook rank 12
        U_fb = [[0] * 12 for _ in range(n * m)]
        V_fb = [[0] * 12 for _ in range(m * p)]
        W_fb = [[0] * 12 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
