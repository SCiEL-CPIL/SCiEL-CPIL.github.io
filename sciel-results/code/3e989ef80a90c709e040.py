import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        for idx, val in u_sparse:
            u[idx] = val
        for idx, val in v_sparse:
            v[idx] = val
        for idx, val in w_sparse:
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def add_schoolbook(i, j, k):
        add_term([(i*m+j, 1)], [(j*p+k, 1)], [(k*n+i, 1)])

    def strassen_2x2(i0, j0, k0):
        """7-term Strassen for 2x2x2 block at offsets (i0,j0,k0)."""
        # M1 = (A11+A22)(B11+B22)
        add_term(
            [(i0*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
            [(j0*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
        )
        # M2 = (A21+A22)B11
        add_term(
            [((i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
            [(j0*p+k0, 1)],
            [(k0*n+(i0+1), 1), ((k0+1)*n+(i0+1), -1)]
        )
        # M3 = A11(B21-B11)
        add_term(
            [(i0*m+j0, 1)],
            [(j0*p+(k0+1), 1), ((j0+1)*p+(k0+1), -1)],
            [(k0*n+i0, 1), ((k0+1)*n+i0, 1)]
        )
        # M4 = A22(B11-B22)
        add_term(
            [((i0+1)*m+(j0+1), 1)],
            [((j0+1)*p+k0, 1), (j0*p+k0, -1)],
            [(k0*n+i0, 1), ((k0+1)*n+i0, 1)]
        )
        # M5 = (A11+A12)B22
        add_term(
            [(i0*m+j0, 1), (i0*m+(j0+1), 1)],
            [((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, -1), ((k0+1)*n+i0, 1)]
        )
        # M6 = (A21-A11)(B11+B12)
        add_term(
            [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
            [(j0*p+k0, 1), (j0*p+(k0+1), 1)],
            [(k0*n+(i0+1), 1)]
        )
        # M7 = (A12-A22)(B21+B22)
        add_term(
            [(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)],
            [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, 1)]
        )

    # 3x3x3 Strassen: 4 corner 2x2 blocks + edge handling + center
    # Corners: (0,0,0), (0,0,1), (0,1,0), (0,1,1) for i in {0,1}, j in {0,1}, k in {0,1}
    # Wait, 3x3 means i,j,k each 0,1,2. 2x2 blocks at (0,0), (0,1), (1,0), (1,1) covering i,j,k in {0,1}
    # That covers i∈{0,1}, j∈{0,1}, k∈{0,1} with 4*7=28 terms.
    # But 3x3x3 Strassen is 19 terms total, not 28+edges.
    
    # Let me use the standard 3x3x3 Strassen algorithm (19 terms)
    # It uses 4 2x2 Strassen blocks (28 terms) but with shared structure
    # Actually the 3x3x3 Strassen is: 
    #   - 4 corner 2x2x2 blocks: but that's 28
    #   - The standard result is 19 for 3x3x3
    
    # Let me just use 2x2 Strassen blocks + schoolbook for the rest
    # and count carefully.
    
    covered = set()
    
    # 2x2 Strassen blocks covering i∈{0,1}, j∈{0,1,2}, k∈{0,1,2}
    for i0 in [0]:
        for j0 in [0, 1]:
            for k0 in [0, 1]:
                strassen_2x2(i0, j0, k0)
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0+di, j0+dj, k0+dk))
    
    # Schoolbook for all remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]
    
    def verify(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    max_err = verify(U, V, W)
    if max_err > 1e-9:
        # Fallback to schoolbook
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
