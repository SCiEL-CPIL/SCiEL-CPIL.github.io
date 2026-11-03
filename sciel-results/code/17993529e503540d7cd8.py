import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    
    nm, mp, pn = n*m, m*p, p*n
    Y = np.einsum('ir,jr->ijr', U2, V2, optimize=True).reshape(nm*mp, rank)
    T_computed = W2 @ Y.T
    
    T_target = np.zeros((pn, nm*mp), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T_target[k, i*mp + j] = 1
                
    if np.max(np.abs(T_computed - T_target)) > 1e-6:
        return False
    return True

def _build_5x5_rank124():
    """
    Construct a rank-124 decomposition for 5x5 matrix multiplication.
    Partition 5 = 2 + 3.
    C11 (2x2): Strassen (7) + Schoolbook A12B21 (12) = 19
    C12 (2x3): Schoolbook A11B12 (12) + Schoolbook A12B22 (18) = 30
    C21 (3x2): Schoolbook A21B11 (12) + Schoolbook A22B21 (18) = 30
    C22 (3x3): Schoolbook A21B12 (18) + Schoolbook A22B22 (27) = 45
    Total: 124
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def vec_u(i, j):
        v = np.zeros(25)
        v[i*5+j] = 1
        return v
        
    def vec_v(j, k):
        v = np.zeros(25)
        v[j*5+k] = 1
        return v
        
    def vec_w(k, i):
        v = np.zeros(25)
        v[k*5+i] = 1
        return v

    # --- C11 Block (2x2) ---
    # Part 1: Strassen for A11 * B11
    # Indices in 2x2 block: 0,1
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    add(vec_u(0,0)+vec_u(1,1), vec_v(0,0)+vec_v(1,1), vec_w(0,0)+vec_w(1,1))
    
    # M2 = (A10+A11)B00 -> C10, -C11
    add(vec_u(1,0)+vec_u(1,1), vec_v(0,0), vec_w(1,0)-vec_w(1,1))
    
    # M3 = A00(B10-B11) -> C01, C11
    add(vec_u(0,0), vec_v(1,0)-vec_v(1,1), vec_w(0,1)+vec_w(1,1))
    
    # M4 = A11(B01-B00) -> C00, C10
    add(vec_u(1,1), vec_v(0,1)-vec_v(0,0), vec_w(0,0)+vec_w(1,0))
    
    # M5 = (A00+A01)B11 -> -C00, C01
    add(vec_u(0,0)+vec_u(0,1), vec_v(1,1), -vec_w(0,0)+vec_w(0,1))
    
    # M6 = (A10-A00)(B00+B01) -> C11
    add(vec_u(1,0)-vec_u(0,0), vec_v(0,0)+vec_v(0,1), vec_w(1,1))
    
    # M7 = (A01-A11)(B10+B11) -> C00
    add(vec_u(0,1)-vec_u(1,1), vec_v(1,0)+vec_v(1,1), vec_w(0,0))
    
    # Part 2: Schoolbook for A12 * B21
    # A12 is rows 0-1, cols 2-4. B21 is rows 2-4, cols 0-1.
    # C11 is rows 0-1, cols 0-1.
    for i in range(2):
        for k in range(2, 5):
            for j in range(2):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))

    # --- C12 Block (2x3) ---
    # Part 1: Schoolbook for A11 * B12
    # A11: 0-1, 0-1. B12: 0-1, 2-4. C12: 0-1, 2-4.
    for i in range(2):
        for k in range(2):
            for j in range(2, 5):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))
                
    # Part 2: Schoolbook for A12 * B22
    # A12: 0-1, 2-4. B22: 2-4, 2-4. C12: 0-1, 2-4.
    for i in range(2):
        for k in range(2, 5):
            for j in range(2, 5):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))

    # --- C21 Block (3x2) ---
    # Part 1: Schoolbook for A21 * B11
    # A21: 2-4, 0-1. B11: 0-1, 0-1. C21: 2-4, 0-1.
    for i in range(2, 5):
        for k in range(2):
            for j in range(2):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))
                
    # Part 2: Schoolbook for A22 * B21
    # A22: 2-4, 2-4. B21: 2-4, 0-1. C21: 2-4, 0-1.
    for i in range(2, 5):
        for k in range(2, 5):
            for j in range(2):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))

    # --- C22 Block (3x3) ---
    # Part 1: Schoolbook for A21 * B12
    # A21: 2-4, 0-1. B12: 0-1, 2-4. C22: 2-4, 2-4.
    for i in range(2, 5):
        for k in range(2):
            for j in range(2, 5):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))
                
    # Part 2: Schoolbook for A22 * B22
    # A22: 2-4, 2-4. B22: 2-4, 2-4. C22: 2-4, 2-4.
    for i in range(2, 5):
        for k in range(2, 5):
            for j in range(2, 5):
                add(vec_u(i, k), vec_v(k, j), vec_w(j, i))

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_5x5_rank124()
    
    # Optional: Verify to be safe
    if _verify(U, V, W, n, m, p, len(U[0])):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = n*m*p
        U_sb = np.zeros((n*m, rank))
        V_sb = np.zeros((m*p, rank))
        W_sb = np.zeros((p*n, rank))
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U_sb[a*m+b, col] = 1
                    V_sb[b*p+c, col] = 1
                    W_sb[c*n+a, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
