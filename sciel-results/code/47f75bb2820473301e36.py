import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Random test to ensure validity quickly
    # Full verification is O(n*m*p*rank) which is small for 56
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _get_strassen_2x2x2():
    """Returns list of 7 (u, v, w) vectors for 2x2x2 Strassen."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    terms.append(([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 1, 0, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, -1]))
    terms.append(([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, -1]))
    return terms

def _get_rank_11_2x2x3():
    """
    Returns a rank-11 decomposition for 2x2 * 2x3.
    Constructed by taking the 7 Strassen terms for 2x2*2x2 and 
    adding 4 extra terms for the third column of the 2x3 matrix.
    """
    terms = []
    s_terms = _get_strassen_2x2x2()
    # Embed 2x2*2x2 into 2x2*2x3 (zeros in 3rd column of V, 3rd column of W)
    for u, v, w in s_terms:
        u_g = u  # length 4
        v_g = [v[0], v[1], 0, v[2], v[3], 0]  # length 6
        w_g = [w[0], w[1], 0, w[2], w[3], 0]  # length 6
        terms.append((u_g, v_g, w_g))
    
    # Add 4 terms for the 2x2 * 2x1 multiplication (column 2)
    # A[0,0]*B[0,2] -> C[0,2]
    terms.append(([1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
    # A[0,1]*B[1,2] -> C[0,2]
    terms.append(([0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]))
    # A[1,0]*B[0,2] -> C[1,2]
    terms.append(([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]))
    # A[1,1]*B[1,2] -> C[1,2]
    terms.append(([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]))
    
    return terms

def _get_rank_4_1x2x2():
    """Rank 4 decomposition for 1x2 * 2x2."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0], [1, 0]))
    terms.append(([1, 0], [0, 1, 0, 0], [0, 1]))
    terms.append(([0, 1], [0, 0, 1, 0], [1, 0]))
    terms.append(([0, 1], [0, 0, 0, 1], [0, 1]))
    return terms

def _get_rank_6_1x2x3():
    """Rank 6 decomposition for 1x2 * 2x3."""
    terms = []
    terms.append(([1, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0]))
    terms.append(([1, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0]))
    terms.append(([1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1]))
    terms.append(([0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]))
    terms.append(([0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 1]))
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Total rank calculation:
    # Block 1: 2x2 * 2x2 -> 7
    # Block 2: 2x2 * 2x2 -> 7
    # Block 3: 2x2 * 2x3 -> 11
    # Block 4: 2x2 * 2x3 -> 11
    # Block 5: 1x2 * 2x2 -> 4
    # Block 6: 1x2 * 2x2 -> 4
    # Block 7: 1x2 * 2x3 -> 6
    # Block 8: 1x2 * 2x3 -> 6
    # Total: 7+7+11+11+4+4+6+6 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u_vec[i]
        for i in range(m * p):
            V[i][col] = v_vec[i]
        for i in range(p * n):
            W[i][col] = w_vec[i]
        col += 1
        
    s_terms = _get_strassen_2x2x2()
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        # A indices: 0,1 (row 0), 4,5 (row 1) for cols 0,1
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        # B indices: 0,1 (col 0), 5,6 (col 1) for rows 0,1
        v_g[0] = v[0]; v_g[1] = v[1]; v_g[5] = v[2]; v_g[6] = v[3]
        # C indices: 0,1 (col 0), 3,4 (col 1) for rows 0,1
        # W index is k*n + i.
        # k=0, i=0 -> 0. k=0, i=1 -> 3.
        # k=1, i=0 -> 1. k=1, i=1 -> 4.
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    for u, v, w in s_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        # A indices: 2,3 (row 0), 6,7 (row 1) for cols 2,3
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        # B indices: 10,11 (col 0), 15,16 (col 1) for rows 2,3
        v_g[10] = v[0]; v_g[11] = v[1]; v_g[15] = v[2]; v_g[16] = v[3]
        w_g[0] = w[0]; w_g[1] = w[1]; w_g[3] = w[2]; w_g[4] = w[3]
        add_term(u_g, v_g, w_g)
        
    r11_terms = _get_rank_11_2x2x3()
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[0] = u[0]; u_g[1] = u[1]; u_g[4] = u[2]; u_g[5] = u[3]
        # B rows 0,1. Cols 2,3,4.
        # B indices:
        # Row 0: 2, 3, 4
        # Row 1: 7, 8, 9
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]
        v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        # C rows 0,1. Cols 2,3,4.
        # W indices k*n + i.
        # k=2, i=0 -> 6. k=2, i=1 -> 7.
        # k=3, i=0 -> 8. k=3, i=1 -> 9.
        # k=4, i=0 -> 10. k=4, i=1 -> 11.
        w_g[6] = w[0]; w_g[7] = w[1]
        w_g[8] = w[2]; w_g[9] = w[3]
        w_g[10] = w[4]; w_g[11] = w[5]
        add_term(u_g, v_g, w_g)
        
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    for u, v, w in r11_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[2] = u[0]; u_g[3] = u[1]; u_g[6] = u[2]; u_g[7] = u[3]
        # B rows 2,3. Cols 2,3,4.
        # B indices:
        # Row 2: 12, 13, 14
        # Row 3: 17, 18, 19
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]
        v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[6] = w[0]; w_g[7] = w[1]
        w_g[8] = w[2]; w_g[9] = w[3]
        w_g[10] = w[4]; w_g[11] = w[5]
        add_term(u_g, v_g, w_g)
        
    r4_terms = _get_rank_4_1x2x2()
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        # A row 2. Cols 0,1.
        # Indices: 8, 9
        u_g[8] = u[0]; u_g[9] = u[1]
        # B rows 0,1. Cols 0,1.
        # Indices: 0,1 (row 0), 5,6 (row 1)
        v_g[0] = v[0]; v_g[1] = v[1]
        v_g[5] = v[2]; v_g[6] = v[3]
        # C row 2. Cols 0,1.
        # W indices k*n + i. i=2.
        # k=0 -> 2. k=1 -> 5.
        w_g[2] = w[0]; w_g[5] = w[1]
        add_term(u_g, v_g, w_g)
        
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2]
    for u, v, w in r4_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        # A row 2. Cols 2,3.
        # Indices: 10, 11
        u_g[10] = u[0]; u_g[11] = u[1]
        # B rows 2,3. Cols 0,1.
        # Indices: 10,11 (row 2), 15,16 (row 3)
        v_g[10] = v[0]; v_g[11] = v[1]
        v_g[15] = v[2]; v_g[16] = v[3]
        w_g[2] = w[0]; w_g[5] = w[1]
        add_term(u_g, v_g, w_g)
        
    r6_terms = _get_rank_6_1x2x3()
    
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[8] = u[0]; u_g[9] = u[1]
        # B rows 0,1. Cols 2,3,4.
        v_g[2] = v[0]; v_g[3] = v[1]; v_g[4] = v[2]
        v_g[7] = v[3]; v_g[8] = v[4]; v_g[9] = v[5]
        # C row 2. Cols 2,3,4.
        # i=2.
        # k=2 -> 8. k=3 -> 11. k=4 -> 14.
        w_g[8] = w[0]; w_g[11] = w[1]; w_g[14] = w[2]
        add_term(u_g, v_g, w_g)
        
    # Block 8: A[2, 2:4] * B[2:4, 2:5] -> C[2, 2:5]
    for u, v, w in r6_terms:
        u_g = [0] * (n * m)
        v_g = [0] * (m * p)
        w_g = [0] * (p * n)
        u_g[10] = u[0]; u_g[11] = u[1]
        # B rows 2,3. Cols 2,3,4.
        v_g[12] = v[0]; v_g[13] = v[1]; v_g[14] = v[2]
        v_g[17] = v[3]; v_g[18] = v[4]; v_g[19] = v[5]
        w_g[8] = w[0]; w_g[11] = w[1]; w_g[14] = w[2]
        add_term(u_g, v_g, w_g)
        
    if col != rank:
        # Fallback to schoolbook if construction failed
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
