import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2."""
    triples = []
    def add(u, v, w):
        triples.append((np.array(u, dtype=float),
                        np.array(v, dtype=float),
                        np.array(w, dtype=float)))
    # m1 = (A00+A11)(B00+B11)
    add([1, 1], [1, 1], [1, 1])
    # m2 = (A10+A11)B00
    add([0, 1], [1, 0], [1, 0])
    # m3 = A00(B01-B11)
    add([1, 0], [1, 0], [0, 1])
    add([1, 0], [1, 0], [-1, 0])
    # m4 = A11(B10-B00)
    add([0, 1], [0, 1], [1, 0])
    add([0, 1], [0, 1], [0, 1]) # Note: B10 is j=1, k=0. B00 is j=0, k=0.
    # Wait, Strassen m4 = A11(B10-B00).
    # A11: u=[0,1].
    # B10: j=1, k=0 -> v=[0,1], w=[1,0].
    # B00: j=0, k=0 -> v=[1,0], w=[1,0].
    # So term is u=[0,1], v=[0,1], w=[1,0] minus u=[0,1], v=[1,0], w=[1,0].
    # My previous code had:
    # add([0, 1], [0, 1], [1, 0])  # This is A11 * B10
    # add([0, 1], [0, 1], [-0, 1]) # This is A11 * (something k=1)
    # This was wrong in the seed. I must implement Strassen correctly.
    
    # Let's re-derive Strassen carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # u is over i (rows of A), v over j (cols of A / rows of B), w over k (cols of B).
    # A[i,j], B[j,k], C[i,k].
    
    triples = []
    # m1 = (A00+A11)(B00+B11)
    # u=[1,1], v=[1,1], w=[1,1]
    triples.append((np.array([1, 1]), np.array([1, 1]), np.array([1, 1])))
    # m2 = (A10+A11)B00
    # A10+A11: u=[0,1] (row 1 of A). Wait. A10 is A[1,0], A11 is A[1,1].
    # So u should be [0,1] if we sum over j? No.
    # u[i] is coeff for A[i, *].
    # A10 is i=1, j=0. A11 is i=1, j=1.
    # So u=[0,1] selects row 1. But we need to sum j?
    # The term is (sum_j u[i] A[i,j]) * ...
    # For m2: (A[1,0] + A[1,1]) * B[0,0].
    # u[i] = 1 if i=1, else 0. u=[0,1].
    # v[j] = 1 if j=0, else 0? No, we sum A[1,j] for j=0,1.
    # So v[j] = 1 for j=0,1. v=[1,1].
    # w[k] = 1 if k=0, else 0. w=[1,0].
    triples.append((np.array([0, 1]), np.array([1, 1]), np.array([1, 0])))
    # m3 = A00(B01-B11)
    # A00: i=0, j=0. u=[1,0], v=[1,0].
    # B01: j=0, k=1. w=[0,1].
    # B11: j=1, k=1. w=[0,1] with v=[0,1].
    # So term 1: u=[1,0], v=[1,0], w=[0,1].
    # Term 2: u=[1,0], v=[0,1], w=[-0,1] -> w=[0,1] with coeff -1.
    triples.append((np.array([1, 0]), np.array([1, 0]), np.array([0, 1])))
    triples.append((np.array([1, 0]), np.array([0, 1]), np.array([0, 1]))) # Coeff -1?
    # The triple is (u, v, w). The sum is sum u[i]v[j]w[k].
    # To subtract, we can use negative coefficients in w or v.
    # Let's use w=[0,1] and v=[0,1] with coeff -1.
    # Actually, just append (u, v, -w).
    triples[-1] = (np.array([1, 0]), np.array([0, 1]), np.array([0, -1]))
    
    # m4 = A11(B10-B00)
    # A11: i=1, j=1. u=[0,1], v=[0,1].
    # B10: j=1, k=0. w=[1,0].
    # B00: j=0, k=0. w=[1,0] with v=[1,0].
    # Term 1: u=[0,1], v=[0,1], w=[1,0].
    # Term 2: u=[0,1], v=[1,0], w=[-1,0].
    triples.append((np.array([0, 1]), np.array([0, 1]), np.array([1, 0])))
    triples.append((np.array([0, 1]), np.array([1, 0]), np.array([-1, 0])))
    
    # m5 = (A00+A01)B11
    # A00+A01: i=0. u=[1,0]. v=[1,1].
    # B11: j=1, k=1. w=[0,1].
    triples.append((np.array([1, 0]), np.array([1, 1]), np.array([0, 1])))
    
    # m6 = (A10-A00)(B00+B01)
    # A10-A00: i=1 for A10, i=0 for A00.
    # A10: i=1, j=0. u=[0,1], v=[1,0].
    # A00: i=0, j=0. u=[1,0], v=[1,0].
    # So u=[-1,1], v=[1,0].
    # B00+B01: j=0. v=[1,0]. w=[1,1].
    triples.append((np.array([-1, 1]), np.array([1, 0]), np.array([1, 1])))
    
    # m7 = (A01-A11)(B10+B11)
    # A01-A11: A01 (i=0, j=1) - A11 (i=1, j=1).
    # u=[1,-1], v=[0,1].
    # B10+B11: j=1. v=[0,1]. w=[1,1].
    triples.append((np.array([1, -1]), np.array([0, 1]), np.array([1, 1])))
    
    return triples

def _schoolbook_rect(n, m, p):
    """Schoolbook rank-1 triples for n x m * m x p -> n x p."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0.0]*n; u[i] = 1.0
                v = [0.0]*m; v[j] = 1.0
                w = [0.0]*p; w[k] = 1.0
                triples.append((np.array(u), np.array(v), np.array(w)))
    return triples

def _verify_triples(triples, n, m, p, tol=1e-9):
    T = np.zeros((n, m, p))
    for u, v, w in triples:
        T += np.einsum('i,j,k->ijk', u, v, w)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _build_global():
    n, m, p = 4, 5, 5
    global_triples = []
    
    # Index maps
    # A rows: 0,1 -> Top (2), 2,3 -> Bottom (2)
    # A cols: 0,1 -> Left (2), 2,3,4 -> Right (3)
    # B rows: 0,1 -> Top (2), 2,3,4 -> Bottom (3)
    # B cols: 0,1 -> Left (2), 2,3,4 -> Right (3)
    # C rows: 0,1 -> Top (2), 2,3 -> Bottom (2)
    # C cols: 0,1 -> Left (2), 2,3,4 -> Right (3)
    
    I_TOP = [0, 1]
    I_BOT = [2, 3]
    J_LEFT = [0, 1]
    J_RIGHT = [2, 3, 4]
    K_LEFT = [0, 1]
    K_RIGHT = [2, 3, 4]
    
    def add_block(dec, i_map, j_map, k_map):
        for (u, v, w) in dec:
            u_g = np.zeros(n)
            v_g = np.zeros(m)
            w_g = np.zeros(p)
            for li, gi in enumerate(i_map):
                u_g[gi] = u[li]
            for lj, gj in enumerate(j_map):
                v_g[gj] = v[lj]
            for lk, gk in enumerate(k_map):
                w_g[gk] = w[lk]
            global_triples.append((u_g, v_g, w_g))

    # Get decompositions
    strassen = _strassen_2x2()
    if not _verify_triples(strassen, 2, 2, 2):
        strassen = _schoolbook_rect(2, 2, 2)
        
    # 2x2x2
    dec_222 = strassen
    # 2x2x3
    dec_223 = _schoolbook_rect(2, 2, 3)
    # 2x3x2
    dec_232 = _schoolbook_rect(2, 3, 2)
    # 2x3x3 (Optimized: 17)
    # Split 3 into 2+1
    # A(2x3) = A1(2x2) A2(2x1)
    # B(3x3) = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
    # Terms:
    # A1 B11 (2x2x2): Rank 7
    # A1 B12 (2x2x1): Rank 4
    # A2 B21 (2x1x2): Rank 4
    # A2 B22 (2x1x1): Rank 2
    # We need to map these local indices to the 2x3x3 block indices.
    # Local i: 0,1
    # Local j: 0,1,2 (0,1 for first part, 2 for second)
    # Local k: 0,1,2 (0,1 for first part, 2 for second)
    
    dec_233 = []
    # A1 B11: i in 0,1; j in 0,1; k in 0,1
    for u, v, w in dec_222:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(3); w_g[0:2] = w
        dec_233.append((u_g, v_g, w_g))
        
    # A1 B12: i in 0,1; j in 0,1; k in 2
    # B12 is 2x1. Local j 0,1. Local k 2.
    # This is 2x2x1.
    dec_221 = _schoolbook_rect(2, 2, 1)
    for u, v, w in dec_221:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(3); w_g[2] = w[0]
        dec_233.append((u_g, v_g, w_g))
        
    # A2 B21: i in 0,1; j in 2; k in 0,1
    # A2 is 2x1. Local j 2.
    # B21 is 1x2. Local j 2, k 0,1.
    # This is 2x1x2.
    dec_212 = _schoolbook_rect(2, 1, 2)
    for u, v, w in dec_212:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(3); w_g[0:2] = w
        dec_233.append((u_g, v_g, w_g))
        
    # A2 B22: i in 0,1; j in 2; k in 2
    # This is 2x1x1.
    dec_211 = _schoolbook_rect(2, 1, 1)
    for u, v, w in dec_211:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(3); w_g[2] = w[0]
        dec_233.append((u_g, v_g, w_g))
        
    if not _verify_triples(dec_233, 2, 3, 3):
        dec_233 = _schoolbook_rect(2, 3, 3)

    # 2x3x2 (Optimized: 11)
    # A(2x3) = A1(2x2) A2(2x1)
    # B(3x2) = [B1(2x2); B2(1x2)]
    # A1 B1 (2x2x2): Rank 7
    # A2 B2 (2x1x2): Rank 4
    dec_232_opt = []
    # A1 B1: i 0,1; j 0,1; k 0,1
    for u, v, w in dec_222:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[0:2] = v
        w_g = np.zeros(2); w_g[:] = w
        dec_232_opt.append((u_g, v_g, w_g))
        
    # A2 B2: i 0,1; j 2; k 0,1
    for u, v, w in dec_212:
        u_g = np.zeros(2); u_g[:] = u
        v_g = np.zeros(3); v_g[2] = v[0]
        w_g = np.zeros(2); w_g[:] = w
        dec_232_opt.append((u_g, v_g, w_g))
        
    if not _verify_triples(dec_232_opt, 2, 3, 2):
        dec_232_opt = _schoolbook_rect(2, 3, 2)

    # Assemble
    # C11 = A11 B11
    add_block(dec_222, I_TOP, J_LEFT, K_LEFT)
    
    # C12 = A11 B12 + A12 B22
    # A11 (2x2) B12 (2x3) -> 2x2x3
    add_block(dec_223, I_TOP, J_LEFT, K_RIGHT)
    # A12 (2x3) B22 (3x3) -> 2x3x3
    add_block(dec_233, I_TOP, J_RIGHT, K_RIGHT)
    
    # C21 = A21 B11 + A22 B21
    # A21 (2x2) B11 (2x2) -> 2x2x2
    add_block(dec_222, I_BOT, J_LEFT, K_LEFT)
    # A22 (2x3) B21 (3x2) -> 2x3x2
    add_block(dec_232_opt, I_BOT, J_RIGHT, K_LEFT)
    
    # C22 = A21 B12 + A22 B22
    # A21 (2x2) B12 (2x3) -> 2x2x3
    add_block(dec_223, I_BOT, J_LEFT, K_RIGHT)
    # A22 (2x3) B22 (3x3) -> 2x3x3
    add_block(dec_233, I_BOT, J_RIGHT, K_RIGHT)
    
    rank = len(global_triples)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_g, v_g, w_g) in enumerate(global_triples):
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = u_g[i] * v_g[j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = v_g[j] * w_g[k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = w_g[k] * u_g[i]
                
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W = _build_global()
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0

    if not _verify(U, V, W, n, m, p, tol=1e-8):
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb))
        V = np.zeros((m * p, rank_sb))
        W = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
