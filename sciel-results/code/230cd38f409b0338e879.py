import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook_rect(n, m, p):
    """Schoolbook: returns list of (P, Q, R) where P is n x m, Q is m x p, R is p x n.
       sum_r P_r[i,j]*Q_r[j,k]*R_r[k,i] = 1 for all i,j,k."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                P = np.zeros((n, m)); P[i, j] = 1.0
                Q = np.zeros((m, p)); Q[j, k] = 1.0
                R = np.zeros((p, n)); R[k, i] = 1.0
                terms.append((P, Q, R))
    return terms

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2 -> 2x2.
       Returns list of (P, Q, R) with P 2x2, Q 2x2, R 2x2."""
    terms = []
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = ae + bg, C[0,1] = af + bh, C[1,0] = ce + dg, C[1,1] = cf + dh
    
    # m1 = (a+d)(e+h)
    P1 = np.array([[1,0],[1,0]], dtype=float)
    Q1 = np.array([[1,1],[0,0]], dtype=float)
    R1 = np.array([[1,1],[1,1]], dtype=float)  # R[k,i]: R[0,0]=1, R[0,1]=1, R[1,0]=1, R[1,1]=1
    terms.append((P1, Q1, R1))
    
    # m2 = (c+d)e
    P2 = np.array([[0,0],[1,1]], dtype=float)
    Q2 = np.array([[1,0],[1,0]], dtype=float)
    R2 = np.array([[0,1],[0,1]], dtype=float)  # contributes to C[1,0] and C[1,1]
    terms.append((P2, Q2, R2))
    
    # m3 = a(f-h)
    P3 = np.array([[1,0],[0,0]], dtype=float)
    Q3 = np.array([[0,1],[1,0]], dtype=float)
    R3 = np.array([[0,1],[1,0]], dtype=float)  # a*f -> C[0,1], a*(-h) -> -C[0,1]... 
    # Actually: a*f contributes to C[0,1], a*(-h) contributes to -C[0,1]. Wait no.
    # m3 = a(f-h). f is B[0,1], h is B[1,1].
    # a*f: A[0,0]*B[0,1] -> C[0,1]
    # a*(-h): A[0,0]*(-B[1,1]) -> -C[0,1]... that's not right either.
    # Let me reconsider. C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = a*f + b*h
    # C[0,0] = a*e + b*g
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # m3 = a(f-h): this is A[0,0] * (B[0,1] - B[1,1])
    # So P3[0,0]=1, Q3[0,1]=1, Q3[1,1]=-1
    # The result a*(f-h) needs to be added to C. Which C entries?
    # In Strassen: C[0,0] = m1 - m2 - m4 + m7
    #              C[0,1] = m3 + m5
    #              C[1,0] = m2 + m4
    #              C[1,1] = m1 - m3 - m5 + m6
    # So m3 contributes +1 to C[0,1] and -1 to C[1,1].
    # R3[k,i]: R3[1,0] = 1 (k=1,i=0 means C[0,1]), R3[1,1] = -1 (k=1,i=1 means C[1,1])
    R3 = np.array([[0,0],[1,-1]], dtype=float)
    terms.append((P3, Q3, R3))
    
    # m4 = d(g-e)
    P4 = np.array([[0,0],[0,1]], dtype=float)
    Q4 = np.array([[0,0],[1,1]], dtype=float)  # B[1,0]-B[0,0] = g-e
    # m4 contributes +1 to C[1,0] and -1 to C[0,0]
    R4 = np.array([[-1,0],[0,1]], dtype=float)
    terms.append((P4, Q4, R4))
    
    # m5 = (a+b)h
    P5 = np.array([[1,1],[0,0]], dtype=float)
    Q5 = np.array([[0,0],[0,1]], dtype=float)
    # m5 contributes +1 to C[0,1] and -1 to C[1,1]
    R5 = np.array([[0,0],[1,-1]], dtype=float)
    terms.append((P5, Q5, R5))
    
    # m6 = (c-a)(e+f)
    P6 = np.array([[-1,0],[1,0]], dtype=float)
    Q6 = np.array([[1,1],[0,0]], dtype=float)
    # m6 contributes +1 to C[1,1]
    R6 = np.array([[0,0],[0,1]], dtype=float)
    terms.append((P6, Q6, R6))
    
    # m7 = (b-d)(g+h)
    P7 = np.array([[0,1],[0,-1]], dtype=float)
    Q7 = np.array([[1,0],[1,0]], dtype=float)
    # m7 contributes +1 to C[0,0]
    R7 = np.array([[1,0],[0,0]], dtype=float)
    terms.append((P7, Q7, R7))
    
    return terms

def _verify_terms(terms, n, m, p, tol=1e-9):
    T = np.zeros((n, m, p))
    for P, Q, R in terms:
        T += np.einsum('ij,jk,ki->ijk', P, Q, R)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _embed_term(P_local, Q_local, R_local, n, m, p, i_map, j_map, k_map):
    """Embed a local term into global dimensions."""
    P_g = np.zeros((n, m))
    Q_g = np.zeros((m, p))
    R_g = np.zeros((p, n))
    for li, gi in enumerate(i_map):
        for lj, gj in enumerate(j_map):
            P_g[gi, gj] = P_local[li, lj]
    for lj, gj in enumerate(j_map):
        for lk, gk in enumerate(k_map):
            Q_g[gj, gk] = Q_local[lj, lk]
    for lk, gk in enumerate(k_map):
        for li, gi in enumerate(i_map):
            R_g[gk, gi] = R_local[lk, li]
    return P_g, Q_g, R_g

def _build_global():
    n, m, p = 4, 5, 5
    all_terms = []
    
    I_TOP = [0, 1]
    I_BOT = [2, 3]
    J_LEFT = [0, 1]
    J_RIGHT = [2, 3, 4]
    K_LEFT = [0, 1]
    K_RIGHT = [2, 3, 4]
    
    # Get decompositions
    strassen_222 = _strassen_2x2()
    if not _verify_terms(strassen_222, 2, 2, 2):
        strassen_222 = _schoolbook_rect(2, 2, 2)
    
    # 2x2 * 2x2 -> 2x2: use Strassen (rank 7)
    dec_222 = strassen_222
    
    # 2x2 * 2x3 -> 2x3: schoolbook (rank 12)
    dec_223 = _schoolbook_rect(2, 2, 3)
    
    # 2x3 * 3x2 -> 2x2: schoolbook (rank 12)
    dec_232 = _schoolbook_rect(2, 3, 2)
    
    # 2x3 * 3x3 -> 2x3: 
    # Partition: A=[2x2 2x1], B=[[3x2 3x1],[1x2 1x1]]... 
    # Actually A is 2x3, B is 3x3.
    # C = A*B = [2x2 2x1] * [[3x2],[1x2]] ... no.
    # A (2x3) * B (3x3) -> C (2x3)
    # Partition A into A_left (2x2) and A_right (2x1)
    # Partition B into B_top (2x3) and B_bot (1x3)
    # C = A_left * B_top + A_right * B_bot
    # A_left (2x2) * B_top (2x3) -> 2x3: schoolbook rank 12
    # A_right (2x1) * B_bot (1x3) -> 2x3: schoolbook rank 6
    # Total: 18. But we can do better?
    # For now, use schoolbook for 2x3*3x3: rank 18
    dec_233 = _schoolbook_rect(2, 3, 3)
    
    # 2x3 * 3x2 -> 2x2: schoolbook rank 12
    dec_232_sb = _schoolbook_rect(2, 3, 2)
    
    # 2x1 * 1x2 -> 2x2: schoolbook rank 4
    dec_212 = _schoolbook_rect(2, 1, 2)
    
    # 2x1 * 1x3 -> 2x3: schoolbook rank 6
    dec_213 = _schoolbook_rect(2, 1, 3)
    
    # 2x2 * 2x1 -> 2x1: schoolbook rank 4
    dec_221 = _schoolbook_rect(2, 2, 1)
    
    # 2x3 * 3x1 -> 2x1: schoolbook rank 6
    dec_231 = _schoolbook_rect(2, 3, 1)
    
    # Assemble the full 4x5 * 5x5 -> 4x5
    # A (4x5) partitioned as:
    #   A_top (2x5) = [A_tl (2x2) | A_tr (2x3)]
    #   A_bot (2x5) = [A_bl (2x2) | A_br (2x3)]
    # B (5x5) partitioned as:
    #   B_left (5x2) = [B_tl (2x2); B_bl (3x2)]
    #   B_right (5x3) = [B_tr (2x3); B_br (3x3)]
    
    # C (4x5) = A * B
    # C_tl (2x2) = A_tl * B_tl
    # C_tr (2x3) = A_tl * B_tr + A_tr * B_br
    # C_bl (2x2) = A_bl * B_tl + A_br * B_bl
    # C_br (2x3) = A_bl * B_tr + A_br * B_br
    
    # C_tl (2x2) = A_tl (2x2) * B_tl (2x2): use Strassen
    for P, Q, R in dec_222:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_TOP, J_LEFT, K_LEFT)
        all_terms.append((P_g, Q_g, R_g))
    
    # C_tr (2x3) = A_tl (2x2) * B_tr (2x3) + A_tr (2x3) * B_br (3x3)
    # A_tl * B_tr: 2x2 * 2x3 -> 2x3
    for P, Q, R in dec_223:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_TOP, J_LEFT, K_RIGHT)
        all_terms.append((P_g, Q_g, R_g))
    
    # A_tr * B_br: 2x3 * 3x3 -> 2x3
    for P, Q, R in dec_233:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_TOP, J_RIGHT, K_RIGHT)
        all_terms.append((P_g, Q_g, R_g))
    
    # C_bl (2x2) = A_bl (2x2) * B_tl (2x2) + A_br (2x3) * B_bl (3x2)
    # A_bl * B_tl: 2x2 * 2x2 -> 2x2
    for P, Q, R in dec_222:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_BOT, J_LEFT, K_LEFT)
        all_terms.append((P_g, Q_g, R_g))
    
    # A_br * B_bl: 2x3 * 3x2 -> 2x2
    for P, Q, R in dec_232_sb:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_BOT, J_RIGHT, K_LEFT)
        all_terms.append((P_g, Q_g, R_g))
    
    # C_br (2x3) = A_bl (2x2) * B_tr (2x3) + A_br (2x3) * B_br (3x3)
    # A_bl * B_tr: 2x2 * 2x3 -> 2x3
    for P, Q, R in dec_223:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_BOT, J_LEFT, K_RIGHT)
        all_terms.append((P_g, Q_g, R_g))
    
    # A_br * B_br: 2x3 * 3x3 -> 2x3
    for P, Q, R in dec_233:
        P_g, Q_g, R_g = _embed_term(P, Q, R, n, m, p, I_BOT, J_RIGHT, K_RIGHT)
        all_terms.append((P_g, Q_g, R_g))
    
    rank = len(all_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (P_g, Q_g, R_g) in enumerate(all_terms):
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = P_g[i, j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = Q_g[j, k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = R_g[k, i]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _schoolbook_full(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    U, V, W = _build_global()
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        U, V, W = _schoolbook_full(n, m, p)
    
    # Round to half-integers
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        U, V, W = _schoolbook_full(n, m, p)
    
    if time.time() > deadline:
        pass  # Shouldn't happen but guard
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
