import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115
    
    def _time_ok():
        return time.time() < deadline
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        recon = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            recon += np.einsum('i,j,k->ijk', U_arr[:,c], V_arr[:,c], W_arr[:,c])
        
        return np.allclose(recon, target, atol=1e-6)

    def add_strassen(i_base, j_base, k_base, U, V, W, col):
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        # m1
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        # m2
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        # m3
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        # m4
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        # m5
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        # m6
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        # m7
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1
        return col

    def add_schoolbook(i, j, k, U, V, W, col):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        return col + 1

    # Strategy 1: Current Best (66)
    rank1 = 66
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    col1 = 0
    for j in range(4):
        for k in range(6):
            col1 = add_schoolbook(0, j, k, U1, V1, W1, col1)
    for k_block in range(3):
        for j_block in range(2):
            col1 = add_strassen(1, j_block * 2, k_block * 2, U1, V1, W1, col1)
            
    best_rank = 72
    best_U = None
    best_V = None
    best_W = None
    
    if verify(U1, V1, W1, rank1):
        best_rank = rank1
        best_U, best_V, best_W = U1, V1, W1

    if not _time_ok():
        if best_rank == 72:
            U_fb = [[0] * 72 for _ in range(n * m)]
            V_fb = [[0] * 72 for _ in range(m * p)]
            W_fb = [[0] * 72 for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col_fb = add_schoolbook(i, j, k, U_fb, V_fb, W_fb, col_fb)
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        return {"U": best_U, "V": best_V, "W": best_W}

    # Strategy 2: Try rank 65 with one Strassen block replaced by 6 schoolbook terms
    # This is unlikely to work but let's try a few structured variations
    
    # Try: row 0 schoolbook (24), row 1 Strassen for j=0,1 and j=2,3 and j=4,5 (21), row 2 schoolbook (24)
    # Total = 24 + 21 + 24 = 69 (worse)
    
    # Try: row 0 schoolbook (24), row 1 Strassen for all 3 blocks (21), row 2 schoolbook (24) = 69
    
    # The key insight: we need to find a way to use fewer than 66 terms.
    # Let's try a different partition: use Strassen for a 2x4x4 block and handle the rest.
    
    # Actually, let's try to use the "splitter" idea: find a low-rank factorization.
    # For now, let's try a few more structured patterns.
    
    # Pattern: row 0 schoolbook (24), row 1 Strassen (21), row 2: mix
    # If we can do row 2 in 20 terms instead of 24, we get 65.
    
    # Let's try: row 2 uses Strassen for j=0,1 (7 terms) and schoolbook for j=2,3,4,5 (12 terms) = 19 terms
    # Total = 24 + 21 + 19 = 64!
    
    rank2 = 64
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    col2 = 0
    # Row 0: schoolbook
    for j in range(4):
        for k in range(6):
            col2 = add_schoolbook(0, j, k, U2, V2, W2, col2)
    # Row 1: Strassen
    for k_block in range(3):
        for j_block in range(2):
            col2 = add_strassen(1, j_block * 2, k_block * 2, U2, V2, W2, col2)
    # Row 2: Strassen for j=0,1 and schoolbook for j=2,3,4,5
    for k_block in range(3):
        col2 = add_strassen(2, 0, k_block * 2, U2, V2, W2, col2)
    for j in range(2, 6):
        for k in range(6):
            col2 = add_schoolbook(2, j, k, U2, V2, W2, col2)
    
    if col2 == rank2 and verify(U2, V2, W2, rank2):
        best_rank = rank2
        best_U, best_V, best_W = U2, V2, W2

    if not _time_ok():
        if best_rank == 72:
            U_fb = [[0] * 72 for _ in range(n * m)]
            V_fb = [[0] * 72 for _ in range(m * p)]
            W_fb = [[0] * 72 for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col_fb = add_schoolbook(i, j, k, U_fb, V_fb, W_fb, col_fb)
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try more variations: row 2 Strassen for j=0,1,2 and schoolbook for j=3,4,5
    # Strassen for 3 j-blocks (but j has size 4, so we can only do 2 blocks of 2)
    # So row 2: Strassen j=0,1 (7) + Strassen j=2,3 (7) + schoolbook j=4,5 (12) = 26 terms
    # Total = 24 + 21 + 26 = 71 (worse)
    
    # Let's try: row 0 Strassen for j=0,1 (7) + schoolbook for j=2,3,4,5 (24) = 31 terms
    # Row 1 Strassen (21), Row 2 Strassen (21)
    # Total = 31 + 21 + 21 = 73 (worse)
    
    # The best we found is 66. Let's try to verify if we can do 65 or lower with a different approach.
    
    # For now, return the best we have.
    if best_rank == 72:
        U_fb = [[0] * 72 for _ in range(n * m)]
        V_fb = [[0] * 72 for _ in range(m * p)]
        W_fb = [[0] * 72 for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col_fb = add_schoolbook(i, j, k, U_fb, V_fb, W_fb, col_fb)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": best_U, "V": best_V, "W": best_W}
