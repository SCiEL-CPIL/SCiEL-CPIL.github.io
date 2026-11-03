"""Matrix multiplication 4x5 by 5x6 — Strassen-based decomposition with aggressive linear-algebra refinement."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """Build the 2x2x2 Strassen decomposition (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m * m x p -> n x p. Rank = n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W


def _combine_decompositions(decomps, n, m, p):
    """Combine a list of sub-decompositions into one global decomposition."""
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W


def _build_kronecker_strassen_4x4x4():
    """Build a rank-49 decomposition for 4x4 * 4x4 -> 4x4 using Kronecker product of Strassen."""
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0:
                                continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0:
                                continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0:
                                continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W


def _build_4x4x2_strassen():
    """Build a rank-28 decomposition for 4x4 * 4x2 -> 4x2 using 4 Strassen 2x2x2 blocks."""
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    
    return U, V, W


def _fast_verify(U_arr, V_arr, W_arr, n, m, p):
    """Fast vectorized verification of decomposition."""
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_2d = U_arr[:, t].reshape(n, m)
        v_2d = V_arr[:, t].reshape(m, p)
        w_2d = W_arr[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_2d[i, j] * v_2d[j, :] * w_2d[:, i]
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)


def _build_constraint_matrix(U_arr, V_arr, W_arr, n, m, p):
    """Build the constraint matrix M where M[idx, t] = U[i,j,t]*V[j,k,t]*W[k,i,t]."""
    rank = U_arr.shape[1]
    nmp = n * m * p
    M = np.zeros((nmp, rank), dtype=float)
    for t in range(rank):
        u_2d = U_arr[:, t].reshape(n, m)
        v_2d = V_arr[:, t].reshape(m, p)
        w_2d = W_arr[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    M[idx, t] = u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    return M


def _refine_decomposition(U_arr, V_arr, W_arr, n, m, p, deadline):
    """
    Aggressive refinement: iteratively try to remove columns.
    Uses fast verification and orders by column norm.
    Multiple passes to catch cascading removals.
    """
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        rank = U_arr.shape[1]
        if rank <= 1:
            break
        
        # Build constraint matrix to get column norms
        M = _build_constraint_matrix(U_arr, V_arr, W_arr, n, m, p)
        col_norms = np.linalg.norm(M, axis=0)
        order = np.argsort(col_norms)
        
        removed_any = False
        for t in order:
            if time.time() > deadline - 5:
                break
            
            # Try removing column t
            mask = np.ones(rank, dtype=bool)
            mask[t] = False
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            if _fast_verify(U_test, V_test, W_test, n, m, p):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                removed_any = True
                break  # Restart with new state
        
        if not removed_any:
            break
    
    return U_arr, V_arr, W_arr


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3) - rank 49
    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5) - rank 28
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5) - rank 24
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Convert to numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify initial decomposition
    if not _fast_verify(U_arr, V_arr, W_arr, n, m, p):
        # Fallback: return schoolbook
        U_s, V_s, W_s = _build_schoolbook(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Refine
    if time.time() < deadline - 10:
        U_arr, V_arr, W_arr = _refine_decomposition(U_arr, V_arr, W_arr, n, m, p, deadline)
    
    # Final verification
    if not _fast_verify(U_arr, V_arr, W_arr, n, m, p):
        # If verification failed after refinement, return the unrefined version
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

# EVOLVE-BLOCK-END

