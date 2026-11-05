import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(Ua[idx_u], Va[idx_v] * Wa[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28(n, m, p):
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    r = 0
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: A1 B11 (j=0,1, k=0,1) Strassen
    U[0][r]=1; U[4][r]=1; V[0][r]=1; V[6][r]=1; W[0][r]=1; W[3][r]=1; r+=1
    U[1][r]=1; U[4][r]=1; V[0][r]=1; W[1][r]=1; W[3][r]=-1; r+=1
    U[0][r]=1; V[1][r]=1; V[6][r]=-1; W[2][r]=1; W[3][r]=1; r+=1
    U[4][r]=1; V[5][r]=1; V[0][r]=-1; W[0][r]=1; W[1][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[6][r]=1; W[0][r]=-1; W[2][r]=1; r+=1
    U[1][r]=1; U[0][r]=-1; V[0][r]=1; V[1][r]=1; W[3][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[5][r]=1; V[6][r]=1; W[0][r]=1; r+=1
    # Part 4: A1 B12 (j=0,1, k=2,3) Strassen
    U[0][r]=1; U[4][r]=1; V[2][r]=1; V[8][r]=1; W[4][r]=1; W[7][r]=1; r+=1
    U[1][r]=1; U[4][r]=1; V[2][r]=1; W[5][r]=1; W[7][r]=-1; r+=1
    U[0][r]=1; V[3][r]=1; V[8][r]=-1; W[6][r]=1; W[7][r]=1; r+=1
    U[4][r]=1; V[7][r]=1; V[2][r]=-1; W[4][r]=1; W[5][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[8][r]=1; W[4][r]=-1; W[6][r]=1; r+=1
    U[1][r]=1; U[0][r]=-1; V[2][r]=1; V[3][r]=1; W[7][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[7][r]=1; V[8][r]=1; W[4][r]=1; r+=1

    return U, V, W

def _try_remove_term(U, V, W, n, m, p, remove_idx):
    """Try to remove term remove_idx by absorbing its contribution into other terms."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # Create copies
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Remove the term
    for i in range(n * m):
        U_new[i].pop(remove_idx)
    for j in range(m * p):
        V_new[j].pop(remove_idx)
    for k in range(p * n):
        W_new[k].pop(remove_idx)
    
    # The removed term had: U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] for all i,j,k
    # We need to add corrections to remaining terms so that the sum still equals 1.
    # This is a complex algebraic problem. For now, just verify if removal works (shouldn't).
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def _try_pairwise_removal(U, V, W, n, m, p, idx1, idx2):
    """Try to remove two terms simultaneously."""
    rank = len(U[0])
    if rank <= 2:
        return None
    
    # Remove term idx1 first, then idx2 (adjust index)
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Remove idx1
    for i in range(n * m):
        U_new[i].pop(idx1)
    for j in range(m * p):
        V_new[j].pop(idx1)
    for k in range(p * n):
        W_new[k].pop(idx1)
    
    # Adjust idx2
    if idx2 > idx1:
        idx2_adj = idx2 - 1
    else:
        idx2_adj = idx2
    
    # Remove idx2_adj
    for i in range(n * m):
        U_new[i].pop(idx2_adj)
    for j in range(m * p):
        V_new[j].pop(idx2_adj)
    for k in range(p * n):
        W_new[k].pop(idx2_adj)
    
    if _verify(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    U, V, W = _build_rank28(n, m, p)
    
    # Verify base decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback: schoolbook rank n*m*p = 30
        rank_sb = n * m * p
        U_sb = [[0.0] * rank_sb for _ in range(n * m)]
        V_sb = [[0.0] * rank_sb for _ in range(m * p)]
        W_sb = [[0.0] * rank_sb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_sb[i * m + j][r] = 1
                    V_sb[j * p + k][r] = 1
                    W_sb[k * n + i][r] = 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try removing single terms
    if time.time() < deadline:
        for r_idx in range(best_rank):
            result = _try_remove_term(U, V, W, n, m, p, r_idx)
            if result is not None:
                new_U, new_V, new_W = result
                if len(new_U[0]) < best_rank:
                    best_U, best_V, best_W = new_U, new_V, new_W
                    best_rank = len(best_U[0])
                break  # Only one removal per pass for simplicity
    
    # Try pairwise removals on the best so far
    if time.time() < deadline and best_rank > 1:
        for r1 in range(best_rank):
            if time.time() >= deadline:
                break
            for r2 in range(r1 + 1, best_rank):
                result = _try_pairwise_removal(best_U, best_V, best_W, n, m, p, r1, r2)
                if result is not None:
                    new_U, new_V, new_W = result
                    if len(new_U[0]) < best_rank:
                        best_U, best_V, best_W = new_U, new_V, new_W
                        best_rank = len(best_U[0])
                    break
            if time.time() >= deadline:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
