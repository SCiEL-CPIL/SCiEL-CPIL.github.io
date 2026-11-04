import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W


def _build_8x8x8_70():
    """
    Build 8x8x8 matrix multiplication tensor decomposition with rank 70.
    Uses Strassen recursion: 8x8x8 = 7 * (4x4x4), each 4x4x4 = 7 * (2x2x2).
    Total rank = 7^3 = 343... no wait.
    
    Actually for 8x8x8 via Strassen: split into 2x2 blocks of 4x4.
    Each 4x4x4 block product uses rank 49 (7x7 Strassen levels).
    So 8x8x8 uses 7 * 49 = 343 terms.
    
    But we want rank 70. Let me reconsider.
    
    For 8x8x8: we can use the fact that 8 = 2*4, so we treat it as 2x2 blocks of 4x4.
    The 2x2x2 Strassen uses 7 terms, each involving 4x4x4 multiplications.
    Each 4x4x4 can be done with rank 49 (Strassen).
    Total: 7 * 49 = 343.
    
    Alternatively, 8 = 4*2, treat as 4x4 blocks with 2x2 structure... same thing.
    
    Hmm, rank 70 doesn't come directly from standard Strassen for 8x8x8.
    Let me think differently.
    
    Actually, the known result is that 8x8x8 can be done with rank 218 (or better).
    The standard Strassen gives 7^3 = 343 for 8x8x8.
    
    Wait - maybe the "rank 70" refers to something else. Let me reconsider the strategy.
    
    Perhaps: use a 4x4x4 decomposition (rank 49) and extend it to handle 4x4x8.
    The 4x4x8 problem: A(4x4) * B(4x8) = C(4x8).
    Split B into B1(4x4) and B2(4x4): C = A*B1 + A*B2.
    Each A*B is 4x4x4 with rank 49. Total rank = 98.
    
    To do better, we need to share computation between A*B1 and A*B2.
    
    Let me try a different approach: use the 8x8x8 Strassen decomposition and extract
    the relevant sub-problem.
    
    For 8x8x8 Strassen (rank 343), we embed:
    - A (4x4) in top-left 4x4 of an 8x8 matrix
    - B (4x8) in top-right... no, that doesn't work dimensionally.
    
    Let me try yet another approach: direct construction using bilinear structure.
    """
    # Build 4x4x4 with rank 49 via Strassen
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    A_TL = [0, 1, 4, 5]
    A_TR = [2, 3, 6, 7]
    A_BL = [8, 9, 12, 13]
    A_BR = [10, 11, 14, 15]
    
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]
    
    C_TL = [0, 1, 4, 5]
    C_TR = [2, 3, 6, 7]
    C_BL = [8, 9, 12, 13]
    C_BR = [10, 11, 14, 15]
    
    U_blocks = {
        0: (A_TL + A_BR, [1]*8),
        1: (A_BL + A_BR, [1]*8),
        2: (A_TL, [1]*4),
        3: (A_BR, [1]*4),
        4: (A_TL + A_TR, [1]*8),
        5: (A_BR + A_BL, [1]*8),
        6: (A_TR + A_BR, [1]*4 + [-1]*4),
    }
    
    V_blocks = {
        0: (B_TL + B_BR, [1]*8),
        1: (B_TL, [1]*4),
        2: (B_TR + B_BR, [1]*4 + [-1]*4),
        3: (B_BL + B_TL, [1]*4 + [-1]*4),
        4: (B_BR, [1]*4),
        5: (B_TR + B_BR, [1]*8),
        6: (B_BL + B_TL, [1]*8),
    }
    
    C_blocks_for_M = {
        0: [(C_TL, 1), (C_BR, 1)],
        1: [(C_TR, 1), (C_BR, -1)],
        2: [(C_BL, 1), (C_BR, 1)],
        3: [(C_TL, 1)],
        4: [(C_TR, 1), (C_BR, 1)],
        5: [(C_TL, -1), (C_BL, 1)],
        6: [(C_TL, 1)],
    }
    
    for outer_t in range(7):
        u_idx_list, u_coefs = U_blocks[outer_t]
        v_idx_list, v_coefs = V_blocks[outer_t]
        
        for inner_s in range(7):
            col = outer_t * 7 + inner_s
            
            for local_i in range(4):
                val = U2[local_i, inner_s]
                if val != 0:
                    for bi, coef in zip(u_idx_list, u_coefs):
                        U4[bi, col] += val * coef
            
            for local_j in range(4):
                val = V2[local_j, inner_s]
                if val != 0:
                    for bj, coef in zip(v_idx_list, v_coefs):
                        V4[bj, col] += val * coef
            
            for local_k in range(4):
                val = W2[local_k, inner_s]
                if val != 0:
                    for c_list, c_coef in C_blocks_for_M[outer_t]:
                        for bk in c_list:
                            W4[bk, col] += val * c_coef
    
    return U4, V4, W4


def _build_4x4x8_via_8x8x8():
    """
    Build 4x4x8 decomposition by embedding in 8x8x8 Strassen.
    
    Strategy: 
    - 8x8x8 Strassen has rank 343 (7^3).
    - We embed A(4x4) in rows 0-3, cols 0-3 of 8x8.
    - We embed B(4x8) in rows 0-3, cols 0-7 of 8x8.
    - Output C(4x8) is in rows 0-3, cols 0-7 of 8x8 result.
    
    But 8x8x8 Strassen rank is 343, which is worse than 98.
    
    Alternative: Use a smarter embedding.
    Treat 4x4x8 as part of a 4x4x8 problem within a larger structure.
    
    Let me try: 4x4x8 = 4x4x4 + 4x4x4 (split B into two 4x4 halves).
    Rank = 49 + 49 = 98.
    
    To improve: use the fact that both products share the same A.
    In Strassen terms, we can combine the two 4x4x4 problems.
    
    Actually, let me try a completely different approach:
    Use the "laser method" or "Coppersmith-Winograd" style decomposition.
    
    For now, let me try to find a rank < 98 decomposition through
    a structured search.
    """
    # For this version, I'll try to construct a rank-90 decomposition
    # by finding linear dependencies in the 98-term solution.
    U4, V4, W4 = _build_4x4x4_49()
    
    # Build the 98-term solution
    U98 = np.zeros((16, 98), dtype=np.int64)
    V98 = np.zeros((32, 98), dtype=np.int64)
    W98 = np.zeros((32, 98), dtype=np.int64)
    U98[:, 0:49] = U4
    V98[0:16, 0:49] = V4
    W98[0:16, 0:49] = W4
    U98[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            V98[j*8+kl+4, 49:98] = V4[j*4+kl, 0:49]
    for kl in range(4):
        for i in range(4):
            W98[(kl+4)*4+i, 49:98] = W4[kl*4+i, 0:49]
    
    return U98, V98, W98


def _build_4x4x4_49():
    """4x4x4 via 49-term Strassen recursion."""
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    A_TL = [0, 1, 4, 5]
    A_TR = [2, 3, 6, 7]
    A_BL = [8, 9, 12, 13]
    A_BR = [10, 11, 14, 15]
    
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]
    
    C_TL = [0, 1, 4, 5]
    C_TR = [2, 3, 6, 7]
    C_BL = [8, 9, 12, 13]
    C_BR = [10, 11, 14, 15]
    
    U_blocks = {
        0: (A_TL + A_BR, [1]*8),
        1: (A_BL + A_BR, [1]*8),
        2: (A_TL, [1]*4),
        3: (A_BR, [1]*4),
        4: (A_TL + A_TR, [1]*8),
        5: (A_BR + A_BL, [1]*8),
        6: (A_TR + A_BR, [1]*4 + [-1]*4),
    }
    
    V_blocks = {
        0: (B_TL + B_BR, [1]*8),
        1: (B_TL, [1]*4),
        2: (B_TR + B_BR, [1]*4 + [-1]*4),
        3: (B_BL + B_TL, [1]*4 + [-1]*4),
        4: (B_BR, [1]*4),
        5: (B_TR + B_BR, [1]*8),
        6: (B_BL + B_TL, [1]*8),
    }
    
    C_blocks_for_M = {
        0: [(C_TL, 1), (C_BR, 1)],
        1: [(C_TR, 1), (C_BR, -1)],
        2: [(C_BL, 1), (C_BR, 1)],
        3: [(C_TL, 1)],
        4: [(C_TR, 1), (C_BR, 1)],
        5: [(C_TL, -1), (C_BL, 1)],
        6: [(C_TL, 1)],
    }
    
    for outer_t in range(7):
        u_idx_list, u_coefs = U_blocks[outer_t]
        v_idx_list, v_coefs = V_blocks[outer_t]
        
        for inner_s in range(7):
            col = outer_t * 7 + inner_s
            
            for local_i in range(4):
                val = U2[local_i, inner_s]
                if val != 0:
                    for bi, coef in zip(u_idx_list, u_coefs):
                        U4[bi, col] += val * coef
            
            for local_j in range(4):
                val = V2[local_j, inner_s]
                if val != 0:
                    for bj, coef in zip(v_idx_list, v_coefs):
                        V4[bj, col] += val * coef
            
            for local_k in range(4):
                val = W2[local_k, inner_s]
                if val != 0:
                    for c_list, c_coef in C_blocks_for_M[outer_t]:
                        for bk in c_list:
                            W4[bk, col] += val * c_coef
    
    return U4, V4, W4


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify_decomposition(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reconstructs T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=atol)


def _try_prune_columns(U, V, W, T, deadline):
    """
    Try to reduce rank by finding and removing linearly dependent columns.
    Uses a greedy approach with verification.
    """
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    keep = [True] * rank
    changed = True
    iter_count = 0
    max_iter = 30
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        if deadline and time.time() > deadline - 15:
            break
        
        for r in range(rank):
            if not keep[r]:
                continue
            
            others = [s for s in range(rank) if s != r and keep[s]]
            if len(others) == 0:
                continue
            
            U_test = U[:, others]
            V_test = V[:, others]
            W_test = W[:, others]
            
            if _verify_decomposition(U_test, V_test, W_test, T, atol=1e-6):
                keep[r] = False
                changed = True
    
    idx = [i for i in range(rank) if keep[i]]
    return U[:, idx], V[:, idx], W[:, idx]


def _try_als_optimization(T, n, m, p, target_rank, deadline, n_restarts=5):
    """
    Try to find a rank-`target_rank` decomposition using ALS (Alternating Least Squares).
    This is a different optimization approach than Gauss-Newton.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    best_residual = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 10:
            break
        
        np.random.seed(42 + restart)
        
        U = np.random.randn(dim_a, target_rank) * 0.5
        V = np.random.randn(dim_b, target_rank) * 0.5
        W = np.random.randn(dim_c, target_rank) * 0.5
        
        # Normalize
        for r in range(target_rank):
            U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
            V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
            W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
        
        for it in range(30):
            if deadline and time.time() > deadline - 8:
                break
            
            # Update U
            M_U = np.zeros((dim_b * dim_c, target_rank))
            for r in range(target_rank):
                M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            T_flat = T.reshape(dim_a, dim_b * dim_c)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat, rcond=1e-10)
            except:
                break
            T_recon = np.einsum('ar,br,cr->abc', U_new, V, W)
            if np.linalg.norm(T - T_recon) < np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W)):
                U = U_new
            
            # Update V
            M_V = np.zeros((dim_a * dim_c, target_rank))
            for r in range(target_rank):
                M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            T_flat_v = T.reshape(dim_b, dim_a * dim_c)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_v, rcond=1e-10)
            except:
                break
            T_recon = np.einsum('ar,br,cr->abc', U, V_new, W)
            if np.linalg.norm(T - T_recon) < np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W)):
                V = V_new
            
            # Update W
            M_W = np.zeros((dim_a * dim_b, target_rank))
            for r in range(target_rank):
                M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            T_flat_w = T.reshape(dim_c, dim_a * dim_b)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_w, rcond=1e-10)
            except:
                break
            T_recon = np.einsum('ar,br,cr->abc', U, V, W_new)
            if np.linalg.norm(T - T_recon) < np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W)):
                W = W_new
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        res = np.linalg.norm(T - T_recon)
        
        if res < best_residual:
            best_residual = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-8:
            break
    
    return best_U, best_V, best_W, best_residual


def _try_structured_reduction(U, V, W, T, deadline):
    """
    Try to find a lower-rank decomposition by looking for structure.
    The 4x4x8 problem has the form: C[i,k] = sum_j A[i,j] * B[j,k]
    where i in 0..3, j in 0..3, k in 0..7.
    
    We can try to find a decomposition where some rank terms are shared
    between different (i,k) pairs.
    """
    # This is a placeholder for a more sophisticated structural search.
    # For now, just return the input.
    return U, V, W


def _to_half_integer_list(mat):
    """Convert a matrix to a list of lists with half-integer or integer entries."""
    mat = np.asarray(mat, dtype=np.float64)
    mat2 = mat * 2
    rounded2 = np.round(mat2)
    if np.allclose(mat2, rounded2, atol=1e-6):
        return (rounded2 / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 8
    dim_a, dim_b, dim_c = n*m, m*p, p*n

    # Build the full 4x4x8 multiplication tensor
    T = _build_tensor(n, m, p)

    # Fallback: Schoolbook (rank 128)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_a, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_b, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_c, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Step 1: Build the 98-rank Strassen-based solution
    if time.time() < deadline - 40:
        try:
            U98, V98, W98 = _build_4x4x8_via_8x8x8()
            if _verify_decomposition(U98, V98, W98, T, atol=1e-9):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass

    # Step 2: Try column pruning on the 98-term solution
    if best_rank == 98 and time.time() < deadline - 25:
        try:
            U_pr, V_pr, W_pr = _try_prune_columns(
                best_U.astype(np.float64),
                best_V.astype(np.float64),
                best_W.astype(np.float64),
                T,
                deadline
            )
            if U_pr.shape[1] < best_rank:
                if _verify_decomposition(U_pr, V_pr, W_pr, T, atol=1e-6):
                    best_U, best_V, best_W = U_pr, V_pr, W_pr
                    best_rank = U_pr.shape[1]
        except Exception:
            pass

    # Step 3: Try ALS optimization to find lower-rank solutions
    if best_rank > 90 and time.time() < deadline - 15:
        try:
            for target in [90, 85, 80]:
                if time.time() > deadline - 10:
                    break
                U_als, V_als, W_als, res = _try_als_optimization(
                    T, n, m, p, target, deadline, n_restarts=3
                )
                if res is not None and res < 1e-6:
                    T_recon = np.einsum('ar,br,cr->abc', U_als, V_als, W_als)
                    if np.allclose(T_recon, T, atol=1e-6):
                        best_U, best_V, best_W = U_als, V_als, W_als
                        best_rank = target
                        break
        except Exception:
            pass

    # Final verification
    T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if not np.allclose(T_recon, T, atol=1e-6):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook

    U_out = _to_half_integer_list(best_U)
    V_out = _to_half_integer_list(best_V)
    W_out = _to_half_integer_list(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
