"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u, v, w) where u, v, w are length-4 vectors
    indexed as [i*2+j] for i,j in {0,1}.
    """
    factors = []
    # M1 = (A11+A22)*(B11+B22)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A21+A22)*B11
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A11*(B21-B22)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = A22*(B11-B22)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # M5 = (A11+A12)*B22
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (A21-A11)*(B11+B12)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A12-A22)*(B21+B22)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using recursive Strassen.
    
    The 4x4 * 4x8 multiplication is split into two 2x4 * 2x8 blocks (top/bottom halves of A),
    each of which is further split using Strassen's 2x2 * 2x2 pattern applied to 2x2 sub-blocks.
    
    Actually, let's think more carefully:
    - A is 4x4, B is 4x8, C is 4x8
    - Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    - Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x4)
    - But Strassen works for square matrices... 
    
    Let me reconsider. The standard approach for 4x4 * 4x8:
    - View as two independent 2x4 * 4x8 problems? No.
    - Better: use the fact that we can apply Strassen recursively.
    
    Actually, for a 4x4 * 4x8 matrix multiply:
    - Split rows of A into two groups of 2, rows of B into two groups of 2
    - A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    - B = [[B11, B12], [B21, B22]] where B11, B12 are 2x4 and B21, B22 are 2x4
    - C = [[C11, C12], [C21, C22]] where each Cij is 2x4
    
    Now C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each of these is a sum of two 2x2 * 2x4 products.
    
    For a 2x2 * 2x4 product, we can use a "half-Strassen" approach or just do it directly.
    Actually, a 2x2 * 2x4 can be viewed as two 2x2 * 2x2 problems side by side, which would cost 7+7=14 per product.
    
    So total: 4 products * 14 = 56? That doesn't match 98.
    
    Let me reconsider the recursive structure. Perhaps the intended decomposition is:
    - Treat 4x4 * 4x8 as a 4x4 * 4x4 * 2 (i.e., two 4x4*4x4 multiplies)
    - Each 4x4*4x4 uses Strassen recursively: 7 * (2x2*2x2) = 7*7 = 49
    - Total: 2 * 49 = 98
    
    Yes! That's it. Split the 4x8 matrix B into two 4x4 matrices B_left and B_right.
    Then C_left = A * B_left and C_right = A * B_right, each a 4x4*4x4 multiply.
    Each 4x4*4x4 multiply uses recursive Strassen with rank 49.
    Total rank = 98.
    
    For the 4x4*4x4 Strassen:
    - Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    - Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    - Use 7 Strassen products, each of which is a 2x2*2x2 multiply (rank 7)
    - Total: 7 * 7 = 49
    
    So the full decomposition has rank 2 * 7 * 7 = 98.
    """
    n, m, p = DIMENSIONS
    s2 = _strassen_2x2_factors()

    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Two halves of the output (left 4 columns and right 4 columns)
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r

                u_block = s2[s][0]
                v_block = s2[s][1]
                w_block = s2[s][2]
                u_scalar = s2[r][0]
                v_scalar = s2[r][1]
                w_scalar = s2[r][2]

                # U: maps (i, j) -> coefficient
                # i in 0..3, j in 0..3
                # Top-level Strassen: split i into {0,1} and {2,3}, j into {0,1} and {2,3}
                # u_block indexes the top-level 2x2 block (bi, bj) where bi, bj in {0,1}
                # u_scalar indexes the inner 2x2 (ei, ej) where ei, ej in {0,1}
                # So i = 2*bi + ei, j = 2*bj + ej
                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += cb * cs

                # V: maps (j, k) -> coefficient
                # j in 0..3, k in 0..7
                # Top-level: j split into {0,1} and {2,3}
                # Inner: for the left half (t=0), k in 0..3; for right half (t=1), k in 4..7
                # v_block indexes (bj, bk) where bj in {0,1} for j-block, bk in {0,1} for k-block within the 4x4
                # v_scalar indexes (ej, ek) where ej in {0,1} for j-inner, ek in {0,1} for k-inner
                # So j = 2*bj + ej, k = 4*t + 2*bk + ek
                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += cb * cs

                # W: maps (k, i) -> coefficient (output factor W[k*n+i, r])
                # k in 0..7, i in 0..3
                # w_block indexes (ck, ci) where ck in {0,1} for k-block, ci in {0,1} for i-block
                # w_scalar indexes (ek, ei) where ek in {0,1} for k-inner, ei in {0,1} for i-inner
                # So k = 4*t + 2*ck + ek, i = 2*ci + ei
                for ci in range(2):
                    for cj in range(2):
                        cb = w_block[ci * 2 + cj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += cb * cs

    return {"U": U, "V": V, "W": W}

def _reduce_rank_linear(U_arr, V_arr, W_arr, max_time=60.0):
    """
    Greedily remove columns whose contribution is a linear combination of the remaining columns.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    tol = 1e-8
    
    while True:
        if time.time() - start_time > max_time:
            break
        rank = U_arr.shape[1]
        if rank <= 1:
            break
        
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        col_contribs = np.stack([U3[:, :, s] * V3[:, :, s] * W3[:, :, s] for s in range(rank)], axis=-1)
        col_contribs_flat = col_contribs.reshape(-1, rank)
        
        removed = False
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            
            target_flat = col_contribs_flat[:, r]
            mask = [s for s in range(rank) if s != r]
            other_T_flat = col_contribs_flat[:, mask]
            
            try:
                c, residuals, rank_ls, sv = np.linalg.lstsq(other_T_flat, target_flat, rcond=None)
            except Exception:
                continue
            
            residual = other_T_flat @ c - target_flat
            if np.linalg.norm(residual) < tol:
                U_arr = np.delete(U_arr, r, axis=1)
                V_arr = np.delete(V_arr, r, axis=1)
                W_arr = np.delete(W_arr, r, axis=1)
                removed = True
                break
        
        if not removed:
            break
    
    return U_arr, V_arr, W_arr

def _polish_decomposition(U_arr, V_arr, W_arr, max_time=30.0):
    """
    Try to polish the decomposition to enable further rank reductions.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = U_arr.shape[1]
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    np.random.seed(42)
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        R = np.random.randn(rank, rank)
        try:
            det = np.linalg.det(R)
            if abs(det) < 1e-6:
                continue
        except Exception:
            continue
        
        U_pert = U_arr @ R
        V_pert = V_arr @ R
        W_pert = W_arr @ R
        
        U3_pert = U_pert.reshape(n, m, rank)
        V3_pert = V_pert.reshape(m, p, rank)
        W3_pert = W_pert.reshape(p, n, rank)
        T_pert = np.einsum('ijr,jkr,kir->ijk', U3_pert, V3_pert, W3_pert)
        
        U3_orig = U_arr.reshape(n, m, rank)
        V3_orig = V_arr.reshape(m, p, rank)
        W3_orig = W_arr.reshape(p, n, rank)
        T_orig = np.einsum('ijr,jkr,kir->ijk', U3_orig, V3_orig, W3_orig)
        
        if not np.allclose(T_pert, T_orig, atol=1e-6):
            continue
        
        U_new, V_new, W_new = _reduce_rank_linear(
            U_pert.tolist(), V_pert.tolist(), W_pert.tolist(),
            max_time=3.0
        )
        
        new_rank = U_new.shape[1]
        if new_rank < best_rank:
            best_rank = new_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    return best_U, best_V, best_W

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_rank = float('inf')
    
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
            
            if time.time() < deadline - 20:
                U_arr = np.array(result["U"], dtype=float)
                V_arr = np.array(result["V"], dtype=float)
                W_arr = np.array(result["W"], dtype=float)
                
                U_arr, V_arr, W_arr = _reduce_rank_linear(
                    U_arr, V_arr, W_arr,
                    max_time=25.0
                )
                new_rank = U_arr.shape[1]
                
                if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": U_arr.tolist(),
                            "V": V_arr.tolist(),
                            "W": W_arr.tolist(),
                        }
                
                if time.time() < deadline - 15:
                    U_arr, V_arr, W_arr = _polish_decomposition(
                        U_arr, V_arr, W_arr,
                        max_time=25.0
                    )
                    new_rank = U_arr.shape[1]
                    
                    if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {
                                "U": U_arr.tolist(),
                                "V": V_arr.tolist(),
                                "W": W_arr.tolist(),
                            }
                
                if time.time() < deadline - 5:
                    U_arr, V_arr, W_arr = _reduce_rank_linear(
                        U_arr, V_arr, W_arr,
                        max_time=8.0
                    )
                    new_rank = U_arr.shape[1]
                    
                    if _verify(U_arr.tolist(), V_arr.tolist(), W_arr.tolist()):
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {
                                "U": U_arr.tolist(),
                                "V": V_arr.tolist(),
                                "W": W_arr.tolist(),
                            }
    except Exception as e:
        pass
    
    if best_result is not None:
        return best_result
    
    return _schoolbook()
# EVOLVE-BLOCK-END
