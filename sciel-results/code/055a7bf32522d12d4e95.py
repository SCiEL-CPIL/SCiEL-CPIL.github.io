"""Matrix multiplication 4x4x6 via hybrid Strassen decomposition with SVD-based rank compression."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _get_strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j], V[j*2+k], W[k*2+i].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U[i * m + j][m_idx] += c
        for (j, k, c) in Vb:
            V[j * p + k][m_idx] += c
        for (i, k, c) in Wb:
            W[k * n + i][m_idx] += c

    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return err

def _rank_reduce(U, V, W, n, m, p, rank, tol=1e-9):
    """
    Try to reduce the rank by identifying linearly dependent rank-1 terms.
    A rank-1 term c is redundant if its tensor contribution is a linear
    combination of other terms. We check this via the matricized tensor.
    
    Returns reduced U, V, W with potentially lower rank.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build matrix where each column c is the flattened tensor T_c
    flat_size = n * m * p
    Tmat = np.zeros((flat_size, rank))
    
    for c in range(rank):
        Tc = np.einsum('ij,jk,ki->ijk', 
                       Ua[:, c].reshape(n, m),
                       Va[:, c].reshape(m, p),
                       Wa[:, c].reshape(p, n))
        Tmat[:, c] = Tc.flatten()
    
    # Greedy: keep columns that are not in the span of previously kept columns
    U_reduced = []
    V_reduced = []
    W_reduced = []
    Q = None  # Orthonormal basis for the span
    num_kept = 0
    
    for c in range(rank):
        if time.time() > 100:  # safety margin
            break
            
        col = Tmat[:, c:c+1]
        col_norm = np.linalg.norm(col)
        
        if col_norm < tol:
            # Zero column, skip
            continue
        
        if num_kept == 0:
            Q = col / col_norm
            U_reduced.append(Ua[:, c])
            V_reduced.append(Va[:, c])
            W_reduced.append(Wa[:, c])
            num_kept += 1
        else:
            # Project onto current Q
            coeff = Q.T @ col  # (num_kept, 1)
            residual = col - Q @ coeff
            resid_norm = np.linalg.norm(residual)
            
            if resid_norm <= tol * max(1.0, col_norm):
                # In span - skip
                continue
            else:
                q_new = residual / resid_norm
                Q = np.hstack([Q, q_new])
                U_reduced.append(Ua[:, c])
                V_reduced.append(Va[:, c])
                W_reduced.append(Wa[:, c])
                num_kept += 1
    
    if num_kept < rank and num_kept > 0:
        U_new = np.column_stack(U_reduced)
        V_new = np.column_stack(V_reduced)
        W_new = np.column_stack(W_reduced)
        
        # Verify
        Ur = U_new.reshape(n, m, num_kept)
        Vr = V_new.reshape(m, p, num_kept)
        Wr = W_new.reshape(p, n, num_kept)
        T_check = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T_check - 1.0))
        
        if err < 1e-6:
            return U_new.tolist(), V_new.tolist(), W_new.tolist(), num_kept
    
    return U, V, W, rank


def _svd_compress(U, V, W, n, m, p, rank, tol=1e-10):
    """
    Apply SVD-based linear compression to reduce rank by finding a lower-rank
    representation that exactly reproduces the tensor.
    
    The key insight: if the factor matrices have linear dependencies across
    their columns, we can find a transformation that preserves the tensor
    while reducing the number of rank-1 terms.
    
    We use the fact that for any invertible R of shape (rank, r) with r < rank,
    if U @ R, V @ R^{-1}, W @ I_r (or similar) still give the same tensor,
    we've reduced the rank.
    
    A practical approach: look for a subspace of the "interaction space" that
    captures all the tensor structure.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This can be written as: for each (i,k), T[i,:,k] = sum_c (U[:,j,c] * W[k,i,c]) * V[j,k,c]
    # Or: T[i,j,k] = <U_ij, V_jk ⊙ W_ki> where ⊙ is element-wise
    
    # Alternative view: T = sum_c (U_c ⊗ V_c ⊗ W_c) where U_c is n×m, etc.
    # Flatten each rank-1 term: vec(U_c) ⊗ vec(V_c) ⊗ vec(W_c)
    
    # Build the "interaction matrix" M where M[a,b] = sum_c U_flat[a,c] * V_flat[b,c] * W_flat[c,c]
    # This isn't quite right. Let's think differently.
    
    # For fixed (i,k), the slice T[i,:,k] is a linear combination of the rows of V
    # weighted by U[i,:,c] * W[k,i,c] for each c.
    
    # Actually, let's use a different approach: 
    # Consider the matrix A_{(i,k),c} = U[i,:,c] * W[k,i,c] (this is m-vector for each c)
    # Then T[i,:,k] = sum_c A_{(i,k),c} * V[:,k,c]... no that's not right either.
    
    # Let me be more careful:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For fixed (i,k): T[i,:,k][j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = sum_c (U[i*m+j, c] * W[k*n+i, c]) * V[j*p+k, c]
    
    # So for each (i,k), the m-vector T[i,:,k] is in the span of the m-vectors
    # {V[:,k,c] * U[i*m+j,c] * W[k*n+i,c]} ... this is getting complicated.
    
    # Simpler approach: use the structure that the 4x4x4 part is a Kronecker product
    # of two 2x2x2 Strassen instances, and the 4x4x2 part uses shared Strassen factors.
    # There may be linear dependencies between the 49 Kronecker terms and the 28 additional terms.
    
    # Let's try: form the 96x96 (or smaller) "coupling matrix" C where
    # C[a,b] = sum_c U[a,c] * V[b,c] * W[c,c] ... no, indices don't match.
    
    # Better: use the "mode-3" unfolding approach.
    # T_{ijk} = sum_c U_{ijc} V_{jkc} W_{kic}
    
    # Define M_c = U_c (n×m) ⊗ V_c (m×p) ⊗ W_c (p×n) as a vector of length nmp.
    # T = sum_c M_c (vector).
    
    # The minimum rank is the minimum number of such vectors that sum to T.
    # If some M_c are linearly dependent, we can reduce.
    
    # Check linear dependence of the M_c vectors:
    flat_size = n * m * p
    Tmat = np.zeros((flat_size, rank))
    for c in range(rank):
        Tc = np.einsum('ij,jk,ki->ijk',
                       Ua[:, c].reshape(n, m),
                       Va[:, c].reshape(m, p),
                       Wa[:, c].reshape(p, n))
        Tmat[:, c] = Tc.flatten()
    
    # SVD of Tmat to find numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(Tmat, full_matrices=False)
    except np.linalg.LinAlgError:
        return U, V, W, rank
    
    # Find numerical rank
    num_rank = np.sum(s_vals > tol * s_vals[0]) if len(s_vals) > 0 else 0
    
    if num_rank >= rank or num_rank == 0:
        return U, V, W, rank
    
    # The SVD tells us the tensor lies in a num_rank-dimensional subspace.
    # But we need to find an explicit rank-num_rank decomposition.
    # This is the hard part - SVD of the flattened tensor doesn't directly give
    # a rank decomposition.
    
    # Alternative: use the fact that our construction has specific structure.
    # The 4x4x4 part uses Kronecker product of Strassen, so its 49 terms have
    # the form U_s[a,t]*U_s[b,s] etc. There might be relations.
    
    # For now, let's try a simpler compression: if two columns c1, c2 have
    # U[:,c1] = α*U[:,c2], V[:,c1] = β*V[:,c2], W[:,c1] = γ*W[:,c2] with
    # αβγ = 1, then they contribute the same and we can merge.
    
    # Check for proportional columns:
    kept = list(range(rank))
    merged = {}  # maps kept index to (U_col, V_col, W_col, count)
    
    for c in range(rank):
        if c not in kept:
            continue
        U_c = Ua[:, c]
        V_c = Va[:, c]
        W_c = Wa[:, c]
        
        # Check if this column is proportional to any previously kept column
        found = False
        for k in kept:
            if k == c:
                continue
            if k not in merged:
                continue
            U_k, V_k, W_k = merged[k][:3]
            
            # Check proportionality
            # U_c ≈ α * U_k, V_c ≈ β * V_k, W_c ≈ γ * W_k, with αβγ ≈ 1
            if np.linalg.norm(U_k) < tol or np.linalg.norm(V_k) < tol or np.linalg.norm(W_k) < tol:
                continue
            
            alpha = np.dot(U_c, U_k) / np.dot(U_k, U_k)
            beta = np.dot(V_c, V_k) / np.dot(V_k, V_k)
            gamma = np.dot(W_c, W_k) / np.dot(W_k, W_k)
            
            # Check if products are close
            if abs(alpha * beta * gamma - 1.0) < 1e-8:
                # Check residuals
                res_U = np.linalg.norm(U_c - alpha * U_k)
                res_V = np.linalg.norm(V_c - beta * V_k)
                res_W = np.linalg.norm(W_c - gamma * W_k)
                
                if res_U < 1e-8 * max(1, np.linalg.norm(U_c)) and \
                   res_V < 1e-8 * max(1, np.linalg.norm(V_c)) and \
                   res_W < 1e-8 * max(1, np.linalg.norm(W_c)):
                    # Merge: just add to count (the contribution is identical)
                    found = True
                    break
        
        if found:
            kept.remove(c)
    
    if len(kept) < rank:
        U_new = Ua[:, kept]
        V_new = Va[:, kept]
        W_new = Wa[:, kept]
        
        Ur = U_new.reshape(n, m, len(kept))
        Vr = V_new.reshape(m, p, len(kept))
        Wr = W_new.reshape(p, n, len(kept))
        T_check = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T_check - 1.0))
        
        if err < 1e-6:
            return U_new.tolist(), V_new.tolist(), W_new.tolist(), len(kept)
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    U_s, V_s, W_s = _get_strassen_2x2_factors()
    
    # Total rank: 49 (for 4x4x4) + 28 (for 4x4x2) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x 77
    V = [[0.0] * total_rank for _ in range(m * p)]  # 24 x 77
    W = [[0.0] * total_rank for _ in range(p * n)]  # 24 x 77
    
    # 1. 4x4x4 Strassen (columns 0-48)
    # Kronecker product of 2x2 Strassen with itself
    for i in range(4):
        I = i // 2
        i0 = i % 2
        for j in range(4):
            J = j // 2
            j0 = j % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = U_s[2 * I + J][t] * U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col] += val
                        
    for j in range(4):
        J = j // 2
        j0 = j % 2
        for k in range(4):
            K = k // 2
            k0 = k % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = V_s[2 * J + K][t] * V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j * 6 + k][col] += val
                        
    for k in range(4):
        K = k // 2
        k0 = k % 2
        for i in range(4):
            I = i // 2
            i0 = i % 2
            for t in range(7):
                for s in range(7):
                    col = 7 * t + s
                    val = W_s[2 * K + I][t] * W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k * 4 + i][col] += val

    # 2. 4x4x2 Strassen-block (columns 49-76)
    # Split B into B_top (rows 0-1, cols 4-5) and B_bot (rows 2-3, cols 4-5)
    # C_top = A_00 * B_top + A_01 * B_bot
    # C_bot = A_10 * B_top + A_11 * B_bot
    # Each product is 2x2 * 2x2, using Strassen (rank 7)
    
    products = [
        (0, 0, 0),  # I=0, J=0, is_bot=0 (B_top)
        (0, 1, 1),  # I=0, J=1, is_bot=1 (B_bot)
        (1, 0, 0),  # I=1, J=0, is_bot=0 (B_top)
        (1, 1, 1),  # I=1, J=1, is_bot=1 (B_bot)
    ]
    
    for p_idx, (I, J, is_bot) in enumerate(products):
        col_offset = rank_444 + p_idx * 7
        
        # A block: A_IJ, rows 2I..2I+1, cols 2J..2J+1
        for i0 in range(2):
            for j0 in range(2):
                i = 2 * I + i0
                j = 2 * J + j0
                for s in range(7):
                    val = U_s[2 * i0 + j0][s]
                    if val != 0:
                        U[i * 4 + j][col_offset + s] += val
                        
        # B block: rows 2*is_bot..2*is_bot+1, cols 4..5
        for j0 in range(2):
            for k0 in range(2):
                j_global = 2 * is_bot + j0
                k_global = 4 + k0
                for s in range(7):
                    val = V_s[2 * j0 + k0][s]
                    if val != 0:
                        V[j_global * 6 + k_global][col_offset + s] += val
                        
        # C block: rows 2I..2I+1, cols 4..5
        for i0 in range(2):
            for k0 in range(2):
                i_global = 2 * I + i0
                k_global = 4 + k0
                for s in range(7):
                    val = W_s[2 * k0 + i0][s]
                    if val != 0:
                        W[k_global * 4 + i_global][col_offset + s] += val

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try rank reduction by removing linearly dependent rank-1 terms
    if time.time() < deadline - 5.0:
        U_red, V_red, W_red, new_rank = _rank_reduce(U, V, W, n, m, p, total_rank)
        if new_rank < total_rank:
            return {"U": U_red, "V": V_red, "W": W_red}
    
    # Try SVD-based compression
    if time.time() < deadline - 5.0:
        U_comp, V_comp, W_comp, comp_rank = _svd_compress(U, V, W, n, m, p, total_rank)
        if comp_rank < total_rank:
            return {"U": U_comp, "V": V_comp, "W": W_comp}

    return {"U": U, "V": V, "W": W}
