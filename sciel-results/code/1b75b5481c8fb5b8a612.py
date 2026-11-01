import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def _rank11_solution():
    """Return the known rank-11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # 1. (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # 4. A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11) -> M7
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 8. A00 B02 -> M8
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # 9. A01 B12 -> M9
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # 10. A10 B02 -> M10
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # 11. A11 B12 -> M11
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _try_slice_absorption(n, m, p, T, deadline):
    """
    Slice absorption: Try to find rank-10 by absorbing the third output column
    (k=2) into a modified version of the rank-11 factors.
    
    Strategy: The rank-11 solution has 4 terms for columns k=0,1 (M1-M7) and 
    4 terms for column k=2 (M8-M11). We try to merge some of these.
    
    Alternative: Use the algebraic structure. For 2x2x3, the tensor has a 
    special structure. We parameterize a rank-10 decomposition with constraints.
    """
    target_rank = 10
    
    # The key insight: C[:,2] = A * B[:,2]
    # B[:,2] = [B02, B12]^T
    # C[0,2] = A00*B02 + A01*B12
    # C[1,2] = A10*B02 + A11*B12
    
    # These are 4 bilinear terms. In rank-11, they use 4 separate rank-1 terms.
    # Can we do better? The 2x2 submatrix for column 2 is a full rank-4 tensor.
    # But maybe we can share factors with the other columns.
    
    # Let's try a different approach: parameterized search with structure.
    # Use the fact that the first two columns (k=0,1) form a 2x2x2 tensor (rank 7 by Strassen)
    # and the third column adds 4 more terms.
    
    # Actually, let's try: keep the Strassen structure for k=0,1 (7 terms) 
    # and find 3 additional terms that handle k=2 plus any corrections.
    # Total: 10 terms.
    
    # The 2x2x2 tensor (for k=0,1, ignoring k=2) has rank 7.
    # The k=2 column needs: A00*B02 + A01*B12 -> C02, A10*B02 + A11*B12 -> C12
    # That's 4 bilinear terms. If we can do it in 3, total is 10.
    
    # Can 4 terms of a 2x2 matrix multiplication (for one output column) 
    # be done in 3 bilinear forms? No, rank of 2x2x1 tensor is 4 (full rank).
    # Wait, the tensor for C[:,2] = A * B[:,2] is:
    # T[a,b,c] where a=(i,j), b=(j,2), c=(2,i)
    # T[0,3,4] = 1 (A00*B02 -> C02)  [a=00, b=02, c=20]
    # T[1,5,4] = 1 (A01*B12 -> C02)  [a=01, b=12, c=20]
    # T[2,3,5] = 1 (A10*B02 -> C12)  [a=10, b=02, c=21]
    # T[3,5,5] = 1 (A11*B12 -> C12)  [a=11, b=12, c=21]
    # This is a 4x4x2 tensor (only 4 nonzero entries in the a,b plane, spread over 2 c values).
    # Its rank is 4 (it's like a permutation matrix structure).
    
    # So we need at least 4 terms for the k=2 column alone.
    # Plus 7 for k=0,1. Total 11.
    # Unless we can share some factors.
    
    # Let's try: modify the Strassen factors to also contribute to k=2.
    # This requires the U and V factors to have nonzero entries for j=0 or j=1 
    # paired with k=2 in B, and the W factors to have entries for k=2.
    
    # This is a complex optimization. Let's do a structured search.
    
    # Parameterization: Start from rank-11, perturb, and try to nullify one term.
    
    U11, V11, W11 = _rank11_solution()
    
    # Try to find rank-10 by making one of the 11 terms zero through perturbation.
    # Or by finding a linear dependence.
    
    # Alternative: Use the "border rank" idea. The tensor might have border rank 10.
    # Let's try to find a rank-10 approximation that is actually exact.
    
    # Structured random search with warm start from rank-11 (dropping one term)
    for drop_idx in range(11):
        if time.time() > deadline:
            break
        
        # Create rank-10 init by dropping term drop_idx and perturbing
        U0 = np.delete(U11, drop_idx, axis=1)
        V0 = np.delete(V11, drop_idx, axis=1)
        W0 = np.delete(W11, drop_idx, axis=1)
        
        # Perturb
        U = U0 + 0.1 * np.random.randn(n*m, target_rank)
        V = V0 + 0.1 * np.random.randn(m*p, target_rank)
        W = W0 + 0.1 * np.random.randn(p*n, target_rank)
        
        # Alternating least squares
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U
            X_u = np.zeros((m*p*p*n, target_rank))
            for rr in range(target_rank):
                X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
            U, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
            U = U.T
            
            # Update V
            X_v = np.zeros((n*m*p*n, target_rank))
            for rr in range(target_rank):
                X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
            V, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
            V = V.T
            
            # Update W
            X_w = np.zeros((n*m*m*p, target_rank))
            for rr in range(target_rank):
                X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
            W, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
            W = W.T
        
        # Check
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        
        err = np.linalg.norm(T - T_rec)
        if err < 1e-6:
            # Check half-integer property
            is_half_int = True
            for mat in [U, V, W]:
                for val in mat:
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-3:
                        is_half_int = False
                        break
                if not is_half_int:
                    break
            
            if is_half_int:
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                
                T_rec_exact = np.zeros((n*m, m*p, p*n))
                for rr in range(target_rank):
                    T_rec_exact += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                
                if np.allclose(T_rec_exact, T, atol=1e-9):
                    return U_s, V_s, W_s
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try slice absorption / structured rank-10 search
    result = _try_slice_absorption(n, m, p, T, deadline)
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-11
    U, V, W = _rank11_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
