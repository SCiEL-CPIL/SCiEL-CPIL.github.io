import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.
    
    The tensor T has entries T[a,b,c] = 1 where a=(i,j), b=(j,k), c=(k,i)
    with i in {0,1}, j in {0,1}, k in {0,...,7}.
    
    We construct the rank-28 Strassen-based decomposition and then attempt
    to reduce the rank by checking for linear dependencies among the
    Kronecker-product terms. If a lower rank is found, we return it;
    otherwise we return the rank-28 decomposition.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16

    # Strassen products M1..M7 for 2x2x2
    # Index convention: A indexed by (i,j) with i=row, j=col -> flat idx = i*m+j
    # B indexed by (j,k) with j=row, k=col -> flat idx = j*p+k
    # W indexed by (k,i) -> flat idx = k*n+i

    # For each Strassen term s (0..6), define:
    #   A_forms[s]: dict from flat A-index to coefficient
    #   B_forms[s]: dict from flat B-index (within one 2x2 block) to coefficient
    #   C_coeffs[c_local][s]: coefficient for C output c_local in block

    A_forms = [
        {0: 1, 3: 1},       # M1: A[0,0] + A[1,1]
        {2: 1, 3: 1},       # M2: A[1,0] + A[1,1]
        {0: 1},             # M3: A[0,0]
        {3: 1},             # M4: A[1,1]
        {0: 1, 1: 1},       # M5: A[0,0] + A[0,1]
        {2: 1, 0: -1},      # M6: A[1,0] - A[0,0]
        {1: 1, 3: -1}       # M7: A[0,1] - A[1,1]
    ]

    B_forms = [
        {0: 1, 3: 1},       # M1: B[0,0] + B[1,1] (within 2x2 block)
        {0: 1},             # M2: B[0,0]
        {1: 1, 3: -1},      # M3: B[0,1] - B[1,1]
        {2: 1, 0: -1},      # M4: B[1,0] - B[0,0]
        {3: 1},             # M5: B[1,1]
        {0: 1, 1: 1},       # M6: B[0,0] + B[0,1]
        {2: 1, 3: 1}        # M7: B[1,0] + B[1,1]
    ]

    # C_coeffs[c_local][s] for c_local in {0,1,2,3} = {(0,0),(0,1),(1,0),(1,1)}
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C[0,0]
        {2: 1, 4: 1},               # C[0,1]
        {1: 1, 3: 1},               # C[1,0]
        {0: 1, 1: -1, 2: 1, 5: 1}   # C[1,1]
    ]

    # Build rank-28 decomposition: 4 blocks (for k0 = 0,2,4,6) x 7 Strassen terms
    rank_28 = 28
    U = np.zeros((nm, rank_28), dtype=np.float64)
    V = np.zeros((mp, rank_28), dtype=np.float64)
    W = np.zeros((pn, rank_28), dtype=np.float64)

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s

            # Fill U
            for idx, val in A_forms[s].items():
                i = idx // m
                j = idx % m
                U[i * m + j, r] = val

            # Fill V
            for idx, val in B_forms[s].items():
                j = idx // 2  # within 2x2 block
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global, r] = val

            # Fill W
            for c_local in range(4):
                if s in C_coeffs[c_local]:
                    coeff = C_coeffs[c_local][s]
                    i = c_local // 2
                    k_local = c_local % 2
                    k_global = k0 + k_local
                    W[k_global * n + i, r] = coeff

    def time_check():
        return time.time() - start_time > deadline

    # Build the target tensor for verification
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Flatten T_target to a vector
    T_flat_target = T_target.flatten()

    # Build the 1024 x 28 matrix whose columns are the Kronecker products
    T_flat = np.zeros((nm * mp * pn, rank_28), dtype=np.float64)
    for r in range(rank_28):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T_flat[:, r] = np.kron(np.kron(u, v), w)
        if time_check():
            break

    # Check rank of the column space
    try:
        rank_val = np.linalg.matrix_rank(T_flat, tol=1e-10)
    except:
        rank_val = rank_28

    if rank_val < rank_28 and not time_check():
        # Find a basis of independent columns
        Q, R = np.linalg.qr(T_flat, mode='complete')
        diag_R = np.abs(np.diag(R))
        tol = 1e-10
        pivots = np.where(diag_R > tol)[0]
        
        if len(pivots) == rank_val:
            U_red = U[:, pivots]
            V_red = V[:, pivots]
            W_red = W[:, pivots]
            
            # Verify
            T_decomposed = np.zeros((nm, mp, pn))
            for r in range(len(pivots)):
                T_decomposed += np.einsum('a,b,c->abc', U_red[:, r], V_red[:, r], W_red[:, r])
            
            if np.allclose(T_decomposed, T_target, atol=1e-10):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    # Try to find a lower rank by adding linear combinations
    # The idea: if some columns of T_flat are linearly dependent,
    # we can express one as a combo of others and eliminate it.
    # But since we already checked rank, let's try a different approach:
    # Try to find if we can reduce by 1 using a null vector.
    
    if not time_check():
        # Compute null space of T_flat^T (i.e., vectors x such that T_flat^T @ x = 0)
        try:
            _, S, Vt = np.linalg.svd(T_flat, full_matrices=True)
            # S has shape (min(1024,28),) = (28,)
            # Vt has shape (28, 28)
            # Null space vectors are rows of Vt where S is near zero
            tol = 1e-10
            null_mask = S < tol
            if np.any(null_mask):
                null_vecs = Vt[null_mask, :]  # shape (k, 28)
                # For each null vector x (length 28), we have:
                # sum_r x[r] * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0
                # This means the 28 tensors are linearly dependent.
                # But we already used this to find rank_val.
                # Let's try to find a decomposition with fewer terms by
                # absorbing the dependency.
                
                # Actually, the rank of the set of tensors IS rank_val.
                # So we can't go below rank_val with these particular U,V,W.
                # But we might be able to find different U,V,W with lower rank.
                
                # For now, let's just use the reduced rank if possible.
                pass
        except:
            pass

    # Attempt: use the fact that the tensor has block structure to try
    # a smarter decomposition. The tensor is T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    # where k ranges over 0..7. This is like 4 copies of the 2x2x2 tensor
    # but with the k-index extended.
    
    # Actually, let's try a direct approach: see if rank 21 (3*7) or 14 (2*7) works.
    # For rank 14: use 2 Strassen terms per block instead of 7? Unlikely.
    
    # Let's try a random search for lower rank.
    best_rank = rank_28
    best_U, best_V, best_W = U, V, W

    if not time_check():
        # Try rank values from 21 down to 14
        for target_rank in range(21, 13, -1):
            if time_check():
                break
            success = False
            # Try a few random initializations
            for attempt in range(10):
                if time_check():
                    break
                # Initialize randomly
                U_rand = np.random.randn(nm, target_rank) * 0.5
                V_rand = np.random.randn(mp, target_rank) * 0.5
                W_rand = np.random.randn(pn, target_rank) * 0.5
                
                # Do a few ALS iterations
                for it in range(50):
                    if time_check():
                        break
                    # Update W: fix U, V
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For each (a,b): sum_r U[a,r] V[b,r] W[:,r] = T[a,b,:]
                    # This is a linear system: (U * V^T) @ W = T_ab
                    # Actually: sum_r U[a,r] V[b,r] W[c,r] = (U V^T)[a,b] ... no
                    # Let's define M[c, a, b] = sum_r W[c,r] U[a,r] V[b,r]
                    # For fixed a,b: M[:, a, b] = W @ (U[a,:] ⊙ V[b,:])^T
                    # where ⊙ is elementwise product
                    # So W @ diag(U[a,:] ⊙ V[b,:])^T = T[:, a, b]
                    # Actually: M[:, a, b] = W (U[a,:] * V[b,:])^T
                    # So W (U[a,:] * V[b,:])^T = T[:, a, b]
                    # This is pn x target_rank @ target_rank x 1 = pn x 1
                    # For all (a,b), we have mp*nm equations.
                    # Stack: for each (a,b), W @ (U[a,:] * V[b,:])^T = T[:, a, b]
                    # Let A_als = [U[a,:] * V[b,:] for all a,b] shape (nm*mp, target_rank)
                    # B_als = T.reshape(pn, nm*mp).T shape (nm*mp, pn)
                    # Then W^T A_als^T = B_als => A_als W^T = B_als^T
                    # So W^T = A_als^+ B_als^T
                    # W = (A_als^+ B_als^T)^T
                    
                    UaVb = (U_rand[:, :, None, None] * V_rand[None, None, :, :])  # (nm, target_rank, mp, target_rank) - wrong shape
                    # Better: for each r, U[a,r] * V[b,r]
                    # A_als[a*mp+b, r] = U[a,r] * V[b,r]
                    A_als = np.zeros((nm * mp, target_rank))
                    for a in range(nm):
                        for b in range(mp):
                            A_als[a * mp + b, :] = U_rand[a, :] * V_rand[:, b]
                    
                    B_als = T_target.reshape(pn, nm, mp).transpose(2, 1, 0).reshape(nm * mp, pn)
                    # B_als[ab, c] = T[a, b, c]
                    
                    # Solve A_als @ W^T = B_als
                    W_new, _, _, _ = np.linalg.lstsq(A_als, B_als, rcond=None)
                    W_rand = W_new.T  # (pn, target_rank)
                    
                    # Update U
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed b,c: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                    # U @ (V[b,:] * W[c,:])^T = T[:, b, c]
                    A_als2 = np.zeros((mp * pn, target_rank))
                    for b in range(mp):
                        for c in range(pn):
                            A_als2[b * pn + c, :] = V_rand[b, :] * W_rand[c, :]
                    B_als2 = T_target.reshape(nm, mp, pn).transpose(1, 2, 0).reshape(mp * pn, nm)
                    U_new, _, _, _ = np.linalg.lstsq(A_als2, B_als2, rcond=None)
                    U_rand = U_new.T
                    
                    # Update V
                    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                    # For fixed a,c: sum_r V[b,r] U[a,r] W[c,r] = T[a,b,c]
                    A_als3 = np.zeros((nm * pn, target_rank))
                    for a in range(nm):
                        for c in range(pn):
                            A_als3[a * pn + c, :] = U_rand[a, :] * W_rand[c, :]
                    B_als3 = T_target.reshape(nm, mp, pn).transpose(0, 2, 1).reshape(nm * pn, mp)
                    V_new, _, _, _ = np.linalg.lstsq(A_als3, B_als3, rcond=None)
                    V_rand = V_new.T
                
                # Check if we converged
                T_decomposed = np.zeros((nm, mp, pn))
                for r in range(target_rank):
                    T_decomposed += np.einsum('a,b,c->abc', U_rand[:, r], V_rand[:, r], W_rand[:, r])
                
                if np.allclose(T_decomposed, T_target, atol=1e-8):
                    # Verify more precisely
                    if np.allclose(T_decomposed, T_target, atol=1e-10):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                        success = True
                        break
            if success:
                break

    if best_rank < rank_28:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback to rank 28
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
