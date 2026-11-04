"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_outer_terms():
    """
    Build the 11 outer terms for 2x2 * 2x3 -> 2x3 block multiplication.
    7 terms from Strassen (cols 0,1) + 4 terms schoolbook (col 2).
    Each term: dict with 'U', 'V', 'W' mapping (block_row, block_col) -> coeff.
    """
    sU, sV, sW = _strassen_2x2()
    terms = []
    
    # 7 Strassen terms for the 2x2 * 2x2 part (block cols 0,1)
    for t in range(7):
        U_d, V_d, W_d = {}, {}, {}
        for a in range(2):
            for b in range(2):
                c = sU[a*2+b, t]
                if c: U_d[(a,b)] = U_d.get((a,b), 0) + c
        for b in range(2):
            for c in range(2):
                cv = sV[b*2+c, t]
                if cv: V_d[(b,c)] = V_d.get((b,c), 0) + cv
        for a in range(2):
            for c in range(2):
                cw = sW[c*2+a, t]
                if cw: W_d[(a,c)] = W_d.get((a,c), 0) + cw
        terms.append({'U': U_d, 'V': V_d, 'W': W_d})
    
    # 4 schoolbook terms for block col 2
    for a in range(2):
        for b in range(2):
            terms.append({'U': {(a,b): 1}, 'V': {(b,2): 1}, 'W': {(a,2): 1}})
    
    return terms


def _compute_tensor(U, V, W, n, m, p):
    """Compute the full tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]"""
    return np.einsum('ir,jr,kr->ijk', U, V, W)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    # For random A (n x m) and B (m x p), check A@B = sum_r (U[:,r]^T A) * (B V[:,r]) * W[:,r]
    T = _compute_tensor(U, V, W, n, m, p)
    
    # Check against known matrix multiplication tensor
    # T[i,j,k] should equal 1 if i==k and j==j (i.e., A[i,j]*B[j,k])
    # Actually, the tensor for matrix mult is: T[i,j,k] = 1 if the (i,j) element of A 
    # multiplied by (j,k) element of B contributes to C[i,k]
    # In our indexing: A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k where this is the contribution
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each triple (i,j,k) represents A[i,j]*B[j,k] contributing to C[i,k]
    expected = np.ones((n, m, p))
    
    # But wait - our indexing uses flattened indices
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U row index: i*m + j for A[i,j]
    # V row index: j*p + k for B[j,k]  
    # W row index: k*n + i for C[i,k] (trace convention)
    
    # Let me re-derive. The tensor T satisfies:
    # T[im+j, jp+k, kn+i] = 1 for all i,j,k
    # i.e., sum_r U[im+j, r] * V[jp+k, r] * W[kn+i, r] = 1
    
    # Check a few random instances
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition
        C_comp = np.zeros((n, p))
        for r in range(U.shape[1]):
            # U[:,r] gives linear combination of A
            # V[:,r] gives linear combination of B
            # W[:,r] gives how to combine to get C
            Au = (A.reshape(n*m) @ U[:, r])  # scalar
            Bv = (B.reshape(m*p) @ V[:, r])  # scalar
            # W[:,r] has shape (p*n,), indexed as k*n+i
            w_vals = W[:, r].reshape(p, n)  # w_vals[k, i]
            C_comp += Au * Bv * w_vals.T  # w_vals.T has shape (n, p), so C_comp[i,k] += Au*Bv*w_vals[k,i]
        
        if not np.allclose(C_comp, C_expected, atol=tol):
            return False
    
    return True


def _try_remove_column(U, V, W, n, m, p, col_idx, tol=1e-9):
    """
    Try to remove column col_idx by expressing it as a linear combination of others.
    Returns (success, new_U, new_V, new_W)
    """
    R = U.shape[1]
    if R <= 1:
        return False, U, V, W
    
    # We need to find alpha such that:
    # U[:,col_idx] = sum_{r != col_idx} alpha[r] * U[:,r]
    # V[:,col_idx] = sum_{r != col_idx} alpha[r] * V[:,r]  (with appropriate scaling)
    # W[:,col_idx] = sum_{r != col_idx} alpha[r] * W[:,r]
    
    # Actually, we can't just do that. The decomposition requires:
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    # If we remove column r0, we need the remaining columns to still sum to T.
    # So we need: U[:,r0]*V[:,r0]*W[:,r0] = sum_{r != r0} beta[r] * U[:,r]*V[:,r]*W[:,r]
    
    # This is a tensor equation, not a matrix one. It's hard to solve directly.
    
    # Alternative: use the fact that if U[:,r0] is in the span of other U columns,
    # we can try to find a transformation.
    
    # Let's try a simpler approach: check if column r0 is "redundant" by seeing
    # if removing it and adjusting others can still produce the tensor.
    
    # For now, just check if the column is zero
    if np.allclose(U[:, col_idx], 0) and np.allclose(V[:, col_idx], 0) and np.allclose(W[:, col_idx], 0):
        new_U = np.delete(U, col_idx, axis=1)
        new_V = np.delete(V, col_idx, axis=1)
        new_W = np.delete(W, col_idx, axis=1)
        return True, new_U, new_V, new_W
    
    return False, U, V, W


def _coordinate_descent_reduction(U, V, W, n, m, p, max_iter=50, tol=1e-9, deadline=None):
    """
    Try to reduce rank via coordinate descent: iteratively try to zero out
    columns and redistribute their contribution to remaining columns.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    R = current_U.shape[1]
    
    # Compute target tensor
    T_target = _compute_tensor(U, V, W, n, m, p)
    
    # Try to remove each column one at a time
    removed = 0
    for col_idx in range(R - 1, -1, -1):
        if time.time() > deadline:
            break
        
        R_cur = current_U.shape[1]
        if R_cur <= 1:
            break
        
        # Check if this column can be removed
        success, new_U, new_V, new_W = _try_remove_column(
            current_U, current_V, current_W, n, m, p, col_idx, tol
        )
        
        if success:
            # Verify the new decomposition still works
            if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol):
                current_U, current_V, current_W = new_U, new_V, new_W
                removed += 1
    
    return current_U, current_V, current_W, removed


def _grassmannian_optimization(U, V, W, n, m, p, target_rank, max_iter=100, lr=0.01, deadline=None):
    """
    Riemannian gradient descent on the Grassmannian to find a lower-rank decomposition.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    # Start from the current decomposition and try to project to lower rank
    R = U.shape[1]
    if R <= target_rank:
        return U, V, W
    
    # Truncate to target rank via SVD of the "flattened" representation
    # This is a heuristic initialization
    
    # Compute tensor
    T = _compute_tensor(U, V, W, n, m, p)
    
    # Unfold and use SVD to get a lower-rank approximation
    T_flat = T.reshape(n*m, m*p*p*n)  # This is too large, let's be smarter
    
    # Actually, let's use the structure: try different truncation strategies
    # For now, just return the original if we can't find a better one quickly
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    outer_terms = _build_outer_terms()
    sU, sV, sW = _strassen_2x2()
    R = len(outer_terms) * 7  # 77
    
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)
    
    for oi, ot in enumerate(outer_terms):
        for t in range(7):
            col = oi * 7 + t
            for (ai, bj), cu in ot['U'].items():
                if cu == 0: continue
                for ai2 in range(2):
                    for bj2 in range(2):
                        ci = sU[ai2*2 + bj2, t]
                        if ci:
                            U[(2*ai+ai2)*4 + (2*bj+bj2), col] += cu * ci
            for (bj, ck), cv in ot['V'].items():
                if cv == 0: continue
                for bj2 in range(2):
                    for ck2 in range(2):
                        ci = sV[bj2*2 + ck2, t]
                        if ci:
                            V[(2*bj+bj2)*6 + (2*ck+ck2), col] += cv * ci
            for (ai, ck), cw in ot['W'].items():
                if cw == 0: continue
                for ai2 in range(2):
                    for ck2 in range(2):
                        ci = sW[ck2*2 + ai2, t]
                        if ci:
                            W[(2*ck+ck2)*4 + (2*ai+ai2), col] += cw * ci
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but handle gracefully
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try coordinate descent reduction
    if time.time() < deadline - 20:
        U_red, V_red, W_red, removed = _coordinate_descent_reduction(
            U, V, W, n, m, p, max_iter=100, deadline=deadline - 20
        )
        if removed > 0 and _verify_decomposition(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
    
    # Try Grassmannian optimization if time permits
    if time.time() < deadline - 10:
        current_rank = U.shape[1]
        for target_rank in range(current_rank - 1, max(current_rank - 5, 10), -1):
            if time.time() > deadline - 10:
                break
            U_opt, V_opt, W_opt = _grassmannian_optimization(
                U, V, W, n, m, p, target_rank, max_iter=50, deadline=deadline - 10
            )
            if U_opt.shape[1] < U.shape[1]:
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    U, V, W = U_opt, V_opt, W_opt
                break
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to the original verified decomposition
        U = np.zeros((n*m, 77), dtype=float)
        V = np.zeros((m*p, 77), dtype=float)
        W = np.zeros((p*n, 77), dtype=float)
        for oi, ot in enumerate(outer_terms):
            for t in range(7):
                col = oi * 7 + t
                for (ai, bj), cu in ot['U'].items():
                    if cu == 0: continue
                    for ai2 in range(2):
                        for bj2 in range(2):
                            ci = sU[ai2*2 + bj2, t]
                            if ci:
                                U[(2*ai+ai2)*4 + (2*bj+bj2), col] += cu * ci
                for (bj, ck), cv in ot['V'].items():
                    if cv == 0: continue
                    for bj2 in range(2):
                        for ck2 in range(2):
                            ci = sV[bj2*2 + ck2, t]
                            if ci:
                                V[(2*bj+bj2)*6 + (2*ck+ck2), col] += cv * ci
                for (ai, ck), cw in ot['W'].items():
                    if cw == 0: continue
                    for ai2 in range(2):
                        for ck2 in range(2):
                            ci = sW[ck2*2 + ai2, t]
                            if ci:
                                W[(2*ck+ck2)*4 + (2*ai+ai2), col] += cw * ci
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
