"""
Matrix multiplication 2x4x7 via tensor decomposition.
Starts with rank-50 Strassen-based decomposition and attempts to reduce rank
by finding and eliminating linearly dependent rank-1 terms.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for k=0..5
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify(U_m, V_m, W_m):
        """Verify the decomposition is exact."""
        np.random.seed(12345)
        for _ in range(10):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(U_m.shape[1]):
                # C[i,k] += sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Vectorized: for each r, compute contribution
                for i in range(n):
                    for k in range(p):
                        C_comp[i, k] += np.dot(U_m[i*m:(i+1)*m, r], V_m[0:m*p:m*p, r]) * W_m[k*n+i, r] if False else 0
                # Actually let's do it properly
            # Better vectorized approach
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(U_m.shape[1]):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    def try_reduce_rank(U_m, V_m, W_m, max_iter=500):
        """Try to reduce rank by eliminating dependent columns."""
        U_cur = U_m.copy()
        V_cur = V_m.copy()
        W_cur = W_m.copy()
        rank_cur = U_m.shape[1]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            removed = False
            for r in range(rank_cur):
                # Try to eliminate column r by expressing it as combination of others
                # Fix U and W, solve for V_r as combination of V_s
                # Actually, try: can we find alpha_s such that
                # u_r[i,j]*v_r[j,k]*w_r[k,i] = sum_s alpha_s * u_s[i,j]*v_s[j,k]*w_s[k,i]
                # This is nonlinear. Instead, try fixing two matrices.
                
                # Strategy: fix U and W, see if V_r can be written as combo
                # For each (j,k): v_r[j,k] * (sum over i where u_r[i,j]*w_r[k,i] != 0) 
                # This is complicated. Let's try a simpler heuristic.
                
                # Check if column r is "redundant" by checking if the tensor
                # contribution of r can be matched by others.
                # Build the tensor contribution of column r
                # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Check if T_r is in the span of other T_s
                # Flatten all tensors into vectors
                # Dimension of space: n*m*p = 2*4*7 = 56
                # But only 56 entries are non-zero in the structure tensor
                
                # Build matrix M where M[:,s] = vec(T_s) for s in 0..rank_cur-1
                # Then check if M[:,r] is in span of M[:,s] for s != r
                dim = n * m * p
                M = np.zeros((dim, rank_cur))
                for s in range(rank_cur):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                M[i*m*p + j*p + k, s] = (U_cur[i*m+j, s] * 
                                                         V_cur[j*p+k, s] * 
                                                         W_cur[k*n+i, s])
                
                # Check if column r is in span of others
                if rank_cur > 1:
                    others = [s for s in range(rank_cur) if s != r]
                    M_others = M[:, others]
                    # Check if M[:,r] is in column space of M_others
                    # Use least squares
                    try:
                        sol, residuals, rank_others, sv = np.linalg.lstsq(M_others, M[:, r], rcond=None)
                        residual = np.linalg.norm(M[:, r] - M_others @ sol)
                        norm_r = np.linalg.norm(M[:, r])
                        if norm_r > 1e-12 and residual < 1e-8 * max(1, norm_r):
                            # Column r is redundant! Remove it.
                            # But we need to actually adjust the decomposition.
                            # The issue is that T_r = sum alpha_s T_s doesn't mean
                            # we can simply remove r - we need the U,V,W to still work.
                            # This approach only works if the tensor vectors are linearly
                            # dependent AND we can redistribute the contributions.
                            # For now, just note it.
                            pass
                    except:
                        pass
            
            # Alternative: try to find a rank-reducing transformation
            # If we have two columns r1, r2 such that their U vectors are proportional
            # or their V vectors are proportional, we might merge them.
            
            # Try: for each pair (r1, r2), check if U[:,r1] is proportional to U[:,r2]
            # If so, we can potentially merge.
            for r1 in range(rank_cur):
                if time.time() > deadline:
                    break
                for r2 in range(r1+1, rank_cur):
                    # Check if U columns are proportional
                    u1 = U_cur[:, r1]
                    u2 = U_cur[:, r2]
                    if np.allclose(u1, 0) or np.allclose(u2, 0):
                        continue
                    # Check proportionality
                    ratio = u1 / (u2 + 1e-15)
                    if np.std(ratio[u2 != 0]) < 1e-10:
                        # They're proportional, might be mergeable
                        pass
            
            if not removed:
                break
        
        return U_cur, V_cur, W_cur, rank_cur
    
    # Try the reduction
    U_reduced, V_reduced, W_reduced, rank_reduced = try_reduce_rank(U, V, W)
    
    # Verify the result
    if verify(U_reduced, V_reduced, W_reduced):
        # Convert to lists
        U_out = U_reduced.tolist()
        V_out = V_reduced.tolist()
        W_out = W_reduced.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Return the original rank-50
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
