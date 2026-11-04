import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Build the matrix multiplication tensor T[i, j, k] where
    # i indexes (row of A, col of A) -> 2*4=8
    # j indexes (row of B, col of B) -> 4*8=32
    # k indexes (row of C, col of C) -> 2*8=16
    # T[i,j,k] = 1 if the (i,j) pair contributes to C[k]
    # A is 2x4, B is 4x8, C = A*B is 2x8
    # C[r,c] = sum_{s=0}^{3} A[r,s] * B[s,c]
    # i = r*4+s, j = s*8+c, k = r*8+c
    T = np.zeros((8, 32, 16))
    for r in range(2):
        for s in range(4):
            for c in range(8):
                i = r * 4 + s
                j = s * 8 + c
                k = r * 8 + c
                T[i, j, k] = 1.0
    
    # Flatten T to a matrix of shape (8*16, 32) for rank estimation
    # T reshaped: for each j, we have a vector in R^{8*16}
    T_mat = T.reshape(128, 32)
    
    # The rank of T_mat gives a lower bound on tensor rank
    # But tensor rank >= matrix rank
    # Let's try to find a low-rank decomposition
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid rank-r decomposition of T"""
        r = U.shape[1]
        # Reconstruct T from decomposition
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Check: for all i,j,k, T[i,j,k] == sum_r U[i,r]*V[j,r]*W[k,r]
        # Efficient check: T_mat @ V.T should relate to outer products
        # Actually: for each r, the contribution to T is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Reshape: T[i,j,k] = sum_r (U[i,r]*V[j,r])*W[k,r]
        # Let M[j,k] = sum_r V[j,r]*W[k,r], then T[i,j,k] = sum_j' ... no
        # Better: T.reshape(8, 32*16) vs U @ (V.reshape(32,r) * W.reshape(16,r).T)
        # T[i, j*16+k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # = sum_r U[i,r] * (V[j,r] * W[k,r])
        # Let B[j,k] = sum_r V[j,r]*W[k,r], then T[i, j*16+k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is not simply U @ B because B depends on r through V and W separately.
        # Correct check:
        recon = np.zeros((8, 32, 16))
        for r_idx in range(r):
            recon += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
        return np.allclose(recon, T, atol=1e-10)
    
    def verify_fast(U, V, W):
        """Faster verification using einsum"""
        r = U.shape[1]
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(recon, T, atol=1e-10)
    
    # Start with the known rank-56 Strassen decomposition
    def strassen_decomp():
        U = np.zeros((8, 56))
        V = np.zeros((32, 56))
        W = np.zeros((16, 56))
        
        col = 0
        blocks = [
            (0, 0, 0),
            (0, 0, 2),
            (0, 0, 4),
            (0, 0, 6),
            (1, 0, 0),
            (1, 0, 2),
            (1, 0, 4),
            (1, 0, 6),
        ]
        
        for (a_col_off, b_row_off, c_col_off) in blocks:
            a00 = 0*4 + a_col_off
            a01 = 0*4 + a_col_off + 1
            a10 = 1*4 + a_col_off
            a11 = 1*4 + a_col_off + 1
            
            b00 = b_row_off*8 + c_col_off
            b01 = b_row_off*8 + c_col_off + 1
            b10 = (b_row_off+1)*8 + c_col_off
            b11 = (b_row_off+1)*8 + c_col_off + 1
            
            c00 = c_col_off*2 + 0
            c01 = (c_col_off+1)*2 + 0
            c10 = c_col_off*2 + 1
            c11 = (c_col_off+1)*2 + 1
            
            U[a00, col] += 1; U[a11, col] += 1
            V[b00, col] += 1; V[b11, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1
            
            U[a10, col] += 1; U[a11, col] += 1
            V[b00, col] += 1
            W[c10, col] += 1; W[c11, col] += -1
            col += 1
            
            U[a00, col] += 1
            V[b01, col] += 1; V[b11, col] += -1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1
            
            U[a11, col] += 1
            V[b10, col] += 1; V[b00, col] += -1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1
            
            U[a00, col] += 1; U[a01, col] += 1
            V[b11, col] += 1
            W[c00, col] += -1; W[c01, col] += 1
            col += 1
            
            U[a10, col] += 1; U[a00, col] += -1
            V[b00, col] += 1; V[b01, col] += 1
            W[c11, col] += 1
            col += 1
            
            U[a01, col] += 1; U[a11, col] += -1
            V[b10, col] += 1; V[b11, col] += 1
            W[c00, col] += 1
            col += 1
        
        return U, V, W
    
    U0, V0, W0 = strassen_decomp()
    best_rank = 56
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy: try to reduce rank by finding linear dependencies
    # If column r is a linear combination of other columns in U, V, W simultaneously,
    # we can eliminate it.
    
    # Approach: Randomized search for lower rank
    # Try ranks from 48 down to some lower bound
    
    # Lower bound: The tensor T has a certain structure.
    # For matrix multiplication m×n × n×p, the rank is at least max(m*n, n*p, m*p) = max(8,32,16)=32
    # But that's a weak bound. The actual minimum for 2x4 x 4x8 is unknown but likely around 48-56.
    
    # Let's try a structured search: modify the Strassen decomposition
    # to see if we can merge blocks or reduce rank.
    
    # Alternative: Use the fact that we have 8 blocks of 7 = 56.
    # Can we do better by sharing some factors across blocks?
    
    # Let's try: what if we use a different block structure?
    # Instead of 8 independent 2x2 Strassen blocks, can we exploit
    # the structure of the 2x4 and 4x8 dimensions?
    
    # Key observation: A is 2x4, B is 4x8.
    # C = A*B: C[r,c] = sum_s A[r,s]*B[s,c]
    # We can think of this as: for each output column c of C,
    # C[:,c] = A * B[:,c] (matrix-vector product 2x4 * 4x1)
    # There are 8 such column computations.
    
    # Each column computation is a 2x4 * 4x1 = 2x1 matrix-vector product.
    # The rank of a single 2x4 * 4x1 multiplication tensor is at most 2*4=8 (schoolbook)
    # or less with Strassen-like methods.
    
    # But we have 8 columns, and they share the same A matrix.
    # This sharing is what allows rank reduction below 8*8=64.
    
    # Let's try a different approach: formulate as a linear algebra problem.
    # We want to find the minimum r such that there exist U (8×r), V (32×r), W (16×r)
    # with T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    # This is a hard NP-hard problem in general, but for small sizes we can search.
    
    # Try: start from rank 48 and see if we can find a valid decomposition.
    # Use random initialization + iterative refinement.
    
    np.random.seed(42)
    
    def try_rank(r, max_iters=500):
        """Try to find a rank-r decomposition using random init + ALS-like updates"""
        if time.time() > deadline - 10:
            return None
        
        # Random initialization
        U = np.random.randn(8, r)
        V = np.random.randn(32, r)
        W = np.random.randn(16, r)
        
        # Normalize to avoid scale issues
        U /= np.sqrt(np.sum(U**2, axis=0, keepdims=True))
        V /= np.sqrt(np.sum(V**2, axis=0, keepdims=True))
        W /= np.sqrt(np.sum(W**2, axis=0, keepdims=True))
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: fix V, W, solve for U
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: T[i,:,:] = sum_r U[i,r] * (V[:,r] ⊗ W[:,r])
            # Let B_r[j,k] = V[j,r]*W[k,r], then T[i,:,:] = sum_r U[i,r] * B_r
            # This is a linear system: T_i = U_i @ B where B is (r, 32*16)
            # Actually: T.reshape(8, 32*16) = U @ (V.reshape(32,r) * W.reshape(16,r).T).T
            # Let's define M[j*16+k, r] = V[j,r] * W[k,r]
            M = np.zeros((32*16, r))
            for r_idx in range(r):
                M[:, r_idx] = np.outer(V[:, r_idx], W[:, r_idx])
            # T_mat[i, j*16+k] = sum_r U[i,r] * M[j*16+k, r]
            # So T_mat = U @ M.T
            # Solve for U: U = T_mat @ M.T @ (M @ M.T)^{-1} (if M has full row rank)
            # Or use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M.T, T_mat, rcond=None)
            except:
                break
            
            # Update V: fix U, W
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed j: T[:,:,j] = sum_r V[j,r] * (U[:,r] ⊗ W[:,r])
            N = np.zeros((8*16, r))
            for r_idx in range(r):
                N[:, r_idx] = np.outer(U[:, r_idx], W[:, r_idx])
            # T.reshape(8*16, 32) -> T_reshaped[i*16+k, j] = sum_r V[j,r] * N[i*16+k, r]
            # So T_reshaped.T = V @ N.T
            T_reshaped = T.reshape(128, 32).T  # (32, 128)
            try:
                V_new, _, _, _ = np.linalg.lstsq(N.T, T_reshaped, rcond=None)
            except:
                break
            
            # Update W: fix U, V
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed k: T[:,:,k] = sum_r W[k,r] * (U[:,r] ⊗ V[:,r])
            P = np.zeros((8*32, r))
            for r_idx in range(r):
                P[:, r_idx] = np.outer(U[:, r_idx], V[:, r_idx])
            # T.reshape(8*32, 16) -> T_reshaped2[i*32+j, k] = sum_r W[k,r] * P[i*32+j, r]
            # So T_reshaped2.T = W @ P.T
            T_reshaped2 = T.reshape(256, 16).T  # (16, 256)
            try:
                W_new, _, _, _ = np.linalg.lstsq(P.T, T_reshaped2, rcond=None)
            except:
                break
            
            # Compute loss
            recon = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            loss = np.sum((recon - T)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if best_loss < 1e-10:
                break
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try various ranks
    for r in range(40, 57):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, max_iters=200)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # If no better found, use the Strassen rank-56
    if best_rank > 56:
        best_rank = 56
        best_U, best_V, best_W = U0, V0, W0
    
    # Round to half-integers if close
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Verify after rounding
    if not verify_fast(best_U, best_V, best_W):
        # Use original Strassen
        best_rank = 56
        best_U, best_V, best_W = U0, V0, W0
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
