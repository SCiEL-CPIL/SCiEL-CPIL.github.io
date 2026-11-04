"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=object)
    V = np.zeros((m * p, schoolbook_rank), dtype=object)
    W = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1

    def tensor_value(U_mat, V_mat, W_mat, rank):
        """Compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    T[(i,j,k)] = s
        return T

    def verify(U_mat, V_mat, W_mat, rank):
        T = tensor_value(U_mat, V_mat, W_mat, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[(i,j,k)] != Fraction(1):
                        return False
        return True

    def verify_float(U_mat, V_mat, W_mat, rank, tol=1e-9):
        """Fast float verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Verify schoolbook
    if not verify(U, V, W, schoolbook_rank):
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}

    # === Winograd-style structured decomposition ===
    # Map 3x3x4 matmul to convolutional structure.
    # Key idea: use polynomial interpolation to reduce rank.
    # 
    # For each output element C[i,k] = sum_j A[i,j]*B[j,k],
    # we can group terms and use Winograd transforms.
    #
    # Winograd F(4,3,3): multiply 4x3 by 3x3 using 4 mults per row of output.
    # But for 3x3x4, we need a different partition.
    #
    # Alternative: use the fact that the tensor is a sum of 12 rank-1 terms (schoolbook),
    # and look for a rank-r decomposition with r < 36 using structured patterns.
    #
    # Strategy: Build a candidate decomposition with rank R using Winograd-like
    # structured entries, then use local search (column merging/splitting) to minimize R.

    def build_winograd_candidate():
        """Build a structured rank-reduced decomposition using Winograd ideas."""
        # Use the identity: for computing dot products of length 3,
        # Winograd F(4,3,3) uses 4 multiplications instead of 3 per pair,
        # but shares across multiple outputs.
        #
        # For 3x3x4: C (3x4) = A (3x3) * B (3x4)
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # Winograd approach: treat each row of A and each column of B as vectors.
        # For row i of A: a_i = (A[i,0], A[i,1], A[i,2])
        # For column k of B: b_k = (B[0,k], B[1,k], B[2,k])
        # C[i,k] = a_i . b_k
        #
        # We can use the Winograd transform for length-3 vectors:
        # Transform matrices (from Winograd F(4,3,3)):
        # G (for A, 4x3): maps 3-vector to 4-vector
        # F^T (for B, 3x4): maps 4-vector to 3-vector
        #
        # G = [[1,0,0], [1,1,0], [0,1,1], [0,0,1]]
        # F = [[1,0,0,0], [0,1,1,0], [0,0,1,1]]
        #
        # Actually, for F(4,3,3):
        # G = [[1,0,0], [1,1,0], [0,1,1], [0,0,1]]  (4x3)
        # F^T = [[1,0,0,0], [0,1,1,0], [0,0,1,1]]  (3x4)
        #
        # This gives: for each (i,k) pair, 4 multiplications.
        # Total: 3*4*4 = 48? That's worse.
        #
        # The key is SHARING: the transformed A values are shared across k,
        # and transformed B values are shared across i.
        #
        # Let me use a different approach: direct structured rank reduction.
        #
        # Try: group the 12 dot products into groups and use the identity
        # (x+y)(z+w) = xz + xw + yz + yw to merge 4 terms into 4 mults
        # but with shared partial products.
        
        # Let's try a concrete construction with rank 24:
        # Partition the 12 terms into 3 groups of 4.
        # Group 1: terms where i=0 (C[0,0], C[0,1], C[0,2], C[0,3])
        #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        # Group 2: terms where i=1
        # Group 3: terms where i=2
        
        # For each group, use Winograd F(4,3,3):
        # Transform A row: g_i = G * a_i (4 values)
        # Transform B columns: for each k, f_k = F^T * b_k (3 values)
        # But this doesn't directly help with sharing.
        
        # Alternative: use the "Kronecker product" structure.
        # The matmul tensor T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}
        # This is the standard matrix multiplication tensor.
        
        # Let's try a randomized local search approach with good initialization.
        return None

    # === Local search approach ===
    # Start with schoolbook (rank 36), try to reduce rank by:
    # 1. Merging pairs of columns that have "compatible" tensor contributions
    # 2. Splitting columns to enable further merges
    # 3. Random perturbations + exact verification

    np.random.seed(42)
    
    def column_contribution(U_f, V_f, W_f, r):
        """Tensor contribution of column r as flat vector (float)"""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    vec[idx] = U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
        return vec

    def try_merge_columns(U_f, V_f, W_f, rank, r1, r2):
        """Try to replace columns r1 and r2 with a single column.
        Need: T_r1 + T_r2 = T_new where T_new is a rank-1 tensor.
        This means there exist vectors u, v, w such that:
        u[i*m+j] * v[j*p+k] * w[k*n+i] = T_r1[i,j,k] + T_r2[i,j,k] for all i,j,k
        This is generally not possible, but we can try to find approximate solutions.
        """
        # Compute target tensor
        target = column_contribution(U_f, V_f, W_f, r1) + column_contribution(U_f, V_f, W_f, r2)
        
        # Try to factorize target as u ⊗ v ⊗ w (rank-1 tensor)
        # Reshape to 3D
        T3 = target.reshape(n, m, p)
        
        # Try CP decomposition of rank 1: use ALS or SVD-based approach
        # For rank-1: maximize <T, u⊗v⊗w> over unit vectors
        # Use power iteration
        u = np.ones(n) / np.sqrt(n)
        v = np.ones(m) / np.sqrt(m)
        w = np.ones(p) / np.sqrt(p)
        
        for _ in range(200):
            # Update u: u_new[i] = sum_{j,k} T3[i,j,k] * v[j] * w[k]
            u_new = np.zeros(n)
            for i in range(n):
                u_new[i] = np.sum(T3[i,:,:] * np.outer(v, w))
            if np.linalg.norm(u_new) > 1e-15:
                u = u_new / np.linalg.norm(u_new)
            
            # Update v
            v_new = np.zeros(m)
            for j in range(m):
                v_new[j] = np.sum(T3[:,j,:] * np.outer(u, w))
            if np.linalg.norm(v_new) > 1e-15:
                v = v_new / np.linalg.norm(v_new)
            
            # Update w
            w_new = np.zeros(p)
            for k in range(p):
                w_new[k] = np.sum(T3[:,:,k] * np.outer(u, v))
            if np.linalg.norm(w_new) > 1e-15:
                w = w_new / np.linalg.norm(w_new)
        
        # Check if u⊗v⊗w approximates T3
        approx = np.outer(np.outer(u, v), w).reshape(n, m, p)
        err = np.linalg.norm(T3 - approx) / max(1, np.linalg.norm(T3))
        
        if err < 1e-8:
            # Found exact rank-1 factorization!
            # Scale: u⊗v⊗w should equal T3
            # Set new column to (u, v, w) scaled appropriately
            # We need: sum over the new column = target
            # The new column contributes u[i*m+j] * v[j*p+k] * w[k*n+i]
            # So we set U_new[i*m+j] = u[i], V_new[j*p+k] = v[j], W_new[k*n+i] = w[k]
            U_new = np.zeros(n * m)
            V_new = np.zeros(m * p)
            W_new = np.zeros(p * n)
            for i in range(n):
                for j in range(m):
                    U_new[i*m+j] = u[i]
            for j in range(m):
                for k in range(p):
                    V_new[j*p+k] = v[j]
            for k in range(p):
                for i in range(n):
                    W_new[k*n+i] = w[k]
            
            # Verify
            check = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        check[(i*m+j)*p+k] = U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i]
            if np.allclose(check, target, atol=1e-10):
                return U_new, V_new, W_new, True
        return None, None, None, False

    def local_search(U_f, V_f, W_f, rank, max_iters=5000):
        """Local search to reduce rank"""
        best_rank = rank
        best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try all pairs of columns for merging
            for r1 in range(rank):
                if improved or time.time() > deadline - 10:
                    break
                for r2 in range(r1 + 1, rank):
                    if time.time() > deadline - 10:
                        break
                    
                    U_new, V_new, W_new, success = try_merge_columns(U_f, V_f, W_f, rank, r1, r2)
                    if success:
                        # Replace columns r1 and r2 with new column
                        # Remove r2 first (if r2 > r1), then r1
                        if r2 > r1:
                            U_f = np.delete(U_f, r2, axis=1)
                            V_f = np.delete(V_f, r2, axis=1)
                            W_f = np.delete(W_f, r2, axis=1)
                            r2 = r1  # r1 is still at position r1
                        # Now replace column r1 with new column
                        U_f[:, r1] = U_new
                        V_f[:, r1] = V_new
                        W_f[:, r1] = W_new
                        rank -= 1
                        best_rank = rank
                        best_U, best_V, best_W = U_f.copy(), V_f.copy(), W_f.copy()
                        improved = True
                        break
        
            if not improved:
                # Try random perturbation to escape local optimum
                if time.time() > deadline - 15:
                    break
                # Pick a random column and perturb it
                r = np.random.randint(rank)
                scale = 0.1
                U_f[:, r] += np.random.randn(n * m) * scale
                V_f[:, r] += np.random.randn(m * p) * scale
                W_f[:, r] += np.random.randn(p * n) * scale
                
                # Verify still valid (approximately)
                if not verify_float(U_f, V_f, W_f, rank, tol=0.01):
                    # Revert
                    U_f[:, r] -= np.random.randn(n * m) * scale  # This won't work properly
                    # Better: save and restore
                    U_f[:, r] = best_U[:, r] if r < best_U.shape[1] else 0
                    V_f[:, r] = best_V[:, r] if r < best_V.shape[1] else 0
                    W_f[:, r] = best_W[:, r] if r < best_W.shape[1] else 0
                    break
        
        return best_U, best_V, best_W, best_rank

    # Start with schoolbook
    U_f = U.astype(float)
    V_f = V.astype(float)
    W_f = W.astype(float)
    current_rank = schoolbook_rank

    # Run local search
    U_f, V_f, W_f, current_rank = local_search(U_f, V_f, W_f, current_rank, max_iters=3000)

    # Verify final result exactly
    U_obj = U_f.astype(object)
    V_obj = V_f.astype(object)
    W_obj = W_f.astype(object)
    
    if verify(U_obj, V_obj, W_obj, current_rank):
        return {"U": [[float(x) for x in row] for row in U_obj], 
                "V": [[float(x) for x in row] for row in V_obj], 
                "W": [[float(x) for x in row] for row in W_obj]}
    else:
        # Fall back to schoolbook
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
# EVOLVE-BLOCK-END
