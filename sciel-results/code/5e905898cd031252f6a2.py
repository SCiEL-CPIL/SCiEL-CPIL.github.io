import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Build the multiplication tensor T[i,j,k] = delta[i,j,k] where
    # C[i,k] = sum_j A[i,j]*B[j,k], so T has T[i,j,k]=1 iff valid
    # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k) in support
    # Actually the tensor is: T[a,b,c] = 1 if a<b and b<c (indices into flattened A,B,C)
    # Let's use the standard: T[i,j,k] = 1 where i is (row of A, col of A) = (i1,i2)
    # j is (row of B, col of B) = (j1,j2), k is (row of C, col of C) = (k1,k2)
    # T = 1 iff i2==j1 and j2==k1 and i1==k2... no.
    # Standard: C[i1,k2] = sum_j1 A[i1,j1]*B[j1,k2]
    # So tensor T[i1,j1,k2, j1... ] hmm let me think more carefully.
    
    # The matmul tensor: T[i,j,k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    # T[i,j,k] = 1 if and only if the (i,j) pair contributes to k
    # i = i1*m + i2 (row i1, col i2 of A)
    # j = j1*p + j2 (row j1, col j2 of B)  
    # k = k1*n + k2 (row k1, col k2 of C)  -- WAIT, the convention is W[k*n+i,r]
    # So k indexes C with k = k1*n + k2 where k1 is col, k2 is row? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed as W[k*n+i, r] where k is column index and i is row index of C
    # So for C[i,k] (row i, col k), the index into W is k*n + i
    
    # Let me rebuild: 
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U index: i*m + j (row i, col j of A)
    # V index: j*p + k (row j, col k of B)
    # W index: k*n + i (col k, row i of C) -- per trace convention
    
    # Tensor T has support where T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # i in [0,n), j in [0,m), k in [0,p)
    
    # For n=2, m=4, p=4:
    # U has 2*4=8 rows, V has 4*4=16 rows, W has 4*2=8 rows
    # Number of non-zero entries in T: n*m*p = 2*4*4 = 32
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i*m + j
                vj = j*p + k
                wk = k*n + i
                T[ui, vj, wk] = 1.0
    
    # Now try MPS/TNT approach:
    # Reshape T into a matrix by grouping modes
    # Mode 1 (size 8), Mode 2 (size 16), Mode 3 (size 8)
    
    # Try to find a low-rank CP decomposition
    # First, let's get the MPS structure via SVD of unfoldings
    
    # Unfold T as (n*m) x (m*p * p*n) matrix
    M1 = T.reshape(n*m, m*p * p*n)
    U1, S1, Vt1 = np.linalg.svd(M1, full_matrices=False)
    
    # Unfold T as (m*p) x (n*m * p*n) matrix  
    M2 = T.reshape(m*p, n*m * p*n)
    U2, S2, Vt2 = np.linalg.svd(M2, full_matrices=False)
    
    # Unfold T as (p*n) x (n*m * m*p) matrix
    M3 = T.reshape(p*n, n*m * m*p)
    U3, S3, Vt3 = np.linalg.svd(M3, full_matrices=False)
    
    # The tensor has rank structure. Let's try a direct CP search.
    # Start with the Strassen rank-28 solution and try to reduce.
    
    # Build Strassen rank-28 as fallback
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=float)
    V28 = np.zeros((m * p, rank_28), dtype=float)
    W28 = np.zeros((p * n, rank_28), dtype=float)
    
    def strassen_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0, U28, V28, W28)
    apply_strassen(1, 1, 0, 0, 7, U28, V28, W28)
    apply_strassen(0, 0, 1, 1, 14, U28, V28, W28)
    apply_strassen(1, 1, 1, 1, 21, U28, V28, W28)
    
    # Verify Strassen solution
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i*m + j
                    vj = j*p + k
                    wk = k*n + i
                    acc = 0.0
                    for r in range(rank):
                        acc += U[ui, r] * V[vj, r] * W[wk, r]
                    if abs(acc - 1.0) > 1e-9:
                        return False
        # Also verify non-support entries are 0
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    ui, vj, wk = i, j, k
                    # Check if this is a support entry
                    i1, i2 = divmod(i, m)
                    j1, j2 = divmod(j, p)
                    k1, k2 = divmod(k, n)
                    if i2 == j1 and j2 == k1 and i1 == k2:
                        continue  # support entry, already checked
                    acc = 0.0
                    for r in range(rank):
                        acc += U[i, r] * V[j, r] * W[k, r]
                    if abs(acc) > 1e-9:
                        return False
        return True
    
    if not verify(U28, V28, W28, 28):
        # Return as-is if verification fails (shouldn't happen)
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Now try to find rank 27 by perturbing the rank-28 solution
    # and using ALS (Alternating Least Squares) for CP decomposition
    
    # Strategy: try to merge two Strassen terms or find a linear dependency
    # The key insight: in the 2x4x4 case, there might be a linear relation
    # among the 28 rank-1 terms that allows merging.
    
    # Try: remove one column and solve for the remaining 27 to still work
    # This is a system of 32*8*16 = 4096 equations (but only 32 are non-zero in support)
    # Actually we have constraints: for each (i,j,k) in support, sum_r U[i,r]V[j,r]W[k,r] = 1
    # and for each (i,j,k) not in support, sum_r U[i,r]V[j,r]W[k,r] = 0
    
    # Let's try a different approach: use the MPS structure to find a compressed representation
    # The tensor T can be written as T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # If we fix mode 1, T[:,j,k] = U^T * (V[j,:] ⊙ W[k,:])
    
    # Try rank 27 via optimization: start from rank 28, zero out one column, 
    # and re-solve the remaining factors
    
    best_U, best_V, best_W, best_rank = U28, V28, W28, 28
    
    # Try removing each column and solving
    np.random.seed(42)
    
    for col_to_remove in range(28):
        if time.time() > deadline:
            break
            
        # Remove this column
        mask = [r for r in range(28) if r != col_to_remove]
        U_try = U28[:, mask].copy()
        V_try = V28[:, mask].copy()
        W_try = W28[:, mask].copy()
        
        # Now we have 27 terms but they won't satisfy the tensor exactly
        # Try to fix them via ALS iterations
        # The constraint is: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all (i,j,k)
        
        # ALS: fix two factors, solve for the third
        # If U and W are fixed, V[j,:] should satisfy:
        # For each j: sum_k W[k,:]^T * (U * T[:,j,k]) = V[j,:] * sum_k W[k,:] ... 
        # This is nonlinear. Let's use a linearized approach.
        
        # Actually, let's try a simpler approach: perturb and use gradient descent
        # on the smooth objective: sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r] - T[i,j,k])^2
        
        # Initialize with the perturbed solution
        U_cur = U_try.copy()
        V_cur = V_try.copy()
        W_cur = W_try.copy()
        
        # Add small random perturbation to break symmetry
        U_cur += 0.01 * np.random.randn(n*m, 27)
        V_cur += 0.01 * np.random.randn(m*p, 27)
        W_cur += 0.01 * np.random.randn(p*n, 27)
        
        # ALS iterations
        lr = 0.1
        for iteration in range(200):
            if time.time() > deadline:
                break
                
            # Update U: for each i, U[i,:] should minimize sum_{j,k} (U[i,:]·(V[j,:]⊙W[k,:]) - T[i,j,k])^2
            # This is: U[i,:] = argmin sum_{j,k} (U[i,:]·(V[j,:]⊙W[k,:]) - T[i,j,k])^2
            # Gradient: sum_{j,k} (U[i,:]·(V[j,:]⊙W[k,:]) - T[i,j,k]) * (V[j,:]⊙W[k,:])
            # Hessian approximation: sum_{j,k} (V[j,:]⊙W[k,:])^T (V[j,:]⊙W[k,:])
            
            # For efficiency, precompute
            # U[i,:] update: minimize ||M_i - T_i||^2 where M_i = U[i,:] @ diag(V[j,:]) @ W[k,:]... 
            # Actually: (U V W)[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Fix V and W: for each i, we want U[i,:] such that
            # sum_r U[i,r] * (sum_j V[j,r] * (sum_k W[k,r] * ... )) hmm this is getting complex.
            
            # Let's just do a simple gradient step
            # Compute current tensor
            # T_current[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            
            # For U update (fix V, W):
            # d/dU[i,a] sum_{j,k} (sum_r U[i,r]V[j,r]W[k,r] - T[i,j,k])^2
            # = 2 * sum_{j,k} (sum_r U[i,r]V[j,r]W[k,r] - T[i,j,k]) * V[j,a]*W[k,a]
            
            # This is expensive to compute naively. Let's use the fact that
            # sum_{j,k} V[j,a]*W[k,a]*(...) = V[:,a] @ (something)
            
            # Actually let's just do a direct computation for small sizes
            # n*m=8, m*p=16, p*n=8, rank=27
            # 8*16*8 = 1024 terms per factor update, 27 rank -> manageable
            
            # Compute residuals
            # R[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] - T[i,j,k]
            
            # For speed, compute using einsum-like operations
            # R = U @ (V^T ⊙ W^T) ... not quite.
            # R[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # This is 8*16*8*27 = 27648 multiplications - very fast
            
            R = np.einsum('ir,jr,kr->ijk', U_cur, V_cur, W_cur) - T
            R_flat = R.ravel()
            
            # Gradient for U: G_U[i,a] = sum_{j,k} R[i,j,k]*V[j,a]*W[k,a]
            # = np.einsum('ijk,ja,ka->ia', R, V_cur, W_cur)
            G_U = np.einsum('ijk,ja,ka->ia', R, V_cur, W_cur)
            
            # Gradient for V: G_V[j,a] = sum_{i,k} R[i,j,k]*U[i,a]*W[k,a]
            G_V = np.einsum('ijk,ia,ka->ja', R, U_cur, W_cur)
            
            # Gradient for W: G_W[k,a] = sum_{i,j} R[i,j,k]*U[i,a]*V[j,a]
            G_W = np.einsum('ijk,ia,ja->ka', R, U_cur, V_cur)
            
            # Gradient descent step
            U_cur -= lr * G_U
            V_cur -= lr * G_V
            W_cur -= lr * G_W
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-10:
                break
        
        # Check if we got a valid solution
        R_final = np.einsum('ir,jr,kr->ijk', U_cur, V_cur, W_cur) - T
        if np.max(np.abs(R_final)) < 1e-6:
            # We have a rank-27 solution!
            # Try to round to integers/half-integers
            U_rounded = np.round(U_cur * 2) / 2
            V_rounded = np.round(V_cur * 2) / 2
            W_rounded = np.round(W_cur * 2) / 2
            
            R_rounded = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded) - T
            if np.max(np.abs(R_rounded)) < 1e-6:
                best_U, best_V, best_W, best_rank = U_rounded, V_rounded, W_rounded, 27
                break
            else:
                # Keep the floating point solution
                best_U, best_V, best_W, best_rank = U_cur, V_cur, W_cur, 27
                break
    
    # If we didn't find rank 27, try rank 26 with more aggressive search
    if best_rank == 28 and time.time() < deadline - 30:
        # Try a fresh random initialization for rank 26
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            rank_try = 26
            U_try = np.random.randn(n*m, rank_try) * 0.5
            V_try = np.random.randn(m*p, rank_try) * 0.5
            W_try = np.random.randn(p*n, rank_try) * 0.5
            
            # Use the SVD structure to get a better initialization
            # Project onto the low-rank structure
            # The tensor has a specific structure - use the MPS bonds
            
            lr = 0.05
            for iteration in range(500):
                if time.time() > deadline:
                    break
                    
                R = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
                G_U = np.einsum('ijk,ja,ka->ia', R, V_try, W_try)
                G_V = np.einsum('ijk,ia,ka->ja', R, U_try, W_try)
                G_W = np.einsum('ijk,ia,ja->ka', R, U_try, V_try)
                
                U_try -= lr * G_U
                V_try -= lr * G_V
                W_try -= lr * G_W
                
                if np.max(np.abs(R)) < 1e-10:
                    break
            
            R_final = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try) - T
            if np.max(np.abs(R_final)) < 1e-6:
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, 26
                break
    
    # Convert to lists
    U_out = np.array(best_U, dtype=float)
    V_out = np.array(best_V, dtype=float)
    W_out = np.array(best_W, dtype=float)
    
    # Final verification
    R_check = np.einsum('ir,jr,kr->ijk', U_out, V_out, W_out) - T
    if np.max(np.abs(R_check)) > 1e-6:
        # Fallback to Strassen
        U_out = U28
        V_out = V28
        W_out = W28
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
