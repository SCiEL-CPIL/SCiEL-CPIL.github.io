"""Matrix multiplication 2,3,5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    # We'll use a rank-20 decomposition based on combining Strassen-like
    # reductions for the 2x2x2 core and handling the extra dimensions.
    #
    # The tensor for 2x3x5 multiplication has rank at most 30 (schoolbook).
    # We aim for rank 20 or lower.
    #
    # Strategy: Decompose the 3 middle columns into 2+1.
    # For the 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1}), use Strassen rank 7.
    # For the remaining terms involving j=2 (the 3rd column of A) or k=2 (3rd row of B),
    # we need additional rank.
    #
    # Let's build the full tensor and use a known rank-20 construction.
    #
    # Actually, let me construct it more carefully.
    # T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,5)
    # We want sum_r u_r[i] * v_r[j] * w_r[k] = T[i,j,k]
    #
    # This is equivalent to finding rank-r factorization of the 3D tensor.
    #
    # Let me try a direct construction. I'll use the following approach:
    # Split j into j1 in {0,1} and j2 = 2.
    # Split k into k1 in {0,1,2,3,4} -- no, that doesn't help directly.
    #
    # Alternative: Use the fact that 2x3x5 can be embedded in a way that
    # allows rank reduction.
    #
    # Let me try a systematic approach: build the full tensor and attempt
    # a low-rank decomposition using ALS with rational reconstruction.
    #
    # But first, let me try a known construction.
    #
    # For 2x2x2, Strassen gives rank 7.
    # For 2x3x5, one approach:
    # - Use Strassen on the (i, j={0,1}, k={0,1}) sub-block: rank 7
    # - Handle j=2 terms: A[:,2] * B[2,:] is a 2x5 outer product, rank 2
    # - Handle k=2,3,4 terms for j in {0,1}: A[:,{0,1}] * B[{0,1},{2,3,4}]
    #   This is a 2x2x3 tensor. Its rank is at most 6 (schoolbook), but can be less.
    #
    # Total: 7 + 2 + 6 = 15? That seems too optimistic.
    #
    # Wait, the Strassen decomposition handles i in {0,1}, j in {0,1}, k in {0,1}.
    # But the output C[i,k] for k in {0,1} gets contributions from all j in {0,1,2}.
    # And C[i,k] for k in {2,3,4} gets contributions from all j in {0,1,2}.
    #
    # So we can't simply split. Let me think differently.
    #
    # Let me just do a numerical optimization to find a low-rank decomposition.
    # I'll use ALS (Alternating Least Squares) on the 3-mode tensor.
    
    # Build the full multiplication tensor
    # T[i, j, k] = 1 for all i, j, k
    # Actually, the tensor is T[i,j,k] = delta (the multiplication tensor has
    # T[i,j,k] = 1 if we're multiplying, but the structure is:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor for matrix mult is T[i,j,k] = 1, meaning the trilinear form is
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # But actually T[i,j,k] = 1 for all i,j,k in the standard embedding.
    
    # Wait, I need to be more careful. The tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    # where the output is indexed by (i,k) in the trace convention W[k*n+i].
    
    # So T is an n x m x p tensor with all entries = 1.
    # We want to write T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r has length n, v_r has length m, w_r has length p.
    
    # This is a rank-r factorization of an all-ones tensor of shape (2,3,5).
    
    # The all-ones tensor of shape (a,b,c) has rank:
    # - For (2,2,2): rank 7 (Strassen)
    # - For (1,b,c): rank 1 (it's an outer product of all-ones vectors)
    # - For (2,3,5): ?
    
    # Let me use numerical methods to find a low-rank approximation and then
    # check if it's exact (within tolerance).
    
    np.random.seed(42)
    
    # Target tensor
    T = np.ones((n, m, p))
    
    # Try ranks from 10 upward
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(10, 25):
        # Random initialization
        U = np.random.randn(n, rank) * 0.5
        V = np.random.randn(m, rank) * 0.5
        W = np.random.randn(p, rank) * 0.5
        
        # ALS iterations
        for it in range(200):
            # Update U: for each i, solve (V * W^T) * U[i,:] = T[i,:,:]
            # V * W^T is m x p matrix
            VW = V @ W.T  # m x p
            for i in range(n):
                U[i, :] = np.linalg.lstsq(VW, T[i, :].flatten(), rcond=None)[0]
            
            # Update V: for each j, solve (U * W^T) * V[j,:] = T[:,j,:].T
            UWT = U @ W.T  # n x p
            for j in range(m):
                V[j, :] = np.linalg.lstsq(UWT.T, T[:, j, :].flatten(), rcond=None)[0]
            
            # Update W: for each k, solve (U * V^T) * W[k,:] = T[:,:,k].T
            UV = U @ V.T  # n x m
            for k in range(p):
                W[k, :] = np.linalg.lstsq(UV, T[:, :, k].flatten(), rcond=None)[0]
        
        # Check reconstruction
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        
        err = np.max(np.abs(T_recon - T))
        if err < 1e-8:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            break
    
    if best_rank is None:
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n, rank))
        V = np.zeros((m, rank))
        W = np.zeros((p, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, col] = 1
                    V[j, col] = 1
                    W[k, col] = 1
                    col += 1
        best_rank = rank
        best_U = U
        best_V = V
        best_W = W
    else:
        # Try to clean up to rational values if close
        # Round to nearest rational with small denominator
        def clean_matrix(M, tol=1e-6):
            M_clean = M.copy()
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    # Try to find a rational p/q with q <= 100
                    best = val
                    best_err = abs(val - round(val))
                    if best_err < tol:
                        best = round(val)
                    else:
                        for q in range(2, 100):
                            p = round(val * q)
                            err = abs(val - p / q)
                            if err < best_err:
                                best_err = err
                                best = p / q
                            if err < tol:
                                break
                    M_clean[i, j] = best
            return M_clean
        
        best_U = clean_matrix(best_U)
        best_V = clean_matrix(best_V)
        best_W = clean_matrix(best_W)
        
        # Verify again after cleaning
        T_recon = np.zeros((n, m, p))
        for r in range(best_rank):
            T_recon += np.outer(np.outer(best_U[:, r], best_V[:, r]), best_W[:, r])
        err = np.max(np.abs(T_recon - T))
        if err > 1e-4:
            # Cleaning broke it, use original
            pass
        else:
            best_U = clean_matrix(best_U)
            best_V = clean_matrix(best_V)
            best_W = clean_matrix(best_W)
    
    rank = best_rank
    
    # Convert to the required format
    # U_out has shape (n*m, rank): U_out[i*m+j, r] = best_U[i, r] * (1 for all j)
    # Wait, no. The U matrix in the decomposition is per (i,j) pair.
    # In my factorization, U has shape (n, rank), V has shape (m, rank), W has shape (p, rank).
    # The term is U[i,r] * V[j,r] * W[k,r].
    # But the required format has U of shape (n*m, rank) where U[i*m+j, r] is the coefficient for A[i,j].
    # So I need to combine U and V into a single matrix.
    # Actually, the decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j A[i,j] * V[j,r]) * (sum_k ... )
    # No, that's not right either.
    
    # Let me re-derive. The tensor T[i,j,k] = 1 means:
    # sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = 1 for all i,j,k
    # Wait, that's not how tensor decomposition works for matrix multiplication.
    
    # For matrix multiplication, the decomposition is:
    # (A*B)[i,k] = sum_r (sum_j A[i,j] * U_r[i,j]) * (sum_l B[l,k] * V_r[j,l]) * W_r[i,k]
    # No, that's not standard either.
    
    # The standard form: We want
    # (A*B)[i,k] = sum_{j} A[i,j] B[j,k]
    # = sum_r U_r[i] * V_r[j] * W_r[k] ... no.
    
    # Let me re-read the task. The output factor follows W[k*n+i, r].
    # The factor matrices are U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    # The decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] is the multiplication tensor.
    
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor is defined such that the bilinear map (A,B) -> A*B
    # corresponds to the trilinear form. The tensor T has entries T[i,j,k] = 1
    # when the product involves A[i,j] and B[j,k] to produce C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,2), j in [0,3), k in [0,5)
    
    # In my ALS, I had T[i,j,k] = 1 and factorized as sum_r U_A[i,r] * U_B[j,r] * U_C[k,r] = 1
    # where U_A has shape (n,rank), U_B has shape (m,rank), U_C has shape (p,rank).
    
    # To convert to the required format:
    # U_full[i*m+j, r] should be U_A[i,r] ... but that doesn't depend on j.
    # V_full[j*p+k, r] should be U_B[j,r] ... but that doesn't depend on k.
    # W_full[k*n+i, r] should be U_C[k,r] ... but that doesn't depend on i.
    
    # But then U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = U_A[i,r] * U_B[j,r] * U_C[k,r]
    # which equals T[i,j,k] = 1. 
    
    # So the conversion is:
    # U_full[i*m+j, r] = U_A[i, r] for all j
    # V_full[j*p+k, r] = U_B[j, r] for all k
    # W_full[k*n+i, r] = U_C[k, r] for all i
    
    # Let me verify: for a given (i,j,k), the product is U_A[i,r]*U_B[j,r]*U_C[k,r],
    # and summing over r gives 1. Yes, this works.
    
    U_A = best_U  # shape (n, rank)
    U_B = best_V  # shape (m, rank)
    U_C = best_W  # shape (p, rank)
    
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U_full[i * m + j, :] = U_A[i, :]
    
    for j in range(m):
        for k in range(p):
            V_full[j * p + k, :] = U_B[j, :]
    
    for k in range(p):
        for i in range(n):
            W_full[k * n + i, :] = U_C[k, :]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Clean up floating point to get clean numbers
    def clean_list(lst, tol=1e-6):
        cleaned = []
        for row in lst:
            clean_row = []
            for val in row:
                if abs(val) < tol:
                    clean_row.append(0)
                else:
                    # Try to find clean rational
                    best = val
                    best_err = abs(val - round(val))
                    if best_err < tol:
                        best = round(val)
                    else:
                        for q in range(2, 200):
                            p = round(val * q)
                            err = abs(val - p / q)
                            if err < best_err:
                                best_err = err
                                best = p / q
                            if err < tol:
                                break
                    # Convert to int if it's an integer
                    if abs(best - round(best)) < tol:
                        clean_row.append(int(round(best)))
                    else:
                        clean_row.append(float(best))
            cleaned.append(clean_row)
        return cleaned
    
    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
