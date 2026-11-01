"""Matrix multiplication 2,2,12 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 145.0  # safety margin
    
    # Schoolbook baseline: rank = n*m*p = 48
    U_sb = np.zeros((n*m, n*m*p), dtype=np.float64)
    V_sb = np.zeros((m*p, n*m*p), dtype=np.float64)
    W_sb = np.zeros((p*n, n*m*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m + j*p + k
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
    
    # The tensor T is defined by T[i,j,k] with indices:
    # U index: i*m+j (row i of A, column j)
    # V index: j*p+k (row j of B, column k)  
    # W index: k*n+i (output index)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We want to find rank r such that T = sum_{c=1}^r U[:,c] * V[:,c] * W[:,c]
    # i.e., T[a,b,c_idx] = sum_c U[a,c]*V[b,c]*W[c_idx,c]
    
    # For 2x2 x 2x12:
    # U has 4 rows (indices 0,1,2,3 for (i,j) in {0,1}x{0,1})
    # V has 24 rows (indices j*12+k for j in {0,1}, k in {0,...,11})
    # W has 24 rows (indices k*2+i for k in {0,...,11}, i in {0,1})
    
    # Key insight: The multiplication tensor for 2x2 x 2xp has a specific structure.
    # Let's think about it differently. We're computing C = A * B where:
    # A is 2x2, B is 2x12, C is 2x12
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor has entries T[i,j,k] = 1 for the multiplication pattern.
    # In the trace convention W[k*n+i], the output index is k*n+i.
    
    # For a 2x2 matrix times 2xp matrix, we can potentially do better than schoolbook.
    # The schoolbook rank is 4*12 = 48.
    
    # Let's try to find a lower rank decomposition.
    # The tensor T is a 4x24x24 tensor (or equivalently, we can reshape).
    
    # Actually, let's think about this more carefully.
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is nonzero only when the j index in a and b match, and the k in b and i in c are consistent.
    
    # For a=0 (i=0,j=0): b must be j*p+k = 0*12+k = k for k=0..11, c = k*n+i = k*2+0
    # For a=1 (i=0,j=1): b must be j*p+k = 1*12+k = 12+k for k=0..11, c = k*2+0
    # For a=2 (i=1,j=0): b must be j*p+k = 0*12+k = k for k=0..11, c = k*2+1
    # For a=3 (i=1,j=1): b must be j*p+k = 1*12+k = 12+k for k=0..11, c = k*2+1
    
    # So T[0, k, 2k] = 1 for k=0..11
    #    T[1, 12+k, 2k] = 1 for k=0..11
    #    T[2, k, 2k+1] = 1 for k=0..11
    #    T[3, 12+k, 2k+1] = 1 for k=0..11
    
    # We need to decompose this 4x24x24 tensor.
    
    # Strategy: Try rank r from lower values upward using numerical methods.
    # For each rank r, we set up the system and try to solve.
    
    # The decomposition is: T[a,b,c] = sum_{t=1}^r U[a,t] * V[b,t] * W[c,t]
    
    # This is a bilinear system in (U,V) for fixed W, or similar.
    # We can use alternating least squares or other approaches.
    
    # Let's try a direct approach: for small rank, enumerate or use optimization.
    
    # First, let's check if rank 24 works (p * n = 12*2 = 24, which is the "naive" output dimension)
    # Actually, the minimum rank for 2x2 x 2xp matrix multiplication...
    
    # Known results: For 2x2 x 2xp, the rank is at most 2p + 4 (using Strassen-like tricks)
    # But let's try to find the actual minimum.
    
    # Let's use a numerical approach: try ranks from 24 upward and check if a solution exists.
    
    # For a tensor T of size (I,J,K), rank r decomposition means:
    # T = sum_{t=1}^r u_t * v_t * w_t
    # where u_t in R^I, v_t in R^J, w_t in R^K
    
    # We can fix one set of vectors and solve for the others.
    # Let's try: fix W, then for each a, T[a,:,:] = sum_t U[a,t] * v_t * w_t^T
    # This is linear in U[a,t] and v_t given w_t.
    
    # Actually, let's use a simpler approach: since the tensor has a very specific structure,
    # let's try to construct a decomposition directly.
    
    # Observation: The tensor can be written as:
    # T = sum_{k=0}^{11} (e_{0,0} + e_{1,1}) * (e_{0,k} + e_{1,k}) * something
    # No, that's not quite right.
    
    # Let me be more careful:
    # T[0, k, 2k] = 1   (i=0,j=0)
    # T[1, 12+k, 2k] = 1   (i=0,j=1)  
    # T[2, k, 2k+1] = 1   (i=1,j=0)
    # T[3, 12+k, 2k+1] = 1   (i=1,j=1)
    
    # We can group by output index c = k*n+i:
    # For c = 2k (i=0, even): T[0,k,2k]=1, T[1,12+k,2k]=1
    # For c = 2k+1 (i=1, odd): T[2,k,2k+1]=1, T[3,12+k,2k+1]=1
    
    # This suggests that for each k, we need 2 "terms" in the sum, but they share structure.
    
    # Let's try rank r = 2p = 24:
    # For each k, have 2 terms:
    # Term 2k: U[0,2k]=1, U[2,2k]=1; V[k,2k]=1, V[12+k,2k]=1; W[2k,2k]=1, W[2k+1,2k]=1
    # Wait, that doesn't work because W[2k,2k]=1 and W[2k+1,2k]=1 would give cross terms.
    
    # Let me think differently. For each k, the "slice" is:
    # T[:, :, 2k] has nonzero entries at (0,k) and (1,12+k), both equal to 1.
    # T[:, :, 2k+1] has nonzero entries at (2,k) and (3,12+k), both equal to 1.
    
    # So for output index c=2k: the 4x24 slice is nonzero only at positions (0,k) and (1,12+k).
    # For output index c=2k+1: the 4x24 slice is nonzero only at positions (2,k) and (3,12+k).
    
    # This means each output "fiber" has rank 1 (as a matrix in the (a,b) indices).
    # So the tensor is a sum of 24 rank-1 terms? No, that would be rank 24 if they were independent.
    
    # Actually, the tensor rank is at most the sum of ranks of fibers, but can be less.
    
    # Let's try: rank 24 decomposition:
    # For each k=0..11:
    #   Term for output 2k: u = e_0 + e_1 (in U space, but only if V matches)
    #   Actually, let's just set:
    #   Term 2k: U[0,2k]=1, V[k,2k]=1, W[2k,2k]=1
    #            U[1,2k]=1, V[12+k,2k]=1, W[2k,2k]=1
    #   But this gives T[0,k,2k] += 1*1*1 = 1, T[1,12+k,2k] += 1*1*1 = 1
    #   But also T[0,12+k,2k] += 1*1*1 = 1 (WRONG!), T[1,k,2k] += 1*1*1 = 1 (WRONG!)
    
    # So we can't do that. We need the U and V vectors to be orthogonal in some sense.
    
    # Better approach: for each k, use 2 separate terms:
    #   Term A_k: U[0,A_k]=1, V[k,A_k]=1, W[2k,A_k]=1  -> gives T[0,k,2k]=1
    #   Term B_k: U[1,B_k]=1, V[12+k,B_k]=1, W[2k,B_k]=1  -> gives T[1,12+k,2k]=1
    #   Term C_k: U[2,C_k]=1, V[k,C_k]=1, W[2k+1,C_k]=1  -> gives T[2,k,2k+1]=1
    #   Term D_k: U[3,D_k]=1, V[12+k,D_k]=1, W[2k+1,D_k]=1  -> gives T[3,12+k,2k+1]=1
    # Total: 4*12 = 48 terms = schoolbook.
    
    # To do better, we need to share terms across k values.
    
    # Key insight: The V vectors for different k's are in different positions, so they're already orthogonal.
    # The constraint is that U and W must be consistent.
    
    # Let's try a different structure. Note that:
    # T[0,k,2k] = 1 for all k
    # T[2,k,2k+1] = 1 for all k
    # These both have V[k,:] nonzero. So if we use a term with V[k,t] != 0, 
    # it contributes to both T[0,k,2k] and T[2,k,2k+1] (and possibly others).
    
    # Suppose we have a term t with V[k,t] = 1 (for some k). Then:
    # Contribution to T[0,k,c] = U[0,t] * W[c,t]
    # Contribution to T[2,k,c] = U[2,t] * W[c,t]
    # We want T[0,k,2k] = 1 and T[2,k,2k+1] = 1, and all other T[*,k,c] = 0.
    
    # So we need: U[0,t]*W[2k,t] = 1, U[0,t]*W[c,t] = 0 for c != 2k
    #              U[2,t]*W[2k+1,t] = 1, U[2,t]*W[c,t] = 0 for c != 2k+1
    # This requires W[2k,t] and W[2k+1,t] to be nonzero, but W[c,t] = 0 for other c.
    # So W[:,t] has at most 2 nonzero entries.
    
    # If we let U[0,t] = a, U[2,t] = b, then W[2k,t] = 1/a, W[2k+1,t] = 1/b.
    # This works for a single term covering both (0,k,2k) and (2,k,2k+1).
    
    # Similarly, for V[12+k,t] != 0:
    # Contribution to T[1,12+k,c] = U[1,t] * W[c,t]
    # Contribution to T[3,12+k,c] = U[3,t] * W[c,t]
    # We want T[1,12+k,2k] = 1 and T[3,12+k,2k+1] = 1.
    # So U[1,t]*W[2k,t] = 1, U[3,t]*W[2k+1,t] = 1.
    
    # If we use the same term t for both V[k,t] and V[12+k,t], then:
    # W[2k,t] = 1/a = 1/U[0,t]
    # W[2k+1,t] = 1/b = 1/U[2,t]
    # Also: U[1,t]*W[2k,t] = 1 => U[1,t]/a = 1 => U[1,t] = a
    #        U[3,t]*W[2k+1,t] = 1 => U[3,t]/b = 1 => U[3,t] = b
    
    # So if U[0,t] = U[1,t] = a and U[2,t] = U[3,t] = b, and V[k,t] = V[12+k,t] = 1,
    # and W[2k,t] = 1/a, W[2k+1,t] = 1/b, this term gives:
    # T[0,k,2k] += a*1*(1/a) = 1
    # T[0,k,2k+1] += a*1*(1/b) = a/b  (should be 0!)
    # T[1,12+k,2k] += a*1*(1/a) = 1
    # T[1,12+k,2k+1] += a*1*(1/b) = a/b  (should be 0!)
    # T[2,k,2k] += b*1*(1/a) = b/a  (should be 0!)
    # T[2,k,2k+1] += b*1*(1/b) = 1
    # T[3,12+k,2k] += b*1*(1/a) = b/a  (should be 0!)
    # T[3,12+k,2k+1] += b*1*(1/b) = 1
    
    # So we get cross terms unless a/b = 0 or b/a = 0, which is impossible.
    
    # So we can't merge them that way. Let's try rank 36 (3 terms per k):
    # Term 1: U[0]=1, U[2]=0; V[k]=1; W[2k]=1  -> T[0,k,2k]=1
    # Term 2: U[1]=1, U[3]=0; V[12+k]=1; W[2k]=1  -> T[1,12+k,2k]=1
    # Term 3: U[0]=0, U[2]=1; V[k]=1; W[2k+1]=1  -> T[2,k,2k+1]=1
    # Term 4: U[1]=0, U[3]=1; V[12+k]=1; W[2k+1]=1  -> T[3,12+k,2k+1]=1
    # That's 4 terms per k = 48 total. Still schoolbook.
    
    # Let me try a numerical optimization approach for lower ranks.
    
    def check_decomposition(U, V, W, T, tol=1e-10):
        """Check if U @ V @ W^T approximates T."""
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # Reshape T to (I, J*K) and compare with (U*W) @ V^T or similar
        I, J, K = T.shape
        r = U.shape[1]
        # Compute T_hat[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # = (U * W)[a,t] summed with V[b,t]
        # T_hat = np.einsum('at,bt,ct->abc', U, V, W)
        # But this is expensive for large r. Let's do it in chunks.
        T_hat = np.zeros((I, J, K), dtype=np.float64)
        for t in range(r):
            T_hat += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
        return np.max(np.abs(T_hat - T))
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try rank r and see if we can find a decomposition
    # Use a randomized approach: initialize randomly, then use ALS (Alternating Least Squares)
    
    def try_rank(r, n_iter=50, n_restarts=3, seed=42):
        np.random.seed(seed)
        best_err = float('inf')
        best_UVW = None
        for restart in range(n_restarts):
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            for it in range(n_iter):
                if time.time() > deadline:
                    return None, float('inf')
                # Update U: for each a, T[a,:,:] = sum_t U[a,t] * (V*W^T)[:,t]
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # Fix V,W: T[a] = sum_t U[a,t] * (V[:,t] * W[:,t]^T)
                # T[a].reshape(J*K) = sum_t U[a,t] * (V[:,t] * W[:,t]^T).reshape(J*K)
                # So T_a = U[a] @ M where M[t, bk] = V[b,t]*W[k,t]
                M = np.einsum('bt,kt->bkt', V, W).reshape(m*p, p*n, r)
                # T[a] (shape J*K) = sum_t U[a,t] * M[:,:,t] (shape J*K)
                # So T_a = U[a] @ M_flat where M_flat[t, bk] = M[b,k,t]
                M_flat = M.transpose(2,0,1).reshape(r, m*p * p*n)
                T_flat = T.reshape(n*m, m*p * p*n)
                U = np.linalg.lstsq(M_flat, T_flat, rcond=None)[0].T  # shape (r, n*m) -> transpose to (n*m, r)
                
                # Update V: for each b, T[:,b,:] = sum_t V[b,t] * (U[:,t] * W[:,t]^T)
                M2 = np.einsum('at,kt->akt', U, W).reshape(n*m, p*n, r)
                M2_flat = M2.transpose(2,0,1).reshape(r, n*m * p*n)
                T2 = T.transpose(1,0,2).reshape(m*p, n*m * p*n)
                V = np.linalg.lstsq(M2_flat, T2, rcond=None)[0].T
                
                # Update W: for each c, T[:,:,c] = sum_t W[c,t] * (U[:,t] * V[:,t]^T)
                M3 = np.einsum('at,bt->abt', U, V).reshape(n*m, m*p, r)
                M3_flat = M3.transpose(2,0,1).reshape(r, n*m * m*p)
                T3 = T.transpose(2,0,1).reshape(p*n, n*m * m*p)
                W = np.linalg.lstsq(M3_flat, T3, rcond=None)[0].T
                
                err = check_decomposition(U, V, W, T)
                if err < best_err:
                    best_err = err
                    best_UVW = (U.copy(), V.copy(), W.copy())
                if err < 1e-8:
                    break
        if best_UVW is not None and best_err < 1e-8:
            return best_UVW, best_err
        return None, best_err
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: The tensor has rank at least max(I,J,K) = 24? No, that's not right.
    # The rank is at least the max of the ranks of any matrix unfolding.
    # The (I, J*K) unfolding has rank at most min(I, J*K) = 4.
    # So the tensor rank is at least 4, but that's very loose.
    
    # For 2x2 x 2xp matrix multiplication, known results suggest rank around 3p-2 or similar.
    # Let's try ranks from 24 to 48.
    
    result = None
    for r in range(24, 49):
        if time.time() > deadline:
            break
        UVW, err = try_rank(r, n_iter=30, n_restarts=2, seed=r)
        if UVW is not None and err < 1e-8:
            result = (r, UVW, err)
            break
        if r % 4 == 0:
            print(f"Rank {r}: best err = {err:.2e}")
    
    if result is not None:
        r, (U, V, W), err = result
        # Convert to lists for the return
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
