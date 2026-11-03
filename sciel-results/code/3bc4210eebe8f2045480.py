import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    # Start from a deadline well before the 150s wall-clock limit.
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 3, 3, 4

    # Build the matrix multiplication tensor T_{i,j,k} = delta(i,j) for C[i,k] = sum_j A[i,j]B[j,k]
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if we are computing
    # C[i,k] = sum_j A[i,j]*B[j,k] ... actually the tensor is:
    # T_{(i,j),(j,k),(k,i)} ... no.
    # The tensor for matrix mult is: for each (i,j,k), the coefficient is 1 if we're forming C[i,k] from A[i,j]*B[j,k].
    # So the tensor T has indices (i,j,k) with T[i,j,k] = 1 for all i,j,k? No.
    # The bilinear map is: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: T[i,j,k] = 1 for the "matching" structure. Specifically, the tensor is defined such that
    # <T, A ⊗ B> = C where (A ⊗ B)[i,j,k] = A[i,j]*B[j,k] and we contract over j.
    # Actually the tensor entries are: T[i,j,k] = δ_{j,j} ... it's just 1 for all (i,j,k) in the sense that
    # the structure is: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k].
    # So the "tensor" is: T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no. The tensor is more subtle. Let me think again.
    #
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor of dimensions (n*m) × (m*p) × (p*n)
    # Wait, that's not right either. Let me re-read the task.
    #
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U is (9, r), V is (12, r), W is (12, r)
    # The decomposition is: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    #
    # The matrix multiplication tensor T[a, b, c] = 1 if the indices are "compatible" i.e.
    # a = i*m+j, b = j'*p+k, c = k'*n+i' and j=j' and k=k' and i=i' ... 
    # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # And T is 0 otherwise.
    
    # Build the target tensor
    N, P, Q = n*m, m*p, p*n  # 9, 12, 12
    T = np.zeros((N, P, Q), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # We need to find a rank-r decomposition with integer or half-integer entries.
    # Start with the schoolbook rank 36, then try to reduce.
    
    # First, let's try to find a low-rank decomposition using numerical methods,
    # then check if it can be made exact with integer/half-integer values.
    
    # Strategy: use the fact that we can search over small ranks.
    # The known lower bound for 3x3 by 3x4 is not well-established, but let's try ranks from 25 up.
    
    def try_rank(r):
        """Try to find a rank-r decomposition with integer/half-integer entries."""
        # We'll use a random initialization and then project to integer/half-integer lattice.
        # But first, let's see if we can find a real-valued decomposition, then round.
        
        # Use ALS (Alternating Least Squares) or similar approach
        # Actually, let's use a simpler approach: random init, then optimize.
        
        np.random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(N, r)
        V = np.random.randn(P, r)
        W = np.random.randn(Q, r)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U) * np.sqrt(r)
        V /= np.linalg.norm(V) * np.sqrt(r)
        W /= np.linalg.norm(W) * np.sqrt(r)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # ALS iterations
        for it in range(500):
            # Update U: for each row a, solve least squares
            for a in range(N):
                # T[a, :, :] = sum_r U[a,r] * (V[:, r] outer W[:, r])
                # We want: sum_r U[a,r] * (V[:,r] * W[:,r]) ≈ T[a,:,:]
                # Let M_r = V[:,r] outer W[:,r], then we want U[a,:] such that sum_r U[a,r] M_r ≈ T[a,:,:]
                # This is a linear system: [M_0, M_1, ..., M_{r-1}] * U[a,:]^T ≈ T[a,:,:].ravel()
                M = np.zeros((P*Q, r))
                for rr in range(r):
                    M[:, rr] = (V[:, rr] * W[:, rr]).ravel()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, T[a, :, :].ravel(), rcond=None)
                    U[a, :] = sol
                except:
                    pass
            
            # Update V
            for b in range(P):
                M = np.zeros((N*Q, r))
                for rr in range(r):
                    M[:, rr] = (U[:, rr] * W[:, rr]).ravel()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, T[:, b, :].ravel(), rcond=None)
                    V[b, :] = sol
                except:
                    pass
            
            # Update W
            for c in range(Q):
                M = np.zeros((N*P, r))
                for rr in range(r):
                    M[:, rr] = (U[:, rr] * V[:, rr]).ravel()
                try:
                    sol, _, _, _ = np.linalg.lstsq(M, T[:, :, c].ravel(), rcond=None)
                    W[c, :] = sol
                except:
                    pass
            
            # Compute error
            err = 0.0
            for rr in range(r):
                err += np.sum((U[:, rr] * V[:, rr] * W[:, rr]) - T * (rr == 0))  # This is wrong
            # Actually compute properly
            reconstruction = np.zeros((N, P, Q))
            for rr in range(r):
                reconstruction += U[:, rr, None] * V[:, None, rr] * W[:, None, None, rr]  # This is wrong shape
            # Let me fix: reconstruction[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            reconstruction = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.sum((reconstruction - T)**2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if time.time() > deadline:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try to find a decomposition at various ranks
    # Start with known good: schoolbook rank 36
    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((P, rank))
        W = np.zeros((Q, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    # Now try to reduce the rank using the ALS approach
    # Try ranks from 25 to 35
    best_result = None
    best_rank = n * m * p
    best_err = 0.0
    
    for r in range(25, 36):
        if time.time() > deadline - 10:
            break
        U_t, V_t, W_t, err = try_rank(r)
        if err < 1e-10:
            best_result = (U_t, V_t, W_t, r)
            best_rank = r
            break
    
    # If no exact solution found, use the schoolbook
    if best_result is None:
        U_out, V_out, W_out = U_sb, V_sb, W_sb
    else:
        U_out, V_out, W_out, best_rank = best_result
    
    # Now, the entries should be integer or half-integer. Let's check and fix.
    # Multiply by 2 to see if we get integers
    U2 = 2 * U_out
    V2 = 2 * V_out
    W2 = 2 * W_out
    
    # Check if close to integer
    U_int = np.round(U2)
    V_int = np.round(V2)
    W_int = np.round(W2)
    
    U_fixed = U_int / 2.0
    V_fixed = V_int / 2.0
    W_fixed = W_int / 2.0
    
    # Verify the decomposition
    reconstruction = np.einsum('ir,jr,kr->ijk', U_fixed, V_fixed, W_fixed)
    err = np.sum((reconstruction - T)**2)
    
    if err > 1e-10:
        # Fall back to schoolbook
        U_out, V_out, W_out = U_sb, V_sb, W_sb
    
    # Convert to lists of floats
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
