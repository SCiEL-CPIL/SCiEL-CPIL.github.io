import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # A: n x m = 2 x 3, B: m x p = 3 x 3, C: p x n = 3 x 2
    # A[i,j] index: i*m + j = i*3 + j
    # B[j,k] index: j*p + k = j*3 + k
    # C[k,i] index: k*n + i = k*2 + i (trace convention)
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if we compute C[k,i] += A[i,j]*B[j,k]
    # We seek a decomposition T = sum_r U[i,r] * V[j,r] * W[k,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    
    # Start with the known rank-17 decomposition (Strassen + schoolbook)
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for 2x2x2 block (rows 0,1 of A, cols 0,1 of A; rows 0,1 of B, cols 0,1 of B)
    # This computes C[0,0], C[0,1], C[1,0], C[1,1]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for 2x1x2 (A col 2, B row 2)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Part 3: Schoolbook for 2x3x1 (B col 2)
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # Verify the decomposition
    def verify(U, V, W):
        # Build the tensor from U, V, W
        nm, r = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Build the true tensor
        T_true = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i, j, k] = 1.0
        
        return np.allclose(T_decomp, T_true, atol=1e-9)
    
    if not verify(U, V, W):
        # Should not happen, but fall back to pure schoolbook
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, r] = 1
                    V[j*p + k, r] = 1
                    W[k*n + i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce rank by finding linear dependencies
    # We can try rank 16 by removing one column and seeing if we can adjust the others
    
    def try_reduce_rank(U, V, W, target_rank):
        """Try to find a valid decomposition of target_rank by removing one column and adjusting."""
        nm, cur_rank = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        
        # Try each column to remove
        for remove_idx in range(cur_rank):
            if time.time() - start_time > deadline - 10:
                break
            
            # Remove the column
            U_sub = np.delete(U, remove_idx, axis=1)
            V_sub = np.delete(V, remove_idx, axis=1)
            W_sub = np.delete(W, remove_idx, axis=1)
            
            # The error tensor is T_error = -U_sub * V_sub * W_sub + T_true
            # We need to find a rank-1 tensor that cancels this error
            # Error[i,j,k] = T_true[i,j,k] - sum_r' U_sub[i,r']*V_sub[j,r']*W_sub[k,r']
            T_true = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_true[i, j, k] = 1.0
            
            T_decomp = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
            T_error = T_true - T_decomp
            
            # Check if T_error is rank 1: T_error[i,j,k] = u_i * v_j * w_k
            # A tensor is rank 1 iff all 2x2 subdeterminants of any unfolding are zero
            # Simple check: for any i1,i2,j1,j2,k1,k2, the determinant condition
            # But a simpler approach: check if the tensor can be factored
            
            # Flatten and try to factor
            # For rank 1, T_error[i,j,k] = a[i]*b[j]*c[k]
            # We can try: if T_error[0,0,0] != 0, then a[i] = T_error[i,0,0]/T_error[0,0,0], etc.
            
            found = False
            if abs(T_error[0, 0, 0]) > 1e-10:
                a = T_error[:, 0, 0] / T_error[0, 0, 0]
                b = T_error[0, :, 0] / T_error[0, 0, 0]
                c = T_error[0, 0, :] / T_error[0, 0, 0]
                # Verify
                T_reconstructed = np.einsum('i,j,k->ijk', a, b, c)
                if np.allclose(T_error, T_reconstructed, atol=1e-9):
                    found = True
                    new_U = np.zeros((nm, target_rank))
                    new_V = np.zeros((mp, target_rank))
                    new_W = np.zeros((pn, target_rank))
                    
                    # Copy existing columns
                    for r in range(target_rank):
                        new_U[:, r] = U_sub[:, r]
                        new_V[:, r] = V_sub[:, r]
                        new_W[:, r] = W_sub[:, r]
                    
                    # Add the new rank-1 component
                    new_U[:, target_rank - 1] = a
                    new_V[:, target_rank - 1] = b
                    new_W[:, target_rank - 1] = c
                    
                    if verify(new_U, new_V, new_W):
                        return new_U, new_V, new_W
            
            # Try other pivot points
            for i0 in range(n):
                for j0 in range(m):
                    for k0 in range(p):
                        if abs(T_error[i0, j0, k0]) > 1e-10:
                            a = T_error[:, j0, k0] / T_error[i0, j0, k0]
                            b = T_error[i0, :, k0] / T_error[i0, j0, k0]
                            c = T_error[i0, j0, :] / T_error[i0, j0, k0]
                            T_reconstructed = np.einsum('i,j,k->ijk', a, b, c)
                            if np.allclose(T_error, T_reconstructed, atol=1e-9):
                                new_U = np.zeros((nm, target_rank))
                                new_V = np.zeros((mp, target_rank))
                                new_W = np.zeros((pn, target_rank))
                                
                                for r in range(target_rank):
                                    new_U[:, r] = U_sub[:, r]
                                    new_V[:, r] = V_sub[:, r]
                                    new_W[:, r] = W_sub[:, r]
                                
                                new_U[:, target_rank - 1] = a
                                new_V[:, target_rank - 1] = b
                                new_W[:, target_rank - 1] = c
                                
                                if verify(new_U, new_V, new_W):
                                    return new_U, new_V, new_W
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        return None
    
    # Try to reduce from 17 to 16
    if time.time() - start_time < deadline - 20:
        result = try_reduce_rank(U, V, W, 16)
        if result is not None:
            U, V, W = result
            rank = 16
            # Try to reduce further
            if time.time() - start_time < deadline - 10:
                result = try_reduce_rank(U, V, W, 15)
                if result is not None:
                    U, V, W = result
                    rank = 15
    
    # Return the best decomposition found
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

