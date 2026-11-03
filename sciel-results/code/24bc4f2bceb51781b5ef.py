import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    
    # Target: Rank-23 decomposition for 3x3x3
    # Tensor T of size 3x3x3 where T[i,j,k] = 1 if i==k else 0
    
    T = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j, i] = 1.0
            
    # We want to find U (3xR), V (3xR), W (3xR) such that
    # sum_r U[i,r] * V[j,r] * W[k,r] == T[i,j,k]
    
    # Use ALS to find a low rank approximation
    # Start with R=23
    
    def find_decomposition(target_tensor, target_rank, max_iters=2000, tol=1e-9):
        R = target_rank
        # Initialize U, V, W with small random integers
        # Using {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(3, R))
        V = np.random.choice([-1, 0, 1], size=(3, R))
        W = np.random.choice([-1, 0, 1], size=(3, R))
        
        # Ensure non-zero
        if np.all(U == 0): U[0,0] = 1
        if np.all(V == 0): V[0,0] = 1
        if np.all(W == 0): W[0,0] = 1
        
        for it in range(max_iters):
            # Solve for U: U[i,r] * (V[j,r] * W[k,r]) = T[i,j,k]
            # Flatten V and W
            VW = (V * W.T).reshape(3, 3, R) # V[j,r] * W[k,r]
            # For each i, U[i, :] is the solution to U[i, :] @ VW[i,:,:].reshape(9, R) = T[i,:,:].reshape(9)
            # Actually, sum_r U[i,r] * (V[j,r] W[k,r]) = T[i,j,k]
            # Let M_U[i] be a 9xR matrix where M_U[i][j*3+k, r] = V[j,r] * W[k,r]
            # Then U[i, :] @ M_U[i] = T[i, :, :].reshape(9)
            
            M_U = np.zeros((3, 9, R))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        M_U[i, j*3+k, :] = V[:, :] * W[:, :].T # This is wrong shape
                        # V is 3xR, W is 3xR.
                        # We need V[j,r] * W[k,r] for all r.
                        # This is a vector of length R.
                        # M_U[i, idx, r] = V[j, r] * W[k, r]
                        
            # Re-calculate M_U efficiently
            # M_U[i, j*3+k, r] = V[j, r] * W[k, r]
            # We can do this by broadcasting
            # V is (3, R), W is (3, R)
            # V[:, :, None] is (3, R, 1)
            # W[:, None, :] is (3, 1, R)
            # Product is (3, R, 3) -> V[j, r] * W[k, r]
            # Reshape to (3, 9, R)
            # V[j, r] * W[k, r] -> index (j, k, r)
            # We want index (j*3+k, r)
            prod_VW = V[:, :, None] * W[:, None, :] # (3, R, 3)
            # Transpose to (3, 3, R) -> (j, k, r)
            prod_VW = prod_VW.transpose(0, 2, 1) # (3, 3, R)
            # Reshape to (3, 9, R)
            M_U = prod_VW.reshape(3, 9, R)
            
            # Solve U[i] @ M_U[i] = T[i].reshape(9)
            for i in range(3):
                rhs = T[i, :, :].reshape(9)
                # Least squares
                U[i, :], _, _, _ = np.linalg.lstsq(M_U[i], rhs, rcond=None)
                
            # Solve for V
            # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # M_V[j, i*3+k, r] = U[i, r] * W[k, r]
            prod_UW = U[:, :, None] * W[:, None, :] # (3, R, 3) -> (i, r, k)
            prod_UW = prod_UW.transpose(0, 2, 1) # (3, 3, R) -> (i, k, r)
            # We need (j, i, k, r) -> M_V[j, i*3+k, r]
            # prod_UW is (i, k, r). We need to expand for j? No, V is independent of j in the product?
            # Wait, V[j, r] is the unknown.
            # The term is U[i, r] * V[j, r] * W[k, r].
            # For fixed j, V[j, :] is the unknown vector.
            # The equation is V[j, :] @ M_V[j] = T[:, j, :].reshape(9)
            # M_V[j, i*3+k, r] = U[i, r] * W[k, r]
            # This M_V is the same for all j?
            # Yes, because U and W do not depend on j.
            # So M_V is just prod_UW reshaped.
            M_V = prod_UW.reshape(3, 9, R) # (i, k, r) -> (i*3+k, r)
            # Wait, the index is i*3+k.
            # T[:, j, :] is 3x3. Flatten to 9.
            for j in range(3):
                rhs = T[:, j, :].reshape(9)
                V[j, :], _, _, _ = np.linalg.lstsq(M_V, rhs, rcond=None)
                
            # Solve for W
            # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # M_W[k, i*3+j, r] = U[i, r] * V[j, r]
            prod_UV = U[:, :, None] * V[:, None, :] # (3, R, 3) -> (i, r, j)
            prod_UV = prod_UV.transpose(0, 2, 1) # (3, 3, R) -> (i, j, r)
            M_W = prod_UV.reshape(3, 9, R)
            for k in range(3):
                rhs = T[:, :, k].reshape(9)
                W[k, :], _, _, _ = np.linalg.lstsq(M_W, rhs, rcond=None)
                
            # Check error
            # Reconstruct
            T_rec = np.zeros((3, 3, 3))
            for r in range(R):
                T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(T_rec - T)
            if err < tol:
                break
                
        return U, V, W, err

    # Try to find rank 23
    start_time = time.time()
    U, V, W, err = find_decomposition(T, 23, max_iters=500)
    
    if err > 1e-6:
        # If 23 fails, try 24, 25...
        for r in range(24, 30):
            U, V, W, err = find_decomposition(T, r, max_iters=500)
            if err < 1e-6:
                break
            if time.time() - start_time > 100:
                break

    # If we still have high error, use trivial 27 (or 54 for the whole thing)
    # But 23 should work.
    
    # Round to nearest 0.5
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    T_rec = np.zeros((3, 3, 3))
    R = U.shape[1]
    for r in range(R):
        T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    if not np.allclose(T_rec, T, atol=1e-9):
        # Fallback to trivial rank 27 for 3x3x3 (which is 54 for 3x3x6)
        # Or just use the trivial 54 directly
        R = 27
        U = np.zeros((3, R))
        V = np.zeros((3, R))
        W = np.zeros((3, R))
        col = 0
        for i in range(3):
            for j in range(3):
                U[i, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        # This is rank 9 for 3x3x3? No, 9 terms.
        # Trivial rank for 3x3x3 is 9.
        # Wait, trivial rank is n*m*p = 9.
        # So if 23 fails, use 9.
        # But 9 is worse than 23.
        # If 23 fails, use 27?
        # Let's just use the found U, V, W even if error is small but not zero?
        # No, must be exact.
        # If rounding failed, we can try to solve integer linear programming?
        # For now, if error is not zero, we return a valid but higher rank decomposition.
        # The trivial rank 9 for 3x3x3 is valid.
        # So total rank 18 for 3x3x6.
        # 18 < 54.
        # So even the trivial 3x3x3 decomposition gives rank 18 for the whole problem.
        # Wait, trivial rank for 3x3x3 is 9.
        # Two blocks -> rank 18.
        # So we can always achieve rank 18.
        # Let's check if the trivial 9-term decomposition is valid.
        # T[i,j,k] = delta_ik.
        # Sum_r U[i,r] V[j,r] W[k,r].
        # If we use U[i,r]=delta_ir, V[j,r]=delta_jr, W[k,r]=delta_kr?
        # Then sum_r delta_ir delta_jr delta_kr = delta_ij delta_jk = delta_ik.
        # Yes, rank 9 works.
        
        # So if Bini 23 fails, we use Rank 9 for each block, total Rank 18.
        # 18 is much better than 54.
        
        U = np.zeros((3, 9))
        V = np.zeros((3, 9))
        W = np.zeros((3, 9))
        for r in range(9):
            i = r // 3
            j = r % 3
            U[i, r] = 1
            V[j, r] = 1
            W[i, r] = 1
        R = 9

    # Construct the final matrices for 3x3x6
    # U is 3xR, V is 3xR, W is 3xR for one block.
    # Total Rank = 2 * R
    Total_Rank = 2 * R
    
    # U_total: size (n*m) x Total_Rank = 9 x 2R
    # V_total: size (m*p) x Total_Rank = 18 x 2R
    # W_total: size (p*n) x Total_Rank = 18 x 2R
    
    U_total = np.zeros((9, Total_Rank))
    V_total = np.zeros((18, Total_Rank))
    W_total = np.zeros((18, Total_Rank))
    
    # Block 1 (k=0,1,2)
    # U: A is 3x3. U_block is 3xR.
    # Map U_block to U_total.
    # U_total[i*3+j, r] = U_block[i, r]
    for i in range(3):
        for j in range(3):
            for r in range(R):
                U_total[i*3+j, r] = U[i, r]
                
    # V: B is 3x3. V_block is 3xR.
    # V_total[j*3+k, r] = V_block[j, r] for k=0,1,2
    for j in range(3):
        for k in range(3):
            for r in range(R):
                V_total[j*3+k, r] = V[j, r]
                
    # W: C is 3x3. W_block is 3xR.
    # W_total[k*3+i, r] = W_block[k, r] for k=0,1,2
    for k in range(3):
        for i in range(3):
            for r in range(R):
                W_total[k*3+i, r] = W[k, r]
                
    # Block 2 (k=3,4,5)
    # Use same U
    for i in range(3):
        for j in range(3):
            for r in range(R):
                U_total[i*3+j, R+r] = U[i, r]
                
    # V: B is 3x3. V_block is 3xR.
    # V_total[j*3+(k+3), R+r] = V_block[j, r]
    for j in range(3):
        for k in range(3):
            for r in range(R):
                V_total[j*3+k+3, R+r] = V[j, r]
                
    # W: C is 3x3. W_block is 3xR.
    # W_total[(k+3)*3+i, R+r] = W_block[k, r]
    for k in range(3):
        for i in range(3):
            for r in range(R):
                W_total[k*3+i+9, R+r] = W[k, r]
                
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
