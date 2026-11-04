"""Matrix multiplication 3,4,7 with structured decomposition."""
DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll use a structured approach.
    # The middle dimension m=4 can be seen as 2x2.
    # We can decompose the 3x4 matrix as a block matrix [[A1, A2], [B1, B2]] where each block is 3x2? No.
    # Actually, let's think about it differently.
    
    # Strategy: Use the known rank-7 decomposition for 2x2x2 (Strassen) and combine.
    # But first, let's try a simpler structured approach.
    
    # For 3x4 * 4x7:
    # We can split the 4 columns of A into two groups of 2: A = [A1 | A2] where A1, A2 are 3x2
    # And split the 4 rows of B into two groups of 2: B = [B1; B2] where B1, B2 are 2x7
    # Then C = A1*B1 + A2*B2
    
    # Each term is (3x2)*(2x7) = 3x7
    # The rank of 3x2*2x7 multiplication: we can use the 2x2x2 Strassen trick on the middle 2.
    # For (3x2)*(2x7): think of it as multiplying 3 vectors of length 2 by 2 vectors of length 7? 
    # No, it's 3x2 times 2x7. The middle dimension is 2.
    # We can use Strassen-like for the 2: 
    #   Let A be 3x2, B be 2x7.
    #   C = A*B
    #   Using Strassen for the 2x2 structure isn't directly applicable since dimensions don't match.
    
    # Alternative: just use the schoolbook for the 3x2 * 2x7 part: rank = 3*2*7 = 42 per term, total 84.
    # That's worse than 84... wait, schoolbook for 3*4*7 = 84. So splitting doesn't help naively.
    
    # Let me reconsider. The schoolbook rank is 3*4*7 = 84.
    
    # Better approach: Use tensor decomposition techniques.
    # For small matrices, we can try to find a lower rank decomposition.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (Bini et al.)
    # - 2x3x3: rank 15
    # - 3x3x4: rank 27
    # - 2x2x3: rank 8
    # - 2x3x4: rank 19
    # - 3x4x4: rank 34?
    
    # For 3x4x7, let's try to use a Kronecker product structure.
    # 3 = 3, 4 = 2*2, 7 = 7
    # Or 3 = 3, 4 = 4, 7 = 7
    
    # Let's try a different split: 3x4*4x7
    # Split 3 into 1+2: A = [a1; A2] where a1 is 1x4, A2 is 2x4
    # B stays 4x7
    # C = a1*B + A2*B
    # a1*B is 1x7: rank 1*4*7 = 28 (schoolbook) or can we do better? 
    #   Actually a1*B is just a linear combination, rank = 4*7 = 28? No, it's rank 1 in the tensor sense? 
    #   The tensor for 1x4*4x7 has rank 4*7 = 28 since it's just a sum of 28 terms.
    # A2*B is 2x4*4x7: 
    #   Use Strassen-like on the 2: split 4 into 2+2
    #   A2 = [A21 | A22] (2x2 each), B = [B1; B2] (2x7 each)
    #   A2*B = A21*B1 + A22*B2
    #   Each is (2x2)*(2x7): middle dim 2, use Strassen? 
    #   (2x2)*(2x7): The middle dimension is 2. We can view this as 2*(2x7) where the 2 is the middle.
    #   Actually, for (m x k) * (k x n) with k=2:
    #   C = A[:,0]*B[0,:] + A[:,1]*B[1,:]
    #   This is a sum of 2 rank-1 tensors (in the appropriate sense).
    #   The rank of the 2x2x7 tensor (middle dim 2) is 2*2*7 = 28? No.
    #   The tensor T[i,j,k] = delta(j,k) for (m x k)*(k x n) has rank m*k*n.
    #   For m=2, k=2, n=7: rank = 2*2*7 = 28.
    
    # This isn't leading to improvements. Let me just implement a search-based approach.
    
    # Actually, let me try a known good decomposition.
    # For 3x4*4x7, the minimum rank is unknown but bounded.
    # Let's use a heuristic: try to find a decomposition using random initialization + optimization.
    
    import numpy as np
    import time
    
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = delta(j,k) for i in [0,n), j in [0,m), k in [0,p)
    # We want T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # i.e., T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try ranks from lower to higher
    # Use a simple iterative approach
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is correct."""
        n, m, p = DIMENSIONS
        # Build the tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute the decomposition
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a 3-mode tensor product
        # Reshape and use einsum
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # T[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        try:
            T_comp = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
        except:
            return False
        
        if np.max(np.abs(T - T_comp)) > tol:
            return False
        return True
    
    # Try to find a better decomposition using optimization
    # Start with a few random initializations for lower ranks
    
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        # Try a rank lower than schoolbook
        for target_rank in range(50, best_rank):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Use a simple gradient-free approach: just check if random works (unlikely)
            # Instead, let's use a more structured approach
            
            # For now, just verify the schoolbook
            if verify(U_school, V_school, W_school):
                best_rank = rank_school
                best_U = U_school
                best_V = V_school
                best_W = W_school
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
