def _search_2x3x2_rank(max_rank=12, time_budget=30):
    """Search for a low-rank decomposition of the 2x3x2 tensor."""
    import random
    start = time.time()
    
    # The tensor T[i,j,k] = 1 if this is a valid (i,j,k) triple for 2x3x2
    # i in [0,1], j in [0,1,2], k in [0,1]
    # We need U (6 x r), V (6 x r), W (4 x r) such that
    # sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all (i,j,k)
    
    # Actually, the tensor entries are: T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # No wait, for matrix multiplication, T[i,j,k] = 1 means that A[i,j] * B[j,k] contributes to C[i,k]
    # So the tensor is: sum over j of A[i,j] B[j,k] = C[i,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,1], j in [0,2], k in [0,1]
    # And we need: sum_r U[i*3+j, r] V[j*2+k, r] W[k*2+i, r] = T[i,j,k] = 1
    
    # Wait, that's not right. The decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*3+j,r] A[i,j]) * (sum_j V[j*2+k,r] B[j,k]) * W[k*2+i,r]
    # No, that's not how it works either.
    
    # The correct form is:
    # C[i,k] = sum_r U[i*3+?, r] * V[?, r] * W[k*2+i, r]
    # where U has shape (6, r) indexed by (i,j) for i in [0,1], j in [0,2]
    # V has shape (6, r) indexed by (j,k) for j in [0,2], k in [0,1]
    # W has shape (4, r) indexed by (k,i) for k in [0,1], i in [0,1]
    # And: sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # This is because A[i,j] * B[j,k] should contribute to C[i,k], and the coefficient is 1.
    
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank + 1
    
    for r in range(8, max_rank + 1):
        if time.time() - start > time_budget:
            break
        
        for attempt in range(100):
            if time.time() - start > time_budget:
                break
            
            U = np.random.randn(6, r)
            V = np.random.randn(6, r)
            W = np.zeros((4, r))
            for k in range(2):
                for i in range(2):
                    W[k*2+i, :] = 1.0  # Fix W to 1 for simplicity
            
            # Now solve for U, V such that sum_r U[i*3+j,r] V[j*2+k,r] = 1 for all i,j,k
            # This is a bilinear system, which is hard.
            # Instead, let's use a different approach: fix U and V randomly, and see if we can find W.
            
            # Actually, let's use a simpler approach: try to find a decomposition by solving a linear system.
            # For fixed U and V, we need: sum_r U[i*3+j,r] V[j*2+k,r] W[k*2+i,r] = 1
            # For each (i,j,k), this is a linear equation in W.
            
            # Let's set up the system: for each (i,j,k), we have:
            # sum_r (U[i*3+j,r] * V[j*2+k,r]) * W[k*2+i,r] = 1
            # But W is indexed by (k,i), so for fixed (k,i), W[k*2+i,r] appears in all equations with that (k,i).
            
            # Actually, this is still nonlinear because U and V are also unknowns.
            
            # Let me try a different approach: use the known Strassen decomposition for 2x2x2 and extend it.
            pass
    
    return best_U, best_V, best_W, best_rank
