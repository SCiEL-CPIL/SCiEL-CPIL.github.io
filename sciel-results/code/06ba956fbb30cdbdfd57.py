import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Rank-23 decomposition for 3x3x3
    # We'll use a known construction
    
    # First, let's build a rank-23 decomposition for 3x3x3
    # T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    
    # Known rank-23 decomposition (one of many possible)
    # Let me use a systematic construction
    
    # Actually, let me just use the naive rank-9 for each 3-column block
    # and see if I can do better
    
    # For now, let's construct a valid decomposition
    # and then try to minimize rank
    
    # Strategy: 
    # - Columns 0,1,2: use rank R1 decomposition
    # - Column 3: use rank 9 (matrix-vector)
    # - Columns 4,5,6: use rank R2 decomposition
    
    # For 3x3x3, the minimum rank is 23
    # Let me construct an explicit rank-23 decomposition
    
    # I'll use the following known rank-23 decomposition for 3x3x3:
    # Based on the work of Strassen and others
    
    # Let me just use a simple valid decomposition first
    # and verify it works, then optimize
    
    # Naive: rank 9 for each 3-column block, rank 9 for single column
    # Total: 9 + 9 + 9 = 27? No, that's not right.
    
    # Wait, for 3x3x3, the naive rank is 9 (one term per (i,j,k) but that's 27)
    # No, the naive rank for 3x3x3 is 27 (all possible (i,j,k) triples)
    # But we can do better.
    
    # Let me just construct a valid decomposition and return it
    # Then we can optimize the rank
    
    # For simplicity, let's use the following approach:
    # Each "multiplication" in the algorithm corresponds to a rank-1 term
    
    # Let me use a known rank-23 decomposition
    # From the literature, one rank-23 decomposition of the 3x3x3 tensor:
    
    # I'll construct it explicitly
    
    rank_23 = 23
    rank_mv = 9  # matrix-vector
    
    total_rank = rank_23 + rank_mv + rank_23
    # = 55
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col):
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
    
    # For 3x3x3, we need 23 terms such that for all i,j,k:
    # sum_col U[i*3+j, col] * V[j*3+k, col] * W[k*3+i, col] = 1
    
    # Let me use a known rank-23 construction
    # One approach: use the fact that 3x3x3 can be embedded in 4x4x4
    # and use rank-19 for 4x4x4 (Strassen-like)
    
    # Actually, let me just use a simple valid decomposition
    # and accept a higher rank for now
    
    # Naive decomposition for 3x3x3: rank 27
    # But we want rank 23
    
    # Let me look up or derive a rank-23 decomposition
    
    # For now, let's use rank 27 for each 3-column block
    # Total: 27 + 9 + 27 = 63 (same as naive)
    
    # That's not an improvement. Let me think harder.
    
    # Key insight: for the 3x3x3 tensor, we can use the following trick:
    # C = A*B
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # etc.
    
    # A rank-1 term (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]
    # So we need to cover all (i,k) pairs with the right combinations
    
    # For rank 23, we need to find 23 triples (i,j,k) and coefficients
    # such that for each (i,k), the sum of A[i,j]*B[j,k] over all terms
    # with that (i,k) equals the correct sum
    
    # This is a combinatorial problem
    
    # Let me just use a known explicit rank-23 decomposition
    # From the paper "The rank of the matrix multiplication tensor"
    
    # I'll use the following construction (one of many):
    
    # Actually, let me just construct a valid solution with rank <= 55
    # by using rank-9 for each 3-column block (which is valid but not minimal)
    # and rank-9 for the single column
    
    # Wait, rank-9 for 3x3x3 is NOT valid. The rank of 3x3x3 is at least 19
    # (lower bound) and at most 23 (upper bound)
    
    # So I MUST use at least rank 19 for each 3-column block
    
    # Let me use rank 23 (known achievable)
    
    # I need an explicit rank-23 decomposition. Let me construct one.
    
    # Here's a rank-23 decomposition of the 3x3x3 multiplication tensor:
    # Based on the work of Strassen, the 23 multiplications are:
    
    # I'll use a systematic construction. Let me define the 23 terms.
    
    # Term 1-9: diagonal terms (i,i,k) for all i,k -> 9 terms
    # Term 10-18: off-diagonal (i,j,k) where j != i, specific pattern
    # Term 19-23: additional terms
    
    # Actually, let me just use a known explicit list
    
    # From the literature, one rank-23 decomposition uses these 23 (i,j,k) triples:
    # (0,0,0), (0,0,1), (0,0,2),
    # (0,1,0), (0,1,1), (0,1,2),
    # (0,2,0), (0,2,1), (0,2,2),
    # (1,0,0), (1,0,1), (1,0,2),
    # (1,1,0), (1,1,1), (1,1,2),
    # (1,2,0), (1,2,1), (1,2,2),
    # (2,0,0), (2,0,1), (2,0,2),
    # (2,1,0), (2,1,1), (2,1,2),
    # (2,2,0), (2,2,1), (2,2,2)
    
    # That's 27 terms (naive). We need to reduce to 23.
    
    # The reduction comes from the fact that some combinations can be shared.
    
    # Let me use a different approach: I'll use the known fact that
    # 3x3x3 has rank 23, and I'll construct it using block methods.
    
    # For now, let me just return a valid decomposition with rank 63
    # and mark this as a placeholder. The actual optimization will come later.
    
    # Actually, let me try a simpler approach: use rank 27 for each 3-col block
    # and rank 9 for the single column. Total: 27+9+27 = 63. Same as naive.
    
    # That's not an improvement. I need to actually find a lower rank.
    
    # Let me use alternating minimization to find a low-rank decomposition.
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(40, 64):
        for attempt in range(5):
            np.random.seed(rank * 100 + attempt)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            success = False
            for iteration in range(200):
                # Alternate between updating U, V, W
                # For fixed V, W, solve for U
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For each (i,j), we have: sum_r U[i*m+j, r] * (V[j*p+0:..., r] * W[0:..., r]) = 1
                # This is: U[i*m+j, :] @ diag(V[j*p+0:..., r] * W[0:..., r]) = 1
                # Wait, that's not quite right.
                
                # Actually: for fixed (i,j), the equation is:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let M[i,j] be a p x rank matrix with M[i,j][k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then: U[i*m+j, :] @ M[i,j].T = ones(p)
                # So: U[i*m+j, :] = ones(p) @ M[i,j].T @ pinv(M[i,j].T @ M[i,j].T.T)
                # Or more simply, solve the linear system
                
                # For each (i,j), solve: sum_r U[i*m+j, r] * M[i,j][k, r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                
                # Build the system
                for i in range(n):
                    for j in range(m):
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r] for k in 0..p-1, r in 0..rank-1
                        M = np.zeros((p, rank))
                        for k in range(p):
                            for r in range(rank):
                                M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve U[i*m+j, :] @ M.T = ones(p)
                        # i.e., M @ U[i*m+j, :].T = ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        # For fixed (j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in 0..n-1, r in 0..rank-1
                        M = np.zeros((n, rank))
                        for i in range(n):
                            for r in range(rank):
                                M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        # For fixed (k,i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in 0..m-1, r in 0..rank-1
                        M = np.zeros((m, rank))
                        for j in range(m):
                            for r in range(rank):
                                M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            error += abs(val - 1)
                
                if error < 1e-6:
                    success = True
                    break
            
            if success:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank is not None:
            break
    
    if best_rank is None:
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
