import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _try_remove_columns(U, V, W, max_iterations=50):
    """Try to remove linearly dependent columns from the decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    for _ in range(max_iterations):
        # Build the tensor matrix: each column r gives a vector of size n*m*p
        # T_r[i*m+j, j*p+k, k*n+i] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can check if column r is in the span of others by checking
        # if the outer product structure allows it.
        
        # Simpler: check if removing column r still satisfies the tensor equation
        # For each (i,j,k): sum_{r != removed} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should still equal 1
        
        # This is expensive, so use a randomized approach
        # Pick a random column and check if it's redundant
        
        if rank <= 1:
            break
            
        # Try to find a column that's a linear combination of others
        # Use the fact that if we fix the structure, we can check via the
        # bilinear form
        
        # Check each column: is it in the span of the others?
        # A column r is redundant if for all (i,j,k):
        # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_{s != r} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
        # This is hard to check directly.
        
        # Alternative: try removing each column and check if the tensor still works
        # Only do this for a subset to save time
        removed = False
        for r in range(rank):
            # Quick check: if this column is all zeros, remove it
            if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            
            for i, idx in enumerate(u_idx):
                U[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk[i, r]
                
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    # Strassen 2x2x2: rank 7
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Schoolbook 2x2x1: rank 4
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x2: rank 4
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1: rank 2
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # U indices: A is 3x4, flat index i*m+j where i=0..2, j=0..3
    # V indices: B is 4x5, flat index j*p+k where j=0..3, k=0..4
    # W indices: C is 5x3, flat index k*n+i where k=0..4, i=0..2

    # === 2x4x5 block (A rows 0-1) ===
    # Split B columns into 2+2+1
    
    # Cols 0-1: 2x4x2, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2]
    u_idx = [0, 1, 4, 5]    # A[0,0],A[0,1],A[1,0],A[1,1]
    v_idx = [0, 1, 5, 6]    # B[0,0],B[0,1],B[1,0],B[1,1]
    w_idx = [0, 3, 1, 4]    # C[0,0],C[1,0],C[0,1],C[1,1]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block: A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2]
    u_idx = [2, 3, 6, 7]    # A[0,2],A[0,3],A[1,2],A[1,3]
    v_idx = [10, 11, 15, 16]  # B[2,0],B[2,1],B[3,0],B[3,1]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Cols 2-3: 2x4x2, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 2:4] -> C[2:4, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]    # B[0,2],B[0,3],B[1,2],B[1,3]
    w_idx = [6, 9, 7, 10]   # C[2,0],C[3,0],C[2,1],C[3,1]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Block: A[0:2, 2:4] x B[2:4, 2:4] -> C[2:4, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]  # B[2,2],B[2,3],B[3,2],B[3,3]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Col 4: 2x4x1, split B rows 0:4 as 2+2
    # Block: A[0:2, 0:2] x B[0:2, 4] -> C[4, 0:2]
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]          # B[0,4],B[1,4]
    w_idx = [12, 13]        # C[4,0],C[4,1]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # Block: A[0:2, 2:4] x B[2:4, 4] -> C[4, 0:2]
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]        # B[2,4],B[3,4]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # === 1x4x5 block (A row 2) ===
    # Split B rows 0:4 as 2+2
    
    # B rows 0:2: 1x2x5, split B cols 0:5 as 2+2+1
    # Block: A[2, 0:2] x B[0:2, 0:2] -> C[0:2, 2]
    u_idx = [8, 9]          # A[2,0],A[2,1]
    v_idx = [0, 1, 5, 6]    # B[0,0],B[0,1],B[1,0],B[1,1]
    w_idx = [2, 5]          # C[0,2],C[1,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 0:2] x B[0:2, 2:4] -> C[2:4, 2]
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]    # B[0,2],B[0,3],B[1,2],B[1,3]
    w_idx = [8, 11]         # C[2,2],C[3,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 0:2] x B[0:2, 4] -> C[4, 2]
    u_idx = [8, 9]
    v_idx = [4, 9]          # B[0,4],B[1,4]
    w_idx = [14]            # C[4,2]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    # B rows 2:4: 1x2x5, split B cols 0:5 as 2+2+1
    # Block: A[2, 2:4] x B[2:4, 0:2] -> C[0:2, 2]
    u_idx = [10, 11]        # A[2,2],A[2,3]
    v_idx = [10, 11, 15, 16]  # B[2,0],B[2,1],B[3,0],B[3,1]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 2:4] x B[2:4, 2:4] -> C[2:4, 2]
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]  # B[2,2],B[2,3],B[3,2],B[3,3]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    # Block: A[2, 2:4] x B[2:4, 4] -> C[4, 2]
    u_idx = [10, 11]
    v_idx = [14, 19]        # B[2,4],B[3,4]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    rank = U.shape[1]
    
    # Try to remove linearly dependent columns
    if time.time() - start_time < 120:
        U, V, W = _try_remove_columns(U, V, W)
        rank = U.shape[1]
    
    # Verify correctness
    if time.time() - start_time < 130:
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-10:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
