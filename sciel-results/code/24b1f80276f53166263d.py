"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 by 3x5 matrix multiplication."""
    deadline = time.time() + 145  # safety margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    schoolbook_rank = n * m * p  # 30
    
    # Schoolbook decomposition: rank 30
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition using random initialization and iterative refinement
    # For 2x3x5, the known lower bound is 19 and upper bound is 30.
    # We'll try ranks from 24 down to 20 (conservative to stay within time limits)
    
    import numpy as np
    
    # Define the multiplication tensor T[i,j,k] = 1 for all valid indices
    # The condition for decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    def check_decomposition(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly reproduces the multiplication tensor."""
        # U_mat shape: (n*m, rank)
        # V_mat shape: (m*p, rank)  
        # W_mat shape: (p*n, rank)
        rank = U_mat.shape[1]
        
        # Reshape to 3D tensors
        U3 = U_mat.reshape(n, m, rank)  # U3[i, j, r]
        V3 = V_mat.reshape(m, p, rank)  # V3[j, k, r]
        W3 = W_mat.reshape(p, n, rank)  # W3[k, i, r]
        
        # Compute the contraction: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # For each (i, j, k), we need this to equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def to_integer_or_half_integer(mat):
        """Try to convert entries to integer or half-integer values if close enough."""
        mat = np.array(mat, dtype=float)
        # Round to nearest half-integer
        rounded = np.round(mat * 2) / 2.0
        # Check if close enough
        if np.max(np.abs(mat - rounded)) < 1e-6:
            return rounded
        return mat
    
    # Try various ranks from high to low
    for target_rank in range(29, 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U_init = np.random.randn(n * m, target_rank) * 0.5
            V_init = np.random.randn(m * p, target_rank) * 0.5
            W_init = np.random.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement: update each matrix to minimize error
            # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
            # Fix V and W, update U; fix U and W, update V; fix U and V, update W
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            # Reshape for easier indexing
            def reshape_U(U_mat):
                return U_mat.reshape(n, m, target_rank)
            def reshape_V(V_mat):
                return V_mat.reshape(m, p, target_rank)
            def reshape_W(W_mat):
                return W_mat.reshape(p, n, target_rank)
            
            # Iterative updates
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i, j), solve for U[i,j,:] 
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                # This is a linear system: for fixed i,j, 
                # sum_k (V[j,k,:] * W[k,i,:]) dot U[i,j,:] = 1
                # Let A_k = V[j,k,:] * W[k,i,:] (element-wise product, shape (rank,))
                # Then we need: sum_r U[i,j,r] * (sum_k A_k[r])... wait, no
                # Actually: for each k, sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # So we have p equations in rank unknowns for U[i,j,:]
                # If rank >= p, we can solve this linear system
                
                U3 = reshape_U(U_cur)
                V3 = reshape_V(V_cur)
                W3 = reshape_W(W_cur)
                
                new_U = np.zeros((n, m, target_rank))
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k in range(p),
                        # sum_r new_U[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1
                        # Let B[r] = sum_k V3[j,k,r] * W3[k,i,r]  -- no, that's not right
                        # For each k: sum_r new_U[i,j,r] * (V3[j,k,r] * W3[k,i,r]) = 1
                        # So we have a matrix M of size (p, rank) where M[k, r] = V3[j,k,r] * W3[k,i,r]
                        # And we need M @ new_U[i,j,:] = ones(p)
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                M[k, r] = V3[j, k, r] * W3[k, i, r]
                        # Solve M @ x = ones(p) for x (shape (rank,))
                        # If p <= rank, use least squares
                        try:
                            if p <= target_rank:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            else:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            new_U[i, j, :] = x
                        except:
                            new_U[i, j, :] = U3[i, j, :]
                
                U_cur = new_U.reshape(n * m, target_rank)
                
                # Update V: for each (j, k), solve for V[j,k,:]
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i
                # For fixed j,k: sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r]) = 1 for all i
                # Matrix N of size (n, rank) where N[i, r] = U[i,j,r] * W[k,i,r]
                # N @ V[j,k,:] = ones(n)
                
                V3 = reshape_V(V_cur)
                U3 = reshape_U(U_cur)
                W3 = reshape_W(W_cur)
                
                new_V = np.zeros((m, p, target_rank))
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                N[i, r] = U3[i, j, r] * W3[k, i, r]
                        try:
                            if n <= target_rank:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                            else:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                            new_V[j, k, :] = x
                        except:
                            new_V[j, k, :] = V3[j, k, :]
                
                V_cur = new_V.reshape(m * p, target_rank)
                
                # Update W: for each (k, i), solve for W[k,i,:]
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
                # For fixed k,i: sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r]) = 1 for all j
                # Matrix P of size (m, rank) where P[j, r] = U[i,j,r] * V[j,k,r]
                # P @ W[k,i,:] = ones(m)
                
                W3 = reshape_W(W_cur)
                U3 = reshape_U(U_cur)
                V3 = reshape_V(V_cur)
                
                new_W = np.zeros((p, n, target_rank))
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                P[j, r] = U3[i, j, r] * V3[j, k, r]
                        try:
                            if m <= target_rank:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                            else:
                                x, residuals, rank_sys, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                            new_W[k, i, :] = x
                        except:
                            new_W[k, i, :] = W3[k, i, :]
                
                W_cur = new_W.reshape(p * n, target_rank)
            
            # Check if the decomposition is exact
            if check_decomposition(U_cur, V_cur, W_cur):
                # Convert to integer/half-integer if possible
                U_final = to_integer_or_half_integer(U_cur)
                V_final = to_integer_or_half_integer(V_cur)
                W_final = to_integer_or_half_integer(W_cur)
                
                # Verify after rounding
                if check_decomposition(U_final, V_final, W_final):
                    best_result = {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
                    best_rank = target_rank
                    break
                else:
                    # Use the unrounded version if it's exact
                    best_result = {
                        "U": U_cur.tolist(),
                        "V": V_cur.tolist(),
                        "W": W_cur.tolist()
                    }
                    best_rank = target_rank
                    break
        
        # If we found a lower rank, we can stop early if time allows
        if best_rank < schoolbook_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
