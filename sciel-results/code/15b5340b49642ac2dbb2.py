import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i<n, j<m, k<p? 
    # No, the tensor for matrix multiplication A(2x3) * B(3x4) = C(2x4)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 for all valid i, j, k.
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    T_flat = T.flatten()
    
    # Try ranks from low to high
    # Lower bound is roughly n*p? No.
    # Let's try 20, 21, 22, 23, 24.
    # We want the smallest rank that works.
    
    start_time = time.time()
    deadline = 150.0 # Safety margin
    
    for rank in [20, 21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W
            # Scale factors to avoid issues
            U = np.random.randn(n * rank, 1).reshape(n, rank)
            V = np.random.randn(m * rank, 1).reshape(m, rank)
            W = np.random.randn(p * rank, 1).reshape(p, rank)
            
            # ALS iterations
            for it in range(100):
                # Solve for U
                # A_U shape: (n*m*p, n*rank)
                # A_U[i, j, k, u_idx] = V[j, k] * W[k, u_idx] ?? No.
                # Term t: U[i, t] V[j, t] W[k, t]
                # We are solving for U[i, t].
                # For a fixed i, t, the coefficient in the equation for T[i, j, k] is V[j, t] W[k, t].
                
                # Constructing A_U explicitly is 576 x 120 = ~70k elements. Fast.
                A_U = np.zeros((n * m * p, n * rank))
                for i in range(n):
                    for t in range(rank):
                        # A_U[i, j, k, i*rank + t] = V[j, t] * W[k, t]
                        # Vectorize:
                        # For each i, t, the block of A_U corresponding to this variable is V[:, t] * W[:, t].T
                        # Flattened, it's np.kron(V[:, t], W[:, t])
                        A_U[:, i * rank + t] = np.kron(V[:, t], W[:, t])
                
                U_sol = np.linalg.lstsq(A_U, T_flat, rcond=None)[0]
                U = U_sol.reshape(n, rank)
                
                # Solve for V
                A_V = np.zeros((n * m * p, m * rank))
                for j in range(m):
                    for t in range(rank):
                        A_V[:, j * rank + t] = np.kron(U[:, t], W[:, t])
                V_sol = np.linalg.lstsq(A_V, T_flat, rcond=None)[0]
                V = V_sol.reshape(m, rank)
                
                # Solve for W
                A_W = np.zeros((n * m * p, p * rank))
                for k in range(p):
                    for t in range(rank):
                        A_W[:, k * rank + t] = np.kron(U[:, t], V[:, t])
                W_sol = np.linalg.lstsq(A_W, T_flat, rcond=None)[0]
                W = W_sol.reshape(p, rank)
                
            # Check convergence / exactness
            # Round to nearest half-integer
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify
            T_recon = np.zeros((n, m, p))
            for t in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U_round[:, t], V_round[:, t], W_round[:, t])
            
            if np.allclose(T_recon, T, atol=1e-9):
                # Found a valid decomposition
                # Convert to list of lists as required
                U_list = U_round.flatten().tolist() # Wait, shape (n, rank) -> (n, rank)
                # The interface requires U of shape (n*m, rank)?
                # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
                # My U is (n, rank). I need to map it to (n*m, rank)?
                # No, the standard decomposition is:
                # C[i, k] = sum_t U[i, t] V[k, t]? No.
                # The tensor is T[i, j, k].
                # Decomposition: T[i, j, k] = sum_t U[i, t] V[j, t] W[k, t].
                # The interface says:
                # U shape (n*m, rank). This implies U is indexed by (i, j)?
                # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" from the seed program.
                # Let's check the seed program structure.
                # Seed:
                # U[i * m + j][column] = 1
                # V[j * p + k][column] = 1
                # W[k * n + i][column] = 1
                # This corresponds to T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]?
                # No, the tensor for matrix multiplication is usually defined such that
                # C[i, k] = sum_j A[i, j] B[j, k].
                # The bilinear form is sum_{i, j, k} A[i, j] B[j, k] C[i, k].
                # So the tensor T has indices i, j, k.
                # The seed program sets U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
                # This looks like it's decomposing the identity map or something else?
                # Let's check the trace convention W[k*n+i, r].
                # If we multiply U (n*m x rank) and V (m*p x rank) and W (p*n x rank),
                # how do they combine?
                # Usually, if T[i, j, k] = sum U[i, t] V[j, t] W[k, t],
                # then U is (n x rank), V is (m x rank), W is (p x rank).
                # The seed program has U of size n*m. This suggests U depends on both i and j.
                # This is the "flattened" input representation?
                # If U is (n*m, rank), it means for each pair (i, j) and rank c, there is a weight.
                # Similarly V for (j, k).
                # W for (k, i).
                # This is a decomposition of the tensor T[i, j, k] into sum_c U[i, j, c] V[j, k, c] W[k, i, c]?
                # No, that would be a cyclic decomposition.
                # Let's look at the seed code again.
                # for i in range(n):
                #   for j in range(m):
                #     for k in range(p):
                #       U[i * m + j][column] = 1
                #       V[j * p + k][column] = 1
                #       W[k * n + i][column] = 1
                #       column += 1
                # This sets one non-zero in each of U, V, W for each (i, j, k).
                # So for a specific (i, j, k), there is a term c such that
                # U[i, j, c] = 1, V[j, k, c] = 1, W[k, i, c] = 1.
                # And all other entries for this c are 0?
                # If so, the product sum_c U[i, j, c] V[j, k, c] W[k, i, c]
                # For a fixed (i, j, k), the term c corresponding to (i, j, k) gives 1*1*1 = 1.
                # Other terms c' corresponding to (i', j', k') will have
                # U[i, j, c'] = 0 unless (i, j) = (i', j').
                # So if (i, j) != (i', j'), U is 0.
                # Thus the sum is exactly 1 for the matching (i, j, k) and 0 otherwise?
                # Wait, if I fix i, j, k, I want T[i, j, k] = 1.
                # The sum is over c.
                # Term c_{ijk} has U[i, j, c_{ijk}]=1, V[j, k, c_{ijk}]=1, W[k, i, c_{ijk}]=1. Product 1.
                # Any other term c' has either U[i, j, c']=0 or V[j, k, c']=0 or W[k, i, c']=0?
                # U[i, j, c'] is non-zero only if c' corresponds to some (i', j') with i'=i, j'=j.
                # So c' must be of the form c_{i, j, k'} for some k'.
                # Then V[j, k, c'] is non-zero only if c' corresponds to (j', k') with j'=j, k'=k.
                # So c' must be c_{i, j, k}.
                # Then W[k, i, c'] is non-zero only if c' corresponds to (k', i') with k'=k, i'=i.
                # So c' must be c_{i, j, k}.
                # Thus, only the term c_{ijk} contributes.
                # So T[i, j, k] = 1.
                # This works!
                
                # So the interface expects:
                # U: shape (n*m, rank). Indexing U[i*m+j, c].
                # V: shape (m*p, rank). Indexing V[j*p+k, c].
                # W: shape (p*n, rank). Indexing W[k*n+i, c].
                
                # My ALS solver found U (n, rank), V (m, rank), W (p, rank) such that
                # T[i, j, k] = sum_t U[i, t] V[j, t] W[k, t].
                # This is NOT the same as the interface.
                # The interface uses U[i, j, c], V[j, k, c], W[k, i, c].
                # This is a different tensor structure.
                # The tensor in the interface is T_interface[i, j, k, l]? No.
                # The seed program decomposes the tensor T[i, j, k] using factors U[i, j, c], V[j, k, c], W[k, i, c].
                # This implies the tensor being decomposed is actually the identity tensor?
                # No, the task is "Matrix multiplication 2,3,4".
                # The tensor for matrix multiplication is T[i, j, k] = 1.
                # The seed program constructs a decomposition of this tensor.
                # So I need to find U, V, W of the shapes specified that satisfy:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
                
                # My ALS solver found a decomposition of the form:
                # sum_t U_small[i, t] V_small[j, t] W_small[k, t] == 1.
                # Can I convert this to the required format?
                # If I set U_req[i*m+j, c] = U_small[i, c] * something?
                # No, the indices are different.
                # The required format has U depending on (i, j).
                # My solution has U depending on i.
                # I can set U_req[i*m+j, c] = U_small[i, c] for all j?
                # Then V_req[j*p+k, c] = V_small[j, c] for all k?
                # And W_req[k*n+i, c] = W_small[k, c] for all i?
                # Let's check:
                # sum_c U_small[i, c] * V_small[j, c] * W_small[k, c] = 1.
                # This matches!
                # So I can expand my small matrices to the required shapes by repeating rows/columns.
                # U_req has n*m rows. Rows i*m ... i*m+m-1 correspond to i.
                # So for row r = i*m + j, set U_req[r, c] = U_small[i, c].
                # Similarly for V and W.
                
                # So the plan is valid.
                
                # Construct the required matrices
                U_req = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        U_req[i * m + j, :] = U_round[i, :]
                        
                V_req = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        V_req[j * p + k, :] = V_round[j, :]
                        
                W_req = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        W_req[k * n + i, :] = W_round[k, :]
                        
                # Final verification
                T_check = np.zeros((n, m, p))
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i, j, k] += U_req[i * m + j, c] * V_req[j * p + k, c] * W_req[k * n + i, c]
                                
                if np.allclose(T_check, T, atol=1e-9):
                    return {
                        "U": U_req.tolist(),
                        "V": V_req.tolist(),
                        "W": W_req.tolist()
                    }
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
