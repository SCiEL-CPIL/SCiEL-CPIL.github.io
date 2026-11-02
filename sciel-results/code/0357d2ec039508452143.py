import time
import numpy as np

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    Strategy:
    Decompose the multiplication into two main blocks based on the column index 
    of the second matrix (B) and row index of the first matrix (A) which map 
    to the inner dimension (size 3).
    
    The inner dimension indices j are {0, 1, 2}.
    
    Block 1: j = 2.
    A submatrix is A[:, 2] (size 2x1).
    B submatrix is B[2, :] (size 1x5).
    C contribution is A[:, 2] * B[2, :].
    This is an outer product of a 2-vector and a 5-vector.
    The tensor M_{2,1,5} has rank 10 (since each output element C[i,k] = A[i,2]*B[2,k] 
    requires a unique multiplication term in the sum if we don't combine them, 
    but actually M_{2,1,5} is a sum of 10 rank-1 tensors? 
    Let's check rank of M_{2,1,5}.
    T = sum_{i=0}^1 sum_{k=0}^4 A_i B_k C_{ik}.
    This is the tensor of the bilinear map (x, y) -> x y^T for x in R^2, y in R^5.
    The rank of this tensor is 2 * 5 = 10.
    Each term A[i,2]*B[2,k] corresponds to one rank-1 component.
    So we add 10 columns.
    
    Block 2: j in {0, 1}.
    A submatrix is A[:, 0:2] (size 2x2).
    B submatrix is B[0:2, :] (size 2x5).
    C contribution is A[:, 0:2] * B[0:2, :].
    This is M_{2,2,5}.
    We decompose M_{2,2,5} by splitting the 5 columns of B into three parts:
    k in {0, 1}, k in {2, 3}, k in {4}.
    
    Part 2a: k in {0, 1}. M_{2,2,2}. Rank 7 (Strassen).
    Part 2b: k in {2, 3}. M_{2,2,2}. Rank 7 (Strassen).
    Part 2c: k in {4}. M_{2,2,1}. 
    M_{2,2,1} is multiplying 2x2 by 2x1.
    C is 2x1.
    C[0,0] = A[0,0]B[0,4] + A[0,1]B[1,4]
    C[1,0] = A[1,0]B[0,4] + A[1,1]B[1,4]
    This is 4 multiplications. Rank 4.
    
    Total Rank = 10 (Block 1) + 7 (Part 2a) + 7 (Part 2b) + 4 (Part 2c) = 28.
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    # local indices i, j, k in {0, 1}
    # global indices:
    # i -> global i
    # j -> global j_offset + j
    # k -> global k_offset + k
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10 += M2, C11 -= M2
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00 -= M5, C01 += M5
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        col += 1

    # Block 1: j=2 (A[:, 2] * B[2, :])
    # This is M_{2,1,5}. Rank 10.
    # Terms: A[i, 2] * B[2, k] -> C[k, i]
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j in {0, 1} (A[:, 0:2] * B[0:2, :])
    # Part 2a: k in {0, 1}. M_{2,2,2}. j_offset=0, k_offset=0.
    add_strassen(0, 0, 0)
    
    # Part 2b: k in {2, 3}. M_{2,2,2}. j_offset=0, k_offset=2.
    add_strassen(0, 0, 2)
    
    # Part 2c: k in {4}. M_{2,2,1}. j_offset=0, k_offset=4.
    # Schoolbook for 2x2 * 2x1.
    # A is 2x2 (rows 0,1; cols 0,1). B is 2x1 (rows 0,1; col 4).
    # C is 2x1.
    # C[0,0] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,0] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 1: A[0,0] * B[0,4] -> C[0,0]
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] * B[1,4] -> C[0,0]
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] * B[0,4] -> C[1,0]
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] * B[1,4] -> C[1,0]
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Attempt to find a lower rank decomposition (e.g., rank 27)
    # This is a heuristic search. If it fails, return the rank-28 solution.
    try:
        start_time = time.time()
        max_time = 10.0  # Spend up to 10s on search
        
        # Convert to numpy arrays
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Define the target tensor
        # T[i, j, k] = 1 for all i, j, k
        T = np.ones((n, m, p))
        
        def compute_tensor(U_mat, V_mat, W_mat):
            # Sum over rank dimension
            # U_mat: (n*m, r) -> reshape to (n, m, r)
            # V_mat: (m*p, r) -> reshape to (m, p, r)
            # W_mat: (p*n, r) -> reshape to (p, n, r)
            U_t = U_mat.reshape(n, m, -1)
            V_t = V_mat.reshape(m, p, -1)
            W_t = W_mat.reshape(p, n, -1)
            
            # Contract over r
            # Result[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
            # Note: W_t index is (k, i, r). We need W_t[k, i, r].
            # The current W is indexed by (k*n + i).
            # So W_t[k, i, r] is correct.
            
            # Use einsum for efficiency
            # U_t: (n, m, r)
            # V_t: (m, p, r)
            # W_t: (p, n, r)
            # We want sum_r U_t[i,j,r] V_t[j,k,r] W_t[k,i,r]
            # This is not a standard einsum because of the cyclic indices.
            # Let's expand manually or use a loop if r is small.
            r = U_mat.shape[1]
            result = np.zeros((n, m, p))
            for l in range(r):
                u_col = U_t[:, :, l] # (n, m)
                v_col = V_t[:, :, l] # (m, p)
                w_col = W_t[:, :, l] # (p, n)
                
                # Term: u_col[i,j] * v_col[j,k] * w_col[k,i]
                # This is a 3D tensor.
                # We can compute it as:
                # For each i, j, k:
                # result[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
                
                # Vectorized:
                # u_col[:, :, None] * v_col[None, :, :] * w_col.T[:, :, None]?
                # w_col is (p, n). w_col.T is (n, p).
                # We need w_col[k, i].
                # Let's use broadcasting.
                # u_col[i, j] -> (n, m, 1)
                # v_col[j, k] -> (1, m, p)
                # w_col[k, i] -> (n, 1, p) ? No.
                # w_col[k, i] is indexed by k, i.
                # We want to broadcast to (n, m, p).
                # u_col: (n, m) -> (n, m, 1)
                # v_col: (m, p) -> (1, m, p)
                # w_col: (p, n) -> (n, 1, p) ?
                # w_col[k, i] means row k, col i.
                # If we transpose w_col, we get (n, p).
                # w_col.T[i, k] = w_col[k, i].
                # So w_col.T is (n, p).
                # We want to broadcast w_col.T[i, k] to (n, 1, p).
                
                term = u_col[:, :, np.newaxis] * v_col[np.newaxis, :, :] * w_col.T[:, np.newaxis, :]
                result += term
                
            return result

        # Try rank 27
        # Initialize with rank 28 solution, zero out one column
        U_27 = U_np[:, :27].copy()
        V_27 = V_np[:, :27].copy()
        W_27 = W_np[:, :27].copy()
        
        # Add some noise to escape local minima
        np.random.seed(42)
        U_27 += np.random.normal(0, 0.1, U_27.shape)
        V_27 += np.random.normal(0, 0.1, V_27.shape)
        W_27 += np.random.normal(0, 0.1, W_27.shape)
        
        # ALS Optimization
        best_err = float('inf')
        best_U, best_V, best_W = U_27, V_27, W_27
        
        for iter in range(100):
            if time.time() - start_time > max_time:
                break
                
            # Update U
            # Fix V, W. Minimize || T - sum U V W ||^2
            # This is a linear system for U.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # For fixed i,j, T[i,j,:] = sum_r U[i,j,r] (V[:, :, r] * W.T[r, i, :])?
            # Let's form the matrix A_U for each i,j.
            # A_U[i,j, k, r] = V[j,k,r] * W[k,i,r]
            # Then T[i,j,:] = U[i,j,:] @ A_U[i,j,:,:].T ?
            
            # Simpler: Just do gradient descent or simple ALS if possible.
            # Given time constraints, a simple check if rank 27 is possible via random restarts might be better.
            # But let's try a few ALS steps.
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # A_U[k, r] = V[j, k, r] * W[k, i, r]
                    A_U = np.zeros((p, 27))
                    for k in range(p):
                        for r in range(27):
                            A_U[k, r] = V_27[j*p+k, r] * W_27[k*n+i, r]
                    # Solve U[i,j,:] @ A_U.T = T[i,j,:]
                    # U[i,j,:] is (1, 27). A_U.T is (27, p).
                    # Let x = U[i,j,:]. x @ A_U.T = T[i,j,:]
                    # A_U @ x.T = T[i,j,:].T
                    try:
                        U_27[i*m+j, :] = np.linalg.lstsq(A_U, T[i,j,:].T, rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    # A_V[i, r] = U[i, j, r] * W[k, i, r]
                    A_V = np.zeros((n, 27))
                    for i in range(n):
                        for r in range(27):
                            A_V[i, r] = U_27[i*m+j, r] * W_27[k*n+i, r]
                    # V[j,k,:] @ A_V.T = T[:,j,k]
                    try:
                        V_27[j*p+k, :] = np.linalg.lstsq(A_V, T[:,j,k].T, rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    # A_W[j, r] = U[i, j, r] * V[j, k, r]
                    A_W = np.zeros((m, 27))
                    for j in range(m):
                        for r in range(27):
                            A_W[j, r] = U_27[i*m+j, r] * V_27[j*p+k, r]
                    # W[k,i,:] @ A_W.T = T[i,:,k]
                    try:
                        W_27[k*n+i, :] = np.linalg.lstsq(A_W, T[i,:,k].T, rcond=None)[0]
                    except:
                        pass
                        
            # Check error
            T_approx = compute_tensor(U_27, V_27, W_27)
            err = np.linalg.norm(T - T_approx)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_27.copy(), V_27.copy(), W_27.copy()
                
            if best_err < 1e-6:
                break
                
        if best_err < 1e-6:
            # Found a rank-27 decomposition!
            # Round to integers if close
            U_out = best_U
            V_out = best_V
            W_out = best_W
            
            # Check if entries are close to integers/half-integers
            # If so, round them to get an exact solution
            U_rounded = np.round(U_out * 2) / 2
            V_rounded = np.round(V_out * 2) / 2
            W_rounded = np.round(W_out * 2) / 2
            
            T_rounded = compute_tensor(U_rounded, V_rounded, W_rounded)
            if np.linalg.norm(T - T_rounded) < 1e-6:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
    except Exception as e:
        pass
        
    return {"U": U, "V": V, "W": W}
