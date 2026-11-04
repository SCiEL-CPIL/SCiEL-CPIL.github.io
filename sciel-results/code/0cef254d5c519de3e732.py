import numpy as np
import time
from collections import deque

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s wall-clock

    # Target: try to find rank 32 (or lower) via local search
    # Start from rank 33 solution (Strassen + schoolbook)
    
    rank_start = 33
    U = np.zeros((n * m, rank_start))
    V = np.zeros((m * p, rank_start))
    W = np.zeros((p * n, rank_start))
    
    col = 0
    # Part 1: A_1 (col 0) * B_1 (row 0) - schoolbook
    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
    
    # Part 2: 2x2x2 Strassen on 3 blocks
    for x in range(3):
        c_start = 2 * x
        # M1: (a00+a11)*(b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6+c_start][col] = 1; V[12+c_start+1][col] = 1
        W[2*c_start][col] = 1; W[2*c_start+3][col] = 1
        col += 1
        # M2: (a10+a11)*b00
        U[4][col] = 1; U[5][col] = 1
        V[6+c_start][col] = 1
        W[2*c_start+1][col] = 1; W[2*c_start+3][col] = -1
        col += 1
        # M3: a00*(b01-b11)
        U[1][col] = 1
        V[6+c_start+1][col] = 1; V[12+c_start+1][col] = -1
        W[2*c_start+2][col] = 1; W[2*c_start+3][col] = 1
        col += 1
        # M4: a11*(b10-b00)
        U[5][col] = 1
        V[12+c_start][col] = 1; V[6+c_start][col] = -1
        W[2*c_start][col] = 1; W[2*c_start+1][col] = 1
        col += 1
        # M5: (a00+a01)*b11
        U[1][col] = 1; U[2][col] = 1
        V[12+c_start+1][col] = 1
        W[2*c_start][col] = -1; W[2*c_start+2][col] = 1
        col += 1
        # M6: (a10-a00)*(b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6+c_start][col] = 1; V[6+c_start+1][col] = 1
        W[2*c_start+3][col] = 1
        col += 1
        # M7: (a01-a11)*(b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12+c_start][col] = 1; V[12+c_start+1][col] = 1
        W[2*c_start][col] = 1
        col += 1
    
    # Verify the starting solution is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def tensor_residual(U_mat, V_mat, W_mat):
        """Compute the residual of the tensor decomposition.
        The tensor T[i,j,k] = C[i,j] from A*B where A is n x m, B is m x p
        T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        Actually, the convention is:
        U has shape (n*m, rank) -> U[i*m+j, r]
        V has shape (m*p, rank) -> V[j*p+k, r]
        W has shape (p*n, rank) -> W[k*n+i, r]
        The product C[i,k] = sum_j A[i,j] * B[j,k]
        Tensor: T[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k]
        Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        nm = U_mat.shape[0]
        mp = V_mat.shape[0]
        pn = W_mat.shape[0]
        r = U_mat.shape[1]
        # Compute sum over r of outer products
        residual = np.zeros((n, m, p))
        for rr in range(r):
            u_col = U_mat[:, rr].reshape(n, m)
            v_col = V_mat[:, rr].reshape(m, p)
            w_col = W_mat[:, rr].reshape(p, n)
            # u_col[i,j] * v_col[j,k] * w_col[k,i]
            for i in range(n):
                for j in range(m):
                    if u_col[i,j] != 0:
                        for k in range(p):
                            residual[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        # True tensor
        true_tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    true_tensor[i,j,k] = 1.0  # A[i,j]*B[j,k] where A,B are identity-like
                    # Actually the tensor is T[i,j,k] = 1 if we think of it as
                    # the structure tensor where the product is sum_j A[i,j]B[j,k]
                    # The tensor entries are: T[i,j,k] = 1 for all i,j,k
                    # because A[i,j] and B[j,k] are independent variables
                    # and C[i,k] = sum_j A[i,j] B[j,k]
                    # So the tensor is the "generic" matrix multiplication tensor
                    # T[i,j,k] = delta structure: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
                    # So T[i,j,k] = 1 for all valid i,j,k
        return np.sum((residual - true_tensor)**2)
    
    # Actually let me use a more efficient verification
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        r = U_mat.shape[1]
        nm, mp, pn = U_mat.shape[0], V_mat.shape[0], W_mat.shape[0]
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For the generic matmul tensor, T[i,j,k] = 1 for all i,j,k
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for rr in range(r):
                        actual += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                    err = abs(actual - expected)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    # Check initial solution
    if not verify_decomposition(U_np, V_np, W_np):
        return {"U": [[float(x) for x in U_np[i]] for i in range(n*m)],
                "V": [[float(x) for x in V_np[i]] for i in range(m*p)],
                "W": [[float(x) for x in W_np[i]] for i in range(p*n)]}
    
    # Try local search to reduce rank
    # Strategy: try to eliminate one column at a time by absorbing it into others
    # or by finding a rank-revealing factorization
    
    best_U, best_V, best_W, best_rank = U_np.copy(), V_np.copy(), W_np.copy(), rank_start
    
    # Try to see if rank 32 is possible
    # Approach: use random perturbations and rank-revealing
    # For each pair of columns, try to see if one can be expressed as combination of others
    
    # Try direct rank reduction: remove a column and see if the remaining can be adjusted
    for trial_rank in range(rank_start - 1, max(20, rank_start - 5), -1):
        if time.time() > deadline:
            break
        
        # Try random subsets of columns of size trial_rank
        np.random.seed(42 + trial_rank)
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Randomly select columns to keep
            cols = np.random.choice(rank_start, trial_rank, replace=False)
            U_sub = U_np[:, cols]
            V_sub = V_np[:, cols]
            W_sub = W_np[:, cols]
            
            # Check if this subset still gives a valid decomposition
            if verify_decomposition(U_sub, V_sub, W_sub, tol=1e-10):
                best_rank = trial_rank
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                break
        
        if best_rank < rank_start:
            break
    
    # If we found a better rank, also try local optimization to find even lower
    current_rank = best_rank
    U_cur, V_cur, W_cur = best_U, best_V, best_W
    
    # Try further rank reduction with local search
    while current_rank > 25:
        if time.time() > deadline:
            break
        
        found_improvement = False
        # Try removing each column one at a time
        for col_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            # Remove this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col_idx] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            if verify_decomposition(U_test, V_test, W_test, tol=1e-10):
                U_cur, V_cur, W_cur = U_test, V_test, W_test
                current_rank -= 1
                found_improvement = True
                break
        
        if not found_improvement:
            break
    
    best_rank = current_rank
    best_U, best_V, best_W = U_cur, V_cur, W_cur
    
    # Convert to lists
    U_list = [[float(x) for x in best_U[i]] for i in range(n*m)]
    V_list = [[float(x) for x in best_V[i]] for i in range(m*p)]
    W_list = [[float(x) for x in best_W[i]] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
