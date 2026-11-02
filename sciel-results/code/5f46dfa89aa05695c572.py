import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-8):
        R = U.shape[1]
        V_mat = V.reshape(m, p, R)  # V_mat[j, k, r]
        U_mat = U.reshape(n, m, R)  # U_mat[i, j, r]
        W_mat = W.reshape(p, n, R)  # W_mat[k, i, r]
        
        err = 0.0
        for i in range(n):
            if time.time() > deadline:
                return float('inf')
            for k in range(p):
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                W_k_i = W_mat[k, i, :]
                M_ik = U_i @ np.diag(W_k_i) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def get_rank_55_solution():
        """
        Rank-55 decomposition using Strassen-style identities on 2x2 blocks.
        
        The 2x3 matrix A is viewed as [A_left (2x2), A_right (2x1)].
        The 3x10 matrix B is viewed as [B_top (2x10); B_bottom (1x10)].
        
        Product C = A*B = A_left*B_top + A_right*B_bottom (each 2x10).
        
        For each pair of columns (k0, k1) in B_top, we use Strassen-like
        bilinear identities to compute the 2x2 block A_left * B_top[:, (k0,k1)]
        with fewer multiplications than the naive 4.
        """
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Naive part: A_right (2x1) * B_bottom (1x10) -> 2*10 = 20 multiplications
        # For each i in {0,1}, k in {0,...,9}: C[i,k] += A[i,2] * B[2,k]
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2  # A[i, 2]
                v_idx = 2 * p + k  # B[2, k]
                w_idx = k * n + i  # C[k, i]
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        # Strassen part: A_left (2x2) * B_top (2x10)
        # B_top has 10 columns, grouped into 5 pairs of 2 columns.
        # For each pair (k0, k1), we compute A_left * B_top[:, (k0,k1)] (2x2 result)
        # using 7 multiplications instead of 4 (Strassen for 2x2).
        
        # Actually, let me reconsider. The standard Strassen algorithm computes
        # C = A*B where A,B are 2x2 with 7 multiplications.
        # Here for each pair of columns, B_top[:, (k0,k1)] is 2x2.
        # So we have 5 such 2x2 multiplications, each needing 7 mults = 35.
        # Total: 20 + 35 = 55. Correct!
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # Strassen's 7 multiplications for C = A * B where:
            # A = [[a11, a12], [a21, a22]] (this is A_left)
            # B = [[b11, b12], [b21, b22]] (this is B_top[:, (k0,k1)])
            #
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b12-b22)
            # M4 = a22(b21-b11)
            # M5 = (a11+a12)b22
            # M6 = (a21-a11)(b11+b12)
            # M7 = (a12-a22)(b21+b22)
            #
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            #
            # U indices: a_ij -> row i*m + j for i in {0,1}, j in {0,1}
            # V indices: b_ij -> row i*p + k for i in {0,1}, k in {k0,k1}
            # W indices: c_ij -> row k*n + i for k in {k0,k1}, i in {0,1}
            
            # M1 = (a11+a22)(b11+b22)
            U[0*m+0, col] += 1  # a11
            U[1*m+1, col] += 1  # a22
            V[0*p+k0, col] += 1  # b11
            V[1*p+k1, col] += 1  # b22
            W[k0*n+0, col] += 1  # C11 += M1
            W[k1*n+1, col] += 1  # C22 += M1
            col += 1
            
            # M2 = (a21+a22)b11
            U[1*m+0, col] += 1  # a21
            U[1*m+1, col] += 1  # a22
            V[0*p+k0, col] += 1  # b11
            W[k0*n+1, col] += 1  # C21 += M2
            W[k1*n+1, col] += -1  # C22 -= M2
            col += 1
            
            # M3 = a11(b12-b22)
            U[0*m+0, col] += 1  # a11
            V[0*p+k1, col] += 1  # b12
            V[1*p+k1, col] += -1  # -b22
            W[k1*n+0, col] += 1  # C12 += M3
            W[k1*n+1, col] += 1  # C22 += M3
            col += 1
            
            # M4 = a22(b21-b11)
            U[1*m+1, col] += 1  # a22
            V[1*p+k0, col] += 1  # b21
            V[0*p+k0, col] += -1  # -b11
            W[k0*n+0, col] += 1  # C11 += M4
            W[k0*n+1, col] += 1  # C21 += M4
            col += 1
            
            # M5 = (a11+a12)b22
            U[0*m+0, col] += 1  # a11
            U[0*m+1, col] += 1  # a12
            V[1*p+k1, col] += 1  # b22
            W[k0*n+0, col] += -1  # C11 -= M5
            W[k1*n+0, col] += 1  # C12 += M5
            col += 1
            
            # M6 = (a21-a11)(b11+b12)
            U[1*m+0, col] += 1  # a21
            U[0*m+0, col] += -1  # -a11
            V[0*p+k0, col] += 1  # b11
            V[0*p+k1, col] += 1  # b12
            W[k1*n+1, col] += 1  # C22 += M6
            col += 1
            
            # M7 = (a12-a22)(b21+b22)
            U[0*m+1, col] += 1  # a12
            U[1*m+1, col] += -1  # -a22
            V[1*p+k0, col] += 1  # b21
            V[1*p+k1, col] += 1  # b22
            W[k0*n+0, col] += 1  # C11 += M7
            col += 1
        
        return U, V, W

    # First, get the known rank 55 solution
    U_55, V_55, W_55 = get_rank_55_solution()
    err_55 = check_validity_residual(U_55, V_55, W_55)
    
    if err_55 > 1e-6:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a lower rank by local search
    # Strategy: fix U and V structure, try to reduce rank by finding
    # linear dependencies or by optimizing W
    
    best_U, best_V, best_W = U_55, V_55, W_55
    best_rank = 55
    
    # Attempt 1: Try to reduce rank by checking if columns are linearly dependent
    # and can be merged
    def try_reduce_rank(U, V, W, R, max_attempts=100):
        """Try to reduce rank by finding and removing redundant columns."""
        if time.time() > deadline - 10:
            return U, V, W, R
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Try removing each column and checking if the residual is still zero
            for col_idx in range(R):
                if time.time() > deadline - 10:
                    break
                
                U_reduced = np.delete(U, col_idx, axis=1)
                V_reduced = np.delete(V, col_idx, axis=1)
                W_reduced = np.delete(W, col_idx, axis=1)
                
                err = check_validity_residual(U_reduced, V_reduced, W_reduced)
                if err < 1e-8:
                    return U_reduced, V_reduced, W_reduced, R - 1
        
        return U, V, W, R
    
    # Attempt 2: Random perturbation of U and V, then solve for W linearly
    def try_random_perturb_and_solve(U0, V0, W0, R, max_attempts=50):
        """Perturb U and V slightly, then solve for W via least squares."""
        if time.time() > deadline - 10:
            return U0, V0, W0, R
        
        best_solution = (U0, V0, W0, R)
        best_err = check_validity_residual(U0, V0, W0)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Small random perturbation
            scale = 0.01
            U_pert = U0 + scale * np.random.randn(n * m, R)
            V_pert = V0 + scale * np.random.randn(m * p, R)
            
            # Solve for W: for each (k,i), we need U_pert[i] @ diag(W_ki) @ V_pert[:,k].T = I
            # This is: sum_r U_pert[i,j,r] * W_ki[r] * V_pert[j,k,r] = delta_jk... 
            # Actually, for fixed (i,k), the equation is:
            # sum_r U_pert[i, j, r] * W_ki[r] * V_pert[j', k, r] = delta_{j,j'} for all j,j'
            # This is a linear system in W_ki[r]
            
            W_pert = np.zeros((p * n, R))
            valid = True
            for k in range(p):
                for i in range(n):
                    # Build the linear system
                    # Unknowns: W_ki[0], ..., W_ki[R-1]
                    # Equations: for each (j, j') in {0,1,2}^2:
                    # sum_r U_pert[i*m+j, r] * W_ki[r] * V_pert[j'*p+k, r] = delta_{j,j'}
                    
                    A_mat = np.zeros((m * m, R))
                    b_vec = np.zeros(m * m)
                    
                    for j in range(m):
                        for jp in range(m):
                            eq_idx = j * m + jp
                            A_mat[eq_idx, :] = U_pert[i * m + j, :] * V_pert[jp * p + k, :]
                            if j == jp:
                                b_vec[eq_idx] = 1
                    
                    # Solve least squares
                    try:
                        sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W_pert[k * n + i, :] = sol
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                err = check_validity_residual(U_pert, V_pert, W_pert)
                if err < best_err:
                    best_err = err
                    best_solution = (U_pert, V_pert, W_pert, R)
        
        return best_solution
    
    # Try to reduce rank
    U_cur, V_cur, W_cur, R_cur = U_55, V_55, W_55, 55
    
    # First, try direct column removal
    U_cur, V_cur, W_cur, R_cur = try_reduce_rank(U_cur, V_cur, W_cur, R_cur, max_attempts=200)
    
    # Then, try random perturbations
    if time.time() < deadline - 15:
        U_new, V_new, W_new, R_new = try_random_perturb_and_solve(U_cur, V_cur, W_cur, R_cur, max_attempts=30)
        if R_new < R_cur:
            U_cur, V_cur, W_cur, R_cur = U_new, V_new, W_new, R_new
    
    # Final verification
    if R_cur < 55:
        err_final = check_validity_residual(U_cur, V_cur, W_cur)
        if err_final < 1e-6:
            return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
    
    # Return the rank-55 solution
    return {"U": U_55.tolist(), "V": V_55.tolist(), "W": W_55.tolist()}
