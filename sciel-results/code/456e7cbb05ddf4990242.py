import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
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

    # Try rank 74 with ALS
    target_rank = 74
    
    # Initialize
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1

    # Precompute target tensor values
    # T[i, j, k, i2, k2] = 1 if i2==i and k2==k else 0
    # We solve for U, V, W such that sum_c U[i*m+j, c] V[j*p+k, c] W[k2*n+i2, c] = T
    
    # We can vectorize the ALS steps.
    # For U:
    # For each (i, j), we have a linear system for U[i*m+j, :]
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k2*n+i2, c] = T[i,j,k,i2,k2]
    # This can be written as:
    # U_row @ (V_row_k * W_row_k2) = T_row
    # Where V_row_k is V[j*p+k, :] and W_row_k2 is W[k2*n+i2, :]
    # The product V_row_k * W_row_k2 is an outer product? No, element-wise product.
    # Let M_k_k2 = V[j*p+k, :] * W[k2*n+i2, :] (element-wise)
    # Then sum_c U_c * M_c = T.
    # This is a dot product.
    # So U_row is the solution to U_row @ M^T = T?
    # No, sum_c U_c M_c = T.
    # This is a single equation for each (k, i2, k2).
    # We have p*n equations for each (i,j).
    # And target_rank variables.
    # If p*n < target_rank, it's underdetermined.
    # 5*3 = 15 equations. 74 variables.
    # We can use least squares.

    for iter in range(10):
        if time.time() > deadline - 5:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # Build the system for U[i*m+j, :]
                # Equations for each k, i2, k2
                # sum_c U_c * V[j*p+k, c] * W[k2*n+i2, c] = 1 if i2==i and k2==k else 0
                
                # We can form a matrix A_u of size (p*n, target_rank)
                # and vector b_u of size (p*n)
                A_u = np.zeros((p * n, target_rank))
                b_u = np.zeros(p * n)
                
                idx = 0
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            # Wait, the output index is (i2, k2).
                            # The term is A[i,j] B[j,k].
                            # It contributes to C[i2, k2] only if i2==i and k2==k?
                            # No, C[i, k] = sum_j A[i,j] B[j,k].
                            # So A[i,j] B[j,k] contributes to C[i, k].
                            # So T[i,j,k, i2, k2] = 1 if i2==i and k2==k.
                            
                            # In the sum over c:
                            # sum_c U[i*m+j, c] V[j*p+k, c] W[k2*n+i2, c]
                            # This sum must equal 1 if (i2, k2) == (i, k) and 0 otherwise.
                            
                            # So for fixed i, j, k:
                            # We have equations for all (i2, k2).
                            # If (i2, k2) == (i, k), RHS = 1.
                            # Else RHS = 0.
                            
                            # The coefficient for U_c is V[j*p+k, c] * W[k2*n+i2, c].
                            
                            # So for each (i2, k2), we have one equation.
                            # There are n*p equations.
                            
                            # Let's restructure.
                            # For fixed i, j:
                            # We have p values of k.
                            # For each k, we have n*p equations.
                            # Total p * n * p equations?
                            # No, the equation is for the whole tensor.
                            # The tensor entry T[i,j,k] is a vector of size n*p?
                            # No, T[i,j,k] is a scalar in the decomposition sum?
                            # The decomposition is:
                            # C[i2, k2] = sum_{i,j,k} A[i,j] B[j,k] (sum_c U[i,j,c] V[j,k,c] W[k2,i2,c])
                            # We want C[i2, k2] = sum_j A[i2,j] B[j,k2].
                            # So the coefficient of A[i,j] B[j,k] in C[i2, k2] must be 1 if i2==i and k2==k, else 0.
                            
                            # So for each (i, j, k), the vector (over c) U[i,j,:] * V[j,k,:] * W[:, i2, k2] (element-wise)
                            # must satisfy:
                            # sum_c ... = 1 if (i2,k2)==(i,k)
                            # sum_c ... = 0 otherwise.
                            
                            # This means:
                            # For fixed i, j, k:
                            # Let u = U[i*m+j, :], v = V[j*p+k, :]
                            # For each (i2, k2):
                            # sum_c u_c v_c W[k2*n+i2, c] = delta(i2,i) delta(k2,k)
                            
                            # This is a linear system for u (since v, W are fixed).
                            # We have n*p equations.
                            # Variables: u (size rank).
                            
                            A_u = np.zeros((n * p, target_rank))
                            b_u = np.zeros(n * p)
                            
                            eq_idx = 0
                            for i2 in range(n):
                                for k2 in range(p):
                                    # Coefficients for u_c
                                    # v_c * W[k2*n+i2, c]
                                    coeffs = V[j * p + k, :] * W[k2 * n + i2, :]
                                    A_u[eq_idx, :] = coeffs
                                    if i2 == i and k2 == k:
                                        b_u[eq_idx] = 1.0
                                    else:
                                        b_u[eq_idx] = 0.0
                                    eq_idx += 1
                                    
                            # Solve A_u @ u = b_u
                            # Use least squares
                            u, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                            U[i * m + j, :] = u

        # Similarly update V and W
        # For V[j*p+k, :]:
        # For fixed j, k:
        # Equations for all (i, i2, k2)?
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k2*n+i2, c] = delta(i2,i) delta(k2,k)
        # Here i is fixed? No, i varies in the sum?
        # No, the term is A[i,j] B[j,k].
        # So i is the row of A.
        # For fixed j, k:
        # We have equations for all i, i2, k2.
        # But the LHS depends on U[i*m+j, c] and W[k2*n+i2, c].
        # This is linear in V.
        # Coeffs: U[i*m+j, c] * W[k2*n+i2, c].
        # RHS: 1 if i2==i and k2==k, else 0.
        
        for j in range(m):
            for k in range(p):
                A_v = np.zeros((n * n * p, target_rank))
                b_v = np.zeros(n * n * p)
                
                eq_idx = 0
                for i in range(n):
                    for i2 in range(n):
                        for k2 in range(p):
                            coeffs = U[i * m + j, :] * W[k2 * n + i2, :]
                            A_v[eq_idx, :] = coeffs
                            if i2 == i and k2 == k:
                                b_v[eq_idx] = 1.0
                            else:
                                b_v[eq_idx] = 0.0
                            eq_idx += 1
                            
                v, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V[j * p + k, :] = v

        # For W[k2*n+i2, :]:
        # For fixed i2, k2:
        # Equations for all i, j, k.
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k2*n+i2, c] = delta(i2,i) delta(k2,k)
        # Linear in W.
        # Coeffs: U[i*m+j, c] * V[j*p+k, c].
        # RHS: 1 if i2==i and k2==k, else 0.
        
        for i2 in range(n):
            for k2 in range(p):
                A_w = np.zeros((n * m * p, target_rank))
                b_w = np.zeros(n * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeffs = U[i * m + j, :] * V[j * p + k, :]
                            A_w[eq_idx, :] = coeffs
                            if i2 == i and k2 == k:
                                b_w[eq_idx] = 1.0
                            else:
                                b_w[eq_idx] = 0.0
                            eq_idx += 1
                            
                w, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W[k2 * n + i2, :] = w

    # Check error
    # Compute max error
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for k2 in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k2 * n + i2, :])
                        target = 1.0 if (i2 == i and k2 == k) else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err

    if max_err < 1e-6:
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify snapped solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k2 * n + i2, :])
                            target = 1.0 if (i2 == i and k2 == k) else 0.0
                            err = abs(val - target)
                            if err > max_err:
                                max_err = err
                                
        if max_err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return schoolbook()
