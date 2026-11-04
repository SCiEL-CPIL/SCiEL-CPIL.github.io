"""
Matrix multiplication tensor decomposition for 2x3 * 3x9.
Attempts to find a low-rank decomposition for the 2x3x3 block and replicates it.
Falls back to schoolbook if no improvement is found.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def find_block_decomposition(n_b, m_b, p_b, max_rank, time_limit=5.0):
        """
        Finds a decomposition of rank < max_rank for n_b x m_b x p_b multiplication.
        Returns (U, V, W, rank) or None.
        """
        # Construct the target tensor T of shape (n_b, m_b, p_b)
        # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
        # In the tensor decomposition T = sum U_alpha V_alpha W_alpha,
        # the entry T[i,j,k] is the coefficient.
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k?
        # No, the tensor is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]?
        # Actually, the standard tensor for matmul is T[i,j,k] = 1.
        # Then C[i,k] = sum_j A[i,j] B[j,k] is recovered by contracting U, V, W?
        # Let's check the schoolbook again.
        # Schoolbook U, V, W are 1s at specific spots.
        # The "tensor" T is the identity map on the space of matrices?
        # T_{i,j,k} = 1.
        
        T = np.ones((n_b, m_b, p_b))
        
        best_result = None
        # Try ranks from lower bound up to max_rank
        # Lower bound for 2x3x3 is 9.
        min_rank = max(n_b * m_b, m_b * p_b, p_b * n_b)
        
        for r in range(min_rank, max_rank):
            if time.time() > time_limit:
                break
            # Initialize U, V, W
            U = np.random.randn(n_b * m_b, r)
            V = np.random.randn(m_b * p_b, r)
            W = np.random.randn(p_b * n_b, r)
            
            # ALS
            for it in range(100):
                # Update U
                # Reshape V, W to match
                # We need to solve U * (V W)^T = T_reshaped?
                # T is (n, m, p).
                # Contract T with W to get (n, m, r)?
                # T_ikj = sum_k T[i,j,k] W[k, alpha]?
                # Let's use einsum or manual loops for small sizes.
                
                # Target for U: 
                # sum_alpha U[i, m+j, alpha] V[j, p+k, alpha] W[k, n+i, alpha] = T[i, j, k]
                
                # Fix V, W.
                # For each alpha, the term is u_alpha v_alpha w_alpha.
                # We want to approximate T.
                # This is a multilinear regression.
                
                # Update U:
                # Let R_U[i, j, alpha] = sum_k T[i, j, k] * (V[j, p+k, alpha] * W[k, n+i, alpha])?
                # No, the equation is:
                # T[i, j, k] = sum_alpha U[i*m+j, alpha] V[j*p+k, alpha] W[k*n+i, alpha]
                
                # Fix V, W. Solve for U.
                # For a fixed alpha, we have a variable U_col.
                # The contribution to T from alpha is:
                # sum_alpha U_alpha[i*m+j] * (V_alpha[j*p+k] * W_alpha[k*n+i])
                
                # This is linear in U_alpha.
                # We can set up a least squares problem for each alpha?
                # Or simultaneously?
                # Since it's linear in U for fixed V, W:
                # Let M_U be a matrix where rows are (i, j, k) and cols are alpha.
                # M_U[idx, alpha] = V[j*p+k, alpha] * W[k*n+i, alpha]
                # Then T_flat = U_flat^T M_U^T?
                # T_flat[idx] = sum_alpha M_U[idx, alpha] U_flat[i*m+j, alpha]
                # This is not a simple matrix multiplication because U_flat index depends on i, j.
                
                # We can solve for U by iterating over i, j?
                # For fixed i, j, the equation is:
                # sum_k T[i, j, k] = sum_alpha U[i*m+j, alpha] (sum_k V[j*p+k, alpha] W[k*n+i, alpha])?
                # No, T[i, j, k] is the target for each k.
                
                # Let's use a simple gradient descent or Gauss-Newton?
                # Given the small size, we can use `scipy.optimize.least_squares`?
                # Or just simple ALS with least squares.
                
                # Update U:
                # We want to minimize || T - sum_alpha U_alpha V_alpha W_alpha ||^2
                # Linear in U.
                # Let's construct the design matrix for U.
                # Variables: U[i*m+j, alpha]
                # For each (i, j, k), the coefficient of U[i*m+j, alpha] is V[j*p+k, alpha] * W[k*n+i, alpha].
                
                # We can solve for each (i, j) pair?
                # No, the variables U[i*m+j, alpha] are coupled across k?
                # No, for fixed i, j, the variables are U[i*m+j, :] (vector of length r).
                # The equations are for all k:
                # T[i, j, k] = sum_alpha U[i*m+j, alpha] V[j*p+k, alpha] W[k*n+i, alpha]
                
                # Let A[i, j] be a matrix of size p_b x r.
                # A[i, j][k, alpha] = V[j*p+k, alpha] * W[k*n+i, alpha]
                # Then T[i, j, :] = A[i, j]^T U[i*m+j, :]^T?
                # T[i, j, k] = sum_alpha A[i, j][k, alpha] U[i*m+j, alpha]
                # So T[i, j, :] = A[i, j] @ U[i*m+j, :]
                
                # We can solve U[i*m+j, :] = (A[i, j]^T A[i, j])^-1 A[i, j]^T T[i, j, :]
                
                # Let's implement this.
                
                U_new = np.zeros_like(U)
                for i in range(n_b):
                    for j in range(m_b):
                        # Construct A
                        A = np.zeros((p_b, r))
                        for k in range(p_b):
                            for alpha in range(r):
                                A[k, alpha] = V[j*p_b+k, alpha] * W[k*n_b+i, alpha]
                        
                        # Solve least squares
                        # A @ u = t
                        # u = U[i*m_b+j, :]
                        t = T[i, j, :]
                        try:
                            U[i*m_b+j, :] = np.linalg.lstsq(A, t, rcond=None)[0]
                        except:
                            pass

                # Update V
                V_new = np.zeros_like(V)
                for j in range(m_b):
                    for k in range(p_b):
                        A = np.zeros((n_b, r))
                        for i in range(n_b):
                            for alpha in range(r):
                                A[i, alpha] = U[i*m_b+j, alpha] * W[k*n_b+i, alpha]
                        t = T[:, j, k]
                        try:
                            V[j*p_b+k, :] = np.linalg.lstsq(A, t, rcond=None)[0]
                        except:
                            pass

                # Update W
                W_new = np.zeros_like(W)
                for k in range(p_b):
                    for i in range(n_b):
                        A = np.zeros((m_b, r))
                        for j in range(m_b):
                            for alpha in range(r):
                                A[j, alpha] = U[i*m_b+j, alpha] * V[j*p_b+k, alpha]
                        t = T[i, :, k]
                        try:
                            W[k*n_b+i, :] = np.linalg.lstsq(A, t, rcond=None)[0]
                        except:
                            pass

                # Check error
                error = 0
                for i in range(n_b):
                    for j in range(m_b):
                        for k in range(p_b):
                            val = 0
                            for alpha in range(r):
                                val += U[i*m_b+j, alpha] * V[j*p_b+k, alpha] * W[k*n_b+i, alpha]
                            error += (val - T[i, j, k])**2
                if error < 1e-6:
                    # Found a solution
                    # Check if it can be rounded to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    err_rounded = 0
                    for i in range(n_b):
                        for j in range(m_b):
                            for k in range(p_b):
                                val = 0
                                for alpha in range(r):
                                    val += U_rounded[i*m_b+j, alpha] * V_rounded[j*p_b+k, alpha] * W_rounded[k*n_b+i, alpha]
                                err_rounded += (val - T[i, j, k])**2
                    
                    if err_rounded < 1e-6:
                        return U_rounded, V_rounded, W_rounded, r
                    else:
                        # Try to see if the float solution is valid?
                        # The problem asks for integer/half-integer entries.
                        # So we must return rounded if valid.
                        pass

        return None

    # Try to find a better decomposition for 2x3x3
    block_result = None
    if time.time() < deadline - 10:
        block_result = find_block_decomposition(2, 3, 3, 18, time_limit=10.0)
        
    if block_result is not None:
        U_b, V_b, W_b, r_b = block_result
        # Construct full decomposition
        rank = 3 * r_b
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for b in range(3):
            offset = b * r_b
            for i in range(2):
                for j in range(3):
                    U[i*m + j, offset:offset+r_b] = U_b[i*3 + j, :]
                    
            for j in range(3):
                for k_l in range(3):
                    k_g = 3*b + k_l
                    V[j*p + k_g, offset:offset+r_b] = V_b[j*3 + k_l, :]
                    
            for k_l in range(3):
                for i in range(2):
                    k_g = 3*b + k_l
                    W[k_g*n + i, offset:offset+r_b] = W_b[k_l*2 + i, :]
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = build_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
