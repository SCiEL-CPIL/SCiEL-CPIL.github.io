import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # First, build the rank-28 decomposition as a baseline
    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        # Block 1: j=0, k=0..4 (schoolbook 2x1x5)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = -1.0
            col += 1
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            U[(i_off+1)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+0), col] = -1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            W[(k_off+0)*n + (i_off+1), col] = 1.0
            col += 1
            U[(i_off+0)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = -1.0
            W[(k_off+1)*n + (i_off+0), col] = 1.0
            col += 1
            U[(i_off+1)*m + (j_off+0), col] = 1.0
            U[(i_off+0)*m + (j_off+0), col] = -1.0
            V[(j_off+0)*p + (k_off+0), col] = 1.0
            V[(j_off+0)*p + (k_off+1), col] = 1.0
            W[(k_off+1)*n + (i_off+1), col] = 1.0
            col += 1
            U[(i_off+0)*m + (j_off+1), col] = 1.0
            U[(i_off+1)*m + (j_off+1), col] = -1.0
            V[(j_off+1)*p + (k_off+0), col] = 1.0
            V[(j_off+1)*p + (k_off+1), col] = 1.0
            W[(k_off+0)*n + (i_off+0), col] = 1.0
            col += 1

        add_strassen(i_off=0, j_off=1, k_off=0)

        for i in range(n):
            for j in [1, 2]:
                U[i * m + j, col] = 1.0
                V[j * p + 2, col] = 1.0
                W[2 * n + i, col] = 1.0
                col += 1

        add_strassen(i_off=0, j_off=1, k_off=3)

        assert col == rank
        return U, V, W

    U28, V28, W28 = build_rank_28()

    def verify(U_mat, V_mat, W_mat, r):
        for seed in [42, 123, 999]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            C_decomp = np.zeros((n, p))
            for t in range(r):
                u_t = U_mat[:, t].reshape(n, m)
                v_t = V_mat[:, t].reshape(m, p)
                w_t = W_mat[:, t].reshape(p, n)
                C_decomp += np.sum(A_test * u_t) * np.sum(B_test * v_t) * w_t.T
            if not np.allclose(C_test, C_decomp, atol=1e-9):
                return False
        return True

    # Try to find rank-27 by removing one term and optimizing
    # The tensor equation: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where T[i,j,k] = delta(i,j,k) for the matmul tensor
    # Flatten: for each (i,j,k), T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # Build the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # This is the identity-like structure for matmul

    # Actually, the matmul tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the support, but the structure is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    
    # For the decomposition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal T[i,j,k]
    # But T[i,j,k] = 1 for all i,j,k in the matmul tensor
    
    # Wait, the matmul tensor is actually: T[i,j,k] = 1 for all (i,j,k)
    # No, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p
    # Because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor is all ones! The decomposition must satisfy:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # This is a polynomial system. Let's try to solve it with rank 27.
    
    def try_rank_27():
        r = 27
        # Start from rank-28, remove the last term, and optimize
        # We'll try removing different terms and see which gives the best residual
        
        best_result = None
        best_residual = np.inf
        
        # Try removing each of the 28 terms
        for remove_idx in range(28):
            if time.time() > DEADLINE - 30:
                break
                
            # Create initial guess: take rank-28 without term remove_idx
            U_init = np.zeros((n * m, r), dtype=float)
            V_init = np.zeros((m * p, r), dtype=float)
            W_init = np.zeros((p * n, r), dtype=float)
            
            col = 0
            for t in range(28):
                if t == remove_idx:
                    continue
                U_init[:, col] = U28[:, t]
                V_init[:, col] = V28[:, t]
                W_init[:, col] = W28[:, t]
                col += 1
            
            # Flatten parameters
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def residual(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:2*n*m*r].reshape(m*p, r)
                W = x[2*n*m*r:].reshape(p*n, r)
                
                # Compute the tensor from the decomposition
                T_comp = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_comp[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Residual: T_comp - T (where T is all ones)
                return T_comp - np.ones((n, m, p))
            
            def objective(x):
                r_vec = residual(x)
                return np.sum(r_vec ** 2)
            
            # Optimize
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15})
                if result.success and result.fun < 1e-10:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*r].reshape(n*m, r)
                    V_opt = x_opt[n*m*r:2*n*m*r].reshape(m*p, r)
                    W_opt = x_opt[2*n*m*r:].reshape(p*n, r)
                    
                    # Verify
                    if verify(U_opt, V_opt, W_opt, r):
                        return U_opt, V_opt, W_opt, r
                elif result.fun < best_residual:
                    best_residual = result.fun
                    best_result = (result.x, remove_idx)
            except Exception:
                pass
        
        return None
    
    # Try rank 27
    rank27_result = try_rank_27()
    
    if rank27_result is not None:
        U_opt, V_opt, W_opt, r = rank27_result
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Fallback to rank 28
    if verify(U28, V28, W28, 28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Final fallback: schoolbook rank-30
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col_sb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col_sb] = 1.0
                V_sb[j * p + k, col_sb] = 1.0
                W_sb[k * n + i, col_sb] = 1.0
                col_sb += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
