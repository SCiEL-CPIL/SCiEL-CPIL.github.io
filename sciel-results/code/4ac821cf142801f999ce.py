"""Program to find rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying 
    a 3x3 matrix by a 3x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 7
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition for n x m x p."""
        rank = n * m * p
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
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
        """
        rank = U.shape[1]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i * m + j]
                    if abs(u_val) < 1e-15:
                        continue
                    for k in range(p):
                        v_val = v_col[j * p + k]
                        w_val = w_col[k * n + i]
                        T_reconstructed[i, j, k] += u_val * v_val * w_val
        
        T_expected = np.ones((n, m, p))
        return np.allclose(T_reconstructed, T_expected, atol=tol)
    
    def find_3x3x3_rank7(max_time=60):
        """
        Search for a rank-7 exact decomposition of 3x3x3 matrix multiplication.
        Returns (U, V, W) each of shape (9, 7) if found, else None.
        """
        start = time.time()
        n = m = p = 3
        target_rank = 7
        
        def objective(params):
            """Objective: sum of squared errors for the tensor decomposition."""
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W = params[2*n*m*target_rank:3*n*m*target_rank].reshape(p*n, target_rank)
            
            error = 0.0
            for r in range(target_rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for i in range(n):
                    for j in range(m):
                        u_val = u_col[i * m + j]
                        if abs(u_val) < 1e-10:
                            continue
                        for k in range(p):
                            v_val = v_col[j * p + k]
                            w_val = w_col[k * n + i]
                            prod = u_val * v_val * w_val
                            error += (prod - 1.0) ** 2  # This is wrong, need to sum first
            return error
        
        # Better objective: compute full tensor then compare
        def objective_full(params):
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W = params[2*n*m*target_rank:3*n*m*target_rank].reshape(p*n, target_rank)
            
            T_reconstructed = np.zeros((n, m, p))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i * m + j, r]
                        if abs(u_val) < 1e-12:
                            continue
                        for k in range(p):
                            v_val = V[j * p + k, r]
                            w_val = W[k * n + i, r]
                            T_reconstructed[i, j, k] += u_val * v_val * w_val
            
            T_expected = np.ones((n, m, p))
            return np.sum((T_reconstructed - T_expected) ** 2)
        
        # Try multiple random starts with Nelder-Mead
        best_result = None
        best_error = np.inf
        
        np.random.seed(42)
        max_attempts = 20
        
        for attempt in range(max_attempts):
            if time.time() - start > max_time - 5:
                break
            
            # Random initialization
            scale = 2.0
            x0 = np.random.randn(3 * n * m * target_rank) * scale
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    objective_full,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_result = result.x
                    
                    if best_error < 1e-6:
                        break
                        
            except Exception as e:
                continue
        
        if best_result is not None and best_error < 1e-6:
            U = best_result[:n*m*target_rank].reshape(n*m, target_rank)
            V = best_result[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
            W = best_result[2*n*m*target_rank:3*n*m*target_rank].reshape(p*n, target_rank)
            
            # Verify
            if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        
        return None
    
    def combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine two decompositions:
        - First: n x m x p1 with rank r1
        - Second: n x m x p2 with rank r2
        Result: n x m x (p1+p2) with rank r1+r2
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        
        # U: (n*m, total_rank)
        U_combined = np.zeros((n * m, total_rank))
        U_combined[:, :r1] = U1
        U_combined[:, r1:] = U2
        
        # V: (m*p, total_rank) where p = p1 + p2
        # V1 is (m*p1, r1), V2 is (m*p2, r2)
        V_combined = np.zeros((m * (p1 + p2), total_rank))
        V_combined[:m*p1, :r1] = V1
        V_combined[m*p1:, r1:] = V2
        
        # W: (p*n, total_rank) where p = p1 + p2
        # W1 is (p1*n, r1), W2 is (p2*n, r2)
        W_combined = np.zeros(((p1 + p2) * n, total_rank))
        W_combined[:p1*n, :r1] = W1
        W_combined[p1*n:, r1:] = W2
        
        return U_combined, V_combined, W_combined
    
    # Strategy: Use rank-7 for 3x3x3 + schoolbook for 3x3x4 = rank 7 + 12 = 19
    # Or: schoolbook for 3x3x7 = rank 63
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 63
    
    # First, get schoolbook as baseline
    U_school, V_school, W_school = schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find rank-7 for 3x3x3
    if time.time() < deadline - 30:
        result_3x3x3 = find_3x3x3_rank7(max_time=60)
        
        if result_3x3x3 is not None:
            U1, V1, W1 = result_3x3x3
            # Schoolbook for 3x3x4
            U2, V2, W2 = schoolbook_decomposition(n, m, 4)
            
            U_comb, V_comb, W_comb = combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, 3, 4)
            
            if verify_decomposition(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_comb, V_comb, W_comb
                best_rank = 7 + 12  # 19
    
    # If we have time, try to improve
    if time.time() < deadline - 10 and best_rank > 19:
        # Try rank-8 for 3x3x3 (might be easier to find)
        # Actually, let's try to find a better decomposition for 3x3x4 or 3x3x7 directly
        pass
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
