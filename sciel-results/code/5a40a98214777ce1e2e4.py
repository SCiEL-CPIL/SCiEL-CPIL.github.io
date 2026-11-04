import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # First, let's try to find a lower-rank decomposition
    # The tensor rank is at least max(rank of matricizations)
    # A-mode matricization: (n*m) x (m*p * p*n) = 10 x 250, rank <= 10
    # B-mode matricization: (m*p) x (n*m * p*n) = 25 x 100, rank <= 25  
    # C-mode matricization: (p*n) x (n*m * m*p) = 10 x 250, rank <= 10
    # So tensor rank >= 25
    
    # Let's try to find decompositions for ranks from 25 to 49
    # We'll use a structured approach with local search
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is exact (within tolerance)"""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Check shapes
        if U.shape != (n*m, U.shape[1]):
            return False
        if V.shape != (m*p, V.shape[1]):
            return False
        if W.shape != (p*n, W.shape[1]):
            return False
        if not (U.shape[1] == V.shape[1] == W.shape[1]):
            return False
        
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def try_structured_rank(r, n, m, p, time_budget=10.0):
        """
        Try to find a decomposition of rank r using structured initializations
        and local search.
        """
        if time.time() - start_time > time_budget:
            return None
        
        # Try different structured initializations
        initializations = []
        
        # Initialization 1: Random
        initializations.append(('random', np.random.randn(n*m, r), np.random.randn(m*p, r), np.random.randn(p*n, r)))
        
        # Initialization 2: Based on Kronecker structure
        # For n=2, try to exploit the 2-row structure
        if n == 2:
            # Try: U[i*m+j, c] = a_j * b_{i,c}, V[j*p+k, c] = c_{jk,c}, W[k*n+i, c] = d_{ki,c}
            # This is getting complex, let's try simpler patterns
            
            # Pattern: split columns into groups for different i values
            r_i = r // 2
            r_j = r - r_i
            U_init = np.zeros((n*m, r))
            V_init = np.zeros((m*p, r))
            W_init = np.zeros((p*n, r))
            
            # For i=0, use first r_i columns; for i=1, use last r_j columns
            for j in range(m):
                for c in range(r_i):
                    U_init[0*m+j, c] = 1.0
                    V_init[j*p+c%p, c] = 1.0 if c < p else 0.0
                    W_init[(c%p)*n+0, c] = 1.0
                for c in range(r_j):
                    U_init[1*m+j, r_i+c] = 1.0
                    V_init[j*p+c%p, r_i+c] = 1.0 if c < p else 0.0
                    W_init[(c%p)*n+1, r_i+c] = 1.0
            initializations.append(('split_i', U_init, V_init, W_init))
        
        # Try each initialization
        for name, U_init, V_init, W_init in initializations:
            if time.time() - start_time > time_budget:
                break
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            # Local search: try to reduce the error
            lr = 0.01
            for iteration in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                # Compute gradient of sum of squared errors
                U_grad = np.zeros_like(U_cur)
                V_grad = np.zeros_like(V_cur)
                W_grad = np.zeros_like(W_cur)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            prod = V_cur[b_idx, :] * W_cur[c_idx, :]
                            val = np.dot(U_cur[a_idx, :], prod)
                            err = val - 1.0
                            U_grad[a_idx, :] += err * prod
                            V_grad[b_idx, :] += err * U_cur[a_idx, :] * W_cur[c_idx, :]
                            W_grad[c_idx, :] += err * U_cur[a_idx, :] * V_cur[b_idx, :]
                
                # Update
                U_cur -= lr * U_grad
                V_cur -= lr * V_grad
                W_cur -= lr * W_grad
                
                # Check if we've converged
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a_idx = i * m + j
                            b_idx = j * p + k
                            c_idx = k * n + i
                            val = np.dot(U_cur[a_idx, :], V_cur[b_idx, :] * W_cur[c_idx, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-6:
                    break
            
            # Check if the solution is valid and has integer/half-integer entries
            if time.time() - start_time > time_budget:
                break
            
            # Try to round to nearest integer or half-integer
            U_rounded = np.round(U_cur * 2) / 2
            V_rounded = np.round(V_cur * 2) / 2
            W_rounded = np.round(W_cur * 2) / 2
            
            # Convert to lists
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 25 to 49
    best_result = None
    for r in range(25, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        result = try_structured_rank(r, n, m, p, time_budget=5.0)
        if result is not None:
            best_result = result
            break  # Found the lowest rank, we're done
    
    # If no lower-rank decomposition found, use schoolbook
    if best_result is None:
        rank_final = schoolbook_rank
        U = [[0] * rank_final for _ in range(n * m)]
        V = [[0] * rank_final for _ in range(m * p)]
        W = [[0] * rank_final for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
