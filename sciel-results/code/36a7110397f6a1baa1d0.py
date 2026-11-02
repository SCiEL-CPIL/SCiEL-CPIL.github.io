"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE = 120.0
TIME_LIMIT = 110.0  # safety margin

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = e_{(i,j)} · U[:,c] * e_{(j,k)} · V[:,c] * e_{(k,i)} · W[:,c]
    # Actually, the decomposition is: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that this equals 1 if (i,j,k) corresponds to a valid multiplication index, else 0.
    # Wait, the matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has indices (i, j, k) where T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a batched operation. Let's do it efficiently.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We want for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Reshape approach: 
    # For fixed j, we have U_j of shape (n, rank) where U_j[i, c] = U[i*m+j, c]
    # V_j of shape (p, rank) where V_j[k, c] = V[j*p+k, c]
    # W_j of shape (p*n, rank) -> reshape to (p, n, rank) where W_j[k, i, c] = W[k*n+i, c]
    
    # Let's just loop over j since m is small
    for j in range(m):
        U_j = U_arr[j::m, :]  # shape (n, rank)
        V_j = V_arr[j*p:(j+1)*p, :]  # shape (p, rank)
        W_j = W_arr.reshape(p, n, rank)[j*j*p + ...]  # This is getting complicated
        
    # Simpler: direct computation for all (i,j,k)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Use broadcasting
    # U_idx: for each (i,j), index is i*m+j, shape (n, m)
    # V_idx: for each (j,k), index is j*p+k, shape (m, p)
    # W_idx: for each (k,i), index is k*n+i, shape (p, n)
    
    # Let's compute the tensor directly
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j, :]
                v_val = V_arr[j*p+k, :]
                w_val = W_arr[k*n+i, :]
                T_computed[i, j, k] = np.dot(u_val, v_val * w_val)
    
    # The target tensor is all ones
    T_target = np.ones((n, m, p))
    
    if np.max(np.abs(T_computed - T_target)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use a randomized approach: try various rank values and use optimization
    
    start_time = time.time()
    
    # Try ranks from lower to higher, starting from known lower bounds
    # For 2x3x7, the trivial lower bound is max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    # But we know schoolbook is 42. Let's try to find something better.
    
    # Actually, let's try a different approach: use the fact that we can have fractional entries
    # and try to find a rank-r decomposition using numerical optimization
    
    # For now, let's just return the schoolbook decomposition to be safe
    # But let's try to improve it
    
    # Try rank values from max(n*m, m*p, n*p) to rank_schoolbook
    min_possible_rank = max(n*m, m*p, n*p)
    
    for target_rank in range(min_possible_rank, rank_schoolbook + 1):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Try to find a decomposition of this rank
        # Use random initialization and iterative refinement
        
        best_error = float('inf')
        best_U_r, best_V_r, best_W_r = None, None, None
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            # Random initialization
            U_r = np.random.randn(nm, target_rank) * 0.1
            V_r = np.random.randn(mp, target_rank) * 0.1
            W_r = np.random.randn(pn, target_rank) * 0.1
            
            # Iterative refinement: alternate between fixing two and solving for the third
            # This is similar to ALS (Alternating Least Squares)
            
            for iteration in range(50):
                if time.time() - start_time > TIME_LIMIT:
                    break
                
                # Update U: for each row i*m+j, solve U[i*m+j, :] such that
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is a linear system for each (i,j)
                
                for i in range(n):
                    for j in range(m):
                        # For fixed (i,j), we want:
                        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
                        # Let M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Then we want U[i*m+j, :] * M^T = [1, 1, ..., 1]
                        # So U[i*m+j, :] is the solution to M * U[i*m+j, :].T = ones(p)
                        
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V_r[j*p+k, :] * W_r[k*n+i, :]
                        
                        # Solve M * x = ones(p)
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U_r[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each row j*p+k, solve V[j*p+k, :] such that
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U_r[i*m+j, :] * W_r[k*n+i, :]
                        
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V_r[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each row k*n+i, solve W[k*n+i, :] such that
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                        
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W_r[k*n+i, :] = x
                        except:
                            pass
                
                # Check error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            error += (val - 1.0) ** 2
                
                if error < best_error:
                    best_error = error
                    best_U_r, best_V_r, best_W_r = U_r.copy(), V_r.copy(), W_r.copy()
                
                if error < 1e-6:
                    break
        
        if best_error < 1e-6 and best_U_r is not None:
            # Verify
            if _verify(best_U_r.tolist(), best_V_r.tolist(), best_W_r.tolist(), n, m, p):
                best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
                best_rank = target_rank
                break
    
    # Convert to lists of lists with rational-looking entries
    def to_list(arr):
        # Round to nearest half-integer or integer if close
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Try to round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    new_row.append(rounded)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
