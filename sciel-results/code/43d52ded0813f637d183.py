import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W):
    """
    Check the full decomposition identity.
    For all i, j, k, j', k', i':
    The tensor entry T[(i,j), (j',k), (k',i')] must match.
    T is 1 if j==j', k==k', i==i'. Else 0.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    # Precompute the tensor factors for speed?
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # Check support entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
                s = float(np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check non-support entries (should be 0)
    # We only need to check a subset to be fast, but for correctness we should check all.
    # Total entries: (n*m) * (m*p) * (p*n) = (2*3)*(3*6)*(6*2) = 6*18*12 = 1296.
    # This is small.
    for idx_u in range(n*m):
        i = idx_u // m
        j = idx_u % m
        for idx_v in range(m*p):
            j_p = idx_v // p
            k = idx_v % p
            # If j != j_p, it's non-support for this pair?
            # The tensor is T[(i,j), (j',k), (k',i')].
            # Support if j==j', k==k', i==i'.
            # Here we are summing over r.
            # We need to check if the sum is 0 for non-matching indices.
            # We can iterate over all idx_w too.
            for idx_w in range(p*n):
                k_p = idx_w // n
                i_p = idx_w % n
                
                # Check if this is a support entry
                is_support = (j == j_p and k == k_p and i == i_p)
                
                s = float(np.dot(np.multiply(U[idx_u], V[idx_v]), W[idx_w]))
                
                if is_support:
                    if abs(s - 1.0) > 1e-9:
                        return False
                else:
                    if abs(s) > 1e-9:
                        return False
                        
    return True

def _solve_als(n, m, p, rank, iterations=100, restarts=5, deadline=None):
    """
    Try to find a rank decomposition for M_{n,m,p} using ALS.
    """
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for _ in range(restarts):
        if deadline and time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(iterations):
            if deadline and time.time() > deadline:
                break
                
            # Update U
            # For each (i,j), solve sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1 (if support)
            # We construct a linear system A * U_col = b
            # But U is a matrix. We can solve for each row of U independently?
            # No, U[i*m+j] is a row vector of length rank.
            # The constraint involves V[j*p+k] and W[k*n+i].
            # For a fixed (i,j), the constraints are for all k.
            # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{k, k'}? No.
            # For fixed i,j, the support entries are for k=0..p-1?
            # No, for fixed i,j, the term A[i,j] B[j,k] contributes to C[i,k].
            # So for fixed i,j, we have p equations (one for each k).
            # Eq k: sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = 1.
            
            # Let Z_k = V[j,k,:] * W[k,i,:] (element-wise product)
            # Then sum_r U[i,j,r] * Z_k[r] = 1.
            
            # This is a linear system in U[i,j,:].
            # We have p equations. Rank is 'rank'.
            # If p < rank, we have an underdetermined system. We can use least squares.
            
            for i in range(n):
                for j in range(m):
                    # Build A (p x rank) and b (p)
                    A = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        # V index: j*p + k
                        # W index: k*n + i
                        A[k, :] = V[j*p + k] * W[k*n + i]
                    
                    # Solve A * U_row = b
                    # Use lstsq
                    U[i*m + j], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            # For fixed j, constraints from all i, k.
            # Eq (i,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1.
            # Let Y_{i,k} = U[i,j,:] * W[k,i,:]
            # sum_r V[j,k,r] * Y_{i,k}[r] = 1.
            # We have n*p equations for V[j,k,:].
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n*p, rank))
                    b = np.ones(n*p)
                    idx = 0
                    for i in range(n):
                        for kp in range(p):
                            # Wait, the equation is for specific i, k.
                            # The sum is over r.
                            # The LHS is U[i,j,r] * V[j,k,r] * W[k,i,r].
                            # For fixed j,k, V[j,k,r] is the variable.
                            # The coefficient is U[i,j,r] * W[k,i,r].
                            # But this equation only exists if (i,j,k) is a support?
                            # Yes, for all i.
                            # So for fixed j,k, we have n equations (one for each i).
                            pass
                    
                    # Correction:
                    # For fixed j, k, the variables are V[j,k,r].
                    # The constraints are: for all i, sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
                    # So we have n equations.
                    A = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        A[i, :] = U[i*m + j] * W[k*n + i]
                    
                    V[j*p + k], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            # For fixed k, i, variables W[k,i,r].
            # Constraints: for all j, sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
            # m equations.
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        A[j, :] = U[i*m + j] * V[j*p + k]
                    
                    W[k*n + i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Calculate error
        # Check support and non-support
        err = 0.0
        for idx_u in range(n*m):
            i = idx_u // m
            j = idx_u % m
            for idx_v in range(m*p):
                j_p = idx_v // p
                k = idx_v % p
                for idx_w in range(p*n):
                    k_p = idx_w // n
                    i_p = idx_w % n
                    s = float(np.dot(np.multiply(U[idx_u], V[idx_v]), W[idx_w]))
                    is_support = (j == j_p and k == k_p and i == i_p)
                    target = 1.0 if is_support else 0.0
                    err += (s - target) ** 2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    # Try to find a decomposition for M_{2,3,3}
    # Then build M_{2,3,6}
    
    # We assume M_{2,3,6} is two copies of M_{2,3,3}
    # Rank = 2 * Rank(M_{2,3,3})
    
    # Try ranks for M_{2,3,3}: 12, 11, 10...
    # Schoolbook for 2,3,3 is 18.
    
    # We will try to solve for rank 12 for 2,3,3.
    # If successful, we build 2,3,6 with rank 24.
    
    # If we fail, we fall back to schoolbook 36.
    
    U_233, V_233, W_233, err = _solve_als(2, 3, 3, 12, iterations=50, restarts=10, deadline=deadline)
    
    if U_233 is not None and err < 1e-6:
        # Verify the 2,3,3 solution
        # We need to map it to 2,3,6
        
        # U_233 shape (6, 12)
        # V_233 shape (9, 12)
        # W_233 shape (6, 12)
        
        # Build U_236 (12, 24)
        # U depends on i, j. Same for both halves.
        # So U_236[:, 0:12] = U_233
        # U_236[:, 12:24] = U_233
        
        U_236 = np.zeros((n*m, 24))
        U_236[:, 0:12] = U_233
        U_236[:, 12:24] = U_233
        
        # V_236 shape (18, 24)
        # V depends on j, k.
        # First half k=0,1,2 uses V_233.
        # Second half k=3,4,5 uses V_233 (mapped).
        V_236 = np.zeros((m*p, 24))
        for j in range(m):
            for k in range(3):
                V_236[j*p + k, 0:12] = V_233[j*3 + k]
            for k in range(3, 6):
                V_236[j*p + k, 12:24] = V_233[j*3 + (k-3)]
        
        # W_236 shape (12, 24)
        # W depends on k, i.
        # First half k=0,1,2 uses W_233.
        # Second half k=3,4,5 uses W_233 (mapped).
        W_236 = np.zeros((p*n, 24))
        for k in range(3):
            for i in range(n):
                W_236[k*n + i, 0:12] = W_233[k*n + i]
        for k in range(3, 6):
            for i in range(n):
                W_236[k*n + i, 12:24] = W_233[(k-3)*n + i]
        
        # Verify
        if _verify(U_236, V_236, W_236):
            return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
        
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
