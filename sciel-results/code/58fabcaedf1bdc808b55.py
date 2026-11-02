import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 22  # Try 22 first, if fails, try 23, etc.
    
    # Tensor T is of size n x m x p
    # T[i, j, k] = 1 if this is a valid multiplication term? 
    # Actually, the tensor represents the map (A, B) -> C.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor T has entries T[i, j, k] = 1 if we are computing C[i, k] using A[i, j] and B[j, k].
    # Wait, the tensor is defined such that T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 for all i, j, k? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i, j] B[j, k] contributes to C[i, k].
    # So the tensor T has T[i, j, k] = 1 if the indices match?
    # The decomposition is:
    # C[i, k] = sum_r U[i, j]_r V[j, k]_r W[i, k]_r ? No.
    
    # Let's re-read the interface.
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R).
    # The equation is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta_{i,k} ? No.
    # The product C[i, k] should be sum_j A[i, j] B[j, k].
    # The tensor T is the matrix multiplication tensor.
    # T[i, j, k] = 1 if (i, j, k) is a valid term?
    # Actually, the tensor T is defined by T[i, j, k] = 1 for all i, j, k?
    # No, the tensor is the linear map from A \otimes B to C.
    # T(A, B) = C.
    # C[i, k] = sum_{j} A[i, j] B[j, k].
    # This can be written as C = T(A, B).
    # The components of T are T_{i, j, k} such that C[i, k] = sum_{i', j', k'} T[i', j', k'] A[i', j'] B[j', k'] \delta_{i, k'}?
    # Actually, the standard form is:
    # C[i, k] = sum_{r} U[i, j]_r V[j, k]_r W[i, k]_r is not quite right because U depends on i, j and V on j, k.
    # The correct form is:
    # C[i, k] = sum_{r} U_{i, j}^{(r)} V_{j, k}^{(r)} W_{i, k}^{(r)} is wrong because j is summed over.
    
    # Let's look at the schoolbook implementation.
    # U[i*m + j][column] = 1
    # V[j*p + k][column] = 1
    # W[k*n + i][column] = 1
    # Sum over columns r:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r]
    # For a specific i, j, k, only one column r (where r corresponds to i, j, k) has non-zero values.
    # That term is 1 * 1 * 1 = 1.
    # So the sum is 1 if (i, j, k) is a valid index triple?
    # But the result should be C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor decomposition expresses the *tensor* of multiplication, not the result for specific A, B.
    # The tensor T has entries T[i, j, k].
    # The decomposition is T[i, j, k] = sum_r U[i, j]_r V[j, k]_r W[i, k]_r.
    # Wait, the indices in U, V, W are flattened.
    # U is (n*m, R). Index is i*m + j.
    # V is (m*p, R). Index is j*p + k.
    # W is (p*n, R). Index is k*n + i.
    
    # So the equation is:
    # T[i, j, k] = sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r].
    # What is T[i, j, k]?
    # In the schoolbook algorithm, T[i, j, k] = 1 for all i, j, k?
    # No, if T[i, j, k] = 1 for all i, j, k, then C[i, k] = sum_j A[i, j] B[j, k] is not directly T.
    # The tensor T is the structure tensor of the multiplication.
    # T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Since C[i, k] = sum_j A[i, j] B[j, k], the term A[i, j] B[j, k] contributes to C[i, k] with coefficient 1.
    # So T[i, j, k] = 1 for all i, j, k?
    # Let's check.
    # If T[i, j, k] = 1, then sum_r U[i, j]_r V[j, k]_r W[i, k]_r = 1.
    # This implies that for any i, j, k, the sum is 1.
    # This is a system of n*m*p equations.
    
    # Let's verify with schoolbook.
    # For a fixed (i, j, k), there is exactly one r such that U[i, j]_r = 1, V[j, k]_r = 1, W[i, k]_r = 1.
    # So the sum is 1.
    # So yes, T[i, j, k] = 1 for all i, j, k.
    
    # So we need to find U, V, W of rank R such that for all i, j, k:
    # sum_r U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] = 1.
    
    n, m, p = DIMENSIONS
    target_rank = 22
    
    # We will use a simple iterative solver.
    # Initialize U, V, W with random values.
    # Iterate:
    # 1. Fix V, W. Solve for U.
    #    For each i, j, we have:
    #    sum_r U[i*m + j, r] (V[j*p + k, r] W[k*n + i, r]) = 1 for all k.
    #    This is a linear system for U[i*m + j, :].
    #    Let A_u[i, j, k, r] = V[j*p + k, r] W[k*n + i, r].
    #    We want sum_r U[i*m + j, r] A_u[i, j, k, r] = 1 for all k.
    #    This is a system of p equations for R unknowns (U[i*m + j, 0..R-1]).
    #    We can solve this in the least squares sense.
    
    # 2. Similarly for V and W.
    
    # Let's implement this.
    
    import time
    start_time = time.time()
    deadline = 140  # 140 seconds
    
    best_rank = 30
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 22 to 25
    for R in range(22, 26):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Iterate
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # For each i, j:
                for i in range(n):
                    for j in range(m):
                        # System: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0..p-1
                        A = np.zeros((p, R))
                        for k in range(p):
                            A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        b = np.ones(p)
                        # Solve A^T A U = A^T b
                        try:
                            U[i*m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        b = np.ones(n)
                        try:
                            V[j*p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                            
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b = np.ones(m)
                        try:
                            W[k*n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                            err += (val - 1.0)**2
                if err < 1e-10:
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break
            if best_rank == R:
                break
        if best_rank == R:
            break
            
    if best_U is None:
        # Fall back to schoolbook
        R = n*m*p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
