import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    
    # Try to find a lower rank
    # We will iterate through candidate ranks
    # Lower bound is likely around 60-80?
    # Let's try from 80 up to rank_sb - 1
    
    # To make it faster, we can just try a few specific ranks
    # Or use a random search for U, V, W and then solve for the third?
    
    # Let's use the ALS approach.
    # We need to be careful with the linear solve.
    # A_{ij} is p x r. We solve A_{ij} x = 1.
    # If r > p, there are infinite solutions.
    # We pick the minimum norm solution x = A^T (A A^T)^{-1} 1.
    # This requires inverting a p x p matrix.
    
    def solve_step(A, b):
        # A is (dim, r), b is (dim,)
        # Solve A x = b
        # x is (r,)
        # Minimum norm solution
        try:
            # Use lstsq
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            return x
        except np.linalg.LinAlgError:
            return None

    def check_decomposition(U, V, W):
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        # Check if sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] == 1 for all i,j,k
        # Vectorized check
        # Reshape U to (n, m, r)
        # Reshape V to (m, p, r)
        # Reshape W to (p, n, r)
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Compute T[i, j, k] = sum_l U[i, j, l] V[j, k, l] W[k, i, l]
        # This is a tensor contraction.
        # We can do this in a loop for clarity and to avoid huge intermediate arrays if needed,
        # but n, m, p are small.
        
        # T[i, j, k]
        # Loop over i, j, k
        # Or use einsum?
        # np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        # Wait, indices:
        # U: i, j, l
        # V: j, k, l  -> V[j, k, l]
        # W: k, i, l  -> W[k, i, l]
        # Result: i, j, k
        
        T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        
        # T should be all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def refine(U, V, W, max_iters=20):
        r = U.shape[1]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # For each i, j
            # A[i, j] is p x r
            # A[i, j, k, l] = V[j, k, l] * W[k, i, l]
            # We need to solve A[i, j] U[i, j]^T = 1 (vector of ones length p)
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Construct A
                    # A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                    V_j = V[j*p:(j+1)*p, :] # (p, r)
                    W_i = W[:, i] # (p, r) ? No, W is (p*n, r).
                    # W index is k*n + i.
                    # So for fixed i, W[k*n+i, :] for k in 0..p-1
                    W_i_mat = np.array([W[k*n + i, :] for k in range(p)]) # (p, r)
                    
                    A = V_j * W_i_mat # (p, r)
                    b = np.ones(p)
                    x = solve_step(A, b)
                    if x is not None:
                        U_new[i*m + j, :] = x
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                    # Sum over i?
                    # Equation: sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] = 1 for all i
                    # For fixed j, k, variables are V[j*p+k, l].
                    # A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                    # A is (n, r). b is (n,)
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    x = solve_step(A, b)
                    if x is not None:
                        V_new[j*p + k, :] = x
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                    # Sum over j?
                    # Equation: sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] = 1 for all j
                    # For fixed k, i, variables are W[k*n+i, l].
                    # A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                    # A is (m, r). b is (m,)
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    x = solve_step(A, b)
                    if x is not None:
                        W_new[k*n + i, :] = x
            W = W_new
            
            # Check convergence
            if check_decomposition(U, V, W):
                # Round to half-integers?
                # The problem asks for integer/half-integer entries.
                # If the solution is exact, it might be rational.
                # We can try to round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # If rounding breaks it, maybe return the float? 
                    # "integer/half-integer entries" is a requirement.
                    # If we can't satisfy it, we might need to search for integer solutions.
                    # But for now, let's assume the float solution is close enough 
                    # or we can try to snap to integers.
                    # Let's try snapping to integers first.
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_decomposition(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    
                    # If neither works, return the float version? 
                    # The evaluator might check for integer/half-integer.
                    # If it fails, score 0.
                    # So we must ensure the returned values are half-integers.
                    # If the ALS solution is not half-integer, we might fail.
                    # We can add a local search around the ALS solution to find a half-integer solution.
                    pass
                    
        return None

    # Try ranks
    # Start from a low rank, go up
    # 3x4x8 = 96.
    # Try 80, 85, 90, 95
    for r in range(80, rank_sb):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(r) # Different seed for different ranks
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Run ALS
        result = refine(U, V, W, max_iters=50)
        if result is not None:
            U_res, V_res, W_res = result
            return {
                "U": U_res.tolist(),
                "V": V_res.tolist(),
                "W": W_res.tolist()
            }

    # Fallback to schoolbook
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
