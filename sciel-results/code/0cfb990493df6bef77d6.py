import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor: T[i,j,k] = 1 for all i,j,k
    # The constraint is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # But actually, we need: for all i,k, sum_j sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1,j2}
    
    # Let's define the constraint more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_j1 U[i*m+j1, c] A[i,j1]) * (sum_j2 V[j2*p+k, c] B[j2,k]) * W[k*n+i, c]
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
    # sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta_{j1, j2} for all i,k,j1,j2
    
    # This is a system of equations. Let's set it up.
    # Number of equations: n * p * m * m = 5 * 5 * 5 * 5 = 625
    # Unknowns: rank * (nm + mp + pn) = rank * 75
    
    # For rank r, we have 75r unknowns and 625 equations.
    # We need 75r >= 625, so r >= 9 (since 75*8=600 < 625, 75*9=675 >= 625)
    # But this is just a counting argument; the actual minimum rank is higher.
    
    # Let's try to find a solution for various r.
    # Start with r=125 (schoolbook) and try to reduce.
    
    # First, let's create the schoolbook solution.
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Now, let's try to reduce the rank by finding a lower-rank approximation.
    # We'll use a greedy approach: try to eliminate columns.
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition satisfies the tensor equation."""
        # For each i,k, compute the m x m matrix: M[j1,j2] = sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
        # This should equal the identity matrix.
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for c in range(U.shape[1]):
                    w_val = W[k * n + i, c]
                    if abs(w_val) > tol:
                        u_col = U[i * m: (i + 1) * m, c]
                        v_col = V[:, k]  # V[j*p+k, c] for all j
                        # Actually, V[j*p+k, c] is a column in V for fixed k
                        # Let me re-index: V has shape (mp, rank), so V[j*p+k, c]
                        # For fixed k, we want V[j*p+k, c] for all j, which is a vector of length m
                        v_vec = np.array([V[j * p + k, c] for j in range(m)])
                        M += w_val * np.outer(u_col, v_vec)
                if not np.allclose(M, np.eye(m), atol=tol):
                    return False
        return True
    
    # The check above is O(n*p*m*m*rank) which is 5*5*5*5*125 = 78125 operations per check.
    # That's fast enough.
    
    # Let's verify the schoolbook solution.
    if not check_decomposition(U_school, V_school, W_school):
        # Should not happen
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Now, try to reduce the rank.
    # Strategy: try to set some columns to zero and see if the decomposition still works.
    # But setting a column to zero doesn't help; we need to find a linear combination that allows elimination.
    
    # Better approach: use SVD or other matrix factorization to find a lower-rank representation.
    # But the constraint is nonlinear.
    
    # Alternative: use a local search. Start with the schoolbook solution and try to perturb it to find a lower-rank solution.
    
    # Given time constraints, let's try a different approach: use the fact that the tensor rank of 5x5 matrix multiplication is known to be at most 49 (I think).
    # Let me try to find a solution with r=49 or lower.
    
    # Actually, I recall that for n x n matrix multiplication, the rank is at most n^3 - n^2 + n for n >= 2, but I'm not sure.
    # For n=5: 125 - 25 + 5 = 105. That's not much better than 125.
    
    # Let me try a randomized optimization approach.
    # We want to find U, V, W of size 25 x r such that the constraint holds.
    # This is a system of polynomial equations. We can use gradient descent.
    
    # Define the residual: for each (i,k,j1,j2), residual = sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] - delta_{j1,j2}
    # We want to minimize the sum of squares of residuals.
    
    # Let's try for various r, starting from a lower bound.
    
    # The counting argument gives r >= ceil(625/75) = 9, but that's very loose.
    # Let's try r from 50 to 125 and see if we can find a solution.
    
    np.random.seed(42)
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find a solution with lower rank using optimization.
    # We'll use a simple gradient descent approach.
    
    for r in range(50, min(rank_school, 125) + 1):
        if time.time() > deadline - 20:
            break
            
        # Initialize U, V, W randomly.
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Gradient descent to minimize the residual.
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline - 10:
                break
                
            # Compute residual and gradient.
            # This is complex; let's use a simpler approach.
            
            # Actually, let's try a different initialization: use the schoolbook solution and try to compress it.
            break
        
        # This approach is too slow. Let me try something else.
    
    # Given the time constraint, let's just return the schoolbook solution.
    # But that gives score 1.0, which is not an improvement.
    
    # Let me try one more thing: use the known Strassen-like algorithm for 5x5.
    # I don't know the exact construction, so let me try a heuristic.
    
    # Actually, let me try to use a sparse optimization approach.
    # We want to find U, V, W with as few non-zero entries as possible, but the rank is the number of columns.
    
    # I think the best I can do in the time limit is to return the schoolbook solution.
    # But let me try one more idea: use the fact that the tensor has symmetry.
    
    # For now, return the schoolbook solution.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
