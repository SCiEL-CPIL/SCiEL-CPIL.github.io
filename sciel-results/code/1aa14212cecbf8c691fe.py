import numpy as np
from scipy.linalg import null_space

def construct():
    n, m, p = 3, 5, 6
    N1, N2, N3 = n * m, m * p, p * n  # 15, 30, 18
    
    # Build the full tensor T as a 3D array
    T = np.zeros((N1, N2, N3))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We want to find a rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # Strategy: Use a greedy approach to build up the decomposition.
    # Start with r=1 and try to increase rank until we can exactly represent T.
    #
    # For a given rank r, we can try to find U, V, W such that the tensor is matched.
    # This is a non-convex problem, but we can use a greedy construction:
    #
    # 1. Start with the schoolbook decomposition (rank = N1*N2*N3 = 90)
    # 2. Try to merge/remove terms by checking linear dependencies
    #
    # Actually, let's use a different approach: 
    # The tensor T has a special structure. Let's see if we can exploit it.
    #
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # This is the matrix multiplication tensor. The known upper bound for the rank
    # of the matrix multiplication tensor M_{n,m,p} is n*m*p (schoolbook), but
    # there are better bounds. For small dimensions, we can try to find exact decompositions.
    #
    # Let's try a greedy approach:
    # Start with schoolbook, then try to eliminate terms by checking if their contribution
    # can be absorbed by other terms.
    
    # Schoolbook initialization
    rank = N1 * N2 * N3  # 90
    U = np.zeros((N1, rank))
    V = np.zeros((N2, rank))
    W = np.zeros((N3, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Now try to reduce the rank by identifying linearly dependent columns.
    # The key insight: if a column (u_c, v_c, w_c) is such that the outer product
    # u_c * v_c * w_c (as a tensor) can be expressed as a linear combination of
    # other columns' outer products, then we can remove it.
    #
    # More precisely, we have:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # If we remove column c, we need:
    # u_c ⊗ v_c ⊗ w_c = sum_{c' != c} alpha_{c'} * u_{c'} ⊗ v_{c'} ⊗ w_{c'}
    #
    # This is a system of N1*N2*N3 equations in the unknowns alpha_{c'}.
    # But this is generally not possible because the tensor rank is not simply
    # the dimension of the span.
    #
    # Instead, let's use a different strategy: try to find a low-rank decomposition
    # from scratch using a randomized approach.
    
    # Let's try rank values from a lower bound upward.
    # Lower bound: max(N1, N2, N3) = max(15, 30, 18) = 30
    # Upper bound: 90 (schoolbook)
    
    # For a given rank r, we can try to solve for U, V, W using alternating least squares
    # or a similar method, but we need EXACT integer/half-integer solutions.
    #
    # Given the difficulty, let's try a constructive approach:
    # The matrix multiplication tensor M_{3,5,6} might have a known decomposition.
    #
    # Let's try to use the fact that 5 = 2+3 and 6 = 2*3, and see if we can
    # decompose using block structure.
    
    # Actually, let's just try to use a greedy elimination on the schoolbook.
    # For each column, check if removing it and adjusting the others can still
    # reproduce the tensor. This is equivalent to checking if the column is
    # in the span of the others in the tensor space.
    
    # The tensor space is 15*30*18 = 8100 dimensional.
    # We have 90 columns, each contributing a rank-1 tensor.
    # We want to find the minimum number of columns needed.
    
    # Let's try a simple greedy: for each column, check if it can be removed.
    # To check if column c can be removed, we need to see if the tensor
    # u_c ⊗ v_c ⊗ w_c can be expressed as a linear combination of the others.
    
    # This is a linear system: for each (a,b,c) in the tensor,
    # T[a,b,c] = sum_{r} U[a,r] V[b,r] W[c,r]
    #
    # If we fix all columns except c, and want to see if we can adjust the others
    # to still match T, this is a non-linear problem.
    
    # Given the time constraint, let's just return the schoolbook.
    # But wait - the task says to improve. Let me try one more thing.
    
    # Let's try to see if we can use a decomposition with rank < 90 by
    # exploiting the structure. 
    
    # For M_{n,m,p}, one known result is that the rank is at most n*m*p - (n+m+p-2)
    # for some cases, but I'm not sure of the exact formula.
    
    # Let's try a randomized search: pick random U, V of rank r, then solve for W
    # using the constraint that T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # For each c, this gives a linear system in W[:,c].
    
    # Actually, for a fixed U and V, W can be determined if the system is consistent.
    # T[:, :, c] = sum_r U[:, r] V[:, r] W[r, c]
    # This is a matrix equation: T_c = (U * diag(W[:,c]) * V^T) ... no, that's not right.
    
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed c, this is: T[:, :, c] = sum_r W[c,r] * (U[:, r] ⊗ V[:, r])
    # So T[:, :, c] is a linear combination of the N1*N2 rank-1 matrices U[:,r] ⊗ V[:,r].
    
    # Let's try: for a given rank r, randomly initialize U and V, then for each c,
    # solve for W[c, :] such that T[:, :, c] ≈ sum_r W[c,r] (U[:,r] ⊗ V[:,r]).
    # This is a least squares problem.
    
    # Let's try this for r = 30 to 90 and see if we get an exact solution.
    
    np.random.seed(42)
    
    for r in range(30, 91):
        U_rand = np.random.randn(N1, r)
        V_rand = np.random.randn(N2, r)
        
        # For each c in 0..N3-1, solve for W[c, :]
        W_sol = np.zeros((N3, r))
        
        success = True
        for c in range(N3):
            # T[:, :, c] is a N1 x N2 matrix
            T_c = T[:, :, c]
            # We need: T_c = sum_r W[c,r] * (U_rand[:, r] ⊗ V_rand[:, r])
            # Flatten: T_c.flatten() = sum_r W[c,r] * (U_rand[:, r] * V_rand[:, r])
            # Let M be a N1*N2 x r matrix where M[:, r] = (U_rand[:, r] * V_rand[:, r]).flatten()
            M = np.zeros((N1 * N2, r))
            for rr in range(r):
                M[:, rr] = (U_rand[:, rr] * V_rand[:, rr]).flatten()
            
            # Solve M @ W[c, :] = T_c.flatten()
            try:
                W_c, residuals, rank_M, sv = np.linalg.lstsq(M, T_c.flatten(), rcond=None)
                W_sol[c, :] = W_c
                # Check if the solution is exact
                if np.linalg.norm(M @ W_c - T_c.flatten()) > 1e-10:
                    success = False
                    break
            except:
                success = False
                break
        
        if success:
            # Check if the full tensor is reproduced
            T_recon = np.zeros((N1, N2, N3))
            for rr in range(r):
                T_recon += U_rand[:, rr, np.newaxis] * V_rand[np.newaxis, :, rr] * W_sol[np.newaxis, np.newaxis, :, rr]
            if np.linalg.norm(T_recon - T) < 1e-10:
                # Found a decomposition with rank r
                # Convert to list format
                return {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_sol.tolist()
                }
    
    # If we didn't find a better decomposition, return schoolbook
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
