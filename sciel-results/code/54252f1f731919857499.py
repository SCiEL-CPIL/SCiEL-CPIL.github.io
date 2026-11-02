import numpy as np
import time
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor T[I, J, K] is 1 if (I, J, K) corresponds to (i, j, k)
    # where I = i*m+j, J = j*p+k, K = k*n+i
    # We want to find U (nm x R), V (mp x R), W (pn x R)
    # such that T[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
    
    # Strategy:
    # We know the rank is likely small. The schoolbook rank is n*m*p = 18.
    # Let's try to find a decomposition with rank R.
    # 
    # A direct SDP formulation for tensor rank is non-trivial because the
    # variables are the entries of U, V, W.
    # However, we can use the "Gram matrix" approach if we fix one matrix.
    # Or, we can use a simpler algebraic approach:
    # 
    # Let's try to find a decomposition where U, V, W have integer/half-integer entries.
    # A very effective method for small matrices is to use the "Strassen-like"
    # identity search or simply try to solve the linear system if we fix the structure.
    #
    # Actually, there is a known result for 2x3x3 matrix multiplication.
    # The rank is believed to be 17 or less.
    # 
    # Let's try a different strategy:
    # 1. Start with the schoolbook decomposition (rank 18).
    # 2. Try to merge terms or find a linear dependence.
    # 3. Alternatively, use a randomized algebraic method:
    #    Assume U, V, W have entries in {0, 1/2, 1, 3/2, ...}.
    #    This is a combinatorial search.
    #
    # Let's try a more robust numerical approach:
    # Use `scipy.optimize.differential_evolution` or `basinhopping`?
    # No, the landscape is non-convex.
    
    # Let's try to construct a rank 17 decomposition directly.
    # The tensor is T. We want T = sum_{r=1}^R u_r v_r w_r.
    # This is equivalent to finding a rank-R matrix completion of the unfolding?
    # No, it's a 3-way tensor.
    
    # Let's use the "Pencil Method" or "Tensor Train" ideas?
    # 
    # Actually, let's look at the constraints.
    # There are 18 non-zero entries in the tensor.
    # The number of free parameters in a rank R decomposition is R*(nm + mp + pn) - 2*R (scaling)
    # = R*(6 + 9 + 6) - 2R = 21R - 2R = 19R.
    # We have 18 equations (the non-zero entries) and (nm*mp*pn - 18) zero constraints.
    # The zero constraints are the hard part.
    
    # Let's try to find a solution with R=17.
    # We can use the fact that for small ranks, we can often find solutions
    # by solving a system of polynomial equations.
    # 
    # Alternative: Use the "Hyperdeterminant" or "Resultant" methods? Too complex.
    
    # Let's try a heuristic: 
    # 1. Generate a random rank-17 decomposition that approximately satisfies the non-zero constraints.
    # 2. Project onto the manifold of rank-17 tensors.
    # 3. Check if the zero constraints are satisfied.
    
    # A better idea: Use the "Strassen" algorithm for 2x2x2 (rank 7) and extend?
    # 2x3x3 is not a direct extension.
    
    # Let's try to find a rank 17 solution by modifying the schoolbook rank 18 solution.
    # The schoolbook solution uses 18 terms:
    # T = sum_{i,j,k} E_{i,j} E_{j,k} E_{k,i} (conceptually)
    # In the flattened form, the 18 terms are:
    # r_{i,j,k} = U_{i,j} V_{j,k} W_{k,i}
    
    # If we can find a linear combination of 17 of these that equals the 18th?
    # No, they are linearly independent in the tensor space?
    # The tensor rank is the minimal number of rank-1 tensors.
    
    # Let's try a randomized search for a rank 17 solution with integer/half-integer entries.
    # We will search for U, V, W with entries in {0, 0.5, 1, 1.5, 2, -0.5, -1, ...}
    # This is a large search space.
    
    # Instead, let's use a numerical solver with a good initialization.
    # Initialization: Take the schoolbook rank 18, and try to zero out one column
    # by adjusting the others.
    
    # Let's try R=17.
    # We will use `scipy.optimize.least_squares` but with a specific initialization
    # derived from the schoolbook decomposition.
    
    # Schoolbook decomposition:
    # U has 1s at (i*m+j, i*m+j)
    # V has 1s at (j*p+k, i*m+j) -- wait, indices don't match directly.
    
    # Let's build the schoolbook matrices explicitly.
    # Rank R = 18.
    # For each (i, j, k):
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    # where r is the index of (i, j, k).
    
    U_school = np.zeros((nm, 18))
    V_school = np.zeros((mp, 18))
    W_school = np.zeros((pn, 18))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U_school[I, r] = 1
                V_school[J, r] = 1
                W_school[K, r] = 1
                r += 1
                
    # Now, we want to see if we can reduce the rank to 17.
    # This is equivalent to finding a linear dependence among the 18 rank-1 tensors.
    # T = sum_{r=1}^{18} u_r v_r w_r.
    # If there exist coefficients c_r such that sum c_r (u_r v_r w_r) = 0,
    # and not all c_r are zero, we might be able to eliminate one term?
    # No, that's not how rank works.
    
    # Let's try to find a rank 17 decomposition directly using optimization.
    # We will use the "BFGS" or "L-BFGS-B" method on the residual.
    # To handle the non-convexity, we will use multiple random restarts.
    # To encourage integer/half-integer solutions, we can add a penalty term
    # or simply round and check.
    
    def residual(params, R):
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        
        # Compute tensor
        T = np.einsum('ir,jr,kr->IJK', U, V, W)
        T_flat = T.flatten()
        
        # Target tensor
        # We only need to check the non-zero entries and a subset of zeros
        # to save computation, but for accuracy we should check all.
        # 324 entries is small.
        
        # Create target tensor
        targets = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    targets[I, J, K] = 1.0
                    
        return (T_flat - targets.flatten()).flatten()

    # Try R=17
    R = 17
    num_vars = R * (nm + mp + pn)
    
    # We will use `scipy.optimize.minimize` with method 'BFGS' or 'L-BFGS-B'
    # and a custom objective function.
    # However, the landscape is very rugged.
    
    # Let's try a different approach:
    # Use the fact that the tensor is sparse.
    # We can solve for W given U and V.
    # T[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
    # For fixed I, J, this is a linear system in W[:, r] for the K's that are non-zero.
    # But W is shared across all I, J.
    
    # Let's stick to the optimization approach but with a smarter initialization.
    # Initialize with the schoolbook solution but with one column set to zero
    # and the others adjusted to compensate.
    
    # Actually, let's just try random restarts with `least_squares` again,
    # but with a tighter tolerance and more restarts, and a better initial guess.
    # The previous program used `least_squares` with `method='lm'`.
    # LM is good for non-linear least squares.
    
    # Let's try R=17 with 50 restarts.
    # If it fails, try R=18 (which we know works).
    
    # To ensure we return a valid solution, we will keep the schoolbook solution as fallback.
    
    best_solution = None
    best_rank = 18
    
    # Fallback: Schoolbook
    U_fallback = U_school.tolist()
    V_fallback = V_school.tolist()
    W_fallback = W_school.tolist()
    
    # Try R=17
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + restart)
        
        # Initialization: Random
        x0 = np.random.randn(num_vars) * 0.5
        
        # Add a small bias towards the schoolbook structure?
        # Maybe mix random and schoolbook?
        # Let's try just random first.
        
        try:
            from scipy.optimize import least_squares
            sol = least_squares(residual, x0, args=(R,), method='lm', max_nfev=2000)
        except Exception:
            continue
            
        if sol.cost < 1e-10:
            # Check for half-integer property
            U = sol.x[:nm*R].reshape(nm, R)
            V = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
            W = sol.x[nm*R+mp*R:].reshape(pn, R)
            
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify
            T_round = np.einsum('ir,jr,kr->IJK', U_round, V_round, W_round)
            targets = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        targets[I, J, K] = 1.0
                        
            if np.allclose(T_round, targets, atol=1e-9):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
            # If rounding fails, try to find a scaling
            # This is unlikely to work with random initialization.
            
    # If R=17 fails, return the schoolbook solution (R=18)
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
