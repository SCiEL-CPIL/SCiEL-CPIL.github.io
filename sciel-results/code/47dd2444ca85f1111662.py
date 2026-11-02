import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor entries: T[i, j, k] = 1
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We will search for R from 12 to 17
    # If not found, return schoolbook R=18
    
    def verify(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_find_rank(R, time_limit=5.0):
        start_time = time.time()
        # Number of non-zeros per matrix to try
        # We need at least 1 non-zero per row in U, V, W? Not necessarily.
        # But we need enough degrees of freedom.
        # Total variables: R * (nm + mp + pn)
        # Equations: nm * p
        
        # Strategy: Randomly assign non-zero positions.
        # We want the system to be solvable.
        # We can fix some variables to 1 and solve for others.
        
        # Let's try a structured approach:
        # Use a random subset of columns to be "active".
        # For each active column r, pick random indices for non-zeros in U, V, W.
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize U, V, W as zeros
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            # For each rank index r, pick a random pattern
            # We need to ensure the system is not trivially 0=1.
            # Let's pick 1 non-zero in U, 1 in V, 1 in W for each r?
            # That would be R terms. We need sum of R terms to equal 1 for each (i,j,k).
            # This is like representing the tensor as a sum of rank-1 tensors.
            
            # Better: Pick a random sparse pattern with density d.
            d = 0.2
            for r in range(R):
                # Pick random non-zeros
                # Ensure at least one non-zero in each matrix for this column
                u_idx = np.random.randint(0, nm)
                v_idx = np.random.randint(0, mp)
                w_idx = np.random.randint(0, pn)
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                # Add more random non-zeros
                if random.random() < d:
                    u_idx2 = np.random.randint(0, nm)
                    U[u_idx2, r] = random.choice([-1.0, 1.0])
                if random.random() < d:
                    v_idx2 = np.random.randint(0, mp)
                    V[v_idx2, r] = random.choice([-1.0, 1.0])
                if random.random() < d:
                    w_idx2 = np.random.randint(0, pn)
                    W[w_idx2, r] = random.choice([-1.0, 1.0])
            
            # Now we have a fixed pattern of non-zeros.
            # We need to find values for these non-zeros such that the tensor equation holds.
            # This is a system of bilinear equations, not linear.
            # However, if we fix U and V, it becomes linear in W.
            # Let's fix U and V, and solve for W.
            
            # Build the system for W:
            # For each (i,j,k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # This is linear in W if U and V are fixed.
            # But W[k,i,r] is the variable.
            # The coefficient is U[i,j,r] * V[j,k,r].
            # The variable is W[k*n+i, r].
            
            # We have nm*p equations.
            # Variables are the non-zero entries of W.
            # Let's collect the non-zero indices of W.
            
            # Actually, we can set U and V to random values, and solve for W.
            # But we need W to have the same sparsity pattern?
            # No, we can let W be dense and then sparsify?
            # If we let W be dense, we have pn*R variables.
            # Equations: nm*p = 18.
            # If R=12, variables = 6*12 = 72.
            # We can solve the least squares problem.
            
            # Let's try: Fix U and V randomly (with some sparsity), solve for W (dense).
            # Then check if W can be made sparse? No, the decomposition allows any values.
            # The "rank" is the number of terms R.
            # The entries of U, V, W can be anything.
            # So we just need to find ANY U, V, W of width R that satisfy the equation.
            
            # So:
            # 1. Randomly generate U and V of shape (nm, R) and (mp, R).
            # 2. Solve for W (shape (pn, R)) such that the tensor equation holds.
            # 3. The system is linear in W.
            #    Let A be the matrix of coefficients for W.
            #    For each (i,j,k), the equation is:
            #    sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            #    Let coeff[i,j,k, r] = U[i*m+j, r] * V[j*p+k, r]
            #    Then sum_r coeff[i,j,k, r] * W[k*n+i, r] = 1
            #    This is a linear system C * w_flat = 1, where w_flat is W flattened.
            
            # Build C:
            # C has shape (nm*p, pn*R)
            # Row index: (i*m+j)*p + k ? No, the equations are indexed by (i,j,k).
            # Let's index equations by e = i*m*p + j*p + k.
            # Column index: w_idx = k*n + i, r.
            # col = r * pn + (k*n + i)
            
            C = np.zeros((n*m*p, pn*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i*m*p + j*p + k
                        for r in range(R):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_idx = k*n + i
                            col = r * pn + w_idx
                            C[e, col] = u_val * v_val
            
            b = np.ones(n*m*p)
            
            # Solve C * w_flat = b
            # Use least squares
            try:
                w_flat, residuals, rank, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-5:
                continue
            
            W = w_flat.reshape(pn, R)
            
            # Verify the solution
            if verify(U, V, W):
                # Convert to list of lists
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    # Try ranks from 12 to 17
    for R in range(12, 18):
        result = try_find_rank(R, time_limit=10.0)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
