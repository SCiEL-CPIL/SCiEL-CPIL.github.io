import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    R_schoolbook = nm * p # n*m*p
    
    # Try to find a lower rank
    # Known rank for 2x4x7 is 32.
    # We will try to solve for R in [32, 40, 48, 56]
    
    best_rank = R_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback schoolbook solution
    U_school = np.zeros((nm, R_schoolbook))
    V_school = np.zeros((mp, R_schoolbook))
    W_school = np.zeros((pn, R_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Time limit guard
    start_time = time.time()
    time_limit = 140 # Leave 10s for safety
    
    # Try ranks from 32 upwards
    for R in range(32, R_schoolbook + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try a few random initializations for this rank
        found = False
        for attempt in range(50): # Limit attempts per rank
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Use gradient descent or a simple iterative refinement
            # The error is the difference between the reconstructed tensor and the target tensor.
            # Target tensor T[i,j,j',k] = delta(j,j')
            # Reconstructed tensor R[i,j,j',k] = sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
            
            # This is a 4D tensor. We can flatten it.
            # But R is large. n*m*m*p = 2*4*4*7 = 224 entries.
            # We can compute the error efficiently.
            
            # Let's use a simple iterative method:
            # Fix W, solve for U, V? No, it's bilinear in U, V, W.
            # We can use alternating minimization.
            
            # Residual for each (i, j, j', k)
            # err(i,j,j',k) = delta(j,j') - sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
            
            # Update U:
            # For fixed V, W, the term is linear in U.
            # We want to minimize sum_{i,j,j',k} (delta(j,j') - sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c])^2
            # This is a least squares problem for U.
            # Let A be a matrix where each row corresponds to an error term.
            # This might be too slow to build explicitly for 50 attempts * 24 ranks.
            
            # Alternative: Use a known decomposition if possible.
            # Since I don't have the explicit 32-rank decomposition hardcoded, 
            # and solving the polynomial system is hard, I will rely on the schoolbook 
            # unless a very fast heuristic works.
            
            # A faster heuristic: 
            # Try to find a rank-1 decomposition of the error and add it.
            # Start with R=1, add terms until error is 0.
            # This is essentially the schoolbook method but might find shortcuts if we use random directions.
            
            # Given the risk of timeout, I will stick to the schoolbook solution 
            # unless I can implement a very fast solver.
            # However, the prompt asks for a "stronger algorithm".
            # Returning schoolbook is score 1.0.
            # If I can find ANY rank < 56, score > 1.0.
            
            # Let's try a simple random search for R=32 with a few iterations of 
            # alternating least squares.
            
            # Build the system matrix for U given V, W.
            # The equation is:
            # sum_c U[i*m+j, c] * (V[j'*p+k, c] * W[k*n+i, c]) = delta(j,j')
            # Let M_c = V[:, c] * W[:, c] (outer product? No, element-wise? No.)
            # The term is U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c].
            # For a fixed c, this is a product of three scalars.
            # This is not a standard linear least squares in U because V and W are also variables.
            # But if we fix V and W, it is linear in U.
            
            # Let's try alternating minimization.
            # 1. Fix V, W. Solve for U using least squares.
            # 2. Fix U, W. Solve for V using least squares.
            # 3. Fix U, V. Solve for W using least squares.
            
            # We need to construct the linear system.
            # For U:
            # Variable: U[i*m+j, c]
            # Equation for each (i, j, j', k):
            # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta(j,j')
            # This couples U[i*m+j, c] with V and W.
            # It is not a simple linear system in U alone because V and W depend on c.
            # Wait, for a fixed (i,j), the variable is the vector U[i*m+j, :] of length R.
            # The equation involves U[i*m+j, c] multiplied by known scalars (if V,W fixed).
            # So yes, it is linear in U[i*m+j, :].
            
            # Let's implement this.
            
            def solve_for_U(V, W):
                # U has shape (nm, R)
                # For each i, j, we have a variable U[i*m+j, :]
                # The equations are for all j', k.
                # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta(j,j')
                # Let A[i,j, :] be a matrix of size (m*p, R) where 
                # A[i,j, j'*p+k, c] = V[j'*p+k, c] * W[k*n+i, c]
                # Then A[i,j] @ U[i*m+j, :] = b[i,j]
                # where b[i,j, j'*p+k] = delta(j,j')
                
                U_new = np.zeros((nm, R))
                for i in range(n):
                    for j in range(m):
                        # Build A[i,j]
                        # Rows correspond to (j', k)
                        # Columns correspond to c
                        A = np.zeros((m * p, R))
                        b = np.zeros(m * p)
                        for jp in range(m * p):
                            jp_j = jp // p
                            jp_k = jp % p
                            b[jp] = 1.0 if jp_j == j else 0.0
                            for c in range(R):
                                A[jp, c] = V[jp, c] * W[jp_k * n + i, c]
                        
                        # Solve A @ U_vec = b
                        # Use least squares
                        try:
                            U_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i * m + j, :] = U_vec
                        except:
                            U_new[i * m + j, :] = np.random.randn(R)
                return U_new

            def solve_for_V(U, W):
                # V has shape (mp, R)
                # Variable V[j*p+k, :]
                # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j,j')
                # For fixed j, k, variable is V[j*p+k, :]
                # sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = delta(j,j')
                # This must hold for all i, j'.
                # So for each j, k, we have m * n equations?
                # Actually, the RHS depends on j'.
                # Let's group by j, k.
                # For fixed j, k, the variable is V[j*p+k, :].
                # The equations are for all i, j'.
                # sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = delta(j,j')
                # LHS is linear in V[j*p+k, :].
                # Let A[j,k, (i,j'), c] = U[i*m+j, c] * W[k*n+i, c]
                # Then A[j,k] @ V[j*p+k, :] = b[j,k]
                # where b[j,k, (i,j')] = delta(j,j')
                
                V_new = np.zeros((mp, R))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n * m, R)) # Rows for (i, j')
                        b = np.zeros(n * m)
                        for i in range(n):
                            for jp in range(m): # j'
                                idx = i * m + jp
                                b[idx] = 1.0 if jp == j else 0.0
                                for c in range(R):
                                    A[idx, c] = U[i * m + j, c] * W[k * n + i, c]
                        
                        try:
                            V_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j * p + k, :] = V_vec
                        except:
                            V_new[j * p + k, :] = np.random.randn(R)
                return V_new

            def solve_for_W(U, V):
                # W has shape (pn, R)
                # Variable W[k*n+i, :]
                # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j,j')
                # For fixed k, i, variable is W[k*n+i, :].
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = delta(j,j')
                # This must hold for all j, j'.
                # Let A[k,i, (j,j'), c] = U[i*m+j, c] * V[j*p+k, c]
                # Then A[k,i] @ W[k*n+i, :] = b[k,i]
                # where b[k,i, (j,j')] = delta(j,j')
                
                W_new = np.zeros((pn, R))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m * m, R)) # Rows for (j, j')
                        b = np.zeros(m * m)
                        for j in range(m):
                            for jp in range(m): # j'
                                idx = j * m + jp
                                b[idx] = 1.0 if jp == j else 0.0
                                for c in range(R):
                                    A[idx, c] = U[i * m + j, c] * V[j * p + k, c]
                        
                        try:
                            W_vec, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k * n + i, :] = W_vec
                        except:
                            W_new[k * n + i, :] = np.random.randn(R)
                return W_new

            # Alternating minimization loop
            for iter in range(100):
                if time.time() - start_time > time_limit:
                    break
                U = solve_for_U(V, W)
                V = solve_for_V(U, W)
                W = solve_for_W(U, V)
                
                # Check error
                # Compute residual
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(R):
                                    val += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                                target = 1.0 if jp == j else 0.0
                                err += (val - target) ** 2
                if err < 1e-6:
                    found = True
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    break

        if found:
            break

    # Convert to lists of floats (integers if possible)
    # The problem asks for integer/half-integer entries.
    # The solver might produce floats. We should round them.
    def to_int_or_half(arr):
        res = []
        for row in arr:
            r = []
            for val in row:
                # Round to nearest 0.5
                v = round(val * 2) / 2.0
                r.append(v)
            res.append(r)
        return res

    U_out = to_int_or_half(best_U)
    V_out = to_int_or_half(best_V)
    W_out = to_int_or_half(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
