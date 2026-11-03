"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Build the tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find lower rank decompositions
    # Strategy: for a given rank r, we need U (16 x r), V (32 x r), W (32 x r)
    # such that for all (a,b,c) where T[a,b,c]=1: sum_d U[a,d]*V[b,d]*W[c,d] = 1
    # and for all (a,b,c) where T[a,b,c]=0: sum_d U[a,d]*V[b,d]*W[c,d] = 0
    
    # This is a system of polynomial equations. Let's try small ranks.
    
    # For each candidate rank, try random initializations and gradient descent
    # or use a more structured approach.
    
    # Actually, let's try a smarter approach: use the fact that the tensor has structure.
    # The "middle" index j couples U and V.
    
    # Let's try rank values from 127 down to some minimum
    # Use a simple iterative method: fix two matrices, solve for the third
    
    # For a given r, set up the problem:
    # We want U (16xr), V (32xr), W (32xr)
    # Constraint: for each (i,j,k) with T=1: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # For each (i,j,k) with T=0: same sum = 0
    
    # Let's try a few ranks with random initialization and alternating least squares
    
    np.random.seed(42)
    
    for r in range(rank_school - 1, max(60, rank_school - 20), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Alternating minimization
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix U, V, solve for W
                # For each c in p*n: W[c] = argmin sum_{a,b: T[a,b,c]=1} (1 - sum_d U[a,d]*V[b,d]*W[c,d])^2
                # + sum_{a,b: T[a,b,c]=0} (sum_d U[a,d]*V[b,d]*W[c,d])^2
                # This is a quadratic in W[c]
                
                # Actually, let's do it more carefully.
                # Let M_ab = U[a,:] * V[b,:] (element-wise, shape (r,))
                # Then for fixed a,b: sum_d M_ab[d] * W[c,d] is the contribution to T[a,b,c]
                
                # Let's build the system differently.
                # For each (a,b) pair, define S_ab = U[a,:] * V[b,:] (element-wise product, shape r)
                # Then the prediction for T[a,b,c] is dot(S_ab, W[c,:])
                
                # We want: for all (a,b,c): dot(S_ab, W[c,:]) = T[a,b,c]
                
                # This is a linear system in W if we fix U and V!
                # Each (a,b) gives us r equations... no wait.
                
                # Actually for fixed U,V: we need W[c,:] such that for all (a,b):
                # dot(U[a,:]*V[b,:], W[c,:]) = T[a,b,c]
                
                # This is: for each c, W[c,:] is a vector of length r, and we have n*m*m*p = 16*32 = 512 
                # linear equations in r unknowns (per c)... but they're coupled.
                
                # Let me reformulate: 
                # Flatten: let w = W.flatten() (length p*n*r = 32*r)
                # Each constraint (a,b,c) gives: sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
                # This is linear in W!
                
                # So for fixed U,V, we can solve a linear system for W.
                
                # Build A matrix: for each (a,b,c) constraint, row is:
                # A[constraint_idx, c*r + d] = U[a,d]*V[b,d] if this is the (a,b,c) constraint
                # But that's 16*32*32 = 16384 rows... too many.
                
                # Better: for fixed U,V, the system for W is:
                # For each (a,b): define vec_ab = U[a,:] * V[b,:] (shape r)
                # Then for each c: dot(vec_ab, W[c,:]) = T[a,b,c]
                # 
                # So for each c, we have n*m = 16 equations in r unknowns (W[c,:]):
                # sum_d vec_ab[d] * W[c,d] = T[a,b,c] for each (a,b)
                # 
                # This is: (V_ba @ W[c,:].T) = T[:, :, c].T ... let me think again.
                
                # Let M = np.zeros((n*m, r)) where M[a, d] = sum_b U[a,d]*V[b,d]... no that's wrong.
                
                # Actually: for fixed a, the equation is:
                # sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                # 
                # For fixed a and c: sum_d (U[a,d] * W[c,d]) * V[b,d] = T[a,b,c] for all b
                # Let x_d = U[a,d] * W[c,d], then: V @ x = T[a,:,c]
                # So x = V^+ @ T[a,:,c] (pseudo-inverse)
                # Then W[c,d] = x_d / U[a,d] (if U[a,d] != 0)
                
                # This is getting complicated. Let me just use numpy's lstsq on the full system.
                
                # For fixed U, V: solve for W
                # Build the linear system:
                # Number of constraints: n*m*p*n = 4*4*8*4 = 512 (one per (i,j,k) tuple, but indexed by (a,b,c))
                # Actually the number of (a,b,c) triples where a in [0,16), b in [0,32), c in [0,32) is 16*32*32 = 16384
                # But most T[a,b,c] = 0.
                
                # Let's just build the full system.
                # A has shape (num_constraints, r*p*n)
                # For constraint (a,b,c): A[idx, c*r + d] = U[a,d]*V[b,d]
                # rhs[idx] = T[a,b,c]
                
                # This is 16384 x (r*32) which for r=100 is 16384 x 3200... too big.
                
                # Alternative: exploit structure. For fixed U,V, the W equations decouple by c!
                # For each c, we have n*m = 16 equations in r unknowns.
                # But we need consistency: the same W[c,:] must work for all (a,b).
                
                # For fixed c: we need W[c,:] such that for all (a,b):
                # dot(U[a,:]*V[b,:], W[c,:]) = T[a,b,c]
                
                # Let's define: for each (a,b), let z_ab = U[a,:] * V[b,:] (shape r)
                # Then: z_ab @ W[c,:] = T[a,b,c]
                
                # This is a linear system: (Z_c) @ W[c,:]^T = T[:, :, c].T
                # where Z_c is the matrix with rows z_ab for all (a,b)
                # But Z_c doesn't depend on c! It's the same for all c.
                
                # So: Z @ W[c,:]^T = T[:, :, c].T for each c
                # where Z has shape (n*m, r) ... wait, no.
                
                # Z has shape (n*m * m, r)? No.
                # For each (a,b) pair (16*32 = 512 pairs), we have z_ab of length r.
                # So Z is 512 x r.
                # And for each c, we solve Z @ W[c,:]^T = T[:, :, c].T (shape 512)
                
                # Wait, T is indexed by (a,b,c) where a in [0,16), b in [0,32), c in [0,32)
                # So T[:, :, c] has shape (16, 32) = 512 elements. Yes!
                
                # So: Z is 512 x r, and for each c: solve Z @ w_c = T_flat[:, c] where w_c = W[c,:]^T
                
                # Let's build Z:
                # Z[ab_idx, d] = U[a,d] * V[b,d] where ab_idx = a*32 + b
                
                # Then for each c: w_c = lstsq(Z, T[:, :, c].flatten())
                # And W[c, :] = w_c
                
                # Similarly, we can alternate:
                # Fix V, W: solve for U
                # Fix U, W: solve for V
                
                # Let's implement alternating least squares
                
                success = False
                for it in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    # Fix U, V: solve for W
                    # Build Z: Z[ab_idx, d] = U[a,d] * V[b,d]
                    Z = np.zeros((n*m * m, r))  # 16*4 = 64 rows? No, 16*32 = 512
                    # Wait: a ranges over n*m = 16, b ranges over m*p = 32
                    # So 16*32 = 512 rows
                    for a in range(n*m):
                        for b in range(m*p):
                            Z[a * (m*p) + b, :] = U[a, :] * V[b, :]
                    
                    # For each c in range(p*n): solve Z @ W[c,:]^T = T[:, :, c].flatten()
                    for c in range(p*n):
                        rhs = T[:, :, c].flatten()
                        sol, residuals, rank_z, sv = np.linalg.lstsq(Z, rhs, rcond=None)
                        W[c, :] = sol
                    
                    # Check if solution is good
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute error
                    max_err = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                pred = np.dot(U[a, :] * V[b, :], W[c, :])
                                err = abs(pred - T[a, b, c])
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-8:
                        success = True
                        break
                    
                    # Fix V, W: solve for U
                    # For fixed V, W: 
                    # For each (a, b, c): sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                    # For fixed a: sum_d U[a,d] * (V[b,d] * W[c,d]) = T[a,b,c] for all (b,c)
                    # Let y_bc = V[b,:] * W[c,:] (shape r)
                    # Then: U[a,:] @ y_bc = T[a,b,c]
                    # For fixed a: we have m*p*n = 4*8*4 = 128 equations in r unknowns
                    # Build Z2: Z2[bc_idx, d] = V[b,d] * W[c,d]
                    for a in range(n*m):
                        Z2 = np.zeros((m*p * p*n, r))  # 32*32 = 1024 rows
                        for b in range(m*p):
                            for c in range(p*n):
                                Z2[b * (p*n) + c, :] = V[b, :] * W[c, :]
                        rhs = T[a, :, :].flatten()
                        sol, residuals, rank_z2, sv = np.linalg.lstsq(Z2, rhs, rcond=None)
                        U[a, :] = sol
                    
                    # Fix U, W: solve for V
                    for b in range(m*p):
                        Z3 = np.zeros((n*m * p*n, r))  # 16*32 = 512 rows
                        for a in range(n*m):
                            for c in range(p*n):
                                Z3[a * (p*n) + c, :] = U[a, :] * W[c, :]
                        rhs = T[:, b, :].flatten()
                        sol, residuals, rank_z3, sv = np.linalg.lstsq(Z3, rhs, rcond=None)
                        V[b, :] = sol
                    
                    # Check convergence
                    max_err = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            for c in range(p*n):
                                pred = np.dot(U[a, :] * V[b, :], W[c, :])
                                err = abs(pred - T[a, b, c])
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < 1e-8:
                        success = True
                        break
                
                if success:
                    found = True
                    break
        
        if found:
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # Try to go lower
            continue
        else:
            # If we couldn't find rank r, try r+1, r+2, ... up to schoolbook
            # But break early if time is running out
            break
    
    # Convert to lists with integer/half-integer entries if possible
    # Check if entries are close to integers or half-integers
    def to_clean(val):
        # Round to nearest integer or half-integer
        nearest_int = round(val)
        nearest_half = round(2 * val) / 2.0
        if abs(val - nearest_int) < 1e-6:
            return int(nearest_int)
        elif abs(val - nearest_half) < 1e-6:
            return nearest_half
        else:
            return val
    
    U_list = [[to_clean(best_U[i, j]) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[to_clean(best_V[i, j]) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[to_clean(best_W[i, j]) for j in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

