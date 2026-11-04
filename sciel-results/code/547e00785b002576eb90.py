"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import fsolve

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def make_decomposition(U, V, W):
        return {"U": np.asarray(U).tolist(), "V": np.asarray(V).tolist(), "W": np.asarray(W).tolist()}
    
    def verify(U, V, W, tol=1e-6):
        """Check if U,V,W form a valid decomposition."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # Check all constraints
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Constraint 1: diagonal = 1
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
                    
                    # Constraint 2: off-diagonal in j = 0
                    for j2 in range(m):
                        if j2 == j:
                            continue
                        val = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                        if abs(val) > tol:
                            return False
                    
                    # Constraint 3: off-diagonal in k = 0
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k2*n+i, :])
                        if abs(val) > tol:
                            return False
                    
                    # Constraint 4: off-diagonal in i = 0
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                        if abs(val) > tol:
                            return False
        
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return make_decomposition(U, V, W)
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Number of constraints:
    # Constraint 1: n*m*p = 120 equations
    # Constraint 2: n*m*p*(m-1) = 120*4 = 480 equations
    # Constraint 3: n*m*p*(p-1) = 120*5 = 600 equations
    # Constraint 4: n*m*p*(n-1) = 120*3 = 360 equations
    # Total: 120 + 480 + 600 + 360 = 1560 equations
    
    # Number of unknowns for rank r: n*m*r + m*p*r + p*n*r = r*(n*m + m*p + p*n) = r*(20+30+24) = 74r
    
    # For the system to be solvable, we need 74r >= 1560, so r >= 21.08
    # But this is a necessary condition, not sufficient.
    
    # The lower bound is max(n*m, m*p, p*n) = 30
    # So try ranks from 30 upward
    
    def build_equations(params, r):
        """Build the system of equations for rank r."""
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = params[n*m*r+m*p*r:].reshape(p*n, r)
        
        eqs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Constraint 1: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
                    eqs.append(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]) - 1.0)
                    
                    # Constraint 2: sum_r U[i,j,r] V[j2,k,r] W[k,i,r] = 0 for j2 != j
                    for j2 in range(m):
                        if j2 == j:
                            continue
                        eqs.append(np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :]))
                    
                    # Constraint 3: sum_r U[i,j,r] V[j,k2,r] W[k2,i,r] = 0 for k2 != k
                    for k2 in range(p):
                        if k2 == k:
                            continue
                        eqs.append(np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k2*n+i, :]))
                    
                    # Constraint 4: sum_r U[i,j,r] V[j,k,r] W[k,i2,r] = 0 for i2 != i
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        eqs.append(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :]))
        
        return np.array(eqs)
    
    # Try ranks from lower bound upward
    for r in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        
        num_unknowns = 74 * r
        num_eqs = 1560
        
        # Only try if we have enough unknowns
        if num_unknowns < num_eqs:
            continue
        
        # Initialize with schoolbook-like pattern
        params = np.zeros(num_unknowns)
        
        # Set up initial guess: use first r terms of schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U_idx = (i*m + j) * r + col
                    V_idx = (n*m*r) + (j*p + k) * r + col
                    W_idx = (n*m*r + m*p*r) + (k*n + i) * r + col
                    params[U_idx] = 1.0
                    params[V_idx] = 1.0
                    params[W_idx] = 1.0
                    col += 1
                if col >= r:
                    break
            if col >= r:
                break
        
        # Try to solve
        try:
            def equations(x):
                return build_equations(x, r)
            
            # Use fsolve with a timeout-like approach
            solution, info, ier, msg = fsolve(equations, params, full_output=True, maxfev=10000)
            
            if ier == 1 and np.max(np.abs(build_equations(solution, r))) < 1e-6:
                U = solution[:n*m*r].reshape(n*m, r)
                V = solution[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = solution[n*m*r+m*p*r:].reshape(p*n, r)
                
                if verify(U, V, W):
                    best_result = make_decomposition(U, V, W)
                    best_rank = r
                    print(f"Found decomposition with rank {r}")
                    break
        except Exception:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
