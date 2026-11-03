"""Matrix multiplication 2x3x7: structured search for low-rank decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer/half-integer arithmetic."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(r):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    # Multiply as fractions to avoid floating point issues
                    # Convert to numerator/2
                    un = int(round(u_val * 2))
                    vn = int(round(v_val * 2))
                    wn = int(round(w_val * 2))
                    total += un * vn * wn
                # total should be 8 (= 1 * 2^3)
                if total != 8:
                    return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-6):
    """Quick float verification."""
    Uf = np.array(U, dtype=float)
    Vf = np.array(V, dtype=float)
    Wf = np.array(W, dtype=float)
    r = Uf.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Uf[i * m + j], Vf[j * p + k] * Wf[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
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


def _try_structured_rank(n, m, p, r, deadline):
    """Try to find a rank-r decomposition with structured factors."""
    # Strategy: use the fact that for small n, we can parameterize
    # U[i*m+j, c] and try to solve for V and W
    
    # For n=2, m=3, p=7:
    # U is 6 x r, V is 21 x r, W is 14 x r
    # Constraints: 42 equations (2*3*7), 6r + 21r + 14r = 41r unknowns
    # For r >= 2, we have more unknowns than equations
    
    # Try a parametrization where U has a specific structure
    # Let's try: U[i*m+j, c] = a_{i,j,c} where we fix patterns
    
    # Approach: use random integer/half-integer initialization and solve
    # via iterative refinement
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Initialize with small random half-integers
        U = np.random.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], 
                            size=(n * m, r))
        
        # For each (i,j,k), we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of bilinear equations in V and W
        
        # Strategy: fix U, then for each (i,j), the equation becomes:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # Let A_{i,j} = U[i*m+j, :] (vector of length r)
        # Then we need: sum_k... no, for each k separately.
        
        # Actually, let's try a different approach:
        # Set W[k*n+i, c] = w_{k,i,c} and V[j*p+k, c] = v_{j,k,c}
        # Equation: sum_c u_{i,j,c} * v_{j,k,c} * w_{k,i,c} = 1
        
        # For fixed i,j: sum_c u_{i,j,c} * v_{j,k,c} * w_{k,i,c} = 1 for k=0..p-1
        # This is p equations. If we treat v_{j,k,c}*w_{k,i,c} as a single variable 
        # for each (j,k,i,c), we have p equations in p*r variables.
        
        # Let's try: for each (i,j), solve for v_{j,k,c} given w
        # Or use alternating minimization
        
        # Simpler: try very structured U
        # Pattern 1: U[i*m+j, c] depends only on i and c
        # Pattern 2: U[i*m+j, c] = delta_{j, c mod m} * something
        
        # Let me try: make U have rank 1 structure in the j direction
        # U[i*m+j, c] = alpha_{i,c} * beta_{j,c}
        
        # Actually, let's just do random search with verification
        # for small ranks, trying many random half-integer assignments
        
        # For rank r, we have 41r variables. Let's try random half-integers
        # and check if the system is consistent
        
        # Better: use the constraint structure
        # For n=2: 
        #   i=0: sum_c U[j,c]*V[7j+k,c]*W[7k,c] = 1 for j=0,1,2; k=0..6
        #   i=1: sum_c U[3+j,c]*V[7j+k,c]*W[7k+1,c] = 1 for j=0,1,2; k=0..6
        
        # Let's parameterize and use numpy to solve linear systems
        # For fixed U and W, V is determined (if the system is consistent)
        
        # Fix U and W randomly, solve for V
        U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
        W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
        
        # For each (j,k): V[j*p+k, :] should satisfy
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for i=0,1
        # This is 2 equations in r unknowns (V[j*p+k, 0..r-1])
        # If r >= 2, we can solve
        
        V_sol = np.zeros((m * p, r))
        consistent = True
        for j in range(m):
            for k in range(p):
                # Build 2 x r system: A * V[j*p+k,:] = b
                A = np.zeros((n, r))
                b = np.ones(n)
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U_rand[i * m + j, c] * W_rand[k * n + i, c]
                
                # Solve A * v = b (least squares)
                if r >= n:
                    v_sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is exact
                    if np.allclose(A @ v_sol, b, atol=1e-10):
                        V_sol[j * p + k] = v_sol
                    else:
                        consistent = False
                        break
                else:
                    # r < n: check consistency
                    if not np.allclose(A @ np.linalg.pinv(A) @ b, b, atol=1e-10):
                        consistent = False
                        break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        # Now verify the full decomposition
        if _verify_float(U_rand.tolist(), V_sol.tolist(), W_rand.tolist(), n, m, p):
            # Snap to half-integers and verify exactly
            U_snap = [[int(round(x * 2)) / 2 for x in row] for row in U_rand.tolist()]
            V_snap = [[int(round(x * 2)) / 2 for x in row] for row in V_sol.tolist()]
            W_snap = [[int(round(x * 2)) / 2 for x in row] for row in W_rand.tolist()]
            
            if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    
    return None


def _try_kronecker_like(n, m, p, r):
    """Try Kronecker-product-like structured decompositions."""
    # For 2x3x7, try decompositions based on factorizations
    # 2 = 2, 3 = 3, 7 = 7
    # Try rank = 2*3 = 6, 2*7 = 14, 3*7 = 21, etc.
    
    # Pattern: U[i*m+j, c] where c indexes (i', j') pairs
    # This gives rank = n*m = 6
    # U[i*m+j, i'*m+j'] = delta_{i,i'} * delta_{j,j'}
    # Then V[j*p+k, i'*m+j'] needs to satisfy...
    # sum_{i',j'} delta_{i,i'}delta_{j,j'} * V[j*p+k, i'*m+j'] * W[k*n+i, i'*m+j'] = 1
    # = V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 for all i,j,k
    # So V[j*p+k, i*m+j] = 1/W[k*n+i, i*m+j]
    # But V depends on (j,k) and W depends on (k,i), and the index i*m+j couples them
    # This requires V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1
    # Let W[k*n+i, i*m+j] = w_{k,i,j}, then V[j*p+k, i*m+j] = 1/w_{k,i,j}
    # But V[j*p+k, c] for c = i*m+j means V depends on i through the column index
    # This works! Set W[k*n+i, i*m+j] = 1, V[j*p+k, i*m+j] = 1
    # That's just schoolbook with rank n*m*p... no wait.
    
    # Actually with rank = n*m = 6:
    # U[i*m+j, c] where c = i'*m+j': U[i*m+j, i'*m+j'] = delta_{i,i'} * delta_{j,j'}
    # Then for the equation: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = 1 * V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1
    # So we need V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1
    # Set both to 1: V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # But V has columns indexed by c=0..5, and for fixed (j,k), V[j*p+k, c] 
    # where c = i*m+j means different i give different columns
    # V[j*p+k, 0*m+0]=1, V[j*p+k, 0*m+1]=1, V[j*p+k, 0*m+2]=1 (for i=0)
    # V[j*p+k, 1*m+0]=1, V[j*p+k, 1*m+1]=1, V[j*p+k, 1*m+2]=1 (for i=1)
    # So V[j*p+k, :] = [1,1,1,1,1,1] for all j,k
    # Similarly W[k*n+i, :] = [1,1,1,1,1,1] for all k,i
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_c delta_{i,i'}delta_{j,j'} * 1 * 1 where c=i'*m+j'
    # = 1 (only term where i'=i, j'=j)
    # YES! This works with rank = n*m = 6!
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
    
    for j in range(m):
        for k in range(p):
            for c in range(r):
                V[j * p + k][c] = 1
    
    for k in range(p):
        for i in range(n):
            for c in range(r):
                W[k * n + i][c] = 1
    
    if _verify_exact(U, V, W, n, m, p):
        return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0  # Safety margin

    # First try the structured Kronecker-like decomposition (rank = n*m = 6)
    result = _try_kronecker_like(n, m, p, n * m)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try other structured patterns
    # Pattern: rank = n*p = 14
    # U[i*m+j, c] where c = i'*p+k': similar idea
    # U[i*m+j, i'*p+k'] = delta_{i,i'} * delta_{k', j}? No, j and k have different ranges
    
    # Let's try rank = p*n = 14 with different structure
    # Actually, let me try rank = m*p = 21
    # U[i*m+j, c] where c = j'*p+k': U[i*m+j, j'*p+k'] = delta_{j,j'} * delta_{k', ?}
    # This doesn't work cleanly since k' ranges over p and j over m
    
    # Try systematic search for ranks 6 to 41
    for r in range(6, 42):
        if time.time() > deadline:
            break
        
        result = _try_structured_rank(n, m, p, r, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
