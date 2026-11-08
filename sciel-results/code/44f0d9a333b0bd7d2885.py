import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition patterns
    # Each pattern is a list of (index, value) pairs
    # Index encodes local (row, col) as row*2+col
    strassen_U = [
        [(0, 1), (3, 1)],   # M1: (A00+A11)
        [(2, 1), (3, 1)],   # M2: (A10+A11)
        [(0, 1)],           # M3: A00
        [(3, 1)],           # M4: A11
        [(0, 1), (1, 1)],   # M5: (A00+A01)
        [(2, 1), (0, -1)],  # M6: (A10-A00)
        [(1, 1), (3, -1)],  # M7: (A01-A11)
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # M1: (B00+B11)
        [(0, 1)],           # M2: B00
        [(1, 1), (0, -1)],  # M3: (B10-B00)
        [(1, 1), (3, -1)],  # M4: (B01-B11)
        [(3, 1)],           # M5: B11
        [(0, 1), (1, 1)],   # M6: (B00+B10)
        [(3, 1), (0, 1)],   # M7: (B11+B01)
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # M1: C00+=M1, C11+=M1
        [(1, 1), (3, -1)],  # M2: C10+=M2, C11-=M2
        [(2, 1), (3, 1)],   # M3: C10+=M3, C11+=M3
        [(0, 1), (3, 1)],   # M4: C00+=M4, C11+=M4
        [(0, -1), (2, 1)],  # M5: C00-=M5, C10+=M5
        [(3, 1)],           # M6: C11+=M6
        [(0, 1)],           # M7: C00+=M7
    ]
    
    # Build rank-56 baseline: 8 blocks of 7 Strassen terms
    rank_56 = 56
    U56 = [[0] * rank_56 for _ in range(n * m)]
    V56 = [[0] * rank_56 for _ in range(m * p)]
    W56 = [[0] * rank_56 for _ in range(p * n)]
    
    def add_block_to(base_rank, i_offset, j_offset, k_offset, U, V, W):
        for r in range(7):
            cr = base_rank + r
            for idx, val in strassen_U[r]:
                li, lj = divmod(idx, 2)
                gi, gj = li + i_offset, lj + j_offset
                U[gi * m + gj][cr] += val
            for idx, val in strassen_V[r]:
                lj, lk = divmod(idx, 2)
                gj, gk = lj + j_offset, lk + k_offset
                V[gj * p + gk][cr] += val
            for idx, val in strassen_W[r]:
                lk, li = divmod(idx, 2)
                gk, gi = lk + k_offset, li + i_offset
                W[gk * n + gi][cr] += val
    
    blocks = [(j_off, k_off) for j_off in [0, 2] for k_off in [0, 2, 4, 6]]
    for idx, (j_off, k_off) in enumerate(blocks):
        add_block_to(idx * 7, 0, j_off, k_off, U56, V56, W56)
    
    # Verify baseline
    def verify(U, V, W, tol=1e-9):
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    expected = 1.0 if True else 0.0  # delta is always 1 for valid (i,j,k)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify(U56, V56, W56):
        # Fallback: return trivial rank-64 solution
        rank_t = n * m * p
        Ut = [[0] * rank_t for _ in range(n * m)]
        Vt = [[0] * rank_t for _ in range(m * p)]
        Wt = [[0] * rank_t for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    Ut[i * m + j][r] = 1
                    Vt[j * p + k][r] = 1
                    Wt[k * n + i][r] = 1
        return {"U": Ut, "V": Vt, "W": Wt}
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try rank 48, 40, 32 by solving a least-squares problem
    # For a given rank R, we need to find U, V, W such that the tensor equation holds
    
    # Use a structured approach: fix U from Strassen patterns, optimize V and W
    # Or use iterative refinement
    
    # Try rank 48: use 6 Strassen blocks (drop 2 blocks and redistribute)
    # Better: try to merge some Strassen terms across blocks
    
    # Approach: for each candidate rank, set up the nonlinear system and solve
    # Using the fact that we can fix one factor matrix and solve for others
    
    # Let's try a different structure: use 4 blocks of 2x2x4 (Strassen-like for rectangular)
    # Or use the known rank-48 construction if it exists
    
    # For now, try to find a rank-48 solution by:
    # 1. Starting from rank-56
    # 2. Identifying 8 terms that can be eliminated
    
    # The U vectors repeat across blocks. For term r in block b:
    # U depends only on (local_i, local_j) and r, not on block
    # So U vectors for the same r across all 8 blocks are identical
    
    # This means the U matrix has rank at most 7 (in terms of distinct row patterns per r)
    # But we can't simply merge terms because V and W differ
    
    # Let's try a numerical approach: for rank R in [48, 40, 32],
    # initialize from the rank-56 solution (take first R terms) and try to
    # solve the remaining terms to satisfy the constraints
    
    # Actually, a cleaner approach: use the bilinear structure
    # Fix U, then for each r, the contribution to the tensor is:
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # The sum over r must equal the multiplication tensor
    
    # Let's try to find a lower-rank solution using optimization
    # We'll parameterize and use gradient descent
    
    # Given time constraints, let's try a few specific rank values
    # and use a simple iterative method
    
    # For rank R, we can try:
    # - Initialize U, V, W from the first R columns of the rank-56 solution
    # - Iteratively update one factor at a time (block coordinate descent)
    # - Check convergence
    
    best_result = {"U": U56, "V": V56, "W": W56}
    best_rank = rank_56
    
    # Try rank 48
    for target_rank in [48, 40, 36, 32]:
        if time.time() > deadline - 30:
            break
        
        # Initialize from rank-56 (take first target_rank terms)
        U_curr = [[U56[i][r] for r in range(target_rank)] for i in range(n * m)]
        V_curr = [[V56[i][r] for r in range(target_rank)] for i in range(m * p)]
        W_curr = [[W56[i][r] for r in range(target_rank)] for i in range(p * n)]
        
        # Block coordinate descent: fix two factors, solve for the third
        # This is a least-squares problem
        
        # For U: fix V and W, solve for U
        # For each r, U[:,r] should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        # This is nonlinear. Instead, let's use a simpler approach:
        # Try to find if the first target_rank terms of the rank-56 solution
        # can be adjusted to satisfy the constraints
        
        # Use a simple iterative refinement
        # For each iteration, fix V and W, solve for U (linear least squares)
        # Then fix U and W, solve for V
        # Then fix U and V, solve for W
        
        # The linear system for U given V, W:
        # For each (i,j,k): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
        # This is: A_U * u = b where A_U[i*m+j, (i,j,k), r] = V[j*p+k, r] * W[k*n+i, r]
        
        # This is a large system. Let's use numpy least squares.
        
        # Build the linear system for U
        # Number of equations: n*m*p = 64
        # Number of unknowns: n*m*R = 8*R
        
        # A is 64 x (8*R), b is 64
        # A[(i,j,k), (i',j',r)] = 1 if i'=i and j'=j else 0, times V[j*p+k, r] * W[k*n+i, r]
        
        # Actually, let's restructure. For fixed V and W, the equation is:
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Group by (i,j): for each (i,j), we have p equations:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0,...,p-1
        
        # This is a linear system in U[i*m+j, :] of size p x R
        
        # For each (i,j), build matrix M_ij where M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
        # Then solve M_ij @ U[i*m+j, :] = ones(p)
        
        # Do this for all (i,j), then do the same for V and W
        
        converged = False
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Update U: fix V and W
            for i in range(n):
                for j in range(m):
                    # Build p x R matrix
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            M[k, r] = V_curr[j * p + k][r] * W_curr[k * n + i][r]
                    # Solve M @ u = ones(p)
                    try:
                        u_sol, residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_curr[i * m + j] = [float(x) for x in u_sol]
                    except:
                        pass
            
            # Update V: fix U and W
            for j in range(m):
                for k in range(p):
                    # Build n x R matrix
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            M[i, r] = U_curr[i * m + j][r] * W_curr[k * n + i][r]
                    # Solve M @ v = ones(n)
                    try:
                        v_sol, residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_curr[j * p + k] = [float(x) for x in v_sol]
                    except:
                        pass
            
            # Update W: fix U and V
            for k in range(p):
                for i in range(n):
                    # Build m x R matrix
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U_curr[i * m + j][r] * V_curr[j * p + k][r]
                    # Solve M @ w = ones(m)
                    try:
                        w_sol, residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_curr[k * n + i] = [float(x) for x in w_sol]
                    except:
                        pass
            
            # Check convergence
            if verify(U_curr, V_curr, W_curr, tol=1e-6):
                converged = True
                break
        
        if converged:
            # Verify with tighter tolerance
            if verify(U_curr, V_curr, W_curr, tol=1e-9):
                best_result = {"U": U_curr, "V": V_curr, "W": W_curr}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
