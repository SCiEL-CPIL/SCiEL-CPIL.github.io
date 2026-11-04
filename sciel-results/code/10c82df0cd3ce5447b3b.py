"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Fast verification using numpy
    def verify_fast(U, V, W, rank):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Fix W with structured patterns, solve for U and V linearly
    # For each k (column of the 3x8 matrix), W has n=3 rows
    # We can try W patterns where each row has a few non-zero entries
    
    # Generate candidate W matrices with half-integer entries
    # Focus on sparse patterns to make U, V more likely to be half-integer
    
    def generate_W_candidates(target_rank, max_nonzero_per_row=3):
        """Generate W matrices with sparse half-integer entries"""
        candidates = []
        # Try various sparsity patterns
        for nnz in range(1, max_nonzero_per_row + 1):
            # Randomly choose which columns are non-zero for each row
            for _ in range(50):
                W = np.zeros((p * n, target_rank))
                valid = True
                for row in range(p * n):
                    if np.random.rand() < 0.3:
                        # Skip this row (all zeros) - might not work
                        continue
                    # Choose nnz random columns
                    cols = np.random.choice(target_rank, size=nnz, replace=False)
                    for c in cols:
                        # Half-integer values: -1, -0.5, 0.5, 1
                        W[row, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                if np.any(W):
                    candidates.append(W)
        return candidates
    
    def solve_for_UV(W, target_rank):
        """Given W, solve for U and V such that the decomposition holds"""
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        
        # For each c (column of U, V, W), we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # 
        # This is a system of n*m*p = 72 equations with n*m + m*p = 9 + 24 = 33 unknowns per c
        # Wait, that's not right. The equations are coupled across c.
        # 
        # Actually, for fixed c, the contribution is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We need sum_c of these to equal the multiplication tensor.
        #
        # This is a nonlinear system in general, but if we fix W, we can try to solve.
        #
        # Alternative: For each k, consider the equation as:
        # For fixed k, we have a 3x3 matrix M_k where M_k[i,j] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We need M_k = I_3 for all k.
        #
        # This is still coupled. Let me try a different approach:
        # Fix W, then for each c, the product U[:,c] ⊗ V[:,c] must contribute to the tensor.
        # This is hard to solve directly.
        
        # Better approach: Use the fact that we can solve iteratively or use least squares
        # on the full system.
        
        # Build the full linear system:
        # Unknowns: U (n*m*rank) + V (m*p*rank) variables
        # Equations: n*m*p equations
        # But the system is nonlinear (products of U and V).
        
        # Alternative: Fix U, solve for V linearly, then check.
        # Or fix V, solve for U linearly.
        
        # Let's try: Fix U randomly, solve for V linearly.
        # For fixed U and W, the equations are linear in V.
        # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (j,k): for fixed j,k, we have n equations (one for each i):
        # sum_c (U[i*m+j,c] * W[k*n+i,c]) * V[j*p+k,c] = 1
        # This is a linear system in V[j*p+k, :] for each (j,k).
        
        # So for each (j,k), we have a system:
        # A_{jk} @ V[j*p+k, :] = ones(n)
        # where A_{jk}[i, c] = U[i*m+j, c] * W[k*n+i, c]
        
        # This is solvable if A_{jk} has full row rank (rank n=3).
        
        U = np.random.randint(-4, 5, (n * m, target_rank)) * 0.5
        V = np.zeros((m * p, target_rank))
        
        for j in range(m):
            for k in range(p):
                # Build A_{jk} of size (n, target_rank)
                A_jk = np.zeros((n, target_rank))
                for i in range(n):
                    for c in range(target_rank):
                        A_jk[i, c] = U[i * m + j, c] * W[k * n + i, c]
                
                # Solve A_jk @ V[j*p+k, :] = ones(n)
                try:
                    V_slice, residuals, rank_sol, sv = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                    V_rounded = np.round(V_slice * 2) / 2
                    
                    # Check if rounded solution works
                    check = A_jk @ V_rounded
                    if not np.allclose(check, np.ones(n), atol=1e-9):
                        return None, None, None
                    V[j * p + k, :] = V_rounded
                except:
                    return None, None, None
        
        return U, V, W
    
    # Search for lower rank
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        # Try various approaches
        found = False
        
        # Approach 1: Random U, solve for V given W
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Generate W with sparse pattern
            W = np.zeros((p * n, target_rank))
            for row in range(p * n):
                if np.random.rand() < 0.5:
                    continue
                nnz = np.random.randint(1, 4)
                cols = np.random.choice(target_rank, size=nnz, replace=False)
                for c in cols:
                    W[row, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            if not np.any(W):
                continue
            
            U, V, W_sol = solve_for_UV(W, target_rank)
            if U is not None:
                if verify_fast(U.tolist(), V.tolist(), W_sol.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                    found = True
                    break
        
        if found:
            continue
        
        # Approach 2: Original approach - random U, V, solve for W
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-4, 5, (n * m, target_rank)) * 0.5
            V = np.random.randint(-4, 5, (m * p, target_rank)) * 0.5
            
            W = np.zeros((p * n, target_rank))
            valid_W = True
            
            for k in range(p):
                for i in range(n):
                    M_i = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            M_i[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    try:
                        W_slice_i, _, _, _ = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                        W_rounded = np.round(W_slice_i * 2) / 2
                        check = M_i @ W_rounded
                        if not np.allclose(check, np.ones(m), atol=1e-9):
                            valid_W = False
                            break
                        W[k * n + i, :] = W_rounded
                    except:
                        valid_W = False
                        break
                if not valid_W:
                    break
            
            if valid_W:
                if verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
