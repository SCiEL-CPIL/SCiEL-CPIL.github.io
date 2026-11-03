"""Matrix multiplication 2x3 by 3x3 rank-minimal decomposition search."""
DIMENSIONS = [2, 3, 3]
import numpy as np
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    
    def check_solution(U, V, W, rank, n, m, p):
        # Verify that U @ diag(V) @ W^T gives the correct tensor
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 if this is a valid matrix mult entry
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] should be 1 for each (i,j,k) where the product contributes
        # The bilinear form: C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
        # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # should equal 1 for each (i,j,k) in the schoolbook sense
        
        # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # where i in [0,n), j in [0,m), k in [0,p)
        # No wait - the tensor for matrix multiplication is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the trilinear form is: sum_{i,j,k} C_{k*n+i} * (coeff) * A_{i*m+j} * B_{j*p+k}
        # The tensor T[i,j,k] = 1 for all valid i,j,k (each product A_ij * B_jk contributes to C_ik)
        
        # Let me verify differently:
        # The decomposition says: C = sum_c (U[:,c] outer A_indices) * (V[:,c] outer B_indices) * W[:,c]
        # More precisely: C_{k*n+i} = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * A_{i*m+j} * V[j*p+k, c] / ...
        
        # Actually the standard form is:
        # C_{i,k} = sum_c M_c(A) * M'_c(B) * W_c
        # where M_c(A) = sum_j U_{i,j,c} * A_{i,j}
        # and M'_c(B) = sum_j V_{j,k,c} * B_{j,k}
        
        # So the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # should equal delta for the matrix multiplication tensor
        # For matrix mult, T[i,j,k] = 1 for all i,j,k (each term A_ij * B_jk appears once)
        
        # Let's verify:
        T = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1) > 1e-9:
                        return False
        return True
    
    def try_rank(rank, n, m, p, time_limit=10):
        """Try to find a rank decomposition using random initialization and verification."""
        start = time.time()
        
        # For a given rank, we need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that the tensor decomposes correctly.
        
        # Strategy: Fix U and V to have simple structure, then solve for W.
        # If U and V are chosen such that for each (i,j,k), there's exactly one c
        # with U[i*m+j,c]=1 and V[j*p+k,c]=1, then W[k*n+i,c]=1.
        # This is the schoolbook approach with rank >= n*m*p.
        
        # For rank < n*m*p, we need sharing.
        
        # Let's try a structured approach:
        # For 2x3 * 3x3, n=2, m=3, p=3, schoolbook rank = 18.
        # We want to find if rank 16 or 17 works.
        
        # Approach: Use the fact that the matrix multiplication tensor for (n,m,p)
        # can sometimes be decomposed with rank less than n*m*p by sharing factors.
        
        # For 2x3*3x3, let's think about it as combining a 2x3*3x1 and 2x3*1x3 type structures.
        
        # Actually, let's try a direct algebraic approach for rank 16.
        # We'll set up the equations and try to solve.
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # The constraint is: for all i,j,k:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a system of nm*p = 18 equations in 3*rank*n*m + 3*rank*m*p + 3*rank*p*n unknowns
        # But it's trilinear (degree 3), so it's not linear.
        
        # Alternative: Fix U and V, then the equations become linear in W.
        # For each (i,k): sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = ... 
        # Wait, that's not right either. Let me re-examine.
        
        # The tensor condition is: for each (i,j,k), T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is trilinear in (U,V,W).
        
        # If we fix U and V, then for each triple (i,j,k), we have:
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
        # This is linear in W! The coefficient of W[k*n+i,c] is U[i*m+j,c]*V[j*p+k,c]
        
        # But we need this to hold for ALL (i,j,k) simultaneously, and W[k*n+i,c] appears
        # in multiple equations (for different j values).
        
        # For a fixed (i,k) and c, W[k*n+i,c] appears in the equation for each j.
        # So we have for each (i,k,c): the coefficient from the (i,j,k) equation is U[i*m+j,c]*V[j*p+k,c]
        # But W[k*n+i,c] must satisfy: for all j, U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This means U[i*m+j,c]*V[j*p+k,c] must be the same for all j (if nonzero),
        # which is a strong constraint.
        
        # Actually, the sum is over c, not over j. Let me re-read.
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each (i,j,k)
        
        # For fixed (i,j), as k varies: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (i,k), as j varies: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a set of 18 scalar equations.
        
        # Let me try a different approach: use known results or try specific structures.
        
        # For 2x3 * 3x3, the rank is known to be 16 (I believe).
        # Let me try to construct it.
        
        # Actually, let's just try random searches for small ranks and verify.
        
        for attempt in range(100):
            if time.time() - start > time_limit:
                break
            
            # Random initialization
            np.random.seed(attempt)
            U = np.random.randn(nm, rank).tolist()
            V = np.random.randn(mp, rank).tolist()
            
            # Now solve for W: the system is:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let A_{(i,j,k), (k',i',c)} = U[i*m+j,c] * V[j*p+k,c] if (i'=i, k'=k) else 0
            # Then A * W_flat = ones(18)
            
            # Actually, W has indices (k*n+i, c) for k in [0,p), i in [0,n), c in [0,rank)
            # So W_flat has p*n*rank entries.
            # The equation for (i,j,k) is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This involves only W entries with row index k*n+i.
            # So the equations decouple by (i,k) pairs!
            
            # For each (i,k): sum_{j,c} ... no wait.
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # The W entry is W[k*n+i,c], which depends on (i,k,c).
            # For a fixed (i,k), the equations over j give:
            # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is m equations in rank unknowns (W[k*n+i,0], ..., W[k*n+i,rank-1])
            # Coefficient matrix: B_{j,c} = U[i*m+j,c] * V[j*p+k,c]
            # System: B @ W_ik = ones(m)
            
            # So for each (i,k), we solve a m x rank linear system!
            # m=3, rank=16 (say), so 3 equations in 16 unknowns - underdetermined.
            
            valid = True
            W = [[0]*rank for _ in range(pn)]
            for i in range(n):
                for k in range(p):
                    # Build the 3 x rank matrix B
                    B = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            B[j,c] = U[i*m+j][c] * V[j*p+k][c]
                    
                    # Solve B @ w = ones(3)
                    try:
                        # Least squares solution
                        w, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        # Check if the solution is exact
                        residual = np.abs(B @ w - np.ones(m))
                        if np.max(residual) > 1e-9:
                            valid = False
                            break
                        W[k*n+i] = w.tolist()
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Final verification
            if check_solution(U, V, W, rank, n, m, p):
                return U, V, W
        
        return None
    
    # Try ranks from 16 upward
    for rank in [16, 17, 18]:
        if time.time() > deadline - 30:
            break
        result = try_rank(rank, 2, 3, 3, time_limit=30)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 18
    n, m, p = 2, 3, 3
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
# EVOLVE-BLOCK-END
