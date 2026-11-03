"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We use the fact that for matrix multiplication, the tensor decomposition condition is:
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    We attempt to find a rank-r solution with r < 60 using a structured approach.
    
    Key insight: We can use the Strassen-like idea where we combine multiple products.
    For 2x3 * 3x10, we can try to exploit the structure.
    
    Let's try rank 54 = 6*9 or other values. We'll use a randomized approach with exact arithmetic.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try ranks from 54 down to 50
    for rank in range(54, 49, -1):
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    
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

def _try_rank(n, m, p, rank):
    """
    Try to find a rank-r decomposition using a structured approach.
    
    We fix V to have a specific structure and solve for U and W.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # We'll use a randomized approach:
    # 1. Initialize V with random rational values
    # 2. For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in U for fixed V,W and linear in W for fixed U,V.
    
    # Strategy: Fix V to be a specific matrix, then the condition becomes:
    # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
    # Let M[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
    # Then we need: sum_c U[i*m+j,c] * M[i,j,k,c] = 1
    
    # This is a linear system in U for fixed V and W. But W is also unknown.
    
    # Alternative: Use the fact that we can parameterize.
    # Let's try a different approach: assume U, V, W have entries from {0, 1, -1}
    # and search for a valid assignment.
    
    # For rank r, we have r columns. Each column c has:
    # u_c: vector in {0,1,-1}^nm
    # v_c: vector in {0,1,-1}^mp  
    # w_c: vector in {0,1,-1}^pn
    
    # Condition: for all i,j,k: sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # This is a constraint satisfaction problem. Let's use a smarter approach.
    
    # Key idea: For matrix multiplication, there's a known lower bound.
    # For 2x3 * 3x10, the rank is at least max(2*10, 3*10, 2*3) = 30 by one bound,
    # but tighter bounds exist. The trivial upper bound is 60.
    
    # Let's try to use a bilinear complexity approach.
    # We can view the multiplication as: C = A * B where A is 2x3, B is 3x10.
    
    # One approach: Use the fact that we can compute A*B by first computing
    # intermediate products. For example, we can compute 3*10 = 30 products
    # for each of the 2 rows, giving 60 total. But we can share computations.
    
    # Actually, let's try a direct algebraic approach.
    # We need to find r such that there exist U, V, W with the trilinear condition.
    
    # Let's use a heuristic: try to find a solution where each column of U, V, W
    # has sparse support.
    
    # For rank 54, we need to "save" 6 products. This might be possible if we
    # can find 6 pairs of (i,j,k) that can be combined.
    
    # Let's try a random search with exact arithmetic for small ranks.
    import random
    random.seed(42)
    
    for attempt in range(100):
        # Initialize V with random values from {-1, 0, 1}
        V = np.zeros((mp, rank), dtype=int)
        for j in range(m):
            for k in range(p):
                # Each V[j*p+k, c] is 0, 1, or -1
                V[j*p+k, :] = np.random.choice([-1, 0, 1], size=rank)
        
        # Now we need to find U and W such that for all i,j,k:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is still coupled. Let's try to decouple by fixing W as well.
        W = np.zeros((pn, rank), dtype=int)
        for k in range(p):
            for i in range(n):
                W[k*n+i, :] = np.random.choice([-1, 0, 1], size=rank)
        
        # Now for each (i,j), we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        
        # Let A[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then for each (i,j): sum_c U[i*m+j,c] * A[i,j,k,c] = 1 for all k
        
        # This is a linear system in U[i*m+j, :] for each (i,j).
        # For fixed (i,j), we have p equations in rank unknowns.
        
        # Let's check if this system is consistent.
        U = np.zeros((nm, rank), dtype=int)
        feasible = True
        for i in range(n):
            for j in range(m):
                # Build the p x rank matrix A[i,j,:,:]
                A_mat = np.zeros((p, rank), dtype=int)
                for k in range(p):
                    for c in range(rank):
                        A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                # We need U[i*m+j, :] such that A_mat @ U[i*m+j, :] = ones(p)
                # This is a linear system. Let's check if it's consistent.
                # We need to solve A_mat * x = 1 where x = U[i*m+j, :]
                
                # Use numpy to check consistency
                try:
                    # Check if 1 is in the column space of A_mat
                    # We can do this by checking if rank(A_mat) == rank([A_mat | 1])
                    A_aug = np.column_stack([A_mat, np.ones(p)])
                    r_A = np.linalg.matrix_rank(A_mat)
                    r_aug = np.linalg.matrix_rank(A_aug)
                    if r_A != r_aug:
                        feasible = False
                        break
                except:
                    feasible = False
                    break
        
        if not feasible:
            continue
        
        # If feasible, solve for U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank), dtype=int)
                for k in range(p):
                    for c in range(rank):
                        A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                
                # Solve A_mat * x = 1
                # Use least squares as a heuristic, then check if it's exact
                try:
                    x, residuals, rank_, singular_values = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    # Check if the solution is exact
                    if np.allclose(A_mat @ x, np.ones(p), atol=1e-10):
                        # Round to nearest integer
                        x_rounded = np.round(x)
                        if np.allclose(A_mat @ x_rounded, np.ones(p), atol=1e-10):
                            U[i*m+j, :] = x_rounded
                        else:
                            feasible = False
                            break
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Verify the solution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            feasible = False
                            break
                    if not feasible:
                        break
                if not feasible:
                    break
            
            if feasible:
                # Convert to lists
                U_list = [[int(x) for x in row] for row in U]
                V_list = [[int(x) for x in row] for row in V]
                W_list = [[int(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
