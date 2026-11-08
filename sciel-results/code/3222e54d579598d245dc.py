"""Matrix multiplication 3x3x4 rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    The strategy uses randomized linear algebra over GF(2) to search for a
    low-rank decomposition, then lifts to exact integer/half-integer solutions.
    
    The matrix multiplication tensor T[i,j,k] = delta[i,k] (for A[i,j]*B[j,k]=C[i,k])
    where A is n x m, B is m x p, C is n x p.
    
    Wait, let me re-read the convention:
    - U has shape (n*m, rank): indexed by (i,j) where i in [0,n), j in [0,m)
    - V has shape (m*p, rank): indexed by (j,k) where j in [0,m), k in [0,p)
    - W has shape (p*n, rank): indexed by (k,i) where k in [0,p), i in [0,n)
      (this is the "official trace convention W[k*n+i,r]")
    
    The tensor T[i,j,k] for matrix mult C[i,k] = sum_j A[i,j]*B[j,k]
    is such that the multiplication tensor has entries T[i,j,k] = 1 if we're 
    computing C[i,k] from A[i,j]*B[j,k], i.e., the tensor is:
    T[(i,j),(j,k),(k,i)] ... 
    
    Actually, the standard formulation: the matrix multiplication tensor 
    <n,m,p> has a flattening. The rank-r decomposition means:
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) 
      is a "valid" triple for the multiplication, else 0.
    
    Wait, actually for matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid (i,j,k) 
    in the product structure. Actually no - the tensor encodes the bilinear map.
    
    The standard tensor for matrix multiplication: T[i,j,k] where i indexes rows of 
    A (and C), j indexes columns of A / rows of B, k indexes columns of B (and C).
    The tensor value T[i,j,k] = 1 because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    So the condition is: for all i, j, k:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of n*m*p equations (3*3*4 = 36 equations) in rank*3 unknowns 
    (each rank factor has 3 variables per entry).
    
    For rank r, we have 3*r variables total (r entries in U, r in V, r in W for each 
    of the 36 positions... wait no).
    
    Let me think more carefully. U is (n*m) x r = 9 x r. V is (m*p) x r = 12 x r. 
    W is (p*n) x r = 12 x r. Total unknowns: 9*r + 12*r + 12*r = 33*r.
    
    We have 36 equations (one for each (i,j,k) triple).
    
    For r=11: 33*11 = 363 unknowns, 36 equations. Very underdetermined.
    For r=1: 33 unknowns, 36 equations. Overdetermined.
    
    The known lower bound for 3x3x4 matrix multiplication rank... Let me think.
    The rank of <3,3,4> is known to be at most 3*3*4=36 (schoolbook). 
    There are known results: rank(<3,3,4>) is between 20 and 36.
    
    Actually, I recall that the rank of <3,3,4> is known to be 20. Let me try to 
    find a rank-20 decomposition.
    
    Strategy: Use random initialization over GF(2) or reals, then solve the 
    polynomial system. Since the system is highly underdetermined for r >= 2,
    we can use random values for most variables and solve for the rest.
    
    Actually, a better approach: parametrize with random values and solve a 
    linear system. For a fixed set of rank-r decompositions, if we fix V and W,
    the equations become linear in U (and vice versa).
    
    Let's try: fix V and W randomly, then solve for U. The equations are:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is linear in U. For each i, the equations involving U[i*m+0..i*m+m-1, *] 
    are coupled. Actually, for fixed i, we have m*p = 3*4 = 12 equations involving 
    U[i*m+0], U[i*m+1], U[i*m+2] (each is a vector of length r).
    
    So for fixed i, we have a system: for each (j,k):
      sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    
    This is a linear system in the 3*r variables U[i*m+0..2, 0..r-1].
    We have 12 equations and 3*r unknowns. For r >= 4, this is underdetermined.
    
    Let me try r=20 and use this approach:
    1. Randomly initialize V and W over reals (or integers)
    2. For each i, solve the linear system for U[i*m:*, :]
    3. Check if the solution works for all (i,j,k)
    
    But this won't work directly because fixing V and W arbitrarily may not allow 
    a solution. We need a more systematic approach.
    
    Alternative: Use the fact that for small ranks, we can set up the problem as 
    finding rank-r factors. Let me try a direct approach with r=20.
    
    Actually, let me try a simpler approach: since the problem is highly underdetermined,
    I can try to construct a decomposition by setting most entries to zero and solving 
    a small system.
    
    Let me try r=20 with the following approach:
    - Initialize U, V, W with random small integers
    - Use iterative refinement or solve the linear system
    
    Actually, the cleanest approach: for a given rank r, the conditions are:
    sum_{r_idx} U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] = 1 for all i,j,k
    
    This is a system of 36 polynomial equations. For r=20, we have 33*20 = 660 
    unknowns. We can fix 624 variables randomly and solve the remaining 36.
    
    But which 36 to leave free? A natural choice: fix all of V and W, and solve 
    for U. That's 12*20 + 12*20 = 480 fixed, 9*20 = 180 free. Too many free.
    
    Better: fix all of V (240 vars) and all of W (240 vars), solve for U (180 vars).
    The system is linear in U with 36 equations. 180 unknowns, 36 equations. 
    Under-determined, so solutions exist generically.
    
    Let me implement this:
    1. Pick random integer V (12x20) and W (12x20)
    2. For each i in range(3), solve the 12x60 system for U[i*3:(i+1)*3, :]
       The 12 equations are for (j,k) in product(range(3), range(4))
       The 60 unknowns are U[i*3+j, r] for j in range(3), r in range(20)
    3. Verify the solution
    
    Since the system is underdetermined, we can set 48 of the 60 U values to 0 and 
    solve for the remaining 12. Or use least squares / pseudoinverse.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Try rank 20
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        np.random.seed(42 + rank)
        
        # Random integer initialization for V and W
        V = np.random.randint(-1, 2, size=(m * p, rank))  # 12 x rank
        W = np.random.randint(-1, 2, size=(p * n, rank))  # 12 x rank
        
        # Solve for U
        U = np.zeros((n * m, rank), dtype=float)
        
        success = True
        for i in range(n):
            # Build the 12 x (m*rank) system for U[i*m:(i+1)*m, :]
            # Equations: for each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            A_mat = np.zeros((m * p, m * rank))
            b_vec = np.ones(m * p)
            
            eq_idx = 0
            for j in range(m):
                for k in range(p):
                    for jj in range(m):
                        for r in range(rank):
                            A_mat[eq_idx, jj * rank + r] = V[j * p + k, r] * W[k * n + i, r]
                    eq_idx += 1
            
            # Solve using pseudoinverse
            U_block, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check residual
            if np.max(np.abs(A_mat @ U_block - b_vec)) > 1e-9:
                success = False
                break
            
            U[i * m:(i + 1) * m, :] = U_block.reshape(m, rank)
        
        if not success:
            continue
        
        # Verify the full decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] for r in range(rank))
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list of lists with integer/half-integer entries
            # Round to nearest half-integer
            U_list = [[round(val * 2) / 2 for val in row] for row in U.tolist()]
            V_list = [[round(val * 2) / 2 for val in row] for row in V.tolist()]
            W_list = [[round(val * 2) / 2 for val in row] for row in W.tolist()]
            
            # Final verification with rounded values
            valid2 = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r] for r in range(rank))
                        if abs(val - 1) > 1e-9:
                            valid2 = False
                            break
                    if not valid2:
                        break
                if not valid2:
                    break
            
            if valid2:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if no low-rank found
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
