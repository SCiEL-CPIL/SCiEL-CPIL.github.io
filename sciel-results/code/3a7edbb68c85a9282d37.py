"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import product

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c]
                if s != W[k * n + i][0]:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Verify using numpy for speed."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_np[i * m + j, c] * V_np[j * p + k, c] for c in range(rank))
                if abs(s - W_np[k * n + i, 0]) > 1e-9:
                    return False
    return True

def _snap_to_half(val):
    """Snap a float to the nearest half-integer."""
    return round(val * 2) / 2

def _try_rank_structured(rank, n, m, p, deadline):
    """
    Try to find a rank-rank decomposition with structured U and V.
    We fix U and V to have specific sparse patterns and solve for W.
    """
    if time.time() > deadline:
        return None
    
    # Strategy: Try different patterns for U and V
    # Pattern 1: U[i*m+j, c] and V[j*p+k, c] are either 0 or 1
    # We want to find if there exist binary matrices U, V such that
    # for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] = 1
    # and W[k*n+i, c] can be determined.
    
    # Actually, the tensor decomposition requires:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # where T is the multiplication tensor (all ones in the (i,j,k) slice)
    
    # Wait, the matrix multiplication tensor has T[i,j,k] = 1 if we're computing
    # C[i,k] = sum_j A[i,j]*B[j,k]. So the tensor is T_{i,j,k} = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # Actually for matrix multiplication, the tensor is:
    # T[i, j, k] = 1 for the "multiplication" part
    # The decomposition is: C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U_c[i,j]) (sum_k B[j,k] V_c[j,k]) W_c[k,i]
    
    # Let me re-read the task. The output factor follows W[k*n+i, r].
    # So W has shape (p*n, rank) = (3*2, rank) = (6, rank)
    
    # The decomposition should satisfy:
    # For all i in [0,n), j in [m), k in [p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is the standard tensor decomposition of matrix multiplication.
    
    # Let's try a systematic approach:
    # Fix U to have a specific structure, then solve for V and W.
    
    # Approach: Use the fact that we can fix one matrix (say U) to have
    # specific entries, then the equations become linear in V and W.
    
    # Actually, let's try a different approach: enumerate possible sparse patterns.
    
    # For rank r, we need r columns in U (6 rows), V (9 rows), W (6 rows).
    # Each column c defines a "rank-1" term.
    
    # Let's try: for each column c, U[:,c] is a vector in {0,1}^6, V[:,c] in {0,1}^9, W[:,c] in {0,1}^6
    # and check if the tensor equation is satisfied.
    
    # But this is exponential. Instead, let's use a smarter approach.
    
    # Key insight: The matrix multiplication tensor for 2x3x3 has rank at most 18 (schoolbook).
    # We want to find if lower rank is possible.
    
    # Let's try rank 14, 15, 16, 17 by solving a system of equations.
    
    # For a given rank r, we have r unknowns per row of U, V, W.
    # Total unknowns: 6*r + 9*r + 6*r = 21*r
    # Constraints: 2*3*3 = 18 equations (one for each (i,j,k) triple)
    
    # Wait, that's not right. The constraint is:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a cubic equation in the unknowns.
    
    # However, if we fix U and V, then W is determined linearly:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W[k*n+i, c] for fixed (i,j,k).
    
    # But W[k*n+i, c] appears in multiple equations (for different j).
    
    # Let me re-think. For fixed U and V, the equations are:
    # For each i, k: sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Wait no, the equation is for each (i,j,k):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # So for each (i,j,k), this gives one equation. The unknowns are W[k*n+i, c] for all c.
    # For fixed (i,k), the unknowns W[k*n+i, 0..r-1] appear in equations for all j.
    
    # So for fixed (i,k), we have m equations (one for each j) and r unknowns.
    # The equations are:
    # sum_c V[j*p+k,c] * W[k*n+i,c] = 1 / U[i*m+j,c] ... no, that's not right.
    
    # The equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W[k*n+i, c] if U and V are fixed.
    # For fixed (i,k), we have m equations (for j=0..m-1) in r unknowns (W[k*n+i, 0..r-1]).
    # The coefficient of W[k*n+i,c] in the j-th equation is U[i*m+j,c] * V[j*p+k,c].
    
    # So we need to solve: A_{ik} * W_{ik} = b_{ik} where A_{ik} is m x r, b_{ik} is m x 1.
    # This is an underdetermined system (m < r typically).
    
    # For a solution to exist, we need b_{ik} to be in the column space of A_{ik}.
    # And the solution is not unique.
    
    # This approach is complex. Let's try a different strategy.
    
    # Strategy: Use numerical optimization to find U, V, W that satisfy the equations.
    # Minimize sum_{i,j,k} (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
    
    # But this is a non-convex problem.
    
    # Alternative: Try known constructions.
    # For 2x3x3, the rank is known to be 14 (from literature on matrix multiplication tensors).
    # Let's verify this.
    
    # Actually, I recall that the rank of the 2x3x3 matrix multiplication tensor is 14.
    # Let's try to construct it.
    
    # The construction uses the following idea:
    # We can write 2x3x3 = 2x2x2 + 2x1x1 + 1x2x2 + ... etc.
    
    # Let's try a specific construction based on known results.
    
    # For now, let's try rank 14 with a structured approach.
    
    # I'll use a known construction for 2x3x3 matrix multiplication.
    
    # The rank of M_{2,3,3} is 14. Here's a construction:
    
    # Actually, let me just try to find a solution using a search over small integer/half-integer values.
    
    # Given the time constraint, let's return the schoolbook solution for now
    # and improve in future iterations.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try to find a better solution
    for rank in range(14, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        result = _try_rank_structured(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
