"""Matrix multiplication tensor decomposition for 2x3 * 3x7."""
from fractions import Fraction
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _try_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-`rank` decomposition of the multiplication tensor.
    
    Condition: For all i,j,j',k:
        sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    """
    if time.time() > deadline:
        return None
    
    # Number of equations: n * m * m * p
    n_eq = n * m * m * p
    # Unknowns: rank * (n*m + m*p + p*n)
    n_unk = rank * (n * m + m * p + p * n)
    
    # For small ranks, try to solve numerically
    # Strategy: fix W to random values, then solve for U and V (bilinear system)
    # Or: use the fact that for fixed j', the equations are linear in (U*V) products
    
    # Let's try: for each (j',k), define vector v_{j',k} in R^rank with components V[j'*p+k,c]
    # For each (i,j), define vector u_{i,j} in R^rank with components U[i*m+j,c]
    # For each (i,k), define scalar w_{i,k,c}... no, W is a vector too.
    
    # Actually: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    # Let a_{i,j,c} = U[i*m+j,c], b_{j',k,c} = V[j'*p+k,c], c_{i,k,c} = W[k*n+i,c]
    # sum_c a_{i,j,c} * b_{j',k,c} * c_{i,k,c} = delta_{j,j'}
    
    # This is a 3-tensor equation. Hard to solve directly.
    
    # Alternative: use the matricization. Fix (i,k), get a matrix of size m x m:
    # M_{i,k}[j,j'] = sum_c U[i*m+j,c] * W[k*n+i,c] * V[j'*p+k,c] = delta_{j,j'} = I
    # So for each (i,k), the m x m matrix with entries sum_c U[i*m+j,c]*W[k*n+i,c]*V[j'*p+k,c] must be I.
    
    # Let's define for each (i,k): A_{i,k} = diag(W[k*n+i,:]) and then
    # (U_i * A_{i,k}) * V_k^T = I where U_i is m x rank, V_k is p... 
    # Actually U_i has rows for j=0..m-1, each row is U[i*m+j,:] in R^rank
    # V_k has rows for j'=0..m-1? No, V[j'*p+k,:] for j'=0..m-1, each in R^rank
    # So M_{i,k} = U_i * diag(W_{i,k}) * V_k^T where U_i is m x rank, V_k is m x rank (rows are V[j'*p+k,:])
    
    # We need U_i * diag(W_{i,k}) * V_k^T = I_m for all (i,k).
    
    # This means U_i * diag(W_{i,k}) * V_k^T = I, so V_k^T = (U_i * diag(W_{i,k}))^{-1}
    # This requires U_i * diag(W_{i,k}) to be invertible, which needs rank >= m.
    
    # So rank >= m = 3 is necessary. Let's try rank = 3, 4, 5, ...
    
    # For a given rank r >= m:
    # Fix W arbitrarily (with generic values).
    # For each (i,k), we need U_i * diag(W_{i,k}) * V_k^T = I.
    # This gives V_k^T = (U_i * diag(W_{i,k}))^{-1} for any i, but this must be consistent for all i.
    # So (U_i * diag(W_{i,k}))^{-1} = (U_{i'} * diag(W_{i',k}))^{-1} for all i,i',k.
    # This means U_i * diag(W_{i,k}) = U_{i'} * diag(W_{i',k}) for all i,i',k.
    # So U_i * diag(W_{i,k}) is independent of i!
    
    # Let M_k = U_i * diag(W_{i,k}) (independent of i). Then V_k^T = M_k^{-1}.
    # And U_i * diag(W_{i,k}) = M_k for all i.
    # So U_i = M_k * diag(1/W_{i,k,c}) for each c... 
    # U_i[j,c] = M_k[j,c] / W_{i,k,c}
    
    # But M_k must be the same for all i, and U_i depends on i through W_{i,k}.
    # For consistency: M_k[j,c] / W_{i,k,c} must give the same M_k regardless of i.
    # This is automatically satisfied if we define U_i[j,c] = M_k[j,c] / W_{i,k,c}.
    
    # So the solution is parameterized by:
    # - W_{i,k,c} for all i,k,c (with no zeros)
    # - M_k[j,c] for all k,j,c (the common matrix)
    # Then U_i[j,c] = M_k[j,c] / W_{i,k,c}
    # And V_k[j',c] = (M_k^{-1})[j',c]... wait, V_k^T = M_k^{-1}, so V_k[c,j'] = M_k^{-1}[j',c]
    # V[j'*p+k,c] = M_k^{-1}[j',c]
    
    # Let me verify: sum_c U[i*m+j,c] * W[k*n+i,c] * V[j'*p+k,c]
    # = sum_c (M_k[j,c] / W_{i,k,c}) * W_{i,k,c} * M_k^{-1}[j',c]
    # = sum_c M_k[j,c] * M_k^{-1}[j',c]
    # = (M_k * M_k^{-1})[j,j'] = delta_{j,j'} ✓
    
    # Great! So the solution space is:
    # Choose W_{i,k,c} (nonzero) and M_k (invertible) for each k.
    # Set U[i*m+j,c] = M_k[j,c] / W_{i,k,c}
    # Set V[j'*p+k,c] = M_k^{-1}[j',c]
    
    # The rank is the number of columns c, which is the size of M_k.
    # M_k is m x rank = 3 x rank. For M_k to be invertible in the sense that M_k * M_k^{-1} = I_3,
    # we need M_k to be 3 x rank with a right inverse, so rank >= 3.
    # Actually M_k is 3 x r, and M_k^{-1} is r x 3 (right inverse).
    # M_k * M_k^{-1} = I_3 requires rank(M_k) = 3, so r >= 3.
    
    # So minimum rank is at least 3. Let's try rank = 3.
    
    if rank < m:
        return None
    
    # For rank = r, we need M_k to be 3 x r with a right inverse (r x 3).
    # The simplest case: r = 3, M_k is 3x3 invertible.
    
    # Let's construct a solution:
    # Choose W_{i,k,c} = 1 for all i,k,c (simplest).
    # Choose M_k = I_3 for all k (3x3 identity).
    # Then U[i*m+j,c] = I_3[j,c] / 1 = delta_{j,c}
    # V[j'*p+k,c] = (I_3^{-1})[j',c] = delta_{j',c}
    # W[k*n+i,c] = 1
    
    # Verify: sum_c U[i*m+j,c]*W[k*n+i,c]*V[j'*p+k,c]
    # = sum_c delta_{j,c} * 1 * delta_{j',c} = delta_{j,j'} ✓
    
    # So rank 3 works! Let's verify with the actual formula:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # With U[i*m+j,c] = delta_{j,c}, V[j*p+k,c] = delta_{j,c}, W[k*n+i,c] = 1:
    # C[i,k] = sum_c 1 * A[i,c] * B[c,k] = sum_c A[i,c]*B[c,k]
    # But c ranges from 0 to rank-1 = 0 to 2, and j ranges from 0 to m-1 = 0 to 2.
    # So sum_c A[i,c]*B[c,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] = C[i,k] ✓
    
    # So rank 3 is achievable!
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                if c < m:
                    U[i * m + j][c] = 1 if j == c else 0
                else:
                    U[i * m + j][c] = 0
        for k in range(p):
            for c in range(rank):
                W[k * n + i][c] = 1
    
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                if c < m:
                    V[j * p + k][c] = 1 if j == c else 0
                else:
                    V[j * p + k][c] = 0
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    for rank in range(1, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
