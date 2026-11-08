"""Fresh design: search for rank-minimal exact tensor decomposition of 3x3x4 matmul."""
import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
N = n * m          # 9
M = m * p          # 12
P = p * n          # 12

def _build_T():
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Flatten to vector of length N*M*P
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _w_index(k, i):
    # W[k*n + i]
    return k * n + i

def _u_index(i, j):
    # U[i*m + j]
    return i * m + j

def _v_index(j, k):
    # V[j*p + k]
    return j * p + k

def _verify(U, V, W, rank):
    # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j,k]
    # T[i,j,k] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_solve_with_w_patterns(w_patterns, deadline):
    """
    w_patterns: list of length rank, each is a dict mapping (k,i) -> coefficient (Fraction)
    We fix W columns and solve for U, V.
    Unknowns: U (N*rank) + V (M*rank) = (N+M)*rank
    Equations: N*M*P = 3*3*4 = 36, but each equation is bilinear in U and V.
    
    Actually this is bilinear, not linear. Let me think differently.
    
    Alternative: fix U and W, solve for V linearly. Or use a different approach.
    
    Let me use the approach: fix the W matrix, then the problem becomes:
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is still bilinear in U and V. 
    
    Better approach: Use the fact that we can fix U to be a standard basis-like matrix
    and then solve for V and W linearly? No, that's also not quite right.
    
    Let me use a different strategy: enumerate possible U matrices (sparse, with 1s and 0s),
    then for each U, solve for V and W via a linear system.
    
    Actually, the cleanest approach for small sizes: 
    - Fix the set of "factorizations" (triplets of support patterns).
    - For each factorization pattern, the contribution to T[i,j,k] is a product of three scalars.
    - We need sum over c of these products to equal 1 for all (i,j,k).
    
    Let me try: fix U to have entries in {0, 1, -1} and be sparse. Then for fixed U,
    the equation for V and W is: for each (i,j,k):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is still bilinear in V and W. 
    
    New approach: Fix both U and W, then V is determined linearly? No.
    
    Let me think again. The standard approach for small tensor rank:
    Use the "tensor rank" via checking if T is in the span of rank-1 tensors.
    
    For a concrete search: try rank r from 20 upward. For each rank, try random
    sparse patterns and solve the resulting system.
    
    Actually, let me use a simpler but effective approach:
    1. Start with the schoolbook (rank 36).
    2. Try to find a rank-23 decomposition (n*p + m*p - 1 = 12 + 12 - 1 = 23).
       This is the "Strassen-like" bound.
    3. Use linear algebra: fix W to be a specific sparse matrix, then solve for U and V.
    """
    pass

def _search_rank23(deadline):
    """
    Try to find a rank-23 decomposition.
    Strategy: W has 23 columns. Each column c has support on some (k,i) pairs.
    For the decomposition to work, we need:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    Let's fix W to be a matrix where each column has exactly one nonzero entry (a "basis" pattern),
    but with possible cancellations. Actually, let's try W columns that are unit vectors.
    
    If W[k*n+i, c] = 1 for some specific (k_c, i_c), then the equation becomes:
      sum_c U[i*m+j, c] * V[j*p+k_c, c] * delta(i_c == i) = 1
    
    This only constrains when i = i_c. This is too restrictive.
    
    Let me try a different approach: use known results. The minimal rank of 3x3x4 matrix
    multiplication is not known to be less than 36 in general, but let me try to find
    any improvement.
    
    Actually, for the specific case, let me just try to solve a linear system by fixing
    the W matrix to have a specific structure.
    """
    # Let me try: W is 12x23. Fix W to be the first 12 rows as identity-like.
    # Actually, let me try a random search with verification.
    
    import random
    random.seed(42)
    
    best_rank = n * m * p  # 36
    best_result = None
    
    # First, return the schoolbook as fallback
    def schoolbook():
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
    
    # Try to find rank 23, 22, 21, 20, ... down to some minimum
    for target_rank in range(23, 19, -1):
        if time.time() > deadline - 10:
            break
        # Try random sparse W patterns
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            # Random W: each column has 1-3 nonzeros with values +/-1
            W = [[0] * target_rank for _ in range(P)]
            for c in range(target_rank):
                nnz = random.choice([1, 2, 3])
                positions = random.sample(range(P), nnz)
                for pos in positions:
                    W[pos][c] = random.choice([1, -1])
            
            # Now solve for U and V. This is bilinear, so let's fix U and solve for V.
            # Fix U to be random sparse, then for each (i,j,k) with U[i*m+j,c] != 0,
            # we have a linear equation in V and W... but W is fixed.
            # 
            # Given U and W, the equation for V is:
            #   For each (i,j,k): sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
            # This is linear in V! For fixed U and W, V is determined by a linear system.
            
            # Let's fix U to be sparse random, then solve for V linearly.
            U = [[0] * target_rank for _ in range(N)]
            for i in range(n):
                for j in range(m):
                    nnz = random.choice([1, 2, 3])
                    positions = random.sample(range(target_rank), min(nnz, target_rank))
                    for pos in positions:
                        U[i * m + j][pos] = random.choice([1, -1])
            
            # Build linear system for V: unknowns are V[j*p+k, c] for j in [0,m), k in [0,p), c in [0,rank)
            # That's M * target_rank = 12 * target_rank unknowns.
            # Equations: for each (i,j,k), one equation.
            # Equation (i,j,k): sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
            
            # Let's set up the linear system using fractions.
            num_unknowns = M * target_rank
            # Map unknown index: (j, k, c) -> j*p*target_rank + k*target_rank + c
            def v_idx(j, k, c):
                return j * p * target_rank + k * target_rank + c
            
            # Build A matrix (36 equations x num_unknowns) and b vector
            A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(N * M * P)]
            b = [Fraction(1) for _ in range(N * M * P)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(target_rank):
                            coeff = U[i * m + j][c] * W[k * n + i][c]
                            if coeff != 0:
                                A[eq_idx][v_idx(j, k, c)] = Fraction(coeff)
                        eq_idx += 1
            
            # Solve the linear system A * v = b
            # Use Gaussian elimination with fractions
            v = _solve_linear(A, b, deadline)
            if v is None:
                continue
            
            # Extract V
            V = [[0] * target_rank for _ in range(M)]
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        V[j * p + k][c] = v[v_idx(j, k, c)]
            
            # Verify
            if _verify(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    return None

def _solve_linear(A, b, deadline):
    """Solve A*x = b using Gaussian elimination with fractions. Returns x or None."""
    num_eqs = len(b)
    num_unknowns = len(A[0]) if A else 0
    
    # Augmented matrix
    aug = []
    for i in range(num_eqs):
        row = list(A[i]) + [b[i]]
        aug.append(row)
    
    # Gaussian elimination
    n_rows = len(aug)
    n_cols = num_unknowns
    pivot_row = 0
    pivot_cols = []
    
    for col in range(n_cols):
        if time.time() > deadline - 5:
            return None
        # Find pivot
        pivot = -1
        for row in range(pivot_row, n_rows):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
        # Normalize
        pivot_val = aug[pivot_row][col]
        if pivot_val != 1:
            for j in range(col, n_cols + 1):
                aug[pivot_row][j] = aug[pivot_row][j] / pivot_val
        # Eliminate
        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = aug[row][col]
            if factor != 0:
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[pivot_row][j]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    
    # Check consistency
    for row in range(pivot_row, n_rows):
        for j in range(n_cols):
            if aug[row][j] != 0:
                return None  # Inconsistent
        if aug[row][n_cols] != 0:
            return None  # Inconsistent
    
    # Extract solution (free variables set to 0)
    x = [Fraction(0) for _ in range(n_cols)]
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][n_cols]
    
    return x

def construct():
    start = time.time()
    deadline = start + 120  # Leave margin for evaluation overhead
    
    # Schoolbook fallback
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
    schoolbook_result = {"U": U, "V": V, "W": W}
    
    # Try to find a better (lower rank) decomposition
    if time.time() < deadline - 15:
        result = _search_rank23(deadline)
        if result is not None:
            return result
    
    return schoolbook_result
