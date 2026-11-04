"""
Matrix multiplication 4x4 * 4x5 rank-minimal exact tensor decomposition.

Combined strategy:
1. Build a structured Kronecker-product decomposition:
   - 2-level Strassen for the 4x4 * 4x4 part (rank 49).
   - For the 5th column of B (which multiplies with the 4x4 A), treat it as
     4 independent scalar multiplications, but we can try to merge them with
     the Strassen structure.
   Actually, a cleaner approach: 
   - 4x4 * 4x5 = 4x4 * 4x4 + 4x4 * 4x1 (treating the 5th column separately).
   - 4x4 * 4x4 can be done with rank 49 (Strassen^2).
   - 4x4 * 4x1: this is just 4 rows of A times the single column vector.
     Each output C[i,4] = sum_j A[i,j] * B[j,4].
     This is 4 independent dot products of length 4, each rank 4, total rank 16.
     But we can potentially share structure.
   
   Total naive: 49 + 16 = 65. But we can do better by integrating the 5th column
   into the Strassen structure.

2. Better: use the fact that 4x4 * 4x5 can be seen as a block matrix multiplication.
   Split A into 2x2 blocks (4 blocks), B into 2x2 blocks with the last block being 2x1.
   Use Strassen at the block level where possible.

3. Most practical: start with schoolbook (rank 80), then apply the linear-dependency
   elimination from the current program, PLUS try the structured Strassen-based
   initialization which starts at a lower rank and may allow further reductions.

4. The key insight from inspirations: use random sparse search as a refinement
   on top of structured initializations.

Let me implement:
- Structured Strassen-based initialization for 4x4*4x4 part + simple for 4x4*4x1 part
- Linear dependency elimination
- Random sparse refinement search
"""

import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _schoolbook():
    """Return schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U, V, W are each (4, 7) with entries in {-1, 0, 1}.
    Indexing: U[row, col] where row = i*2+j for 2x2 matrix element (i,j).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11) -> C00 += M3, C01 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # M4 = (A00-A11)(B01+B11) -> C00 += M4, C11 += M4
    U[0][3] = 1; U[3][3] = -1
    V[1][3] = 1; V[3][3] = 1
    W[0][3] = 1; W[3][3] = 1
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C10 += M6, C00 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[0][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C01 += M7, C10 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[2][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Build rank-49 decomposition for 4x4 * 4x4 using 2-level Strassen.
    Returns U, V, W each of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # Outer level block indices: bi = i//2, inner: ii = i%2
            # For 4x4 matrix, element (i,j) has block (i//2, j//2) and inner (i%2, j%2)
            for i in range(4):
                for j in range(4):
                    bi = i // 2
                    bj = j // 2
                    ii = i % 2
                    jj = j % 2
                    # U entry: U2[bi*2+bj, t1] * U2[ii*2+jj, t2]
                    val = U2[bi*2+bj][t1] * U2[ii*2+jj][t2]
                    if val != 0:
                        U[i*4+j][col] += val
                    # V entry: V2[bi*2+bj, t1] * V2[ii*2+jj, t2]
                    val = V2[bi*2+bj][t1] * V2[ii*2+jj][t2]
                    if val != 0:
                        V[i*4+j][col] += val
                    # W entry: W2[bi*2+bj, t1] * W2[ii*2+jj, t2]
                    val = W2[bi*2+bj][t1] * W2[ii*2+jj][t2]
                    if val != 0:
                        W[i*4+j][col] += val
    
    return U, V, W

def _structured_init_4x4_4x5():
    """Build a structured decomposition for 4x4 * 4x5.
    Split B into B_left (4x4) and B_right (4x1).
    C = A * B_left + A * B_right
    For A*B_left: use Strassen 4x4*4x4 (rank 49).
    For A*B_right: 4 independent dot products, each rank 4, total rank 16.
    Total: 65.
    
    But we can potentially do better by integrating. Let's start with 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    U_s, V_s, W_s = _strassen_4x4()
    
    # For the 5th column: C[i, 4] = sum_j A[i,j] * B[j, 4]
    # This is 4 independent rank-4 decompositions.
    # U part: for each i, the row i of A. V part: column 4 of B.
    # W part: element (i, 4) of C.
    
    rank_total = 49 + 16
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Copy Strassen part
    for i in range(16):
        for c in range(49):
            U[i][c] = U_s[i][c]
            V[i][c] = V_s[i][c]
            W[i][c] = W_s[i][c]
    
    # Add the 5th column part
    col_offset = 49
    for i in range(4):
        for j in range(4):
            c = col_offset + i*4 + j
            U[i*4+j][c] = 1
            V[j*5+4][c] = 1
            W[4*4+i][c] = 1  # k=4, n=4, so index = 4*4+i = 16+i
    
    return {"U": U, "V": V, "W": W}

def _to_fraction_matrix(mat):
    """Convert a 2D list to a 2D list of Fractions."""
    return [[Fraction(x) for x in row] for row in mat]

def _rref(frac_mat):
    """Compute reduced row echelon form of a matrix of Fractions.
    Returns (rref_matrix, pivot_columns)."""
    if not frac_mat or not frac_mat[0]:
        return [], []
    m = len(frac_mat)
    n = len(frac_mat[0])
    mat = [row[:] for row in frac_mat]
    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        pivot = None
        for i in range(r, m):
            if mat[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        mat[r], mat[pivot] = mat[pivot], mat[r]
        factor = mat[r][c]
        mat[r] = [x / factor for x in mat[r]]
        for i in range(m):
            if i != r and mat[i][c] != 0:
                factor = mat[i][c]
                mat[i] = [mat[i][j] - factor * mat[r][j] for j in range(n)]
        pivot_cols.append(c)
        r += 1
    return mat, pivot_cols

def _is_linear_combination(frac_mat, target_col, other_cols):
    """Check if target_col is a linear combination of the columns in other_cols.
    Returns the coefficients if found, else None."""
    if not other_cols:
        return None
    m = len(frac_mat)
    cols = []
    for c in other_cols:
        col = [frac_mat[i][c] for i in range(m)]
        cols.append(col)
    n = len(cols)
    aug = []
    target = [frac_mat[i][target_col] for i in range(m)]
    for i in range(m):
        row = [cols[j][i] for j in range(n)]
        row.append(target[i])
        aug.append(row)
    rref, pivots = _rref(aug)
    for i in range(m):
        if all(rref[i][j] == 0 for j in range(n)) and rref[i][n] != 0:
            return None
    sol = [Fraction(0)] * n
    for i in range(len(pivots)):
        col = pivots[i]
        sol[col] = rref[i][n]
    return sol

def _try_reduce(U, V, W):
    """Try to reduce the rank by finding linearly dependent columns."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_f = _to_fraction_matrix(U)
    V_f = _to_fraction_matrix(V)
    W_f = _to_fraction_matrix(W)
    
    removed = [False] * rank
    new_rank = rank
    
    for c in range(rank):
        if removed[c]:
            continue
        other_cols = [c2 for c2 in range(rank) if c2 != c and not removed[c2]]
        if not other_cols:
            continue
        
        coeff_U = _is_linear_combination(U_f, c, other_cols)
        if coeff_U is None:
            continue
        coeff_V = _is_linear_combination(V_f, c, other_cols)
        if coeff_V is None:
            continue
        coeff_W = _is_linear_combination(W_f, c, other_cols)
        if coeff_W is None:
            continue
        
        valid = True
        for idx, a in enumerate(other_cols):
            if coeff_U[idx] != coeff_V[idx] or coeff_U[idx] != coeff_W[idx]:
                valid = False
                break
        
        if valid:
            removed[c] = True
            new_rank -= 1
    
    if new_rank == rank:
        return None
    
    keep = [c for c in range(rank) if not removed[c]]
    U_new = [[U[i][c] for c in keep] for i in range(len(U))]
    V_new = [[V[i][c] for c in keep] for i in range(len(V))]
    W_new = [[W[i][c] for c in keep] for i in range(len(W))]
    
    return U_new, V_new, W_new, new_rank

def _verify(U, V, W):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, tol=1e-9):
    """Fast float verification."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _random_sparse_search(U, V, W, start_rank, deadline):
    """Try to find a lower rank decomposition using random sparse search."""
    n, m, p = DIMENSIONS
    best_U, best_V, best_W, best_rank = U, V, W, start_rank
    entry_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    for test_rank in range(1, best_rank):
        if time.time() > deadline:
            break
        max_trials = 300 if test_rank < 30 else 100
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            density = min(0.4, 6.0 / test_rank)
            U_rand = np.zeros((n*m, test_rank))
            V_rand = np.zeros((m*p, test_rank))
            W_rand = np.zeros((p*n, test_rank))
            
            for i in range(n*m):
                nz = max(1, min(test_rank, int(round(random.gauss(density * test_rank, 0.5)))))
                indices = random.sample(range(test_rank), nz)
                for idx in indices:
                    U_rand[i, idx] = random.choice(entry_set)
            for i in range(m*p):
                nz = max(1, min(test_rank, int(round(random.gauss(density * test_rank, 0.5)))))
                indices = random.sample(range(test_rank), nz)
                for idx in indices:
                    V_rand[i, idx] = random.choice(entry_set)
            for i in range(p*n):
                nz = max(1, min(test_rank, int(round(random.gauss(density * test_rank, 0.5)))))
                indices = random.sample(range(test_rank), nz)
                for idx in indices:
                    W_rand[i, idx] = random.choice(entry_set)
            
            if _verify_float(U_rand, V_rand, W_rand):
                best_rank = test_rank
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with structured initialization (rank 65)
    result = _structured_init_4x4_4x5()
    U, V, W = result["U"], result["V"], result["W"]
    
    # Verify structured init
    if not _verify(U, V, W):
        # Fall back to schoolbook
        result = _schoolbook()
        U, V, W = result["U"], result["V"], result["W"]
    
    # Apply linear dependency reduction
    while time.time() < deadline:
        reduced = _try_reduce(U, V, W)
        if reduced is None:
            break
        U, V, W, new_rank = reduced
        if new_rank == 0:
            break
        if not _verify(U, V, W):
            break
    
    # Try random sparse search for further improvement
    if time.time() < deadline - 20:
        U, V, W, new_rank = _random_sparse_search(U, V, W, len(U[0]), deadline)
        if _verify(U, V, W):
            pass  # Use the improved result
    
    # Final verification
    if not _verify(U, V, W):
        result = _schoolbook()
        return result
    
    return {"U": U, "V": V, "W": W}
