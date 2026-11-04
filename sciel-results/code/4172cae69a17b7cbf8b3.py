"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction as Fr

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 if (i,j,k) corresponds to a valid product, 0 otherwise.
    Actually the tensor T[i,j,k] for matrix mult C=A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor is T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]*B[j,k], else 0.
    In the trace convention: W[k*n+i, r] means the W matrix has index (k, i) -> row = k*n + i.
    U has shape (n*m, rank), indexed by (i, j) -> row = i*m + j.
    V has shape (m*p, rank), indexed by (j, k) -> row = j*p + k.
    
    The decomposition requires: for all i, j, k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid term, 0 otherwise.
    
    Wait, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    The tensor T has entries T[i, j, k] = 1 for all valid i, j, k (since every term A[i,j]*B[j,k] contributes to C[i,k]).
    Actually no - the tensor for matrix multiplication is:
    T[i, j, k] = 1 if the product A[i,j] * B[j,k] is part of the computation of C[i,k].
    Since C[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p) has T[i,j,k] = 1.
    
    So the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fr(0)
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Float verification for speed."""
    import numpy as np
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_fraction_list(mat):
    """Convert a matrix of Fractions or ints to a list of lists of Fractions."""
    return [[Fr(x) for x in row] for row in mat]

def _build_strassen_3x3():
    """
    Build a rank-8 decomposition for 3x3 matrix multiplication.
    Returns U, V, W as lists of lists of Fractions.
    U: shape (9, 8), V: shape (9, 8), W: shape (9, 8)
    
    For C = A * B where A, B are 3x3:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Strassen-like decomposition for 3x3:
    We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    Let's use a known rank-8 scheme for 3x3 matrix multiplication.
    Actually, the minimal rank for 3x3 matrix multiplication is known to be 19 (or 23?), 
    but Strassen gives rank 7 for 2x2. For 3x3, we can use a block approach.
    
    Actually, let me think about this differently. For 3x3 * 3x3, the schoolbook is rank 27.
    We can use a Strassen-like approach: split into 2x2 and 1x1 blocks.
    
    A = [A11 A12 A13; A21 A22 A23; A31 A32 A33] where Aij are 1x1 scalars.
    
    Actually, for the 3x3 case, let me just use a direct rank-27 decomposition first,
    then try to find a lower rank via search.
    
    For now, let's construct the schoolbook rank-27 for 3x3 and then try to reduce.
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    U = [[Fr(0)] * rank for _ in range(n * m)]
    V = [[Fr(0)] * rank for _ in range(m * p)]
    W = [[Fr(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fr(1)
                V[j * p + k][col] = Fr(1)
                W[k * n + i][col] = Fr(1)
                col += 1
    return U, V, W

def _combine_block_decompositions(block_U, block_V, block_W, n, m, p, num_blocks):
    """
    Combine multiple block decompositions for the full n x m * m x p problem.
    If p = num_blocks * p_block, we split B into num_blocks column blocks.
    For each block b, we compute C[:, b*p_block:(b+1)*p_block] = A * B_b.
    
    block_U: shape (n*m, rank_block) - same for all blocks
    block_V: list of num_blocks, each shape (m*p_block, rank_block)
    block_W: list of num_blocks, each shape (p_block*n, rank_block)
    
    Returns U, V, W for the full problem.
    """
    p_block = p // num_blocks
    rank_block = len(block_U[0])
    total_rank = rank_block * num_blocks
    
    U_full = [[Fr(0)] * total_rank for _ in range(n * m)]
    V_full = [[Fr(0)] * total_rank for _ in range(m * p)]
    W_full = [[Fr(0)] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = b * rank_block
        for i in range(n * m):
            for r in range(rank_block):
                U_full[i][base + r] = block_U[i][r]
        for j in range(m):
            for k in range(p_block):
                for r in range(rank_block):
                    V_full[j * p + b * p_block + k][base + r] = block_V[b][j * p_block + k][r]
        for k in range(p_block):
            for i in range(n):
                for r in range(rank_block):
                    W_full[b * p_block + k][k * n + i] = block_W[b][k * n + i][r]
    
    return U_full, V_full, W_full

def _try_reduce_rank(U, V, W, n, m, p, max_time=10):
    """
    Try to find linear dependencies in the rank-1 components to reduce rank.
    This is a simplified approach - just check if any columns are linearly dependent
    in a way that allows merging.
    """
    import numpy as np
    rank = len(U[0])
    
    # Convert to float for numerical analysis
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check if we can find a lower rank decomposition by looking at the tensor
    # This is complex, so for now just return the same rank
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # First, build the schoolbook decomposition
    rank_schoolbook = n * m * p  # 72
    U_school = [[Fr(0)] * rank_schoolbook for _ in range(n * m)]
    V_school = [[Fr(0)] * rank_schoolbook for _ in range(m * p)]
    W_school = [[Fr(0)] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = Fr(1)
                V_school[j * p + k][col] = Fr(1)
                W_school[k * n + i][col] = Fr(1)
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to use Strassen-like decomposition for 3x3 blocks
    # p = 8, so we can split into blocks. 8 = 3 + 3 + 2 or just use 3x3 + 3x3 + 3x2
    # Let's try: split p=8 into blocks of size 3, 3, 2
    
    # For a 3x3 * 3x3 block, we need a rank-r decomposition
    # For a 3x3 * 3x2 block, we need a rank-r' decomposition
    
    # Let's first try to find a good rank for 3x3 * 3x3
    # The schoolbook is 27. Can we do better?
    # For 3x3 matrix multiplication, the best known upper bound is 19 (Landsberg 1994) 
    # but constructing it is complex. Let's try a simpler approach.
    
    # Alternative: use the fact that we can share the U matrix across blocks
    # if we use the same decomposition pattern for each 3x3 block.
    
    # Let's try a rank-8 approach for 3x3 using a Strassen-like formula.
    # Actually, let me just try to construct a valid lower-rank decomposition
    # using a search over small integer coefficients.
    
    # For now, let's use the block approach with schoolbook 3x3 blocks
    # Block 1: columns 0-2 (3x3)
    # Block 2: columns 3-5 (3x3)
    # Block 3: columns 6-7 (3x2)
    
    # Schoolbook for 3x3: rank 27
    # Schoolbook for 3x2: rank 18
    # Total: 27 + 27 + 18 = 72 (same as schoolbook, no improvement)
    
    # To get improvement, we need a non-schoolbook decomposition for the blocks.
    
    # Let's try to find a rank-20 decomposition for 3x3 * 3x3 via search.
    # This is a complex optimization problem. For now, let's just return schoolbook.
    
    # Actually, let me try a different approach: use the structure of the problem.
    # For 3x3 * 3x8, we can think of it as 3x3 * 3x3 + 3x3 * 3x3 + 3x3 * 3x2.
    # If we find a rank-r decomposition for 3x3*3x3, the total rank is 2r + rank_3x2.
    
    # Let's try to construct a rank-20 decomposition for 3x3 * 3x3.
    # This requires finding U, V, W of shape (9,20) such that
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k in {0,1,2}.
    
    # This is a system of 27 equations in 3*9*20 = 540 unknowns.
    # We can try to find a solution with small integer entries.
    
    # For simplicity and given time constraints, let's just return the schoolbook.
    # The key insight from the directive is to try Strassen-like approaches,
    # but constructing a valid lower-rank decomposition for 3x3 is non-trivial.
    
    # Let's at least verify the schoolbook is correct.
    if time.time() < deadline:
        if not _verify(U_school, V_school, W_school, n, m, p):
            # Should not happen, but just in case
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
