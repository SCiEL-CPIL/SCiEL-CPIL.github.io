import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U @ V * W^T reproduces the matrix multiplication tensor.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Output convention: C[k*n + i, j] = sum_j A[i,j] * B[j,k]
    Tensor T[i,j,k] = delta, where the bilinear form is:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid (i,j,k)
      and 0 otherwise (but since every (i,j,k) is valid for matrix mult, just check == 1)
    """
    rank = U.shape[1]
    # Build the full tensor check
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Efficient: use einsum
    # U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's reshape:
    # U_flat: (n*m, rank) -> reshape to (n, m, rank)
    U3 = U.reshape(n, m, rank)
    # V_flat: (m*p, rank) -> reshape to (m, p, rank)
    V3 = V.reshape(m, p, rank)
    # W_flat: (p*n, rank) -> reshape to (p, n, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is: for each i,j,k, dot product of U3[i,j,:], V3[j,k,:], W3[k,i,:]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check that T[i,j,k] == 1 for all i,j,k
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True


def _snap_to_half_integer(arr):
    """Snap array values to nearest half-integer (i.e., k/2 for integer k)."""
    return np.round(arr * 2) / 2.0


def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2x2.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    # Strassen's 7 multiplications for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # U: maps (i,j) of A to 7 factors
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    U = np.zeros((4, 7), dtype=float)
    # V: maps (j,k) of B to 7 factors
    V = np.zeros((4, 7), dtype=float)
    # W: maps (k,i) of C to 7 factors
    W = np.zeros((4, 7), dtype=float)
    
    # m1: (a11+a22)(b11+b22)
    # A: a11->U[0,0]=1, a22->U[3,0]=1
    # B: b11->V[0,0]=1, b22->V[3,0]=1
    # C: c11->W[0,0]=1, c22->W[3,0]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # m2: (a21+a22)b11
    # A: a21->U[2,1]=1, a22->U[3,1]=1
    # B: b11->V[0,1]=1
    # C: c21->W[2,1]=1, c22->W[3,1]=-1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # m3: a11(b12-b22)
    # A: a11->U[0,2]=1
    # B: b12->V[1,2]=1, b22->V[3,2]=-1
    # C: c12->W[1,2]=1, c22->W[3,2]=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # m4: a22(b21-b11)
    # A: a22->U[3,3]=1
    # B: b21->V[2,3]=1, b11->V[0,3]=-1
    # C: c11->W[0,3]=1, c21->W[2,3]=1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # m5: (a11+a12)b22
    # A: a11->U[0,4]=1, a12->U[1,4]=1
    # B: b22->V[3,4]=1
    # C: c11->W[0,4]=-1, c12->W[1,4]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # m6: (a21-a11)(b11+b12)
    # A: a21->U[2,5]=1, a11->U[0,5]=-1
    # B: b11->V[0,5]=1, b12->V[1,5]=1
    # C: c22->W[3,5]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # m7: (a12-a22)(b21+b22)
    # A: a12->U[1,6]=1, a22->U[3,6]=-1
    # B: b21->V[2,6]=1, b22->V[3,6]=1
    # C: c11->W[0,6]=1, c22->W[3,6]=-1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[3,6] = -1
    
    return U, V, W


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _embed_strassen_blocks(n, m, p):
    """
    Partition the 2x5x6 matrix multiplication into 2x2x2 blocks where possible,
    using Strassen for those blocks and schoolbook for the rest.
    
    For 2x5x6: We can partition A(2x5) into blocks of size 2x2, and B(5x6) into 2x2 blocks.
    Since 5 = 2+2+1 and 6 = 2+2+2, we can have:
    - A: 2x5 = rows 0-1 (2 rows), cols 0-4 (5 cols)
    - B: 5x6 = rows 0-4 (5 rows), cols 0-5 (6 cols)
    
    We can group columns of A and rows of B into blocks of size 2 (with remainder 1),
    and columns of B into blocks of size 2.
    
    For each pair of column-blocks (of A/B rows) and column-blocks (of B), we have a 2x2x2 sub-problem.
    
    A has 2 rows, so the "i" dimension is just 2 (one block).
    A has 5 cols -> 2 full blocks of 2 + 1 leftover.
    B has 5 rows -> 2 full blocks of 2 + 1 leftover.
    B has 6 cols -> 3 full blocks of 2.
    
    So we have (2+1) x (2+1) = 9 column/row pairs, each with B having 3 col-blocks of 2.
    Total 9 * 3 = 27 sub-blocks of size 2x2x2.
    
    For the 4 sub-blocks where both A-col-block and B-row-block are full (size 2), use Strassen (rank 7).
    For sub-blocks with a leftover (size 2x1x2 or 2x2x1 or 1x2x2), use schoolbook.
    
    Let's enumerate:
    A-col-blocks: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    B-row-blocks: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    B-col-blocks: [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
    
    For each (a_block, b_row_block, b_col_block):
    - a_block has 2 rows (always, since A has 2 rows)
    - b_row_block has size s_ar (2 or 1)
    - b_col_block has size s_bc (always 2)
    
    Sub-problem: 2 x s_ar x 2
    
    If s_ar = 2: 2x2x2 -> Strassen rank 7
    If s_ar = 1: 2x1x2 -> schoolbook rank 4
    
    Count:
    - (a_block in [0,1,2], b_row_block in [0,1,2], b_col_block in [0,1,2])
    - b_row_block sizes: [2,2,1]
    
    For b_row_block=0 (size 2): 3 a_blocks * 3 b_col_blocks = 9 sub-blocks, each 2x2x2 -> rank 7 each -> 63
    For b_row_block=1 (size 2): 3 a_blocks * 3 b_col_blocks = 9 sub-blocks, each 2x2x2 -> rank 7 each -> 63
    For b_row_block=2 (size 1): 3 a_blocks * 3 b_col_blocks = 9 sub-blocks, each 2x1x2 -> rank 4 each -> 36
    
    Total: 63 + 63 + 36 = 162. That's worse than schoolbook (60).
    
    Wait, I think I'm overcomplicating this. The issue is that Strassen only helps when we have a full 2x2x2 block. But the way I'm partitioning, I'm splitting the matrix into independent sub-problems, which doesn't exploit cross-block structure.
    
    Actually, for 2x5x6, the standard approach would be to use a Kronecker-like decomposition. Let me think differently.
    
    Actually, the key insight is: if we can find a rank-r decomposition for a smaller problem and "embed" it, we can reduce the overall rank. For 2x5x6, one approach is to note that:
    
    2x5x6 can be viewed as combining:
    - A 2x4x6 part (using 2x2x2 Strassen blocks in a structured way)
    - A 2x1x6 part (schoolbook)
    
    Or use a more sophisticated embedding.
    
    Let me try a different approach: use the fact that 5 = 2+2+1 and 6 = 2+2+2.
    
    Actually, let me just try a random search for low-rank decompositions, starting from Strassen-based initializations.
    """
    pass


def _random_search(n, m, p, start_rank, end_rank, time_limit=120):
    """
    Random search for a rank-R decomposition with half-integer entries.
    We initialize U and V randomly (integers in small range), then solve for W.
    The constraint is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    This is a linear system in W given U and V.
    """
    best_result = None
    best_rank = None
    deadline = time.time() + time_limit - 5  # safety margin
    
    U, V = _build_strassen_2x2x2()
    U_school, V_school = _build_schoolbook(n, m, p)
    W_school = np.zeros((p * n, n * m * p), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    for R in range(start_rank, end_rank, -1):
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random U and V with small integer entries
            U_rand = np.random.randint(-2, 3, size=(n * m, R), dtype=float)
            V_rand = np.random.randint(-2, 3, size=(m * p, R), dtype=float)
            
            # Solve for W: for each (k,i) [row of W], we need:
            # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i,j
            # Actually, the constraint is: for each (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # 
            # Let's define for each r: A_r = U[:, r] (size n*m), B_r = V[:, r] (size m*p), c_r = W[:, r] (size p*n)
            # Then the bilinear form is: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
            # 
            # This is: sum_r M_r[i,j,k] * W[k*n+i, r] where M_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r]
            # 
            # For a fixed (k,i), the constraint across all j is:
            # sum_j [sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]] = sum_j ... 
            # No, the constraint is per (i,j,k), not summed over j.
            #
            # So we have n*m*p equations and p*n*R unknowns (entries of W).
            # This is underdetermined in general. We can use least squares.
            
            # Build the linear system:
            # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
            # Equation for (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            num_eqs = n * m * p
            num_unknowns = p * n * R
            
            A_mat = np.zeros((num_eqs, num_unknowns), dtype=float)
            b_vec = np.ones(num_eqs, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            unknown_idx = (k * n + i) * R + r
                            A_mat[eq_idx, unknown_idx] = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            W_sol = W_sol.reshape(p * n, R)
            
            # Snap to half-integers
            W_snapped = _snap_to_half_integer(W_sol)
            
            # Verify
            if _verify(U_rand, V_rand, W_snapped, n, m, p):
                best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_snapped.tolist()}
                best_rank = R
                break  # Found valid for this R, try lower
        
        if best_rank == R:
            continue  # Found for this R, try lower
        # If not found for this R, continue to lower R
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try random search from rank 55 down to 40
    # Start from 55 since we know 54 might be possible with Strassen
    for start_rank in [55, 54, 53, 52, 50]:
        if time.time() > deadline - 30:
            break
        result = _random_search(n, m, p, start_rank, max(40, start_rank - 5), time_limit=30)
        if result is not None:
            # Extract rank
            rank = len(result["U"][0]) if result["U"] else 0
            if rank < best_rank:
                best_rank = rank
                best_result = result
                break
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
