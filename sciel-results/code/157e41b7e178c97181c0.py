"""Program for rank-minimal 5x5x5 matrix multiplication tensor decomposition.
Strategy: Build a rank-124 decomposition via 2+3 block split (Strassen 2x2 + Strassen 3x3),
then use rank-revealing QR on tensor slices to find and remove redundant rank-1 terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly using integer scaling by 2."""
    def scale2(mat):
        arr = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(arr - np.round(arr))) > 1e-9:
            return None
        return np.round(arr).astype(np.int64)

    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False

    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1

    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)

    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True


def _strassen_2x2():
    """Return U, V, W (4x7 matrices) for 2x2 Strassen."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1: (A00+A11)*(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[1,0]=1
    W[0,0]=1; W[3,0]=1
    # M2: (A10+A11)*B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    # M3: A00*(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    # M4: A11*(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    # M5: (A00+A01)*B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    # M6: (A10-A00)*(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7: (A01-A11)*(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1

    return U, V, W


def _strassen_3x3_2plus1():
    """
    Return U, V, W (9x24 matrices) for 3x3 matrix multiplication via 2+1 split.
    Split 3x3 into [2x2 | 2x1; 1x2 | 1x1] blocks.
    Uses 7 (Strassen 2x2) + 4 + 4 + 1 + 4 + 4 = 24 terms.
    
    Actually, let me use a cleaner approach:
    3x3 = [[A00(2x2), A01(2x1)], [A10(1x2), A11(1x1)]]
    
    C = A*B where:
    C00(2x2) = A00*B00 + A01*B10
    C01(2x1) = A00*B01 + A01*B11
    C10(1x2) = A10*B00 + A11*B10
    C11(1x1) = A10*B01 + A11*B11
    
    We compute each sub-product:
    - C00: 2x2 = 2x2 * 2x2 + 2x1 * 1x2 -> use Strassen for first (7), schoolbook for second (2*1*2=4) = 11
    - C01: 2x1 = 2x2 * 2x1 + 2x1 * 1x1 -> 2*2*1=4 + 2*1*1=2 = 6
    - C10: 1x2 = 1x2 * 2x2 + 1x1 * 1x2 -> 1*2*2=4 + 1*1*2=2 = 6
    - C11: 1x1 = 1x2 * 2x1 + 1x1 * 1x1 -> 1*2*1=2 + 1*1*1=1 = 3
    
    Total: 11+6+6+3 = 26. Hmm, that's more than 24.
    
    Let me use a different approach: use the known rank-24 for 3x3.
    Actually, the known best for 3x3 is 23 (Lader). But let me use a simpler construction.
    
    Let me just use schoolbook for 3x3 (27 terms) and rely on the rank-reduction to find savings.
    Actually, let me use a 2+1 split more carefully.
    
    Alternative: use the fact that 3x3 can be done with rank 23 (Lader's algorithm).
    But implementing Lader is complex. Let me use a simpler approach:
    
    Use 2x2 Strassen (7) for the top-left 2x2 block of A times top-left 2x2 of B,
    then handle cross terms with schoolbook.
    
    Actually, for simplicity and correctness, let me just use schoolbook 27 for 3x3
    and let the rank reduction find the savings.
    """
    # Schoolbook 3x3: 27 rank-1 terms
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                r = a*3 + b*3 + c  # 0..26
                U[a*3 + b, r] = 1.0
                V[b*3 + c, r] = 1.0
                W[c*3 + a, r] = 1.0
    
    return U, V, W


def _build_block_decomposition():
    """
    Build a decomposition for 5x5 using 2+3 block split.
    
    A = [[A00(2x2), A01(2x3)], [A10(3x2), A11(3x3)]]
    B = [[B00(2x2), B01(2x3)], [B10(3x2), B11(3x3)]]
    
    C00(2x2) = A00*B00 + A01*B10
    C01(2x3) = A00*B01 + A01*B11
    C10(3x2) = A10*B00 + A11*B10
    C11(3x3) = A10*B01 + A11*B11
    
    Terms:
    - A00*B00: 2x2x2 -> Strassen 7
    - A01*B10: 2x3 * 3x2 -> 2*3*2 = 12
    - A00*B01: 2x2 * 2x3 -> 2*2*3 = 12
    - A01*B11: 2x3 * 3x3 -> 2*3*3 = 18
    - A10*B00: 3x2 * 2x2 -> 3*2*2 = 12
    - A11*B10: 3x3 * 3x2 -> 3*3*2 = 18
    - A10*B01: 3x2 * 2x3 -> 3*2*3 = 18
    - A11*B11: 3x3 * 3x3 -> 3*3*3 = 27
    
    Total: 7+12+12+18+12+18+18+27 = 124
    """
    n, m, p = 5, 5, 5
    nm, mp, pn = 25, 25, 25
    
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _strassen_3x3_2plus1()
    
    # Total rank
    rank_total = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27  # = 124
    
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))
    
    col = 0
    
    # Helper: add terms for a sub-matrix product
    # A block at rows [ar:ar+ar_rows], cols [ac:ac+ac_cols] of A (5x5)
    # B block at rows [br:br+br_rows], cols [bc:bc+bc_cols] of B (5x5)
    # C block at rows [cr:cr+cr_rows], cols [cc:cc+cc_cols] of C (5x5)
    # But C index is (c*n+a), so C[i,j] where i = row of C = col index of A product...
    # Actually C = A*B, so C[i,j] = sum_k A[i,k]*B[k,j]
    # In our indexing: C index = (c*n + a) where a is row of A, c is col of B
    # So C row index = c (from B's column), C col index = a (from A's row)
    # Wait, the problem says W[k*n+i, r] where k is the column index of B, i is row index of A
    # So the output matrix C has C[i,j] where i = row of A (0..4), j = col of B (0..4)
    # And the flattened index is j*n + i (column-major in terms of (i,j) -> j*5+i)
    
    # Let me re-examine: target[a*m+b, b*p+c, c*n+a] = 1
    # i = a*m+b (row of A, col of A)
    # j = b*p+c (row of B, col of B)  
    # k = c*n+a (row of C = col of B, col of C = row of A)
    # So C index k = c*5 + a, meaning C is stored as C[c][a] = (B_col, A_row)
    # This means C^T is the standard matrix product, or C = (A*B)^T
    
    # For our purposes: the "C matrix" has entry at position (c, a) = sum_b A(a,b)*B(b,c)
    # So it's the standard (A*B) but with indices transposed in storage.
    
    # Block structure for A (5x5):
    # A[0:2, 0:2] = A00, A[0:2, 2:5] = A01
    # A[2:5, 0:2] = A10, A[2:5, 2:5] = A11
    
    # Block structure for B (5x5):
    # B[0:2, 0:2] = B00, B[0:2, 2:5] = B01
    # B[2:5, 0:2] = B10, B[2:5, 2:5] = B11
    
    # C = A*B (standard), stored as C[c,a] = sum_b A[a,b]*B[b,c]
    # C[0:2, 0:2] = C00, C[0:2, 2:5] = C01
    # C[2:5, 0:2] = C10, C[2:5, 2:5] = C11
    
    # C00[a,c] = sum_{b=0}^{1} A00[a,b]*B00[b,c] + sum_{b=2}^{4} A01[a,b]*B10[b,c]
    #   where A01 is A[0:2, 2:5] (2x3), B10 is B[2:5, 0:2] (3x2)
    # C01[a,c] = sum_{b=0}^{1} A00[a,b]*B01[b,c] + sum_{b=2}^{4} A01[a,b]*B11[b,c]
    #   where B01 is B[0:2, 2:5] (2x3), B11 is B[2:5, 2:5] (3x3)
    # C10[a,c] = sum_{b=0}^{1} A10[a,b]*B00[b,c] + sum_{b=2}^{4} A11[a,b]*B10[b,c]
    #   where A10 is A[2:5, 0:2] (3x2), A11 is A[2:5, 2:5] (3x3)
    # C11[a,c] = sum_{b=0}^{1} A10[a,b]*B01[b,c] + sum_{b=2}^{4} A11[a,b]*B11[b,c]
    
    def add_block_product(a_row_start, a_col_start, a_rows, a_cols,
                          b_row_start, b_col_start, b_rows, b_cols,
                          c_row_start, c_col_start, c_rows, c_cols,
                          U_sub, V_sub, W_sub, start_col):
        """Add terms from a sub-decomposition to the global matrices."""
        num_terms = U_sub.shape[1]
        for r in range(num_terms):
            for i_local in range(a_rows):
                for j_local in range(a_cols):
                    if U_sub[i_local * a_cols + j_local, r] != 0:
                        global_i = a_row_start + i_local
                        global_j = a_col_start + j_local
                        U[global_i * m + global_j, start_col + r] += U_sub[i_local * a_cols + j_local, r]
            
            for i_local in range(b_rows):
                for j_local in range(b_cols):
                    if V_sub[i_local * b_cols + j_local, r] != 0:
                        global_i = b_row_start + i_local
                        global_j = b_col_start + j_local
                        V[global_i * p + global_j, start_col + r] += V_sub[i_local * b_cols + j_local, r]
            
            # C is stored as C[c,a], so C row = c (col of B), C col = a (row of A)
            for i_local in range(c_rows):
                for j_local in range(c_cols):
                    if W_sub[i_local * c_cols + j_local, r] != 0:
                        global_c = c_row_start + i_local
                        global_a = c_col_start + j_local
                        W[global_c * n + global_a, start_col + r] += W_sub[i_local * c_cols + j_local, r]
        
        return start_col + num_terms
    
    # 1. A00(2x2) * B00(2x2) -> C00(2x2)
    col = add_block_product(0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, U2, V2, W2, col)
    
    # 2. A01(2x3) * B10(3x2) -> C00(2x2)
    # Schoolbook: 2*3*2 = 12 terms
    U_sub = np.zeros((2*3, 12))
    V_sub = np.zeros((3*2, 12))
    W_sub = np.zeros((2*2, 12))
    for a in range(2):
        for b in range(3):
            for c in range(2):
                r = a*3 + b*2 + c
                U_sub[a*3 + b, r] = 1.0
                V_sub[b*2 + c, r] = 1.0
                W_sub[c*2 + a, r] = 1.0
    col = add_block_product(0, 2, 2, 3, 2, 0, 3, 2, 0, 0, 2, 2, U_sub, V_sub, W_sub, col)
    
    # 3. A00(2x2) * B01(2x3) -> C01(2x3)
    U_sub = np.zeros((2*2, 12))
    V_sub = np.zeros((2*3, 12))
    W_sub = np.zeros((3*2, 12))
    for a in range(2):
        for b in range(2):
            for c in range(3):
                r = a*2 + b*3 + c
                U_sub[a*2 + b, r] = 1.0
                V_sub[b*3 + c, r] = 1.0
                W_sub[c*2 + a, r] = 1.0
    col = add_block_product(0, 0, 2, 2, 0, 2, 2, 3, 0, 2, 3, 2, U_sub, V_sub, W_sub, col)
    
    # 4. A01(2x3) * B11(3x3) -> C01(2x3)
    U_sub = np.zeros((2*3, 18))
    V_sub = np.zeros((3*3, 18))
    W_sub = np.zeros((3*2, 18))
    for a in range(2):
        for b in range(3):
            for c in range(3):
                r = a*3 + b*3 + c
                U_sub[a*3 + b, r] = 1.0
                V_sub[b*3 + c, r] = 1.0
                W_sub[c*2 + a, r] = 1.0
    col = add_block_product(0, 2, 2, 3, 2, 2, 3, 3, 0, 2, 3, 2, U_sub, V_sub, W_sub, col)
    
    # 5. A10(3x2) * B00(2x2) -> C10(3x2)
    U_sub = np.zeros((3*2, 12))
    V_sub = np.zeros((2*2, 12))
    W_sub = np.zeros((2*3, 12))
    for a in range(3):
        for b in range(2):
            for c in range(2):
                r = a*2 + b*2 + c
                U_sub[a*2 + b, r] = 1.0
                V_sub[b*2 + c, r] = 1.0
                W_sub[c*3 + a, r] = 1.0
    col = add_block_product(2, 0, 3, 2, 0, 0, 2, 2, 2, 0, 2, 3, U_sub, V_sub, W_sub, col)
    
    # 6. A11(3x3) * B10(3x2) -> C10(3x2)
    U_sub = np.zeros((3*3, 18))
    V_sub = np.zeros((3*2, 18))
    W_sub = np.zeros((2*3, 18))
    for a in range(3):
        for b in range(3):
            for c in range(2):
                r = a*3 + b*2 + c
                U_sub[a*3 + b, r] = 1.0
                V_sub[b*2 + c, r] = 1.0
                W_sub[c*3 + a, r] = 1.0
    col = add_block_product(2, 2, 3, 3, 2, 0, 3, 2, 2, 0, 2, 3, U_sub, V_sub, W_sub, col)
    
    # 7. A10(3x2) * B01(2x3) -> C11(3x3)
    U_sub = np.zeros((3*2, 18))
    V_sub = np.zeros((2*3, 18))
    W_sub = np.zeros((3*3, 18))
    for a in range(3):
        for b in range(2):
            for c in range(3):
                r = a*2 + b*3 + c
                U_sub[a*2 + b, r] = 1.0
                V_sub[b*3 + c, r] = 1.0
                W_sub[c*3 + a, r] = 1.0
    col = add_block_product(2, 0, 3, 2, 0, 2, 2, 3, 2, 2, 3, 3, U_sub, V_sub, W_sub, col)
    
    # 8. A11(3x3) * B11(3x3) -> C11(3x3)
    col = add_block_product(2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, U3, V3, W3, col)
    
    assert col == rank_total, f"Column count mismatch: {col} != {rank_total}"
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list


def _rank_revealing_reduction(U, V, W, n, m, p, deadline):
    """
    Use rank-revealing QR on tensor slices to find redundant rank-1 terms.
    For each fixed value of one index, the remaining tensor is a matrix.
    If a rank-1 term has zero contribution to all slices, it's redundant.
    
    More practically: for each column r, check if its tensor contribution
    T_r[i,j,k] = U[i,r]*V[j,r]*W[k,r] is a linear combination of others.
    We do this by checking on a set of "probe" tensor entries.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    current_rank = U_arr.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1.0
    
    # For each column, compute its tensor contribution
    # T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
    
    # Strategy: for each column r, try to express T_r as a combination of others.
    # Use a set of probe points (i,j,k) where the target is 1.
    
    # Get all (i,j,k) where target[i,j,k] = 1
    probe_points = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                probe_points.append((a*m+b, b*p+c, c*n+a))
    
    num_probes = len(probe_points)  # 125
    
    # Build the contribution matrix: M[r, probe] = U[i,r]*V[j,r]*W[k,r]
    M = np.zeros((current_rank, num_probes), dtype=np.float64)
    for r in range(current_rank):
        for idx, (i, j, k) in enumerate(probe_points):
            M[r, idx] = U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
    
    # Now for each column r, check if M[r,:] is in the span of M[other rows,:]
    # Use QR with column pivoting to find the rank of the matrix
    # If a row is in the span of others, we can remove it.
    
    # More efficient: iteratively remove columns that are linearly dependent.
    # Start with all columns, try to remove one at a time.
    
    active = list(range(current_rank))
    removed = []
    
    for _ in range(50):  # Try to remove up to 50 columns
        if time.time() > deadline - 15:
            break
        if len(active) <= 100:  # Don't go below rank 100
            break
        
        # Find the column with the smallest "importance"
        # Use the norm of the row in the M matrix
        norms = np.array([np.linalg.norm(M[r, :]) for r in active])
        
        # Try removing the column with smallest norm first
        min_idx = np.argmin(norms)
        r_to_remove = active[min_idx]
        
        # Check if M[r_to_remove, :] is in the span of other active rows
        other_active = [r for r in active if r != r_to_remove]
        
        if len(other_active) == 0:
            break
        
        # Set up least squares: M[r_to_remove, :] = alpha^T * M[other_active, :]
        A_mat = M[other_active, :].T  # (num_probes, len(other_active))
        b_vec = M[r_to_remove, :]      # (num_probes,)
        
        try:
            alpha, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            break
        
        predicted = A_mat @ alpha
        residual = np.linalg.norm(b_vec - predicted)
        b_norm = np.linalg.norm(b_vec)
        
        if b_norm > 1e-10 and residual / b_norm < 1e-10:
            # This column is redundant! Remove it.
            active.remove(r_to_remove)
            removed.append(r_to_remove)
            print(f"  Removed column {r_to_remove}, rank now {len(active)}")
        else:
            # Try the next smallest
            # Actually, let's just try all columns in order of increasing norm
            found = False
            for idx in range(len(active)):
                if time.time() > deadline - 15:
                    break
                r_try = active[idx]
                other_active = [r for r in active if r != r_try]
                
                if len(other_active) == 0:
                    break
                
                A_mat = M[other_active, :].T
                b_vec = M[r_try, :]
                
                try:
                    alpha, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    continue
                
                predicted = A_mat @ alpha
                residual = np.linalg.norm(b_vec - predicted)
                b_norm = np.linalg.norm(b_vec)
                
                if b_norm > 1e-10 and residual / b_norm < 1e-10:
                    active.remove(r_try)
                    removed.append(r_try)
                    print(f"  Removed column {r_try}, rank now {len(active)}")
                    found = True
                    break
            
            if not found:
                break
    
    if not removed:
        return None
    
    # Build the reduced matrices
    U_new = np.zeros((nm, len(active)))
    V_new = np.zeros((mp, len(active)))
    W_new = np.zeros((pn, len(active)))
    
    for new_r, old_r in enumerate(active):
        U_new[:, new_r] = U_arr[:, old_r]
        V_new[:, new_r] = V_arr[:, old_r]
        W_new[:, new_r] = W_arr[:, old_r]
    
    U_list = [[float(x) for x in row] for row in U_new]
    V_list = [[float(x) for x in row] for row in V_new]
    W_list = [[float(x) for x in row] for row in W_new]
    
    return U_list, V_list, W_list


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Build the rank-124 block decomposition
    U, V, W = _build_block_decomposition()
    print(f"Initial rank: {len(U[0])}")
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        print("Initial decomposition failed verification, using schoolbook")
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    U_sb[a * m + b][column] = 1
                    V_sb[b * p + c][column] = 1
                    W_sb[c * n + a][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = len(U[0])
    
    # Try rank reduction
    if time.time() < deadline - 30:
        result = _rank_revealing_reduction(U, V, W, n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                new_rank = len(U_new[0])
                print(f"Rank reduced to {new_rank}")
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = new_rank
            else:
                print("Reduced decomposition failed verification, keeping original")
    
    # Try additional random-based reduction
    if time.time() < deadline - 20:
        U_arr = np.array(best_result["U"], dtype=np.float64)
        V_arr = np.array(best_result["V"], dtype=np.float64)
        W_arr = np.array(best_result["W"], dtype=np.float64)
        current_rank = U_arr.shape[1]
        
        nm, mp, pn = n*m, m*p, p*n
        
        target = np.zeros((nm, mp, pn), dtype=np.float64)
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    target[a*m + b, b*p + c, c*n + a] = 1.0
        
        # Try removing pairs of columns and replacing with a single column
        # This is more complex; skip for now and just try single removals again
        # with a different probe set
        
        rng = np.random.RandomState(42)
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random probe set
            num_probes = 200
            probe_indices = []
            for _ in range(num_probes):
                a = rng.randint(n)
                b = rng.randint(m)
                c = rng.randint(p)
                probe_indices.append((a*m+b, b*p+c, c*n+a))
            
            # Build M matrix
            M = np.zeros((current_rank, num_probes), dtype=np.float64)
            for r in range(current_rank):
                for idx, (i, j, k) in enumerate(probe_indices):
                    M[r, idx] = U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
            
            # Try to remove a random column
            r_try = rng.randint(current_rank)
            other = [r for r in range(current_rank) if r != r_try]
            
            A_mat = M[other, :].T
            b_vec = M[r_try, :]
            
            try:
                alpha, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            predicted = A_mat @ alpha
            residual = np.linalg.norm(b_vec - predicted)
            b_norm = np.linalg.norm(b_vec)
            
            if b_norm > 1e-10 and residual / b_norm < 1e-10:
                # Remove column r_try
                keep = [r for r in range(current_rank) if r != r_try]
                U_new = [[float(U_arr[i, r]) for r in keep] for i in range(nm)]
                V_new = [[float(V_arr[i, r]) for r in keep] for i in range(mp)]
                W_new = [[float(W_arr[i, r]) for r in keep] for i in range(pn)]
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    current_rank -= 1
                    U_arr = np.array(U_new, dtype=np.float64)
                    V_arr = np.array(V_new, dtype=np.float64)
                    W_arr = np.array(W_new, dtype=np.float64)
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = current_rank
                    print(f"Random removal: rank now {current_rank}")
                else:
                    break
    
    print(f"Best rank found: {best_rank}")
    return best_result

# EVOLVE-BLOCK-END
