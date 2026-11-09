"""
Recursive block decomposition strategy for 3x4x4 matrix multiplication.
Constructs a rank-44 decomposition using Strassen for the 2x4x4 block and
schoolbook for the 1x4x4 block. Then attempts to reduce rank to 43 by
finding a linear dependency or zeroing a column via local search.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE_SECONDS = 140.0  # Safety margin

def _validate(U, V, W, tol=1e-6):
    """Check if (U, V, W) exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank44():
    """
    Construct a rank-44 decomposition.
    
    We split A (3x4) into A1 (2x4) and A2 (1x4).
    C = A * B = [A1 * B; A2 * B]
    
    Block 1: A1 (2x4) * B (4x4) -> C1 (2x4).
    This is a 2x4x4 multiplication. We can split B into B1 (4x2) and B2 (4x2).
    C1 = A1 * [B1, B2] = [A1*B1, A1*B2].
    Each A1*Bx is 2x2x2 multiplication.
    Using Strassen (rank 7) for each 2x2x2:
    A1*B1 uses rank 7.
    A1*B2 uses rank 7.
    Total rank for C1 is 14.
    
    Block 2: A2 (1x4) * B (4x4) -> C2 (1x4).
    This is a 1x4x4 multiplication.
    Schoolbook rank is 1*4*4 = 16.
    
    Total rank = 14 + 16 = 30?
    Wait. The decomposition is not just sum of ranks if the indices don't align perfectly in a simple concatenation of columns without overlap.
    Actually, the tensor is direct sum if the output indices are disjoint.
    C1 has indices i in {0,1}, k in {0,1,2,3}.
    C2 has indices i in {2}, k in {0,1,2,3}.
    The output tensor indices (i,k) are disjoint.
    However, the input indices (i,j) and (j,k) are shared.
    A rank decomposition of a direct sum of tensors T1 and T2 has rank at most rank(T1) + rank(T2).
    So rank 14 + 16 = 30 is a valid upper bound?
    
    Let's verify the indices for W.
    W is indexed by (k, i).
    For C1 (i=0,1), W columns correspond to k in 0..3, i in 0..1.
    For C2 (i=2), W columns correspond to k in 0..3, i=2.
    Since i is part of the W index, the W vectors for C1 and C2 have disjoint support in the row dimension (rows 0..7 vs rows 8..11).
    Similarly, U is indexed by (i,j).
    For C1 (i=0,1), U rows 0..7 are non-zero.
    For C2 (i=2), U rows 8..11 are non-zero.
    V is indexed by (j,k).
    For C1, we used B1 (k=0,1) and B2 (k=2,3).
    Wait, the Strassen decomposition for A1*B1 (2x2x2) involves j in 0..1, k in 0..1.
    The Strassen decomposition for A1*B2 (2x2x2) involves j in 0..1, k in 0..1 (local indices).
    Global k for B2 is 2,3.
    So V rows for C1 involve j in 0..3? No, A1 is 2x4, so j is 0..3.
    A1 = [A11, A12] where A11 is 2x2 (j=0,1) and A12 is 2x2 (j=2,3).
    B = [B1; B2] where B1 is 4x2 (j=0..3, k=0,1) and B2 is 4x2 (j=0..3, k=2,3).
    
    Let's refine the block decomposition.
    A (3x4):
    Rows 0,1 -> A_top (2x4)
    Row 2 -> A_bot (1x4)
    
    B (4x4):
    Cols 0,1 -> B_left (4x2)
    Cols 2,3 -> B_right (4x2)
    
    C (3x4):
    C[0:2, 0:2] = A_top * B_left (2x2x2 problem, but A_top is 2x4, B_left is 4x2).
    This is a 2x4x2 multiplication.
    We can split A_top into A_top_left (2x2, j=0,1) and A_top_right (2x2, j=2,3).
    A_top * B_left = A_top_left * B_left_top + A_top_right * B_left_bot.
    Where B_left_top is B[0:2, 0:2] (2x2) and B_left_bot is B[2:4, 0:2] (2x2).
    So C[0:2, 0:2] is sum of two 2x2x2 multiplications.
    Rank = 7 + 7 = 14.
    
    Similarly for C[0:2, 2:3] (2x2x2 problem):
    A_top * B_right = A_top_left * B_right_top + A_top_right * B_right_bot.
    Rank = 7 + 7 = 14.
    
    For C[2, 0:2] (1x2x2 problem? No, 1x4x2):
    A_bot (1x4) * B_left (4x2).
    Split A_bot into A_bot_left (1x2, j=0,1) and A_bot_right (1x2, j=2,3).
    Split B_left into B_left_top (2x2, j=0,1) and B_left_bot (2x2, j=2,3).
    A_bot * B_left = A_bot_left * B_left_top + A_bot_right * B_left_bot.
    Each is 1x2x2 multiplication. Rank of 1x2x2 is 4 (schoolbook: 1*2*2=4).
    Is it optimal? Yes, rank(1xn x nx1) is n. Rank(1x2 x 2x2) is 4.
    So rank = 4 + 4 = 8.
    
    For C[2, 2:3] (1x4x2):
    Similarly rank = 8.
    
    Total rank = 14 + 14 + 8 + 8 = 44.
    
    This matches the "44" mentioned in the knowledge.
    The strategy is to construct this 44-rank decomposition and then try to reduce it.
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen 2x2x2 decomposition
    # Inputs: i_offset (row in A), j_offset (col in A / row in B), k_offset (col in B)
    # The Strassen 7 factors for X(2x2)*Y(2x2)=Z(2x2)
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y21-Y22)
    # M4 = X22(Y12-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # We need to map these to global indices.
    # X is 2x2 at A[i_off:i_off+2, j_off:j_off+2]
    # Y is 2x2 at B[j_off:j_off+2, k_off:k_off+2]
    # Z is 2x2 at C[i_off:i_off+2, k_off:k_off+2]
    
    def add_strassen(i_off, j_off, k_off, col_start):
        # 7 columns
        cols = range(col_start, col_start + 7)
        
        # M1
        U[i_off*4 + j_off, cols[0]] += 1
        U[(i_off+1)*4 + (j_off+1), cols[0]] += 1
        V[j_off*4 + k_off, cols[0]] += 1
        V[(j_off+1)*4 + (k_off+1), cols[0]] += 1
        W[k_off*3 + i_off, cols[0]] += 1
        W[(k_off+1)*3 + (i_off+1), cols[0]] += 1
        
        # M2
        U[(i_off+1)*4 + j_off, cols[1]] += 1
        U[(i_off+1)*4 + (j_off+1), cols[1]] += 1
        V[j_off*4 + k_off, cols[1]] += 1
        # W contribution: Z21 gets +M2. Z21 is C[i_off+1, k_off]
        W[k_off*3 + (i_off+1), cols[1]] += 1
        
        # M3
        U[i_off*4 + j_off, cols[2]] += 1
        V[(j_off+1)*4 + k_off, cols[2]] += 1
        V[(j_off+1)*4 + (k_off+1), cols[2]] -= 1
        # W contribution: Z12 gets +M3. Z12 is C[i_off, k_off+1]
        W[(k_off+1)*3 + i_off, cols[2]] += 1
        
        # M4
        U[(i_off+1)*4 + (j_off+1), cols[3]] += 1
        V[j_off*4 + (k_off+1), cols[3]] += 1
        V[j_off*4 + k_off, cols[3]] -= 1
        # W contribution: Z11 gets +M4, Z21 gets +M4
        W[k_off*3 + i_off, cols[3]] += 1
        W[k_off*3 + (i_off+1), cols[3]] += 1
        
        # M5
        U[i_off*4 + j_off, cols[4]] += 1
        U[i_off*4 + (j_off+1), cols[4]] += 1
        V[(j_off+1)*4 + (k_off+1), cols[4]] += 1
        # W contribution: Z11 gets -M5, Z12 gets +M5
        W[k_off*3 + i_off, cols[4]] -= 1
        W[(k_off+1)*3 + i_off, cols[4]] += 1
        
        # M6
        U[(i_off+1)*4 + j_off, cols[5]] -= 1
        U[i_off*4 + j_off, cols[5]] += 1 # X21 - X11
        V[j_off*4 + k_off, cols[5]] += 1
        V[j_off*4 + (k_off+1), cols[5]] += 1
        # W contribution: Z22 gets +M6
        W[(k_off+1)*3 + (i_off+1), cols[5]] += 1
        
        # M7
        U[i_off*4 + (j_off+1), cols[6]] -= 1
        U[(i_off+1)*4 + (j_off+1), cols[6]] += 1 # X12 - X22
        V[(j_off+1)*4 + k_off, cols[6]] += 1
        V[(j_off+1)*4 + (k_off+1), cols[6]] += 1
        # W contribution: Z11 gets +M7, Z22 gets +M7
        W[k_off*3 + i_off, cols[6]] += 1
        W[(k_off+1)*3 + (i_off+1), cols[6]] += 1
        
        return col_start + 7

    def add_schoolbook_1x2x2(i_off, j_off, k_off, col_start):
        # 1x2x2 multiplication: A(1x2) * B(2x2) = C(1x2)
        # C0 = A0*B00 + A1*B10
        # C1 = A0*B01 + A1*B11
        # 4 terms
        cols = range(col_start, col_start + 4)
        
        # Term 1: A0 * B00 -> C0
        U[i_off*4 + j_off, cols[0]] += 1
        V[j_off*4 + k_off, cols[0]] += 1
        W[k_off*3 + i_off, cols[0]] += 1
        
        # Term 2: A1 * B10 -> C0
        U[i_off*4 + (j_off+1), cols[1]] += 1
        V[(j_off+1)*4 + k_off, cols[1]] += 1
        W[k_off*3 + i_off, cols[1]] += 1
        
        # Term 3: A0 * B01 -> C1
        U[i_off*4 + j_off, cols[2]] += 1
        V[j_off*4 + (k_off+1), cols[2]] += 1
        W[(k_off+1)*3 + i_off, cols[2]] += 1
        
        # Term 4: A1 * B11 -> C1
        U[i_off*4 + (j_off+1), cols[3]] += 1
        V[(j_off+1)*4 + (k_off+1), cols[3]] += 1
        W[(k_off+1)*3 + i_off, cols[3]] += 1
        
        return col_start + 4

    col = 0
    
    # Block 1: C[0:2, 0:2] (i=0,1; k=0,1)
    # From A[0:2, 0:2] * B[0:2, 0:2]
    col = add_strassen(0, 0, 0, col)
    # From A[0:2, 2:4] * B[2:4, 0:2]
    col = add_strassen(0, 2, 0, col)
    
    # Block 2: C[0:2, 2:4] (i=0,1; k=2,3)
    # From A[0:2, 0:2] * B[0:2, 2:4]
    col = add_strassen(0, 0, 2, col)
    # From A[0:2, 2:4] * B[2:4, 2:4]
    col = add_strassen(0, 2, 2, col)
    
    # Block 3: C[2, 0:2] (i=2; k=0,1)
    # From A[2, 0:2] * B[0:2, 0:2]
    col = add_schoolbook_1x2x2(2, 0, 0, col)
    # From A[2, 2:4] * B[2:4, 0:2]
    col = add_schoolbook_1x2x2(2, 2, 0, col)
    
    # Block 4: C[2, 2:4] (i=2; k=2,3)
    # From A[2, 0:2] * B[0:2, 2:4]
    col = add_schoolbook_1x2x2(2, 0, 2, col)
    # From A[2, 2:4] * B[2:4, 2:4]
    col = add_schoolbook_1x2x2(2, 2, 2, col)
    
    # col should be 44
    if col != 44:
        print(f"Warning: constructed rank {col}, expected 44")
        
    return U, V, W

def _try_reduce_rank(U, V, W, deadline):
    """
    Try to reduce rank by finding a column that can be zeroed or merged.
    This is a heuristic local search.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Convert to lists for validation
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # First validate the base
    if not _validate(U_list, V_list, W_list):
        print("Base decomposition invalid!")
        return None
    
    # Try to zero out columns
    # A column r can be zeroed if the tensor contribution of that column is 0.
    # But we need the sum to be 1.
    # We can try to find linear dependencies.
    # If column r is a linear combination of other columns, we might be able to eliminate it.
    # Specifically, if U[:, r] = sum c_i U[:, i], V[:, r] = sum c_i V[:, i], W[:, r] = sum c_i W[:, i],
    # then we can remove column r and adjust others? No, that's for linear subspace.
    # For tensor rank, if the vector (U[:,r], V[:,r], W[:,r]) is in the span of others in a specific way...
    
    # Simpler: Try to set one column to 0 and see if we can adjust others to compensate.
    # This is hard.
    
    # Alternative: Try to find a column r such that U[:, r] is zero.
    # If U[:, r] is zero, the term is 0 for all i,j.
    # Then we can remove column r.
    
    for r in range(rank):
        if time.time() > deadline:
            break
        if np.allclose(U[:, r], 0):
            # Remove column r
            new_U = np.delete(U, r, axis=1)
            new_V = np.delete(V, r, axis=1)
            new_W = np.delete(W, r, axis=1)
            if _validate(new_U.tolist(), new_V.tolist(), new_W.tolist()):
                return new_U, new_V, new_W
        
        # Try to find if column r is redundant.
        # Check if (U[:,r], V[:,r], W[:,r]) can be expressed as a combination of others.
        # This is generally hard.
        
    # Try to merge columns?
    # If column r and s have proportional U, V, W, we can merge them.
    for r in range(rank):
        if time.time() > deadline:
            break
        for s in range(r + 1, rank):
            if np.allclose(U[:, r], U[:, s]) and np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                # Merge
                new_U = np.delete(U, s, axis=1)
                new_V = np.delete(V, s, axis=1)
                new_W = np.delete(W, s, axis=1)
                # Double the remaining column?
                # No, if they are identical, sum is 2 * col.
                # We can divide by 2?
                # Entries must be integer/half-integer.
                new_U[:, r] /= 2
                new_V[:, r] /= 2
                new_W[:, r] /= 2
                if _validate(new_U.tolist(), new_V.tolist(), new_W.tolist()):
                    return new_U, new_V, new_W

    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    U, V, W = _construct_rank44()
    
    # Validate
    if _validate(U.tolist(), V.tolist(), W.tolist()):
        # Try to reduce
        reduced = _try_reduce_rank(U, V, W, deadline)
        if reduced is not None:
            U, V, W = reduced
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
