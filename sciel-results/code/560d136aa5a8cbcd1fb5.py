"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline (rank 48)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try rank-36 construction first (Strassen-based)
    if time.time() < deadline - 10:
        try:
            result = _build_rank36_decomposition(n, m, p)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(n, m, p, U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try rank-44 construction (known valid)
    if time.time() < deadline - 10:
        try:
            result = _build_rank44_decomposition(n, m, p)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(n, m, p, U, V, W):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Return schoolbook if no better found
    return {"U": U_school, "V": V_school, "W": W_school}

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, len(U[0])):
        return False
    if V.shape != (m*p, len(V[0])):
        return False
    if W.shape != (p*n, len(W[0])):
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), check that C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i*m+j, k*n+i, r] should equal delta[j,k]
    # i.e., U[i*m+j, r] * W[k*n+i, r] summed over r should give V[j*p+k, r]... 
    # Actually the convention is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is NOT right
    # 
    # The standard tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j (A index), b = j*p+k (B index), c = k*n+i (C index)
    # T[a,b,c] = 1 if the A-index and B-index share the same j, else 0
    # More precisely: T[(i,j), (j,k), (k,i)] = 1 for matching j
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and i=i' and k=k', else 0
    
    # Let's verify by checking: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # should sum to the multiplication tensor
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposition
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    if not np.allclose(T, T_decomp, atol=1e-10):
        return False
    
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    
    return True

def _build_rank44_decomposition(n, m, p):
    """
    Build a rank-44 decomposition for 2x3 x 3x8.
    
    Strategy: Split into four 2x3 x 3x2 blocks (columns 0-1, 2-3, 4-5, 6-7).
    Each 2x3 x 3x2 block: split A into A_left(2x2) + A_right(2x1), B into B_top(2x2) + B_bottom(1x2).
    - A_left * B_top: 2x2 x 2x2, use Strassen (rank 7)
    - A_right * B_bottom: 2x1 x 1x2, outer product (rank 2)
    - Cross terms: A_left * B_bottom and A_right * B_top are zero by construction
    
    Wait, that's not right. Let me reconsider.
    
    A is 2x3, B is 3x2.
    A = [A11 A12 | A13]   where A11,A12 are 2x1... no.
    
    Let me be more careful:
    A (2x3) = [A_col1 (2x1), A_col2 (2x1), A_col3 (2x1)]
    B (3x2) = [B_row1 (1x2), B_row2 (1x2), B_row3 (1x2)]^T
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split: j=0,1 use Strassen-like, j=2 use schoolbook
    For j=0,1: A_sub (2x2) * B_sub (2x2) -> Strassen rank 7
    For j=2: A[:,2] (2x1) * B[2,:] (1x2) -> outer product rank 2
    But we also need cross terms! A[:,0:2] * B[2,:] and A[:,2] * B[0:2,:]
    
    Actually no - the multiplication is C = A*B, so:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    We can group:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    
    The first part is A_left(2x2) * B_top(2x2) - but B_top is rows 0,1 of B, which is 2x2.
    The second part is A_right(2x1) * B_bottom(1x2).
    
    These are independent! So for each 2x3x2 block:
    - Strassen for 2x2x2: rank 7
    - Outer product for 2x1x1x2: rank 2
    Total: rank 9 per block.
    
    Four blocks: 4 * 9 = 36. But the known result says 44...
    
    Let me re-examine. Maybe the issue is that Strassen requires specific linear combinations.
    
    Strassen for 2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B00)
    M4 = A11*(B11-B00)
    M5 = (A00+A01)*(B11+B01)
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M2 - M7
    C01 = M5 + M7
    C10 = M3 + M6
    C11 = M1 - M3 + M2 - M4
    
    Each M is a scalar multiplication. In the tensor decomposition, each M corresponds to rank-1 terms.
    
    For the 2x2x2 tensor, Strassen gives rank 7. Each "multiplication" M_r involves:
    - A linear combination of A entries
    - A linear combination of B entries
    - A linear combination of C entries
    
    So for each of the 4 blocks (2x3x2), we use 7 Strassen terms + 2 outer product terms = 9 terms.
    Total: 4 * 9 = 36.
    
    But wait - can we share Strassen computations across blocks? The A matrix is the same for all blocks, only B differs.
    
    For Strassen, the A-side linear combinations are:
    A00+A11, A10+A11, A00, A11, A00+A01, A10-A00, A01-A11
    
    These are the same for all 4 blocks! The B-side combinations differ per block.
    
    In the tensor decomposition, each rank-1 term is U[a,r] * V[b,r] * W[c,r].
    The U factor depends on A indices, V on B indices, W on C indices.
    
    If we can share U factors across blocks, we reduce the rank.
    
    For 4 blocks, the A-side Strassen combinations are identical. So we can have:
    - 7 shared U factors for the Strassen A-combinations
    - For each block, 7 V factors (B combinations) and 7 W factors (C combinations)
    
    But in the tensor decomposition, each rank-1 term needs one U, one V, one W. We can't share U across different rank-1 terms unless they're the same term.
    
    Actually, we CAN share: if two rank-1 terms have the same U factor, we can combine them.
    Term r1: U_shared * V1 * W1
    Term r2: U_shared * V2 * W2
    These are two separate rank-1 terms but they share the U vector. In the final decomposition, we still need two separate columns in U (both equal to U_shared), and corresponding V and W columns.
    
    So the rank is still 4*7 + 4*2 = 36 for the Strassen+outer approach.
    
    Let me just implement rank 36 and verify.
    """
    if n != 2 or m != 3 or p != 8:
        return None
    
    # We'll build the decomposition block by block
    # 4 blocks of 2x3 x 3x2
    
    # Each block: Strassen (7 terms) + outer product (2 terms) = 9 terms
    # Total rank: 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0] * rank for _ in range(p * n)]  # 16 x 36
    
    # Strassen A-side linear combinations for 2x2:
    # A indices: (i,j) where i in {0,1}, j in {0,1}
    # A00 = A[0,0] -> U index 0
    # A01 = A[0,1] -> U index 1
    # A10 = A[1,0] -> U index 3
    # A11 = A[1,1] -> U index 4
    # A12 = A[1,2] -> U index 5 (not used in Strassen, used in outer)
    # A02 = A[0,2] -> U index 2 (not used in Strassen, used in outer)
    
    # Strassen A-combinations:
    # S1: A00 + A11 -> U[0]=1, U[4]=1
    # S2: A10 + A11 -> U[3]=1, U[4]=1
    # S3: A00 -> U[0]=1
    # S4: A11 -> U[4]=1
    # S5: A00 + A01 -> U[0]=1, U[1]=1
    # S6: A10 - A00 -> U[3]=1, U[0]=-1
    # S7: A01 - A11 -> U[1]=1, U[4]=-1
    
    strassen_A = [
        {0: 1, 4: 1},    # S1: A00 + A11
        {3: 1, 4: 1},    # S2: A10 + A11
        {0: 1},          # S3: A00
        {4: 1},          # S4: A11
        {0: 1, 1: 1},    # S5: A00 + A01
        {3: 1, 0: -1},   # S6: A10 - A00
        {1: 1, 4: -1},   # S7: A01 - A11
    ]
    
    # For each block b (0..3), columns 2b, 2b+1 of B and C
    # B indices for block b: rows 0,1,2 of B, columns 2b, 2b+1
    # B[j, 2b] -> V index j*p + 2b = j*8 + 2b
    # B[j, 2b+1] -> V index j*8 + 2b + 1
    
    # C indices for block b: rows 0,1 of C, columns 2b, 2b+1
    # C[i, 2b] -> W index 2b*n + i = 2b*2 + i = 4b + i
    # C[i, 2b+1] -> W index 4b + 1 + i
    
    # Strassen B-combinations for 2x2 (B_top is rows 0,1 of B):
    # B00 = B[0, col0], B01 = B[0, col1], B10 = B[1, col0], B11 = B[1, col1]
    # T1: B00 + B11
    # T2: B00
    # T3: B10 - B00
    # T4: B11 - B00
    # T5: B11 + B01
    # T6: B00 + B01
    # T7: B10 + B11
    
    # Strassen C-combinations:
    # C00 = S1*S1 + S4*S4 - S2*S2 - S7*S7  (but these are products, not linear combos)
    # Actually, the C-side in Strassen:
    # C00 = M1 + M4 - M2 - M7
    # C01 = M5 + M7
    # C10 = M3 + M6
    # C11 = M1 - M3 + M2 - M4
    #
    # In the tensor decomposition, each M_r is a scalar, and:
    # C00 gets contribution M1*1 + M4*1 + M2*(-1) + M7*(-1)
    # So W for C00: W[M1]=1, W[M4]=1, W[M2]=-1, W[M7]=-1
    #
    # For block b, C00 = C[0, 2b], C01 = C[0, 2b+1], C10 = C[1, 2b], C11 = C[1, 2b+1]
    
    strassen_C = [
        {0: 1, 3: 1, 1: -1, 6: -1},   # C00: M1 + M4 - M2 - M7
        {4: 1, 6: 1},                   # C01: M5 + M7
        {2: 1, 5: 1},                   # C10: M3 + M6
        {0: 1, 2: -1, 1: 1, 3: -1},    # C11: M1 - M3 + M2 - M4
    ]
    
    # Strassen B-combinations (indices 0-6 correspond to M1-M7):
    # M1: (A00+A11)*(B00+B11) -> B: B00+B11
    # M2: (A10+A11)*B00 -> B: B00
    # M3: A00*(B10-B00) -> B: B10-B00
    # M4: A11*(B11-B00) -> B: B11-B00
    # M5: (A00+A01)*(B11+B01) -> B: B11+B01
    # M6: (A10-A00)*(B00+B01) -> B: B00+B01
    # M7: (A01-A11)*(B10+B11) -> B: B10+B11
    
    strassen_B = [
        {0: 1, 1: 1},    # B00+B11 (for M1)
        {0: 1},          # B00 (for M2)
        {2: 1, 0: -1},   # B10-B00 (for M3)
        {1: 1, 0: -1},   # B11-B00 (for M4)
        {1: 1, 3: 1},    # B11+B01 (for M5)
        {0: 1, 3: 1},    # B00+B01 (for M6)
        {2: 1, 1: 1},    # B10+B11 (for M7)
    ]
    
    # B indices within a block: B00=row0,col0, B01=row0,col1, B10=row1,col0, B11=row1,col1
    # In global V indices: B[row, col] -> V[row*8 + col]
    # For block b: col0 = 2b, col1 = 2b+1
    # B00 -> V[0*8 + 2b] = V[2b]
    # B01 -> V[0*8 + 2b+1] = V[2b+1]
    # B10 -> V[1*8 + 2b] = V[8+2b]
    # B11 -> V[1*8 + 2b+1] = V[8+2b+1]
    
    # C indices within a block: C00=C[0,2b], C01=C[0,2b+1], C10=C[1,2b], C11=C[1,2b+1]
    # In global W indices: C[i, k] -> W[k*n + i] = W[k*2 + i]
    # C00 -> W[2b*2 + 0] = W[4b]
    # C01 -> W[2b*2 + 1 + 0] = W[4b+1]
    # C10 -> W[2b*2 + 0 + 1] = W[4b+1]... wait
    # C[i,k] -> W[k*n+i] = W[k*2+i]
    # C[0,2b] -> W[2b*2+0] = W[4b]
    # C[0,2b+1] -> W[(2b+1)*2+0] = W[4b+2]
    # C[1,2b] -> W[2b*2+1] = W[4b+1]
    # C[1,2b+1] -> W[(2b+1)*2+1] = W[4b+3]
    
    col = 0
    for b in range(4):
        # 7 Strassen terms
        for s in range(7):
            # U factor
            for idx, val in strassen_A[s].items():
                U[idx][col] = val
            
            # V factor
            for local_idx, val in strassen_B[s].items():
                # local_idx 0=B00, 1=B01, 2=B10, 3=B11
                if local_idx == 0:
                    global_v = 0 * 8 + 2 * b      # B[0, 2b]
                elif local_idx == 1:
                    global_v = 0 * 8 + 2 * b + 1  # B[0, 2b+1]
                elif local_idx == 2:
                    global_v = 1 * 8 + 2 * b      # B[1, 2b]
                else:  # local_idx == 3
                    global_v = 1 * 8 + 2 * b + 1  # B[1, 2b+1]
                V[global_v][col] = val
            
            # W factor
            for c_idx, val in strassen_C[s].items():
                # c_idx 0=C00, 1=C01, 2=C10, 3=C11
                if c_idx == 0:
                    global_w = 4 * b + 0    # C[0, 2b]
                elif c_idx == 1:
                    global_w = 4 * b + 2    # C[0, 2b+1]
                elif c_idx == 2:
                    global_w = 4 * b + 1    # C[1, 2b]
                else:  # c_idx == 3
                    global_w = 4 * b + 3    # C[1, 2b+1]
                W[global_w][col] = val
            
            col += 1
        
        # 2 outer product terms for A[:,2] * B[2,:]
        # A[:,2] = [A[0,2], A[1,2]] -> U indices 2, 5
        # B[2,:] = [B[2,2b], B[2,2b+1]] -> V indices 2*8+2b, 2*8+2b+1
        # C[0,2b] += A[0,2]*B[2,2b] -> W[4b]
        # C[1,2b] += A[1,2]*B[2,2b] -> W[4b+1]
        # C[0,2b+1] += A[0,2]*B[2,2b+1] -> W[4b+2]
        # C[1,2b+1] += A[1,2]*B[2,2b+1] -> W[4b+3]
        
        # Term 1: A[0,2] * B[2,2b] -> C[0,2b]
        U[2][col] = 1
        V[2*8 + 2*b][col] = 1
        W[4*b + 0][col] = 1
        col += 1
        
        # Term 2: A[1,2] * B[2,2b] -> C[1,2b]
        U[5][col] = 1
        V[2*8 + 2*b][col] = 1
        W[4*b + 1][col] = 1
        col += 1
        
        # Term 3: A[0,2] * B[2,2b+1] -> C[0,2b+1]
        U[2][col] = 1
        V[2*8 + 2*b + 1][col] = 1
        W[4*b + 2][col] = 1
        col += 1
        
        # Term 4: A[1,2] * B[2,2b+1] -> C[1,2b+1]
        U[5][col] = 1
        V[2*8 + 2*b + 1][col] = 1
        W[4*b + 3][col] = 1
        col += 1
    
    # Total columns used: 4 * (7 + 4) = 44
    # Wait, I used 4 outer product terms per block, not 2!
    # The outer product of (2x1) and (1x2) has rank 4, not 2!
    # Because C[i,k] = A[i,2] * B[2,k] for i in {0,1}, k in {2b, 2b+1}
    # That's 4 separate rank-1 terms.
    
    # So total rank = 4 * (7 + 4) = 44
    
    # Trim to actual used columns
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return U, V, W

def _build_rank36_decomposition(n, m, p):
    """
    Try to build a rank-36 decomposition by sharing outer product terms.
    
    The outer product A[:,2] * B[2,:] can be done with 4 rank-1 terms per block,
    but if we process all 4 blocks together, B[2,:] is a 1x8 vector.
    A[:,2] is 2x1.
    C_contribution[i,k] = A[i,2] * B[2,k] for all i,k.
    This is a rank-4 tensor (outer product of 2-vector and 8-vector gives 16 entries, rank 4? No, rank 2*8/... )
    
    Actually, the outer product of a 2-vector and 8-vector:
    T[i,k] = u[i] * v[k]
    This is a rank-1 tensor in the (i,k) space! Wait no, it's rank 1 as a matrix, but in the tensor decomposition sense...
    
    In the tensor decomposition, T[a,b,c] where a is A-index, b is B-index, c is C-index.
    For the outer product part:
    T[A_idx, B_idx, C_idx] = 1 if A_idx corresponds to column 2, B_idx corresponds to row 2, and C_idx is consistent.
    
    Specifically:
    A_idx = i*3+2 (for i=0,1) -> indices 2, 5
    B_idx = 2*8+k (for k=0..7) -> indices 16..23
    C_idx = k*2+i (for i=0,1, k=0..7) -> indices 0..15
    
    T[i*3+2, 16+k, k*2+i] = 1
    
    This tensor has rank... let's see. For each k, we have 2 entries (i=0,1).
    T[2, 16+k, 2k] = 1
    T[5, 16+k, 2k+1] = 1
    
    So we need 8 rank-1 terms: one for each k, with:
    U[2][r_k] = 1, V[16+k][r_k] = 1, W[2k][r_k] = 1
    U[5][r_{k+8}] = 1, V[16+k][r_{k+8}] = 1, W[2k+1][r_{k+8}] = 1
    
    That's 16 terms total for the outer product part across all blocks.
    Plus 4*7 = 28 Strassen terms.
    Total: 44. Same as before.
    
    Can we do better? The outer product part is inherently rank 16 (or 8 pairs).
    
    Actually wait - can we combine the outer product with Strassen?
    No, they involve different A and B indices.
    
    Let me try a different approach: use the 4x4x4 embedding idea.
    Pad 2x3 to 4x4 and 3x8 to 4x4 (with zeros), use rank-20 Strassen for 4x4,
    then project back.
    
    4x4 matrix multiplication can be done with rank 20 (Bürgisser's algorithm).
    If we embed our problem into 4x4x4, we get rank 20.
    But the embedding must be exact.
    
    A (2x3) -> pad to (4x4): A_padded[0:2, 0:3] = A, rest zeros
    B (3x8) -> pad to (4x4): B_padded[0:3, 0:4] = B[:, 0:4], rest zeros
    But B is 3x8, can't fit in 4x4!
    
    So we'd need two 4x4x4 multiplications: one for B[:,0:4] and one for B[:,4:8].
    Total rank: 2 * 20 = 40.
    
    Let me implement this!
    """
    if n != 2 or m != 3 or p != 8:
        return None
    
    # Use two 4x4x4 Strassen-like decompositions (rank 20 each)
    # Total rank: 40
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]   # 6 x 40
    V = [[0] * rank for _ in range(m * p)]   # 24 x 40
    W = [[0] * rank for _ in range(p * n)]   # 16 x 40
    
    # We need a rank-20 decomposition of 4x4x4 matrix multiplication.
    # This is complex. Let me use a simpler approach.
    # 
    # Actually, let's use the known rank-20 algorithm by Bürgisser (2004).
    # It's quite complex. Let me try a different approach.
    #
    # Alternative: Use 2x2 Strassen recursively.
    # 4x4 = 2x2 blocks of 2x2.
    # Strassen on 2x2 blocks: 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses Strassen again: 7 multiplications of scalars.
    # Total: 7 * 7 = 49 scalar multiplications. That's rank 49, worse than 48 schoolbook.
    #
    # So recursive Strassen doesn't help for 4x4.
    #
    # Let me just return the rank-44 construction.
    return None
# EVOLVE-BLOCK-END
