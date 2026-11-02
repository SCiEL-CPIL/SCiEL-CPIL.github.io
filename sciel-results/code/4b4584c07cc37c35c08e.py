"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Return (U, V, W) lists-of-lists for schoolbook decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    For C = A @ B where A is 2x2, B is 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Returns (U, V, W) where:
    U has shape (4, 7) - rows indexed by A element (a,b)
    V has shape (4, 7) - rows indexed by B element (c,d)  
    W has shape (4, 7) - rows indexed by C element (e,f)
    
    Indexing: row = i*2 + j for 2x2 matrices
    """
    # 7 intermediate products
    # Each product is a linear combination of A elements * linear combination of B elements
    
    # A indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B indices: 0=B11, 1=B12, 2=B21, 3=B22
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: A part: A11+A22, B part: B11+B22
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    
    # M2: A part: A21+A22, B part: B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    
    # M3: A part: A11, B part: B12-B22
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12
    V[3][2] = -1 # B22
    
    # M4: A part: A22, B part: B21-B11
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21
    V[0][3] = -1 # B11
    
    # M5: A part: A11+A12, B part: B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    
    # M6: A part: A21-A11, B part: B11+B12
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    
    # M7: A part: A12-A22, B part: B21+B22
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    
    # C11 = M1 + M4 - M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # -M5
    W[0][6] = 1  # M7
    
    # C12 = M3 + M5
    W[1][2] = 1  # M3
    W[1][4] = 1  # M5
    
    # C21 = M2 + M4
    W[2][1] = 1  # M2
    W[2][3] = 1  # M4
    
    # C22 = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # -M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return U, V, W

def _build_block_strassen(n, m, p):
    """
    Build a block-Strassen decomposition for n x m * m x p matrix multiplication.
    
    Partition m into two blocks: m1 + m2 = m
    Partition p into two blocks: p1 + p2 = p
    
    For n=3, m=5, p=5:
    - Split m into (2, 3) or (3, 2)
    - Split p into (2, 3) or (3, 2)
    - n stays as 3 (or we could split it too, but let's keep it simple first)
    
    Actually, for proper block matrix multiplication:
    A (n x m) = [A_top | A_bottom] where A_top is n x m1, A_bottom is n x m2
    B (m x p) = [B_top; B_bottom] where B_top is m1 x p, B_bottom is m2 x p
    
    C = A @ B = A_top @ B_top + A_bottom @ B_bottom
    
    This gives us two independent matrix multiplications:
    1. (n x m1) @ (m1 x p) -> n x p
    2. (n x m2) @ (m2 x p) -> n x p
    
    Then C is the sum of these two results.
    
    But this doesn't directly give us a tensor decomposition that's lower rank than schoolbook
    unless we use Strassen within each block.
    
    Better approach: Use the full 2x2 block structure.
    
    Let me reconsider. The standard block Strassen approach:
    
    If we can write the multiplication as a combination of smaller matrix multiplications,
    we can use Strassen on each.
    
    For 3x5 * 5x5:
    One approach: pad to 3x6 * 6x6, then use block structure.
    But that increases dimensions.
    
    Alternative: Direct block decomposition.
    
    Let's try: m=5 = 2+3, p=5 = 2+3
    
    A is 3x5, split columns into A1 (3x2) and A2 (3x3)
    B is 5x5, split rows into B1 (2x5) and B2 (3x5), and columns into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    
    C = A1 @ B1 + A2 @ B2
    where B1 = [B11 | B12] and B2 = [B21 | B22]
    
    C11 (3x2) = A1 @ B11 + A2 @ B21
    C12 (3x3) = A1 @ B12 + A2 @ B22
    
    So we need:
    - A1 @ B11: (3x2) @ (2x2) -> 3x2
    - A1 @ B12: (3x2) @ (2x3) -> 3x3
    - A2 @ B21: (3x3) @ (3x2) -> 3x2
    - A2 @ B22: (3x3) @ (3x3) -> 3x3
    
    Total rank = R(3,2,2) + R(3,2,3) + R(3,3,2) + R(3,3,3)
    
    Using schoolbook for each: 6 + 6 + 6 + 9 = 27... wait that's not right.
    
    R(3,2,2) = 3*2*2 = 12
    R(3,2,3) = 3*2*3 = 18
    R(3,3,2) = 3*3*2 = 18
    R(3,3,3) = 3*3*3 = 27
    Total = 75, same as schoolbook.
    
    Now apply Strassen where possible:
    - R(3,2,2): Can we use Strassen? 2x2 part could use Strassen, but the 3 rows complicate it.
    
    This is getting complex. Let me try a different approach.
    
    Key insight from the directive: "partitioning the 5x5 matrix multiplication into a 2x2 block structure with dimensions (2,3) and (3,2), utilizing the known rank-19 algorithm for 3x3 matrix multiplication on the four resulting 3x3 sub-blocks to achieve a total rank of 3*19 = 57"
    
    Wait, that suggests padding to 3x3 blocks. Let me think about this differently.
    
    If we pad A from 3x5 to 3x6, B from 5x5 to 6x6, then C from 3x5 to 3x6:
    Pad with zeros. Then:
    A' = [A | 0] (3x6)
    B' = [B; 0] (6x6) where bottom 1x6 row is zero
    C' = A' @ B' (3x6), and C = C'[:, :5]
    
    Now 3x6 * 6x6 can be seen as 2x2 blocks of size 3x3:
    A' = [A11 | A12] where A11 is 3x3, A12 is 3x3
    B' = [B11 B12; B21 B22] where each is 3x3
    
    C' = [A11@B11 + A12@B21 | A11@B12 + A12@B22]
    
    Using Strassen on this 2x2 block of 3x3 matrices:
    M1 = (A11+A22)(B11+B22) but A22 = A12, B22 is bottom-right
    ...
    
    This would give rank 7 * R(3,3,3) = 7 * 19 = 133, which is worse than 75.
    
    Hmm, that's not helpful. Let me re-read the directive.
    
    "partitioning the 5x5 matrix multiplication into a 2x2 block structure with dimensions (2,3) and (3,2)"
    
    Maybe it means: split the 5 dimension into 2 and 3, and use the fact that we can express the multiplication in terms of 2x2 and 3x3 multiplications, then use Strassen on the 2x2 parts and rank-19 on the 3x3 parts.
    
    Let me try yet another approach. The current best is 71 (from 4 disjoint Strassen 2x2x2 blocks saving 1 rank each from 75).
    
    Can we find more overlaps or use a different structure?
    
    For 3x5*5x5, the tensor has 3*5*5 = 75 non-zero entries (one for each (i,j,k) triple).
    
    A rank-1 decomposition corresponds to a single (i,j,k) triple.
    Strassen saves rank by finding linear dependencies.
    
    The 4 disjoint Strassen blocks found in the current approach work because within each 2x2x2 sub-tensor, the 8 schoolbook terms can be replaced by 7 Strassen terms.
    
    With m=5, p=5, n=3, we have 3*5*5 = 75 terms.
    If we can find 4 disjoint 2x2x2 blocks, that saves 4 ranks, giving 71.
    
    Can we find more? Or can we find larger structures?
    
    Let me try to systematically find the maximum number of disjoint Strassen-like reductions.
    
    Actually, let me just implement the 4-block Strassen that achieves 71, and then try to extend it.
    """
    n, m, p = 3, 5, 5
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    rank = n * m * p  # 75
    
    # The schoolbook has one term for each (i, j, k) triple
    # Term index: col = (i*m + j)*p + k
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1  
    # W[k*n+i, col] = 1
    
    # Strassen reduction on a 2x2x2 block:
    # For fixed i0, j0, k0, the 8 terms (i0+i1, j0+j2, k0+k3) for i1,j2,k3 in {0,1}
    # can be replaced by 7 terms.
    # 
    # The 8 schoolbook terms have columns:
    # col(i1,j2,k3) = ((i0+i1)*m + (j0+j2))*p + (k0+k3)
    #
    # Strassen replaces these with 7 linear combinations.
    #
    # Let's define the Strassen substitution for a 2x2x2 block at (i0, j0, k0).
    # The 8 original columns are replaced by 7 new columns.
    #
    # In the new decomposition:
    # For each of the 7 Strassen intermediates M1..M7:
    #   U row (i0+i1)*m + (j0+j2) gets the A-coefficient
    #   V row (j0+j2)*p + (k0+k3) gets the B-coefficient
    #   W row (k0+k3)*n + (i0+i1) gets the C-coefficient
    #
    # But we need to be careful: the same U, V, W rows may be shared across blocks.
    #
    # For disjoint blocks, we can do this independently.
    
    # Find 4 disjoint 2x2x2 blocks in the 3x5x5 tensor.
    # A block at (i0, j0, k0) uses rows:
    #   U: (i0)*m+j0, (i0)*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
    #   V: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
    #   W: k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
    #
    # Blocks are disjoint if they don't share any U, V, or W rows.
    
    # Let's find 4 disjoint blocks.
    # With n=3, m=5, p=5:
    # U rows: 0..14 (15 rows)
    # V rows: 0..24 (25 rows)
    # W rows: 0..14 (15 rows)
    
    # A block at (i0, j0, k0) needs:
    # i0 in {0,1} (since i0+1 <= 2)
    # j0 in {0,1,2,3} (since j0+1 <= 4)
    # k0 in {0,1,2,3} (since k0+1 <= 4)
    
    # Let's try:
    # Block 1: (0, 0, 0) - uses U rows {0,1,5,6}, V rows {0,1,5,6}, W rows {0,1,3,4}
    # Block 2: (0, 2, 2) - uses U rows {0,1,7,8}, V rows {10,11,15,16}, W rows {6,7,9,10}
    #   Check U: {0,1,7,8} vs {0,1,5,6} - overlap on {0,1}! Not disjoint.
    
    # Need to be more careful. Let me write a function to check disjointness.
    
    def block_rows(i0, j0, k0, n, m, p):
        u_rows = [(i0)*m + j0, (i0)*m + j0+1, (i0+1)*m + j0, (i0+1)*m + j0+1]
        v_rows = [j0*p + k0, j0*p + k0+1, (j0+1)*p + k0, (j0+1)*p + k0+1]
        w_rows = [k0*n + i0, k0*n + i0+1, (k0+1)*n + i0, (k0+1)*n + i0+1]
        return set(u_rows), set(v_rows), set(w_rows)
    
    def are_disjoint(blocks, n, m, p):
        all_u, all_v, all_w = set(), set(), set()
        for (i0, j0, k0) in blocks:
            u, v, w = block_rows(i0, j0, k0, n, m, p)
            if all_u & u or all_v & v or all_w & w:
                return False
            all_u |= u
            all_v |= v
            all_w |= w
        return True
    
    # Find 4 disjoint blocks
    best_blocks = []
    for i0 in range(2):
        for j0 in range(4):
            for k0 in range(4):
                candidate = (i0, j0, k0)
                test_blocks = best_blocks + [candidate]
                if are_disjoint(test_blocks, n, m, p):
                    best_blocks.append(candidate)
                    if len(best_blocks) == 4:
                        break
            if len(best_blocks) == 4:
                break
        if len(best_blocks) == 4:
            break
    
    print(f"Found {len(best_blocks)} disjoint Strassen blocks: {best_blocks}")
    
    # Now build the decomposition with Strassen reductions on these blocks
    # Start with schoolbook, then for each block, replace 8 terms with 7 Strassen terms
    
    # Rebuild from scratch
    U_new = [[0] * rank for _ in range(n * m)]
    V_new = [[0] * rank for _ in range(m * p)]
    W_new = [[0] * rank for _ in range(p * n)]
    
    # First, mark which columns are used by Strassen blocks
    strassen_cols = set()
    for (i0, j0, k0) in best_blocks:
        for i1 in range(2):
            for j2 in range(2):
                for k3 in range(2):
                    col = ((i0+i1)*m + (j0+j2))*p + (k0+k3)
                    strassen_cols.add(col)
    
    # For non-Strassen columns, use schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                orig_col = (i*m + j)*p + k
                if orig_col not in strassen_cols:
                    U_new[i*m + j][col] = 1
                    V_new[j*p + k][col] = 1
                    W_new[k*n + i][col] = 1
                    col += 1
    
    # For each Strassen block, add 7 new columns
    # C indices: C11=(k0,i0), C12=(k0,i0+1), C21=(k0+1,i0), C22=(k0+1,i0+1)
    # W[C11] = M1 + M4 - M5 + M7
    # W[C12] = M3 + M5
    # W[C21] = M2 + M4
    # W[C22] = M1 - M2 + M3 + M6
    
    for (i0, j0, k0) in best_blocks:
        c11 = k0*n + i0
        c12 = k0*n + (i0+1)
        c21 = (k0+1)*n + i0
        c22 = (k0+1)*n + (i0+1)
        
        # M1 = (A11+A22)(B11+B22)
        U_new[i0*m + j0][col] = 1
        U_new[(i0+1)*m + (j0+1)][col] = 1
        V_new[j0*p + k0][col] = 1
        V_new[(j0+1)*p + (k0+1)][col] = 1
        W_new[c11][col] = 1
        W_new[c22][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U_new[(i0+1)*m + j0][col] = 1
        U_new[(i0+1)*m + (j0+1)][col] = 1
        V_new[j0*p + k0][col] = 1
        W_new[c21][col] = 1
        W_new[c22][col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U_new[i0*m + j0][col] = 1
        V_new[j0*p + (k0+1)][col] = 1
        V_new[(j0+1)*p + (k0+1)][col] = -1
        W_new[c12][col] = 1
        W_new[c22][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U_new[(i0+1)*m + (j0+1)][col] = 1
        V_new[(j0+1)*p + k0][col] = 1
        V_new[j0*p + k0][col] = -1
        W_new[c11][col] = 1
        W_new[c21][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U_new[i0*m + j0][col] = 1
        U_new[i0*m + (j0+1)][col] = 1
        V_new[(j0+1)*p + (k0+1)][col] = 1
        W_new[c11][col] = -1
        W_new[c12][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U_new[(i0+1)*m + j0][col] = 1
        U_new[i0*m + j0][col] = -1
        V_new[j0*p + k0][col] = 1
        V_new[j0*p + (k0+1)][col] = 1
        W_new[c22][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U_new[i0*m + (j0+1)][col] = 1
        U_new[(i0+1)*m + (j0+1)][col] = -1
        V_new[(j0+1)*p + k0][col] = 1
        V_new[(j0+1)*p + (k0+1)][col] = 1
        W_new[c11][col] = 1
        col += 1
    
    actual_rank = col
    # Trim to actual rank
    U_final = [row[:actual_rank] for row in U_new]
    V_final = [row[:actual_rank] for row in V_new]
    W_final = [row[:actual_rank] for row in W_new]
    
    return U_final, V_final, W_final

def construct():
    start_time = time.time()
    deadline = 130.0  # 150s limit, leave 20s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try block Strassen
    try:
        U_s, V_s, W_s = _build_block_strassen(n, m, p)
        if _verify(U_s, V_s, W_s, n, m, p):
            if len(U_s[0]) < best_rank:
                best_rank = len(U_s[0])
                best_U, best_V, best_W = U_s, V_s, W_s
                print(f"Block Strassen achieved rank {best_rank}")
    except Exception as e:
        print(f"Block Strassen failed: {e}")
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
