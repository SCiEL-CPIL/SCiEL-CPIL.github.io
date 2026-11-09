"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if this is a valid matrix mult term, else 0
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] should be 1 for all (i,j,k) in the schoolbook sense
    # Actually the tensor is: T[i,j,k] = delta(i,j,k) where the product A[i,j]*B[j,k]*C[k,i]... 
    # The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # No wait. The tensor M[i,j,k] for multiplying i×j matrix by j×k matrix is:
    # M[i,j,k] = 1, and the decomposition is: M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Build the tensor and check
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is expensive for large rank, so do it in chunks
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U[:, r]  # shape (n*m,)
        v_col = V[:, r]  # shape (m*p,)
        w_col = W[:, r]  # shape (p*n,)
        # Reshape and multiply
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        # T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # Use einsum
        T += np.einsum('ij,jk,ki->ijk', u_mat, v_mat, w_mat)
    
    # Check all entries are 1
    expected = np.ones((n, m, p), dtype=float)
    if not np.allclose(T, expected, atol=tol):
        diff = np.abs(T - expected)
        max_diff = np.max(diff)
        num_bad = np.sum(diff > tol)
        return False, max_diff, num_bad
    return True, 0.0, 0

def _strassen_4x4x4_decomposition():
    """
    Return the rank-49 decomposition of the 4x4x4 matrix multiplication tensor.
    Uses Strassen recursively: 4x4x4 = (2x2x2)^2, each 2x2x2 has rank 7.
    
    The 2x2x2 Strassen decomposition (rank 7):
    For multiplying 2x2 by 2x2:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # Strassen 2x2x2: 7 terms
    # Each term has coefficients in U, V, W
    # For 2x2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7
    
    # Strassen coefficients for 2x2x2
    # Term r: U[i,j,r], V[j,k,r], W[k,i,r]
    # Indices: i,j in {0,1}, k in {0,1}
    
    # Let me define the 7 Strassen terms explicitly
    # a = [[a00, a01], [a10, a11]]
    # b = [[b00, b01], [b10, b11]]
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U[i,j,r] coefficient, V[j,k,r] coefficient, W[k,i,r] coefficient
    
    # Term 0 (M1): U has (0,0)=1,(1,1)=1; V has (0,0)=1,(1,1)=1; W has (0,0)=1,(0,1)=1,(1,0)=1,(1,1)=1... 
    # Wait, W[k,i,r] means for output position (k,i) where k is row of B, i is col of A... 
    # Actually in the tensor: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This should equal 1 for all i,j,k.
    
    # For Strassen, the "multiplications" are not scalar, they're matrix products.
    # So I need to compose the 2x2x2 Strassen with the block structure.
    
    # For 4x4x4, split into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [[C00, C01], [C10, C11]] where each is 2x2
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mx is a 2x2 matrix product, which itself has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # I need to build U, V, W of size (16, 49), (16, 49), (16, 49)
    # where indices are:
    # U: row index = i*4+j (i in 0..3, j in 0..3) for A[i,j]
    # V: row index = j*4+k (j in 0..3, k in 0..3) for B[j,k]  
    # W: row index = k*4+i (k in 0..3, i in 0..3) for C[k,i]... wait, W[k*n+i]
    
    # For 4x4x4: n=m=p=4
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    # Block indices: 
    # A block (bi, bj) where bi in {0,1} is block row, bj in {0,1} is block col
    # Within block: local indices (li, lj) in {0,1}
    # Global: i = bi*2 + li, j = bj*2 + lj
    
    # Similarly for B: j = bj*2 + lj, k = bk*2 + lk
    # C: k = bk*2 + lk, i = bi*2 + li
    
    # So W index: k*4+i = (bk*2+lk)*4 + (bi*2+li) = bk*8 + lk*4 + bi*2 + li
    
    # For each of the 7 Strassen block-level terms, and within each, 7 sub-terms:
    
    # Block-level Strassen terms:
    # Term t (0..6): 
    #   U_block: which A blocks are included with what sign
    #   V_block: which B blocks are included with what sign
    #   W_block: which C blocks get contributions
    
    # Let me define the 7 block-level terms:
    # t=0 (M1): A blocks: (0,0)+, (1,1)+; B blocks: (0,0)+, (1,1)+; C: C00+, C11-
    # t=1 (M2): A blocks: (1,0)+, (1,1)+; B blocks: (0,0)+; C: C10+
    # t=2 (M3): A blocks: (0,0)+; B blocks: (0,1)+, (1,1)-; C: C01+
    # t=3 (M4): A blocks: (1,1)+; B blocks: (1,0)+, (0,0)-; C: C00+, C10+
    # t=4 (M5): A blocks: (0,0)+, (0,1)+; B blocks: (1,1)+; C: C00-, C01+
    # t=5 (M6): A blocks: (1,0)+, (0,0)-; B blocks: (0,0)+, (0,1)+; C: C11+
    # t=6 (M7): A blocks: (0,1)+, (1,1)-; B blocks: (1,0)+, (1,1)+; C: C00+, C11+
    
    # Wait, I need to be more careful. Let me re-derive.
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # So for each Mx, the A-side blocks and B-side blocks are:
    # M1: A: {00:+, 11:+}, B: {00:+, 11:+}
    # M2: A: {10:+, 11:+}, B: {00:+}
    # M3: A: {00:+}, B: {01:+, 11:-}
    # M4: A: {11:+}, B: {10:+, 00:-}
    # M5: A: {00:+, 01:+}, B: {11:+}
    # M6: A: {10:+, 00:-}, B: {00:+, 01:+}
    # M7: A: {01:+, 11:-}, B: {10:+, 11:+}
    
    # And C contributions:
    # M1: C00:+, C11:-
    # M2: C10:+
    # M3: C01:+
    # M4: C00:+, C10:+
    # M5: C00:-, C01:+
    # M6: C11:+
    # M7: C00:+, C11:+
    
    # Now, within each Mx, we use Strassen on the 2x2 blocks.
    # Each Mx is a product of (sum of some 2x2 A blocks) * (sum of some 2x2 B blocks)
    # = a single 2x2 * 2x2 matrix multiplication
    # Which has rank 7 via Strassen.
    
    # So for each block-level term t (0..6) and each sub-term s (0..6):
    #   We add 1 to the rank
    #   U: for each A block (bi,bj) with sign sa in M_t:
    #       for each local (li,lj): U[(bi*2+li)*4 + (bj*2+lj), col] += sa * U_strassen[li,lj,s]
    #   V: for each B block (bj,bk) with sign sb in M_t:
    #       for each local (lj,lk): V[(bj*2+lj)*4 + (bk*2+lk), col] += sb * V_strassen[lj,lk,s]
    #   W: for each C block (bk,bi) with sign sc in M_t:
    #       for each local (lk,li): W[(bk*2+lk)*4 + (bi*2+li), col] += sc * W_strassen[lk,li,s]
    
    # First, define the 2x2x2 Strassen decomposition (rank 7)
    # For 2x2x2: U2[i,j,s], V2[j,k,s], W2[k,i,s] for i,j,k in {0,1}, s in 0..6
    
    # Strassen 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U2[i,j,s]:
    # s=0 (M1): a00:+, a11:+
    # s=1 (M2): a10:+, a11:+
    # s=2 (M3): a00:+
    # s=3 (M4): a11:+
    # s=4 (M5): a00:+, a01:+
    # s=5 (M6): a10:+, a00:-
    # s=6 (M7): a01:+, a11:-
    
    # V2[j,k,s]:
    # s=0: b00:+, b11:+
    # s=1: b00:+
    # s=2: b01:+, b11:-
    # s=3: b10:+, b00:-
    # s=4: b11:+
    # s=5: b00:+, b01:+
    # s=6: b10:+, b11:+
    
    # W2[k,i,s]:
    # C00 = M1+M4-M5+M7 => W2[0,0,s]: s=0:+, s=3:+, s=4:-, s=6:+
    # C01 = M3+M5 => W2[0,1,s]: s=2:+, s=4:+
    # C10 = M2+M4 => W2[1,0,s]: s=1:+, s=3:+
    # C11 = M1-M2+M3+M6 => W2[1,1,s]: s=0:+, s=1:-, s=2:+, s=5:+
    
    # Build U2, V2, W2 as 4x7 matrices (indices 0..3 for i*2+j or j*2+k or k*2+i)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # U2[i*2+j, s]
    # s=0: (0,0):+1, (1,1):+1
    U2[0, 0] = 1  # a00
    U2[3, 0] = 1  # a11
    # s=1: (1,0):+1, (1,1):+1
    U2[2, 1] = 1  # a10
    U2[3, 1] = 1  # a11
    # s=2: (0,0):+1
    U2[0, 2] = 1  # a00
    # s=3: (1,1):+1
    U2[3, 3] = 1  # a11
    # s=4: (0,0):+1, (0,1):+1
    U2[0, 4] = 1  # a00
    U2[1, 4] = 1  # a01
    # s=5: (1,0):+1, (0,0):-1
    U2[2, 5] = 1  # a10
    U2[0, 5] = -1  # a00
    # s=6: (0,1):+1, (1,1):-1
    U2[1, 6] = 1  # a01
    U2[3, 6] = -1  # a11
    
    # V2[j*2+k, s]
    # s=0: (0,0):+1, (1,1):+1
    V2[0, 0] = 1  # b00
    V2[3, 0] = 1  # b11
    # s=1: (0,0):+1
    V2[0, 1] = 1  # b00
    # s=2: (0,1):+1, (1,1):-1
    V2[1, 2] = 1  # b01
    V2[3, 2] = -1  # b11
    # s=3: (1,0):+1, (0,0):-1
    V2[2, 3] = 1  # b10
    V2[0, 3] = -1  # b00
    # s=4: (1,1):+1
    V2[3, 4] = 1  # b11
    # s=5: (0,0):+1, (0,1):+1
    V2[0, 5] = 1  # b00
    V2[1, 5] = 1  # b01
    # s=6: (1,0):+1, (1,1):+1
    V2[2, 6] = 1  # b10
    V2[3, 6] = 1  # b11
    
    # W2[k*2+i, s]
    # W2[0,0,s] for C00: s=0:+1, s=3:+1, s=4:-1, s=6:+1
    W2[0, 0] = 1
    W2[0, 3] = 1
    W2[0, 4] = -1
    W2[0, 6] = 1
    # W2[0,1,s] for C01: s=2:+1, s=4:+1
    W2[1, 2] = 1
    W2[1, 4] = 1
    # W2[1,0,s] for C10: s=1:+1, s=3:+1
    W2[2, 1] = 1
    W2[2, 3] = 1
    # W2[1,1,s] for C11: s=0:+1, s=1:-1, s=2:+1, s=5:+1
    W2[3, 0] = 1
    W2[3, 1] = -1
    W2[3, 2] = 1
    W2[3, 5] = 1
    
    # Verify 2x2x2
    valid, max_diff, num_bad = _verify_decomposition(U2.tolist(), V2.tolist(), W2.tolist(), 2, 2, 2)
    if not valid:
        print(f"2x2x2 Strassen verification failed: max_diff={max_diff}, num_bad={num_bad}")
        # Fall back to schoolbook
        return None
    
    # Now build 4x4x4 from 2x2x2 blocks
    # Block-level Strassen terms (7 terms), each with 7 sub-terms = 49 total
    
    # Define block-level coefficients
    # For each block term t (0..6):
    #   A_blocks[t]: dict from (bi,bj) to sign
    #   B_blocks[t]: dict from (bj,bk) to sign  
    #   C_blocks[t]: dict from (bk,bi) to sign
    
    A_blocks = [
        {(0,0): 1, (1,1): 1},    # M1
        {(1,0): 1, (1,1): 1},    # M2
        {(0,0): 1},               # M3
        {(1,1): 1},               # M4
        {(0,0): 1, (0,1): 1},    # M5
        {(1,0): 1, (0,0): -1},   # M6
        {(0,1): 1, (1,1): -1},   # M7
    ]
    
    B_blocks = [
        {(0,0): 1, (1,1): 1},    # M1
        {(0,0): 1},               # M2
        {(0,1): 1, (1,1): -1},   # M3
        {(1,0): 1, (0,0): -1},   # M4
        {(1,1): 1},               # M5
        {(0,0): 1, (0,1): 1},    # M6
        {(1,0): 1, (1,1): 1},    # M7
    ]
    
    C_blocks = [
        {(0,0): 1, (1,1): -1},   # M1: C00+, C11-
        {(1,0): 1},               # M2: C10+
        {(0,1): 1},               # M3: C01+
        {(0,0): 1, (1,0): 1},    # M4: C00+, C10+
        {(0,0): -1, (0,1): 1},   # M5: C00-, C01+
        {(1,1): 1},               # M6: C11+
        {(0,0): 1, (1,1): 1},    # M7: C00+, C11+
    ]
    
    # Build 4x4x4 decomposition
    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    col = 0
    for t in range(7):
        for s in range(7):
            # U: for each A block (bi,bj) with sign sa
            for (bi, bj), sa in A_blocks[t].items():
                for li in range(2):
                    for lj in range(2):
                        i = bi * 2 + li
                        j = bj * 2 + lj
                        u_idx = i * 4 + j
                        u2_idx = li * 2 + lj
                        U4[u_idx, col] += sa * U2[u2_idx, s]
            
            # V: for each B block (bj,bk) with sign sb
            for (bj, bk), sb in B_blocks[t].items():
                for lj in range(2):
                    for lk in range(2):
                        j = bj * 2 + lj
                        k = bk * 2 + lk
                        v_idx = j * 4 + k
                        v2_idx = lj * 2 + lk
                        V4[v_idx, col] += sb * V2[v2_idx, s]
            
            # W: for each C block (bk,bi) with sign sc
            for (bk, bi), sc in C_blocks[t].items():
                for lk in range(2):
                    for li in range(2):
                        k = bk * 2 + lk
                        i = bi * 2 + li
                        w_idx = k * 4 + i
                        w2_idx = lk * 2 + li
                        W4[w_idx, col] += sc * W2[w2_idx, s]
            
            col += 1
    
    # Verify 4x4x4
    valid, max_diff, num_bad = _verify_decomposition(U4.tolist(), V4.tolist(), W4.tolist(), 4, 4, 4)
    if not valid:
        print(f"4x4x4 Strassen verification failed: max_diff={max_diff}, num_bad={num_bad}")
        return None
    
    return U4.tolist(), V4.tolist(), W4.tolist()

def _extend_to_4x4x7(U4, V4, W4):
    """
    Extend the 4x4x4 decomposition (rank 49) to 4x4x7.
    
    The 4x4x4 decomposition gives us: for all i,j,k in 0..3:
    sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] = 1
    
    For 4x4x7, we need: for all i in 0..3, j in 0..3, k in 0..6:
    sum_r' U[i*4+j, r'] * V[j*7+k, r'] * W[k*4+i, r'] = 1
    
    Idea: Use the 4x4x4 decomposition for k=0..3, and for k=4..6, 
    we need additional terms. But that's not straightforward.
    
    Alternative: View the 4x4x7 problem as having a "wide" B matrix.
    We can use the fact that the tensor T[i,j,k] for 4x4x7 is:
    T[i,j,k] = 1 for all i,j,k
    
    One approach: Pad the 4x4x4 decomposition. For k in 0..3, use the existing terms.
    For k in 4..6, we need new terms.
    
    Actually, a cleaner approach: The 4x4x7 multiplication can be seen as:
    C[:, k] = A @ B[:, k] for each k
    
    So it's 7 independent 4x4x1 multiplications... no, that's not right either.
    
    Let me think about this differently. The tensor is:
    T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..6
    
    We need: sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r] = 1
    
    Using the 4x4x4 decomposition (rank 49) for k=0..3:
    For k in 0..3: sum_{r=0}^{48} U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] = 1
    
    But V4 has index j*4+k, while we need j*7+k. These are different indexings.
    
    I need to map V4[j*4+k, r] to V[j*7+k, r'] for some r'.
    
    This is getting complicated. Let me try a different approach.
    
    Direct approach: Use the schoolbook for the "extra" dimensions.
    
    Actually, the simplest correct approach: 
    The rank of 4x4x7 is at most 4*4*7 = 112 (schoolbook).
    The rank of 4x4x4 is 49 (Strassen).
    
    For 4x4x7, I can use:
    - For k=0..3: use the 4x4x4 Strassen decomposition, but I need to handle the V indexing.
    
    Let me just build it directly using the block structure.
    
    Split A (4x4) into 2x2 blocks of 2x2.
    Split B (4x7) into 2x2 blocks of 2x7.
    Split C (4x7) into 2x2 blocks of 2x7.
    
    Block-level Strassen (7 terms), each term is a 2x2 x 2x7 product.
    Each 2x2 x 2x7 product has rank 2*2*7 = 28 (schoolbook) or possibly less.
    
    Total rank = 7 * 28 = 196, which is worse than 112.
    
    So the block Strassen approach doesn't help for rectangular matrices.
    
    Let me just use the schoolbook (rank 112) for now, but verify it's correct.
    The current program already does this. Let me verify it works.
    """
    pass

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # First, try to get a 4x4x4 Strassen decomposition
    result_444 = _strassen_4x4x4_decomposition()
    
    if result_444 is None:
        # Fall back to schoolbook
        pass
    
    # For 4x4x7, let's use a hybrid approach:
    # Use Strassen for the 4x4 part where possible, schoolbook for the rest.
    
    # Actually, let me just implement the schoolbook correctly and verify it.
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
    
    # Verify
    valid, max_diff, num_bad = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        print(f"Schoolbook verification failed: max_diff={max_diff}, num_bad={num_bad}")
        # This should never happen, but just in case
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        print(f"Time budget nearly exhausted: {elapsed:.1f}s")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
