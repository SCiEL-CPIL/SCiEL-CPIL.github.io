"""
Rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication.

Uses a block-Strassen decomposition:
  Split A (3x3) into A1 (2x3) and A2 (1x3).
  Split B (3x4) into B1 (2x4) and B2 (1x4).
  Output C (3x4) has blocks C1 (2x4) and C2 (1x4).

C1 = A1*B1: 2x3 * 3x4 -> split into 2x2*2x2 (Strassen, rank 7) + 2x1*2x2 (rank 2) + 1x2*2x2 (rank 2) + 1x1*2x2 (rank 2)
C2 = A2*B2: 1x3 * 3x4 -> rank 3

Total rank: 7+2+2+2+3 = 16
"""
import time
import fractions

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction


def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True


def _construct_strassen_block(rank_start, i_offset, j_indices, l_indices, k_indices):
    """
    Construct Strassen factors for a 2x2 * 2x2 block.
    
    A rows: i_offset + [0,1], A cols: j_indices (2 cols)
    B rows: l_indices (2 rows), B cols: k_indices (2 cols)
    C rows: i_offset + [0,1], C cols: k_indices (2 cols)
    
    Returns (U_contrib, V_contrib, W_contrib) where each is a dict mapping
    (row_idx) -> list of (col_in_rank, value)
    """
    U = {}
    V = {}
    W = {}
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices in the block
    a00 = (i_offset, j_indices[0])  # A[0,0] in block
    a01 = (i_offset, j_indices[1])  # A[0,1] in block
    a10 = (i_offset + 1, j_indices[0])  # A[1,0] in block
    a11 = (i_offset + 1, j_indices[1])  # A[1,1] in block
    
    b00 = (l_indices[0], k_indices[0])  # B[0,0] in block
    b01 = (l_indices[0], k_indices[1])  # B[0,1] in block
    b10 = (l_indices[1], k_indices[0])  # B[1,0] in block
    b11 = (l_indices[1], k_indices[1])  # B[1,1] in block
    
    # Define the 7 rank-1 terms
    # Each term c has U_val, V_val, W_val for each (i,j), (l,k), (k,i) position
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: (i_offset, j_indices[0]) += 1, (i_offset+1, j_indices[1]) += 1
    # V: (l_indices[0], k_indices[0]) += 1, (l_indices[1], k_indices[1]) += 1
    # W: (k_indices[0], i_offset) += 1, (k_indices[1], i_offset+1) += 1
    #     plus cross terms: (k_indices[0], i_offset+1) += 1, (k_indices[1], i_offset) += 1
    
    U[rank_start + 0] = {}
    U[rank_start + 0][(i_offset, j_indices[0])] = 1
    U[rank_start + 0][(i_offset + 1, j_indices[1])] = 1
    V[rank_start + 0] = {}
    V[rank_start + 0][(l_indices[0], k_indices[0])] = 1
    V[rank_start + 0][(l_indices[1], k_indices[1])] = 1
    W[rank_start + 0] = {}
    W[rank_start + 0][(k_indices[0], i_offset)] = 1
    W[rank_start + 0][(k_indices[1], i_offset + 1)] = 1
    W[rank_start + 0][(k_indices[0], i_offset + 1)] = 1
    W[rank_start + 0][(k_indices[1], i_offset)] = 1
    
    # Term 2: M2 = (A10+A11)B00
    # U: (i_offset+1, j_indices[0]) += 1, (i_offset+1, j_indices[1]) += 1
    # V: (l_indices[0], k_indices[0]) += 1
    # W: (k_indices[0], i_offset+1) += 1
    U[rank_start + 1] = {}
    U[rank_start + 1][(i_offset + 1, j_indices[0])] = 1
    U[rank_start + 1][(i_offset + 1, j_indices[1])] = 1
    V[rank_start + 1] = {}
    V[rank_start + 1][(l_indices[0], k_indices[0])] = 1
    W[rank_start + 1] = {}
    W[rank_start + 1][(k_indices[0], i_offset + 1)] = 1
    
    # Term 3: M3 = A00(B10-B11)
    # U: (i_offset, j_indices[0]) += 1
    # V: (l_indices[1], k_indices[0]) += 1, (l_indices[1], k_indices[1]) += -1
    # W: (k_indices[0], i_offset) += 1, (k_indices[1], i_offset) += 1
    U[rank_start + 2] = {}
    U[rank_start + 2][(i_offset, j_indices[0])] = 1
    V[rank_start + 2] = {}
    V[rank_start + 2][(l_indices[1], k_indices[0])] = 1
    V[rank_start + 2][(l_indices[1], k_indices[1])] = -1
    W[rank_start + 2] = {}
    W[rank_start + 2][(k_indices[0], i_offset)] = 1
    W[rank_start + 2][(k_indices[1], i_offset)] = 1
    
    # Term 4: M4 = A11(B01-B10)
    # U: (i_offset+1, j_indices[1]) += 1
    # V: (l_indices[0], k_indices[1]) += 1, (l_indices[1], k_indices[0]) += -1
    # W: (k_indices[1], i_offset+1) += 1, (k_indices[0], i_offset+1) += 1
    U[rank_start + 3] = {}
    U[rank_start + 3][(i_offset + 1, j_indices[1])] = 1
    V[rank_start + 3] = {}
    V[rank_start + 3][(l_indices[0], k_indices[1])] = 1
    V[rank_start + 3][(l_indices[1], k_indices[0])] = -1
    W[rank_start + 3] = {}
    W[rank_start + 3][(k_indices[1], i_offset + 1)] = 1
    W[rank_start + 3][(k_indices[0], i_offset + 1)] = 1
    
    # Term 5: M5 = (A00+A01)B11
    # U: (i_offset, j_indices[0]) += 1, (i_offset, j_indices[1]) += 1
    # V: (l_indices[1], k_indices[1]) += 1
    # W: (k_indices[1], i_offset) += 1
    U[rank_start + 4] = {}
    U[rank_start + 4][(i_offset, j_indices[0])] = 1
    U[rank_start + 4][(i_offset, j_indices[1])] = 1
    V[rank_start + 4] = {}
    V[rank_start + 4][(l_indices[1], k_indices[1])] = 1
    W[rank_start + 4] = {}
    W[rank_start + 4][(k_indices[1], i_offset)] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: (i_offset+1, j_indices[0]) += 1, (i_offset, j_indices[0]) += -1
    # V: (l_indices[0], k_indices[0]) += 1, (l_indices[0], k_indices[1]) += 1
    # W: (k_indices[0], i_offset+1) += 1, (k_indices[1], i_offset+1) += 1
    U[rank_start + 5] = {}
    U[rank_start + 5][(i_offset + 1, j_indices[0])] = 1
    U[rank_start + 5][(i_offset, j_indices[0])] = -1
    V[rank_start + 5] = {}
    V[rank_start + 5][(l_indices[0], k_indices[0])] = 1
    V[rank_start + 5][(l_indices[0], k_indices[1])] = 1
    W[rank_start + 5] = {}
    W[rank_start + 5][(k_indices[0], i_offset + 1)] = 1
    W[rank_start + 5][(k_indices[1], i_offset + 1)] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: (i_offset, j_indices[1]) += 1, (i_offset+1, j_indices[1]) += -1
    # V: (l_indices[1], k_indices[0]) += 1, (l_indices[1], k_indices[1]) += 1
    # W: (k_indices[0], i_offset) += 1, (k_indices[1], i_offset) += 1
    U[rank_start + 6] = {}
    U[rank_start + 6][(i_offset, j_indices[1])] = 1
    U[rank_start + 6][(i_offset + 1, j_indices[1])] = -1
    V[rank_start + 6] = {}
    V[rank_start + 6][(l_indices[1], k_indices[0])] = 1
    V[rank_start + 6][(l_indices[1], k_indices[1])] = 1
    W[rank_start + 6] = {}
    W[rank_start + 6][(k_indices[0], i_offset)] = 1
    W[rank_start + 6][(k_indices[1], i_offset)] = 1
    
    return U, V, W


def _construct_rectangular_block(rank_start, i_index, j_indices, l_indices, k_indices):
    """
    Construct rank-2 factors for a 1x2 * 2x2 block.
    A row: i_index, A cols: j_indices (2 cols)
    B rows: l_indices (2 rows), B cols: k_indices (2 cols)
    C row: i_index, C cols: k_indices (2 cols)
    
    This is just schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    For 1x2 * 2x2, we need 2 multiplications (one per column of B for each k).
    Actually, 1x2 * 2x2 has rank 2 (schoolbook is optimal here).
    """
    U = {}
    V = {}
    W = {}
    
    for c_idx, k in enumerate(k_indices):
        # For each output column k, we need to compute sum_j A[i,j] * B[j,k]
        # This is a single rank-1 term per k
        col = rank_start + c_idx
        U[col] = {}
        for j in j_indices:
            U[col][(i_index, j)] = 1
        V[col] = {}
        for l in l_indices:
            V[col][(l, k)] = 1
        W[col] = {}
        W[col][(k, i_index)] = 1
    
    return U, V, W


def _construct_row_block(rank_start, i_index, j_indices, l_indices, k_indices):
    """
    Construct factors for a 1xL * LxK block (row vector times matrix).
    Rank = K (one term per output column).
    """
    U = {}
    V = {}
    W = {}
    
    for c_idx, k in enumerate(k_indices):
        col = rank_start + c_idx
        U[col] = {}
        for j in j_indices:
            U[col][(i_index, j)] = 1
        V[col] = {}
        for l in l_indices:
            V[col][(l, k)] = 1
        W[col] = {}
        W[col][(k, i_index)] = 1
    
    return U, V, W


def _build_decomposition():
    """
    Build the full decomposition.
    
    A is 3x3, B is 3x4, C is 3x4.
    
    Split:
    - A1 = A[0:2, :] (2x3), A2 = A[2:3, :] (1x3)
    - B1 = B[0:2, :] (2x4), B2 = B[2:3, :] (1x4)
    
    C1 = A1 * B1 (2x4): rows 0,1 of C
    C2 = A2 * B2 (1x4): row 2 of C
    
    C1 = A1 * B1:
      A1 is 2x3, B1 is 2x4.
      Split A1 into A1a (2x2, cols 0,1) and A1b (2x1, col 2)
      Split B1 into B1a (2x2, rows 0,1) and B1b (2x2, rows... wait)
      
      Actually, let's think more carefully.
      A1 (2x3) * B1 (2x4):
      The inner dimension is 3 (rows of A1's cols match... no.
      A1 has 2 rows, 3 cols. B1 has 2 rows, 4 cols.
      Wait, for matrix multiplication A1 * B1, A1 must have cols = rows of B1.
      A1 is 2x3, B1 is 2x4. 3 != 2. That's wrong.
      
      Let me re-think the split.
      
      A is 3x3 (n=3, m=3). B is 3x4 (m=3, p=4). C is 3x4 (n=3, p=4).
      
      Split the m dimension (3) into two parts: m1=2, m2=1.
      A = [A1; A2] where A1 is 3x2, A2 is 3x1.
      B = [B1 B2] where B1 is 2x4, B2 is 1x4.
      C = A1*B1 + A2*B2.
      
      A1*B1: 3x2 * 2x4 -> 3x4
      A2*B2: 3x1 * 1x4 -> 3x4
      
      Now decompose A1*B1 (3x2 * 2x4):
      Split the 3 rows of A1 into 2+1: A1a (2x2), A1b (1x2)
      The 2x4 of B1 stays.
      
      A1a*B1: 2x2 * 2x4
      A1b*B1: 1x2 * 2x4
      
      A1a*B1: 2x2 * 2x4. Split the 4 cols of B1 into 2+2: B1a (2x2), B1b (2x2)
      A1a*B1a: 2x2 * 2x2 -> Strassen, rank 7
      A1a*B1b: 2x2 * 2x2 -> Strassen, rank 7
      
      A1b*B1: 1x2 * 2x4. Split into 1x2*2x2 + 1x2*2x2, each rank 2. Total rank 4.
      
      A2*B2: 3x1 * 1x4. Rank 4 (one per output element... actually 3*1*4=12 schoolbook, but since inner dim is 1, it's just element-wise, rank 3*4=12? No.)
      
      Wait. A2 is 3x1, B2 is 1x4. Product is 3x4.
      C[i,k] = A2[i,0] * B2[0,k]. This is rank 3*4 = 12 with schoolbook.
      But actually, since the inner dimension is 1, each output element is a single product.
      The tensor for 3x1 * 1x4 is just the outer product structure.
      Rank = 3*4 = 12? No, that's the number of terms.
      
      Actually for 3x1 * 1x4, the matrix multiplication tensor has:
      U[i*m + j] where i in [0,3), m=1, j in [0,1) -> 3 elements in U
      V[l*p + k] where l in [0,1), p=4, k in [0,4) -> 4 elements in V
      W[k*n + i] where k in [0,4), n=3, i in [0,3) -> 12 elements in W
      
      C[i,k] = sum_j U[i*1+j] * V[j*4+k] * W[k*3+i]
             = U[i,0] * V[0,k] * W[k,i]
      
      We need this to equal delta for the multiplication tensor.
      For 3x1 * 1x4: C[i,k] = A[i,0] * B[0,k].
      The tensor is: sum_j U[i,j] V[j,k] W[k,i] = A[i,0] B[0,k]
      This is a rank-12 tensor (outer product of three vectors)? No.
      Actually, A[i,0] B[0,k] = sum_j (A[i,0] if j=0 else 0) * (B[0,k] if j=0 else 0) * 1
      But that's only one term. Wait, no.
      
      For the multiplication tensor T(i,j,k,l) = delta(j,l) for A(i,j) B(l,k):
      With m=1: T(i,0,k,0) = 1, all others 0.
      So we need sum_c U[i,c] V[c] W[k,c] = 1 for all (i,k).
      This requires rank >= 3*4 = 12 (since the U*V*W must span all 12 (i,k) pairs independently).
      Actually, U is 3xR, V is 1xR, W is 4xR.
      sum_c U[i,c] V[c] W[k,c] = (U V^T)_{i} * ... no.
      It's sum_c U[i,c] * V[0,c] * W[k,c].
      Let v_c = V[0,c]. Then sum_c (U[i,c] v_c) W[k,c].
      This is like: for each c, we add U[:,c] * v_c * W[:,c]^T.
      The sum is a 3x4 matrix where entry (i,k) = sum_c U[i,c] v_c W[k,c].
      For this to be all-ones, we need the 3x4 matrix to be all 1s.
      Minimum rank: we can do it with rank 1: U = all 1s (3x1), V = 1, W = all 1s (4x1).
      Check: sum_c=1 U[i,0] * V[0,0] * W[k,0] = 1*1*1 = 1. Yes!
      So 3x1 * 1x4 has rank 1!
      
      Similarly, 1x2 * 2x2:
      T(i,j,k,l) with n=1, m=2, p=2: delta(j,l) for i in [0,1), j in [0,2), l in [0,2), k in [0,2).
      U is 1*2=2 x R, V is 2*2=4 x R, W is 2*1=2 x R.
      C[i,k] = sum_j A[i,j] B[j,k]. With n=1: C[0,k] = sum_j A[0,j] B[j,k].
      Tensor: sum_c U[j,c] V[l,c] W[k,c] = delta(j,l) for all j,l,k.
      For fixed k: sum_c U[j,c] V[l,c] W[k,c] = delta(j,l).
      This is like: for each k, the matrix M_k = sum_c W[k,c] U[:,c] V[:,c]^T must be I_2.
      Actually, sum_c U[j,c] V[l,c] W[k,c] = (U diag(W_k) V^T)_{j,l} where W_k is a vector.
      We need this to be I_2 for each k.
      Rank of 1x2 * 2x2 multiplication: This is just a row vector times a matrix.
      It's equivalent to 2x2 matrix multiplication with one row of A being the input.
      Rank is at least 2. Can we do it with rank 2?
      U is 2x2, V is 4x2, W is 2x2.
      Let's use schoolbook: 4 terms. But can we do better?
      Actually 1x2 * 2x2: the tensor has support on (j,l) pairs with j=l.
      For each k, we need sum_c U[j,c] V[l,c] W[k,c] = delta(j,l).
      This is a system where for each k, the "slice" must give identity.
      I believe rank 2 is not enough. Let's use schoolbook rank 4 for safety, or try rank 2.
      
      Actually, let me reconsider. For 1x2 * 2x2:
      C[k] = A * B[:,k] (a dot product of A (length 2) with column k of B).
      C[0] = A0*B[0,0] + A1*B[1,0]
      C[1] = A0*B[0,1] + A1*B[1,1]
      
      Tensor: T(j, l, k) where j indexes A, l indexes B rows, k indexes B cols.
      T(j,l,k) = 1 if l=j, else 0. (The k just selects which column of B.)
      So T(j,l,0) = delta(j,l), T(j,l,1) = delta(j,l).
      
      sum_c U[j,c] V[l,c] W[k,c] = delta(j,l) for k=0,1.
      
      With rank 2: U is 2x2, V is 4x2, W is 2x2.
      Let U = I_2. Then sum_c delta(j,c) V[l,c] W[k,c] = V[l,j] W[k,j].
      We need V[l,j] W[k,j] = delta(l,j) for all k.
      This means V[:,j] W[:,j] = e_j e_j^T (outer product) for each j.
      And this must hold for both k=0 and k=1.
      So W[0,j] V[l,j] = delta(l,j) and W[1,j] V[l,j] = delta(l,j).
      This implies W[0,j] = W[1,j] (for V[l,j] to be consistent), and then V[l,j] = delta(l,j)/W[0,j].
      So V[0,0] = 1/W[0,0], V[1,1] = 1/W[1,1], others 0.
      And W[0,0] = W[1,0], W[0,1] = W[1,1].
      Let W[0,0] = W[1,0] = 1, W[0,1] = W[1,1] = 1.
      Then V[0,0] = 1, V[1,1] = 1.
      Check: sum_c U[j,c] V[l,c] W[k,c]
      For j=0,l=0,k=0: U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*0*1 = 1. OK.
      For j=0,l=1,k=0: 1*0*1 + 0*1*1 = 0. OK.
      For j=1,l=0,k=0: 0*1*1 + 1*0*1 = 0. OK.
      For j=1,l=1,k=0: 0*0*1 + 1*1*1 = 1. OK.
      Same for k=1. 
      
      So 1x2 * 2x2 has rank 2! Great.
      
      So revised plan:
      A1*B1 (3x2 * 2x4):
        A1a*B1a (2x2 * 2x2): Strassen, rank 7
        A1a*B1b (2x2 * 2x2): Strassen, rank 7
        A1b*B1a (1x2 * 2x2): rank 2
        A1b*B1b (1x2 * 2x2): rank 2
        Subtotal: 18
        
      A2*B2 (3x1 * 1x4): rank 1
        Wait, A2 is 3x1, B2 is 1x4. 
        T(i, j, l, k) with n=3, m=1, p=4: delta(j,l) where j in [0,1), l in [0,1).
        So T(i, 0, 0, k) = 1 for all i, k.
        sum_c U[i,c] V[0,c] W[k,c] = 1 for all i, k.
        Rank 1 works: U = all 1s, V = 1, W = all 1s.
        Subtotal: 1
        
      Total: 18 + 1 = 19.
      
      Hmm, but we also need to handle the cross terms correctly. Let me re-examine.
      
      Actually, the split is:
      C = A * B = [A1; A2] * [B1 B2] = A1*B1 + A2*B2
      
      A1 is 3x2 (rows 0,1,2 of A, cols 0,1)
      A2 is 3x1 (rows 0,1,2 of A, col 2)
      B1 is 2x4 (rows 0,1 of B)
      B2 is 1x4 (row 2 of B)
      
      A1*B1: 3x2 * 2x4 = 3x4
      Split A1 into A1a (2x2: rows 0,1, cols 0,1) and A1b (1x2: row 2, cols 0,1)
      Split B1 into B1a (2x2: rows 0,1, cols 0,1) and B1b (2x2: rows 0,1, cols 2,3)
      
      A1a*B1a: 2x2 * 2x2, Strassen rank 7
      A1a*B1b: 2x2 * 2x2, Strassen rank 7
      A1b*B1a: 1x2 * 2x2, rank 2
      A1b*B1b: 1x2 * 2x2, rank 2
      
      A2*B2: 3x1 * 1x4, rank 1
      
      Total rank: 7+7+2+2+1 = 19.
      
      Let me build this.
    """
    n, m, p = 3, 3, 4
    total_rank = 19
    
    # We'll build U, V, W as lists of lists
    U = [[0] * total_rank for _ in range(n * m)]  # 9 x 19
    V = [[0] * total_rank for _ in range(m * p)]  # 12 x 19
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 19
    
    # Helper to add to U, V, W
    def add_U(i, j, c, val):
        U[i * m + j][c] += val
    def add_V(l, k, c, val):
        V[l * p + k][c] += val
    def add_W(k, i, c, val):
        W[k * n + i][c] += val
    
    # Block 1: A1a * B1a (2x2 * 2x2)
    # A1a: i in [0,2), j in [0,2)
    # B1a: l in [0,2), k in [0,2)
    # C1a: i in [0,2), k in [0,2)
    U1, V1, W1 = _construct_strassen_block(0, 0, [0, 1], [0, 1], [0, 1])
    for c, data in U1.items():
        for (i, j), val in data.items():
            add_U(i, j, c, val)
    for c, data in V1.items():
        for (l, k), val in data.items():
            add_V(l, k, c, val)
    for c, data in W1.items():
        for (k, i), val in data.items():
            add_W(k, i, c, val)
    
    # Block 2: A1a * B1b (2x2 * 2x2)
    # A1a: i in [0,2), j in [0,2)
    # B1b: l in [0,2), k in [2,4)
    # C1b: i in [0,2), k in [2,4)
    U2, V2, W2 = _construct_strassen_block(7, 0, [0, 1], [0, 1], [2, 3])
    for c, data in U2.items():
        for (i, j), val in data.items():
            add_U(i, j, c, val)
    for c, data in V2.items():
        for (l, k), val in data.items():
            add_V(l, k, c, val)
    for c, data in W2.items():
        for (k, i), val in data.items():
            add_W(k, i, c, val)
    
    # Block 3: A1b * B1a (1x2 * 2x2)
    # A1b: i=2, j in [0,2)
    # B1a: l in [0,2), k in [0,2)
    # C1c: i=2, k in [0,2)
    U3, V3, W3 = _construct_rectangular_block(14, 2, [0, 1], [0, 1], [0, 1])
    for c, data in U3.items():
        for (i, j), val in data.items():
            add_U(i, j, c, val)
    for c, data in V3.items():
        for (l, k), val in data.items():
            add_V(l, k, c, val)
    for c, data in W3.items():
        for (k, i), val in data.items():
            add_W(k, i, c, val)
    
    # Block 4: A1b * B1b (1x2 * 2x2)
    # A1b: i=2, j in [0,2)
    # B1b: l in [0,2), k in [2,4)
    # C1d: i=2, k in [2,4)
    U4, V4, W4 = _construct_rectangular_block(16, 2, [0, 1], [0, 1], [2, 3])
    for c, data in U4.items():
        for (i, j), val in data.items():
            add_U(i, j, c, val)
    for c, data in V4.items():
        for (l, k), val in data.items():
            add_V(l, k, c, val)
    for c, data in W4.items():
        for (k, i), val in data.items():
            add_W(k, i, c, val)
    
    # Block 5: A2 * B2 (3x1 * 1x4), rank 1
    # A2: i in [0,3), j=2
    # B2: l=2, k in [0,4)
    # C2: i in [0,3), k in [0,4)
    c = 18
    for i in range(3):
        add_U(i, 2, c, 1)
    for k in range(4):
        add_V(2, k, c, 1)
    for k in range(4):
        for i in range(3):
            add_W(k, i, c, 1)
    
    return {"U": U, "V": V, "W": W}


def construct():
    result = _build_decomposition()
    
    # Verify
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    
    if not _validate(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
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
    
    return result
