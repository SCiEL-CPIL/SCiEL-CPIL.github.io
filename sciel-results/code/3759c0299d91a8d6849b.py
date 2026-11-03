"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) as lists for schoolbook n*m*p rank decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
    U shape: 4x7 (rows = i*m+j for 2x2 matrix A)
    V shape: 4x7 (rows = j*p+k for 2x2 matrix B)
    W shape: 4x7 (rows = k*n+i for 2x2 matrix C, trace convention)
    """
    n, m, p = 2, 2, 2
    rank = 7
    # Strassen 7 multiplications:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) -> i*2+j, cols 0..6
    # A = [[A00, A01], [A10, A11]]
    # U[i*2+j, c] = coefficient of A[i,j] in M_c
    U = [[0.0]*7 for _ in range(4)]
    # M1: A00 + A11
    U[0][0] = 1.0  # A00
    U[3][0] = 1.0  # A11
    # M2: A10 + A11
    U[2][1] = 1.0  # A10
    U[3][1] = 1.0  # A11
    # M3: A00
    U[0][2] = 1.0  # A00
    # M4: A11
    U[3][3] = 1.0  # A11
    # M5: A00 + A01
    U[0][4] = 1.0  # A00
    U[1][4] = 1.0  # A01
    # M6: A10 - A00
    U[0][5] = -1.0  # A00
    U[2][5] = 1.0  # A10
    # M7: A01 - A11
    U[1][6] = 1.0  # A01
    U[3][6] = -1.0  # A11
    
    # V: rows indexed by (j,k) -> j*2+k, cols 0..6
    # B = [[B00, B01], [B10, B11]]
    V = [[0.0]*7 for _ in range(4)]
    # M1: B00 + B11
    V[0][0] = 1.0  # B00
    V[3][0] = 1.0  # B11
    # M2: B00
    V[0][1] = 1.0  # B00
    # M3: B01 - B11
    V[1][2] = 1.0  # B01
    V[3][2] = -1.0  # B11
    # M4: B10 - B00
    V[0][3] = -1.0  # B00
    V[2][3] = 1.0  # B10
    # M5: B11
    V[3][4] = 1.0  # B11
    # M6: B00 + B01
    V[0][5] = 1.0  # B00
    V[1][5] = 1.0  # B01
    # M7: B10 + B11
    V[2][6] = 1.0  # B10
    V[3][6] = 1.0  # B11
    
    # W: rows indexed by (k,i) -> k*2+i (trace convention: W[k*n+i]), cols 0..6
    # C = [[C00, C01], [C10, C11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W = [[0.0]*7 for _ in range(4)]
    # Row 0: k=0,i=0 -> C00
    W[0][0] = 1.0   # M1
    W[0][3] = 1.0   # M4
    W[0][4] = -1.0  # M5
    W[0][6] = 1.0   # M7
    # Row 1: k=0,i=1 -> C10
    W[1][1] = 1.0   # M2
    W[1][3] = 1.0   # M4
    # Row 2: k=1,i=0 -> C01
    W[2][2] = 1.0   # M3
    W[2][4] = 1.0   # M5
    # Row 3: k=1,i=1 -> C11
    W[3][0] = 1.0   # M1
    W[3][1] = -1.0  # M2
    W[3][2] = 1.0   # M3
    W[3][5] = 1.0   # M6
    
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    """
    Combine two block decompositions.
    Block 1: n1 x m x p1
    Block 2: n2 x m x p2
    Total: n=(n1+n2) x m x p=(p1+p2)
    U has n*m rows, V has m*p rows, W has p*n rows.
    """
    n1 = len(U1) // len(V1[0])  # This is wrong, let me think differently
    # Actually, U1 shape: (n1*m, r1), V1 shape: (m*p1, r1), W1 shape: (p1*n1, r1)
    # U2 shape: (n2*m, r2), V2 shape: (m*p2, r2), W2 shape: (p2*n2, r2)
    
    n1 = len(U1) // (len(V1[0]) if len(V1) > 0 else 1)  # This is getting complex
    # Let me use a different approach: pass dimensions explicitly
    pass

def construct():
    """
    Construct a decomposition for 3x4x7 matrix multiplication.
    
    Strategy: Partition the 3x4 matrix into a 2x4 top block and 1x4 bottom block.
    Partition the 4x7 matrix into a 4x6 left part and 4x1 right part? 
    Actually, let's think about it as:
    - Top 2 rows of A (2x4) times full B (4x7) -> gives top 2 rows of C (2x7)
    - Bottom 1 row of A (1x4) times full B (4x7) -> gives bottom 1 row of C (1x7)
    
    For the 2x4x7 part, we can use Strassen-like methods.
    For the 1x4x7 part, it's just a dot product, rank 4 (or 7 depending on how we view it).
    
    Actually, a 1x4 times 4x7 is just a row vector times a matrix, which is rank 4 (one scalar per column of A's row).
    
    Let me reconsider. The tensor for n x m x p matrix mult has:
    - U: n*m rows (indexed by i*m+j)
    - V: m*p rows (indexed by j*p+k)  
    - W: p*n rows (indexed by k*n+i)
    
    For 3x4x7:
    - U: 12 rows
    - V: 28 rows
    - W: 21 rows
    
    Block approach:
    - Block A_top: rows 0-1 of A (2x4), indices i=0,1 -> U rows 0-7
    - Block A_bot: row 2 of A (1x4), index i=2 -> U rows 8-11
    
    For A_top (2x4) * B (4x7):
    We can further split B into B_left (4x3) and B_right (4x4) or use Strassen on 2x2 blocks.
    
    Let's use a simpler approach: 
    - 2x4x7 can be done with rank at most 2*4*7 = 56 schoolbook, but we can do better.
    - Using Strassen on 2x2x2 blocks: split 2x4 into two 2x2 blocks, and 4x7 into... hmm, 7 isn't even.
    
    Let me try: split the 4 columns into 2+2, and the 7 rows into 3+4.
    - 2x2 * 2x3: rank at most 2*2*3=12, or with Strassen-like: 2x2x2 part (rank 7) + 2x2x1 part (rank 4) = 11
    - 2x2 * 2x4: rank at most 2*2*4=16, or 2x2x2+2x2x2: 7+7=14
    Total for 2x4x7: 11+14 = 25
    
    For 1x4x7: rank 4 (each of the 4 elements of the row vector times each column of B)
    Actually 1x4 * 4x7: the result is 1x7. Each output C[0,k] = sum_j A[0,j]*B[j,k].
    This is rank 4: for each j, U[0,j]=1, V[j,k]=B[j,k]... wait, no.
    
    In the tensor decomposition framework:
    U[i*m+j, c] for i=0 (only row), j=0..3 -> 4 rows
    V[j*p+k, c] for j=0..3, k=0..6 -> 28 rows
    W[k*n+i, c] for k=0..6, i=0 -> 7 rows
    
    Schoolbook rank = 4*7 = 28 for this block.
    But actually, 1x4*4x7 is just a linear map from R^4 to R^7, which has rank 4 (the 4 input dimensions).
    Wait, no. The tensor rank of 1x4x7 multiplication... 
    
    For 1x4 * 4x7: C[k] = sum_j A[j] * B[j,k]
    This can be decomposed with rank 4: for each j, U[j]=1, V[j,k]=B[j,k]... but V must be constant!
    
    In the tensor decomposition, U, V, W are fixed matrices. The decomposition says:
    C = sum_c (U[:,c]^T * A) * (V[:,c]^T * B) ... no, that's not right either.
    
    Let me recall the correct formulation. For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition with rank r means there exist U (n*m x r), V (m*p x r), W (p*n x r) such that:
    C[i,k] = sum_c U[i*m+j... wait, I need to be more careful.
    
    The standard formulation: the matrix multiplication tensor T has entries T[i,j,k] = delta_{j,k} (in some indexing).
    The decomposition is: T = sum_c u_c * v_c * w_c where u_c is a vector of size n*m, v_c of size m*p, w_c of size p*n.
    
    Specifically, for the entry corresponding to computing C[i,k] from A[i,:] and B[:,k]:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) ... 
    
    Hmm, I think the correct interpretation is:
    C[i,k] = sum_c W[k*n+i, c] * (U[:,c]^T * A_row_i) * (V[:,c]^T * B_col_k)
    
    No, that's not right. Let me look at the schoolbook example:
    U[i*m+j, c] = 1 when c corresponds to (i,j,k)
    V[j*p+k, c] = 1 when c corresponds to (i,j,k)
    W[k*n+i, c] = 1 when c corresponds to (i,j,k)
    
    So C[i,k] = sum_c W[k*n+i, c] * U[i*m+?, c] * V[?*p+k, c]
    
    Actually, I think the contraction is:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    No wait, that would give a product of two sums, which isn't right.
    
    Let me think again. In the schoolbook case with rank n*m*p, each column c corresponds to a triple (i,j,k).
    U[i*m+j, c] = 1 iff the (i,j) index matches.
    V[j*p+k, c] = 1 iff the (j,k) index matches.
    W[k*n+i, c] = 1 iff the (k,i) index matches.
    
    Then C[i,k] should equal sum_j A[i,j] B[j,k].
    
    The formula must be: C[i,k] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
    summed over j as well? No, that doesn't make sense with the indices.
    
    I think the correct interpretation is that the tensor T_{i,j,k} for matrix multiplication is defined such that:
    C[i,k] = sum_{j} A[i,j] B[j,k]
    
    And the rank-r decomposition gives:
    C[i,k] = sum_{c=1}^r (sum_j U[i*m+j, c] A[i,j]) * W[k*n+i, c] * (sum_j V[j*p+k, c] B[j,k])
    
    No, that's a product of three things, not right.
    
    Actually, I believe the correct formulation is:
    C[i,k] = sum_c W[k*n+i, c] * (U[:,c]^T * vec(A)_relevant) ... 
    
    Let me just verify with schoolbook. For a specific (i,j,k), column c has:
    U[i*m+j, c] = 1, all other U entries 0.
    V[j*p+k, c] = 1, all other V entries 0.
    W[k*n+i, c] = 1, all other W entries 0.
    
    If the formula is C[i,k] = sum_c W[k*n+i, c] * U[i*m+j', c] * V[j'*p+k, c] * A[i,j'] * B[j',k] summed over j':
    For the column c corresponding to (i,j,k), W[k*n+i,c]=1, and U[i*m+j',c]=1 only if j'=j, V[j'*p+k,c]=1 only if j'=j.
    So the term contributes A[i,j]*B[j,k]. Summing over all c gives sum_j A[i,j]B[j,k]. Correct!
    
    So the formula is:
    C[i,k] = sum_c sum_j W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
    
    Which can be written as:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    
    Wait, that's not right because the j in U and V are the same j. So it's:
    C[i,k] = sum_c W[k*n+i, c] * sum_j (U[i*m+j, c] * A[i,j]) * (V[j*p+k, c] * B[j,k])
    
    Hmm, but U[i*m+j,c] and V[j*p+k,c] depend on j, and they're multiplied together with A[i,j] and B[j,k]. This is a bilinear form in A and B for each c, weighted by W.
    
    OK so the formula is:
    C[i,k] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k]
    
    For schoolbook, each c picks out one j, so sum_j becomes just the j term.
    
    Now, for the 1x4x7 block (i=2 only):
    C[2,k] = sum_j A[2,j] B[j,k]
    
    Rank 4 decomposition: for each j (0..3), one column c:
    U[2*4+j, c] = 1 (i.e., U[8+j, c] = 1)
    V[j*7+k, c] = 1 for all k (i.e., V[j*7+k, c] = 1)
    W[k*3+2, c] = 1 for all k (i.e., W[k*3+2, c] = 1)
    
    Wait, that gives C[2,k] = sum_c W[k*3+2,c] * sum_j U[8+j,c] * V[j*7+k,c] * A[2,j] * B[j,k]
    = sum_c 1 * 1 * 1 * A[2,j_c] * B[j_c, k] where j_c is the j for column c
    = sum_j A[2,j] B[j,k]
    
    Yes! So rank 4 works for the 1x4x7 block.
    
    For the 2x4x7 block (i=0,1):
    C[i,k] = sum_j A[i,j] B[j,k] for i=0,1.
    
    We can split j into j1 (0,1) and j2 (2,3):
    C[i,k] = sum_{j1=0,1} A[i,j1] B[j1,k] + sum_{j2=2,3} A[i,j2] B[j2,k]
    
    Each part is 2x2x7. For 2x2x7, we can split k into k1 (0..2) and k2 (3..6):
    2x2x3 + 2x2x4
    
    2x2x3: rank at most 2*2*3=12, or use Strassen: 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
    2x2x4: 2x2x2 + 2x2x2, rank 7+7=14
    
    Total for 2x4x7: 11+14 = 25
    Total: 25 + 4 = 29
    
    That's much better than 84! Let me implement this.
    
    Actually, let me be more careful and just use a straightforward block decomposition with Strassen where applicable.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = 3, 4, 7
    
    # We'll build the decomposition in blocks.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1,2} -> 2x2x3
    # Block 2: i in {0,1}, j in {0,1}, k in {3,4,5,6} -> 2x2x4
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1,2} -> 2x2x3
    # Block 4: i in {0,1}, j in {2,3}, k in {3,4,5,6} -> 2x2x4
    # Block 5: i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6} -> 1x4x7
    
    # For 2x2x3: use Strassen for 2x2x2 part + schoolbook for 2x2x1 part
    # For 2x2x4: use Strassen for two 2x2x2 parts
    # For 1x4x7: rank 4 as derived above
    
    def make_2x2x2_strassen(i_off, j_off, k_off):
        """
        Return (U_block, V_block, W_block) for 2x2x2 Strassen at offsets.
        i_off: offset for i index (0 or 1 for the 2x2 block in the 3x4 matrix)
        j_off: offset for j index (0 or 2 for the 2x2 block in the 4x7 matrix)
        k_off: offset for k index (0, 2, 3, or 5 for the 2x2 block in the 7 columns)
        
        U has rows for i in {i_off, i_off+1}, j in {j_off, j_off+1}
        V has rows for j in {j_off, j_off+1}, k in {k_off, k_off+1}
        W has rows for k in {k_off, k_off+1}, i in {i_off, i_off+1}
        """
        U_sub, V_sub, W_sub = _strassen_2x2x2()
        # U_sub is 4x7, rows indexed by (i_local, j_local) -> i_local*2+j_local
        # Map to global: row i_global*m + j_global where i_global = i_off+i_local, j_global = j_off+j_local
        # V_sub is 4x7, rows indexed by (j_local, k_local) -> j_local*2+k_local
        # Map to global: row j_global*p + k_global where j_global = j_off+j_local, k_global = k_off+k_local
        # W_sub is 4x7, rows indexed by (k_local, i_local) -> k_local*2+i_local (trace convention)
        # Map to global: row k_global*n + i_global where k_global = k_off+k_local, i_global = i_off+i_local
        
        U_block = [[0.0]*7 for _ in range(4)]
        V_block = [[0.0]*7 for _ in range(4)]
        W_block = [[0.0]*7 for _ in range(4)]
        
        for il in range(2):
            for jl in range(2):
                ig = i_off + il
                jg = j_off + jl
                U_block[il*2+jl] = U_sub[il*2+jl]  # Same local indexing
        
        for jl in range(2):
            for kl in range(2):
                jg = j_off + jl
                kg = k_off + kl
                V_block[jl*2+kl] = V_sub[jl*2+kl]
        
        for kl in range(2):
            for il in range(2):
                kg = k_off + kl
                ig = i_off + il
                W_block[kl*2+il] = W_sub[kl*2+il]
        
        return U_block, V_block, W_block
    
    def make_2x2x1(i_off, j_off, k_off):
        """
        Return (U_block, V_block, W_block) for 2x2x1 schoolbook at offsets.
        Rank 4.
        """
        U_block = [[0.0]*4 for _ in range(4)]
        V_block = [[0.0]*4 for _ in range(4)]
        W_block = [[0.0]*4 for _ in range(2)]  # 2 rows for k (only one k), 2 i's -> 2*2=4? No, p*n for this block is 1*2=2
        
        # Wait, for 2x2x1: n=2, m=2, p=1
        # U: 2*2=4 rows
        # V: 2*1=2 rows
        # W: 1*2=2 rows
        # Rank: 4 (schoolbook)
        
        U_block = [[0.0]*4 for _ in range(4)]
        V_block = [[0.0]*4 for _ in range(2)]
        W_block = [[0.0]*4 for _ in range(2)]
        
        col = 0
        for il in range(2):
            for jl in range(2):
                U_block[il*2+jl][col] = 1.0
                V_block[jl][col] = 1.0
                W_block[il][col] = 1.0
                col += 1
        
        return U_block, V_block, W_block
    
    # Now assemble the full decomposition.
    # Total rank = 2*11 + 2*14 + 4 = 22 + 28 + 4 = 54? 
    # Wait: Block 1 (2x2x3): 11, Block 2 (2x2x4): 14, Block 3 (2x2x3): 11, Block 4 (2x2x4): 14, Block 5 (1x4x7): 4
    # Total: 11+14+11+14+4 = 54
    
    # Hmm, that's better than 84 but let me double-check the 2x2x4 part.
    # 2x2x4: split into two 2x2x2, each rank 7, total 14. Yes.
    # 2x2x3: split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11. Yes.
    
    # Let's build it.
    rank_total = 11 + 14 + 11 + 14 + 4  # = 54
    
    U = [[0.0] * rank_total for _ in range(n * m)]  # 12 rows
    V = [[0.0] * rank_total for _ in range(m * p)]  # 28 rows
    W = [[0.0] * rank_total for _ in range(p * n)]  # 21 rows
    
    col_offset = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1,2}
    # 2x2x2 part: i in {0,1}, j in {0,1}, k in {0,1}
    U1, V1, W1 = make_2x2x2_strassen(0, 0, 0)
    for r in range(4):
        ig = (r // 2)  # i_local
        jg = (r % 2)   # j_local
        U[ig * m + jg][col_offset : col_offset + 7] = U1[r]
    
    for r in range(4):
        jg = (r // 2)   # j_local
        kg = (r % 2)    # k_local
        V[jg * p + kg][col_offset : col_offset + 7] = V1[r]
    
    for r in range(4):
        kg = (r // 2)   # k_local
        ig = (r % 2)    # i_local
        W[kg * n + ig][col_offset : col_offset + 7] = W1[r]
    col_offset += 7
    
    # 2x2x1 part: i in {0,1}, j in {0,1}, k in {2}
    U2, V2, W2 = make_2x2x1(0, 0, 2)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + jg][col_offset : col_offset + 4] = U2[r]
    
    for r in range(2):
        jg = r
        V[jg * p + 2][col_offset : col_offset + 4] = V2[r]
    
    for r in range(2):
        ig = r
        W[2 * n + ig][col_offset : col_offset + 4] = W2[r]
    col_offset += 4
    
    # Block 2: i in {0,1}, j in {0,1}, k in {3,4,5,6}
    # 2x2x2 part 1: i in {0,1}, j in {0,1}, k in {3,4}
    U3, V3, W3 = make_2x2x2_strassen(0, 0, 3)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + jg][col_offset : col_offset + 7] = U3[r]
    
    for r in range(4):
        jg = (r // 2)
        kg = (r % 2)
        V[jg * p + (3 + kg)][col_offset : col_offset + 7] = V3[r]
    
    for r in range(4):
        kg = (r // 2)
        ig = (r % 2)
        W[(3 + kg) * n + ig][col_offset : col_offset + 7] = W3[r]
    col_offset += 7
    
    # 2x2x2 part 2: i in {0,1}, j in {0,1}, k in {5,6}
    U4, V4, W4 = make_2x2x2_strassen(0, 0, 5)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + jg][col_offset : col_offset + 7] = U4[r]
    
    for r in range(4):
        jg = (r // 2)
        kg = (r % 2)
        V[jg * p + (5 + kg)][col_offset : col_offset + 7] = V4[r]
    
    for r in range(4):
        kg = (r // 2)
        ig = (r % 2)
        W[(5 + kg) * n + ig][col_offset : col_offset + 7] = W4[r]
    col_offset += 7
    
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1,2}
    # 2x2x2 part: i in {0,1}, j in {2,3}, k in {0,1}
    U5, V5, W5 = make_2x2x2_strassen(0, 2, 0)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + (2 + jg)][col_offset : col_offset + 7] = U5[r]
    
    for r in range(4):
        jg = (r // 2)
        kg = (r % 2)
        V[(2 + jg) * p + kg][col_offset : col_offset + 7] = V5[r]
    
    for r in range(4):
        kg = (r // 2)
        ig = (r % 2)
        W[kg * n + ig][col_offset : col_offset + 7] = W5[r]
    col_offset += 7
    
    # 2x2x1 part: i in {0,1}, j in {2,3}, k in {2}
    U6, V6, W6 = make_2x2x1(0, 2, 2)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + (2 + jg)][col_offset : col_offset + 4] = U6[r]
    
    for r in range(2):
        jg = r
        V[(2 + jg) * p + 2][col_offset : col_offset + 4] = V6[r]
    
    for r in range(2):
        ig = r
        W[2 * n + ig][col_offset : col_offset + 4] = W6[r]
    col_offset += 4
    
    # Block 4: i in {0,1}, j in {2,3}, k in {3,4,5,6}
    # 2x2x2 part 1: i in {0,1}, j in {2,3}, k in {3,4}
    U7, V7, W7 = make_2x2x2_strassen(0, 2, 3)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + (2 + jg)][col_offset : col_offset + 7] = U7[r]
    
    for r in range(4):
        jg = (r // 2)
        kg = (r % 2)
        V[(2 + jg) * p + (3 + kg)][col_offset : col_offset + 7] = V7[r]
    
    for r in range(4):
        kg = (r // 2)
        ig = (r % 2)
        W[(3 + kg) * n + ig][col_offset : col_offset + 7] = W7[r]
    col_offset += 7
    
    # 2x2x2 part 2: i in {0,1}, j in {2,3}, k in {5,6}
    U8, V8, W8 = make_2x2x2_strassen(0, 2, 5)
    for r in range(4):
        ig = (r // 2)
        jg = (r % 2)
        U[ig * m + (2 + jg)][col_offset : col_offset + 7] = U8[r]
    
    for r in range(4):
        jg = (r // 2)
        kg = (r % 2)
        V[(2 + jg) * p + (5 + kg)][col_offset : col_offset + 7] = V8[r]
    
    for r in range(4):
        kg = (r // 2)
        ig = (r % 2)
        W[(5 + kg) * n + ig][col_offset : col_offset + 7] = W8[r]
    col_offset += 7
    
    # Block 5: i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # Rank 4: for each j, one column
    for j in range(4):
        U[2 * m + j][col_offset] = 1.0
        for k in range(7):
            V[j * p + k][col_offset] = 1.0
        for k in range(7):
            W[k * n + 2][col_offset] = 1.0
        col_offset += 1
    
    # Verify col_offset matches rank_total
    assert col_offset == rank_total, f"Column offset {col_offset} != rank_total {rank_total}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
