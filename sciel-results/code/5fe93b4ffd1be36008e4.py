import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 4, 5, 6
    
    # We will build the decomposition by splitting the dimensions.
    # C (4x6) = A (4x5) B (5x6)
    # Split m=5 into m1=1, m2=4.
    # C = A[:, :1] B[:1, :] + A[:, 1:] B[1:, :]
    # Part 1: 4x1x6 (Rank 24)
    # Part 2: 4x4x6
    
    # Part 2: Split p=6 into p1=2, p2=4.
    # C2 = A2 B2[:, :2] + A2 B2[:, 2:]
    # Part 2a: 4x4x2 (Rank 28)
    # Part 2b: 4x4x4 (Rank 49)
    
    # Total Rank = 24 + 28 + 49 = 101
    
    rank_total = 101
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Helper to add a base case decomposition
    def add_decomposition(n_sub, m_sub, p_sub, i_off, j_off, k_off, rank_offset, U_mat, V_mat, W_mat):
        # This function assumes U_mat, V_mat, W_mat are the local matrices for the subproblem
        # and writes them into the global U, V, W at the specified offsets and columns
        for r in range(rank_offset, rank_offset + U_mat.shape[1]):
            # Copy U
            U[i_off * m_sub + (r - rank_offset) * 0: i_off * m_sub + m_sub * m_sub, :] = 0 # Placeholder, will do direct copy below
        # Simpler: Direct slice assignment
        # U shape: (n*m, rank). Sub U shape: (n_sub*m_sub, rank_sub)
        # Row offset: i_off * m_sub
        # Col offset: rank_offset
        U[i_off * m_sub : i_off * m_sub + n_sub * m_sub, rank_offset : rank_offset + U_mat.shape[1]] = U_mat
        V[j_off * p_sub : j_off * p_sub + m_sub * p_sub, rank_offset : rank_offset + V_mat.shape[1]] = V_mat
        W[k_off * n_sub : k_off * n_sub + p_sub * n_sub, rank_offset : rank_offset + W_mat.shape[1]] = W_mat
        return rank_offset + U_mat.shape[1]

    def get_strassen_2x2_2x2():
        # Returns U, V, W for 2x2x2 multiplication with rank 7
        # Indices:
        # A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # C: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
        # Note: C index in global W is k*n + i. For n=2, k*2 + i.
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen form for 2x2:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B10-B11)
        # M3 = A11(B01-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        
        # U vectors (coefficients for A)
        # M0: A00=1, A11=1 -> [1, 0, 0, 1]
        # M1: A10=1, A11=1 -> [0, 0, 1, 1]
        # M2: A00=1 -> [1, 0, 0, 0]
        # M3: A11=1 -> [0, 0, 0, 1]
        # M4: A00=1, A01=1 -> [1, 1, 0, 0]
        # M5: A10=1, A00=-1 -> [-1, 0, 1, 0]
        # M6: A01=1, A11=-1 -> [0, 1, 0, -1]
        
        U[:, 0] = [1, 0, 0, 1]
        U[:, 1] = [0, 0, 1, 1]
        U[:, 2] = [1, 0, 0, 0]
        U[:, 3] = [0, 0, 0, 1]
        U[:, 4] = [1, 1, 0, 0]
        U[:, 5] = [-1, 0, 1, 0]
        U[:, 6] = [0, 1, 0, -1]
        
        # V vectors (coefficients for B)
        # M0: B00=1, B11=1 -> [1, 0, 0, 1]
        # M1: B00=1 -> [1, 0, 0, 0]
        # M2: B10=1, B11=-1 -> [0, 0, 1, -1]
        # M3: B01=1, B00=-1 -> [-1, 1, 0, 0]
        # M4: B11=1 -> [0, 0, 0, 1]
        # M5: B00=1, B01=1 -> [1, 1, 0, 0]
        # M6: B10=1, B11=1 -> [0, 0, 1, 1]
        
        V[:, 0] = [1, 0, 0, 1]
        V[:, 1] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        V[:, 3] = [-1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        V[:, 5] = [1, 1, 0, 0]
        V[:, 6] = [0, 0, 1, 1]
        
        # W vectors (coefficients for C)
        # C00 = M0 - M3 + M5 - M6
        # C01 = M4 + M6
        # C10 = M2 + M3
        # C11 = M0 - M1 + M4 + M5
        
        # W indices: 0: C00, 1: C01, 2: C10, 3: C11
        # M0 contributes to C00(1), C11(1) -> [1, 0, 0, 1]
        # M1 contributes to C11(-1) -> [0, 0, 0, -1]
        # M2 contributes to C10(1) -> [0, 0, 1, 0]
        # M3 contributes to C00(-1), C10(1) -> [-1, 0, 1, 0]
        # M4 contributes to C01(1), C11(1) -> [0, 1, 0, 1]
        # M5 contributes to C00(1), C11(1) -> [1, 0, 0, 1]
        # M6 contributes to C00(-1), C01(1) -> [-1, 1, 0, 0]
        
        W[:, 0] = [1, 0, 0, 1]
        W[:, 1] = [0, 0, 0, -1]
        W[:, 2] = [0, 0, 1, 0]
        W[:, 3] = [-1, 0, 1, 0]
        W[:, 4] = [0, 1, 0, 1]
        W[:, 5] = [1, 0, 0, 1]
        W[:, 6] = [-1, 1, 0, 0]
        
        return U, V, W

    # Part 1: 4x1x6
    # n=4, m=1, p=6
    # Rank 24
    # U: 4x1. V: 1x6. W: 6x4.
    # For each i in 0..3, k in 0..5:
    # U[i, 0] = 1
    # V[0, k] = 1
    # W[k*4 + i, col] = 1
    n1, m1, p1 = 4, 1, 6
    rank1 = n1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    c = 0
    for i in range(n1):
        for k in range(p1):
            U1[i * m1][c] = 1 # i*1 + 0
            V1[0 * p1 + k][c] = 1 # 0*6 + k
            W1[k * n1 + i][c] = 1 # k*4 + i
            c += 1
    col = add_decomposition(n1, m1, p1, 0, 0, 0, 0, U1, V1, W1)
    
    # Part 2a: 4x4x2
    # n=4, m=4, p=2
    # Split n into 2+2.
    # Two subproblems: 2x4x2.
    # Split m into 2+2.
    # Four subproblems: 2x2x2.
    n2a, m2a, p2a = 4, 4, 2
    rank2a = 28
    U2a = np.zeros((n2a * m2a, rank2a))
    V2a = np.zeros((m2a * p2a, rank2a))
    W2a = np.zeros((p2a * n2a, rank2a))
    
    # Sub 2a-1: n=2, m=4, p=2, i_off=0, j_off=0, k_off=0
    # Sub 2a-2: n=2, m=4, p=2, i_off=2, j_off=0, k_off=0
    
    def construct_2x4x2():
        # n=2, m=4, p=2
        # Split m into 2+2.
        # Sub 1: 2x2x2 (i_off=0, j_off=0, k_off=0)
        # Sub 2: 2x2x2 (i_off=0, j_off=2, k_off=0)
        n_s, m_s, p_s = 2, 4, 2
        rank_s = 14
        U_s = np.zeros((n_s * m_s, rank_s))
        V_s = np.zeros((m_s * p_s, rank_s))
        W_s = np.zeros((p_s * n_s, rank_s))
        
        U_base, V_base, W_base = get_strassen_2x2_2x2()
        
        # Add sub 1 (m_off=0)
        U_s[0:4, 0:7] = U_base
        V_s[0:4, 0:7] = V_base
        W_s[0:4, 0:7] = W_base
        
        # Add sub 2 (m_off=2)
        # U is same (n=2, m=2 part of m=4? No, U is indexed by i,j.
        # For 2x4x2, U is 2x4.
        # Sub 1 uses j=0,1. Sub 2 uses j=2,3.
        # So U_s[:, 7:14] should be U_base for the second half of m?
        # Wait, U_base is for 2x2x2.
        # In 2x4x2, A is 2x4.
        # Sub 1: A[:, 0:2] B[0:2, :].
        # Sub 2: A[:, 2:4] B[2:4, :].
        # U for Sub 1: non-zero in cols 0,1.
        # U for Sub 2: non-zero in cols 2,3.
        # So we need to shift U_base columns?
        # No, U_base has shape (4, 7). Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
        # For Sub 1, these map to global (0,0),(0,1),(1,0),(1,1).
        # For Sub 2, these map to global (0,2),(0,3),(1,2),(1,3).
        # So we create a new U matrix for Sub 2 by shifting indices.
        
        U_s2 = np.zeros((4, 7))
        # Map local j=0,1 to global j=2,3
        # Local 0 (0,0) -> Global 2 (0,2)
        # Local 1 (0,1) -> Global 3 (0,3)
        # Local 2 (1,0) -> Global 6 (1,2)
        # Local 3 (1,1) -> Global 7 (1,3)
        # U_base rows: 0,1,2,3.
        # U_s2[2, :] = U_base[0, :]
        # U_s2[3, :] = U_base[1, :]
        # U_s2[6, :] = U_base[2, :]
        # U_s2[7, :] = U_base[3, :]
        # But U_s shape is (8, 14). Rows 0..7.
        # U_s[:, 7:14] needs to be filled.
        
        # V for Sub 2:
        # V_base is (4, 7). Indices 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1).
        # For Sub 2, B is 2x2 (rows 2,3 of the 4-row B).
        # So V indices map to global rows 2,3.
        # V_s2[0, :] = V_base[0, :] (local row 0 -> global row 2? No, V is indexed by j,k.
        # j is the row of B.
        # Local j=0 -> Global j=2.
        # Local j=1 -> Global j=3.
        # So V_s[2:4, 7:14] = V_base[0:2, :]?
        # V_base shape (4,7). Rows 0,1,2,3.
        # Row 0: j=0, k=0
        # Row 1: j=0, k=1
        # Row 2: j=1, k=0
        # Row 3: j=1, k=1
        # Global j=2 corresponds to local j=0.
        # Global j=3 corresponds to local j=1.
        # So V_s[2:4, 7:14] = V_base[0:2, :]?
        # No, V_s has shape (8, 14) because m_s=4, p_s=2.
        # Rows 0..7.
        # Global j=2,3 are rows 2,3 of V_s.
        # So V_s[2:4, 7:14] = V_base[0:2, :]?
        # Wait, V_base has 4 rows.
        # We only use the first 2 rows of V_base?
        # No, V_base is for 2x2x2.
        # In Sub 2, B is 2x2.
        # So we use the full V_base, but mapped to global rows 2,3?
        # V_base row 0 (j=0, k=0) -> Global j=2, k=0 -> Index 2*2+0=4?
        # No, V index is j*p + k.
        # p=2.
        # Global j=2, k=0 -> Index 2*2 + 0 = 4.
        # Global j=2, k=1 -> Index 2*2 + 1 = 5.
        # Global j=3, k=0 -> Index 3*2 + 0 = 6.
        # Global j=3, k=1 -> Index 3*2 + 1 = 7.
        # So we need to place V_base into V_s[4:8, 7:14].
        
        # W for Sub 2:
        # W is indexed by k*n + i.
        # n=2.
        # W_base shape (4, 7).
        # Global indices are same as local because i, k are not shifted.
        # So W_s[0:4, 7:14] = W_base.
        
        # Let's re-evaluate U.
        # U_s shape (8, 14).
        # Sub 1: U_s[0:4, 0:7] = U_base.
        # Sub 2:
        # Local (0,0) -> Global (0,2) -> Index 0*4+2=2.
        # Local (0,1) -> Global (0,3) -> Index 0*4+3=3.
        # Local (1,0) -> Global (1,2) -> Index 1*4+2=6.
        # Local (1,1) -> Global (1,3) -> Index 1*4+3=7.
        # So:
        # U_s[2, 7:14] = U_base[0, :]
        # U_s[3, 7:14] = U_base[1, :]
        # U_s[6, 7:14] = U_base[2, :]
        # U_s[7, 7:14] = U_base[3, :]
        
        V_s[4:8, 7:14] = V_base
        W_s[0:4, 7:14] = W_base
        
        return U_s, V_s, W_s
    
    U2a_s1, V2a_s1, W2a_s1 = construct_2x4x2()
    # Place into U2a (4x4x2)
    # Sub 1: i_off=0.
    U2a[0:8, 0:14] = U2a_s1
    V2a[0:8, 0:14] = V2a_s1
    W2a[0:4, 0:14] = W2a_s1
    
    # Sub 2: i_off=2.
    # n=2, m=4, p=2.
    # i_off=2 means global rows 2,3 of A.
    # U2a_s1 is for rows 0,1.
    # We need to shift U rows.
    # Local row 0 (i=0, j=0..3) -> Global row 2 (i=2, j=0..3).
    # Index: i*4 + j.
    # Local 0 (0,0) -> Global 2*4+0=8.
    # Local 1 (0,1) -> Global 2*4+1=9.
    # Local 2 (0,2) -> Global 2*4+2=10.
    # Local 3 (0,3) -> Global 2*4+3=11.
    # Local 4 (1,0) -> Global 3*4+0=12.
    # ...
    U2a_s2 = np.zeros((16, 14))
    U2a_s2[8:12, 0:7] = U2a_s1[0:4, 0:7]
    U2a_s2[12:16, 0:7] = U2a_s1[4:8, 0:7] # Wait, U2a_s1 has 8 rows.
    # U2a_s1 rows 0..3 correspond to i=0. Rows 4..7 correspond to i=1.
    # So:
    # U2a_s2[8:12, :] = U2a_s1[0:4, :]
    # U2a_s2[12:16, :] = U2a_s1[4:8, :]
    
    # V and W are same (j,k and k,i don't shift with i_off?
    # V is j,k. j is col of A, row of B.
    # In 4x4x2, A is 4x4.
    # Sub 1: A[0:2, :]. Sub 2: A[2:4, :].
    # V depends on j (col of A).
    # So V is same for both subproblems?
    # Yes, V is indexed by j,k.
    # j=0..3.
    # So V2a[0:8, 14:28] = V2a_s1
    # W depends on k,i.
    # i shifts.
    # W_base is k*n + i. n=2.
    # Sub 1: i=0,1.
    # Sub 2: i=2,3.
    # Global W index: k*4 + i.
    # Sub 1: k*4 + 0/1.
    # Sub 2: k*4 + 2/3.
    # W2a_s1 has shape (4, 14).
    # Rows 0..3 correspond to k=0,1 and i=0,1?
    # No, W_s1 was constructed for n=2.
    # W_s1 index: k*2 + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # Global W index for n=4: k*4 + i.
    # Sub 1 (i=0,1):
    # Local 0 (k=0, i=0) -> Global 0*4+0=0
    # Local 1 (k=0, i=1) -> Global 0*4+1=1
    # Local 2 (k=1, i=0) -> Global 1*4+0=4
    # Local 3 (k=1, i=1) -> Global 1*4+1=5
    # Sub 2 (i=2,3):
    # Local 0 (k=0, i=0) -> Global 0*4+2=2
    # Local 1 (k=0, i=1) -> Global 0*4+3=3
    # Local 2 (k=1, i=0) -> Global 1*4+2=6
    # Local 3 (k=1, i=1) -> Global 1*4+3=7
    
    W2a_s2 = np.zeros((16, 14)) # W2a has shape (2*4, 28) = (8, 28).
    # Wait, W2a shape is (p*n, rank) = (8, 28).
    # W2a_s1 shape is (4, 14).
    
    # Place Sub 1
    W2a[0:2, 0:14] = W2a_s1[0:2, :] # Local 0,1 -> Global 0,1
    W2a[4:6, 0:14] = W2a_s1[2:4, :] # Local 2,3 -> Global 4,5
    
    # Place Sub 2
    W2a[2:4, 14:28] = W2a_s1[0:2, :] # Local 0,1 -> Global 2,3
    W2a[6:8, 14:28] = W2a_s1[2:4, :] # Local 2,3 -> Global 6,7
    
    # V2a
    V2a[0:8, 14:28] = V2a_s1
    
    # U2a
    U2a[8:12, 14:28] = U2a_s1[0:4, :]
    U2a[12:16, 14:28] = U2a_s1[4:8, :]
    
    col = add_decomposition(n2a, m2a, p2a, 0, 1, 0, col, U2a, V2a, W2a)
    # Note: j_off=1 because we split m=5 into 1+4. The 4x4x2 part uses m indices 1..4.
    # k_off=0. i_off=0.
    
    # Part 2b: 4x4x4
    # n=4, m=4, p=4
    # Rank 49
    # Constructed from 7 blocks of 2x2x2?
    # No, Strassen for 4x4 uses 7 multiplications of 2x2.
    # We need to map these to the global indices.
    
    n2b, m2b, p2b = 4, 4, 4
    rank2b = 49
    U2b = np.zeros((n2b * m2b, rank2b))
    V2b = np.zeros((m2b * p2b, rank2b))
    W2b = np.zeros((p2b * n2b, rank2b))
    
    # Strassen 4x4:
    # Split A, B, C into 2x2 blocks.
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    # C00, C01, C10, C11
    
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 product.
    # We use the 2x2x2 Strassen decomposition for each Mk.
    # This means for each Mk, we add 7 rank-1 tensors.
    # Total 49.
    
    # We need to define the linear forms for A and B for each Mk,
    # and then apply the 2x2 Strassen coefficients within that block.
    
    # This is complex to hardcode.
    # Instead, we can use the recursive property.
    # But for efficiency, we'll just expand the 7 blocks.
    
    # For each of the 7 Strassen terms, we have a 2x2x2 multiplication.
    # Let's define the offsets for the 2x2 blocks.
    # Block (bi, bj):
    # A block (bi, bj): rows 2*bi, cols 2*bj.
    # B block (bj, bk): rows 2*bj, cols 2*bk.
    # C block (bi, bk): rows 2*bi, cols 2*bk.
    
    # We'll iterate through the 7 terms.
    # Term 0: A_blocks {00, 11}, B_blocks {00, 11}, C_blocks {00, 01, 10, 11}
    # The coefficients for A and B are sums/diffs of blocks.
    # This requires building U and V that span these blocks.
    
    # Given the time constraint, I will use the schoolbook for 4x4x4 (Rank 64) if the recursive construction is too slow to write.
    # Schoolbook 4x4x4 Rank 64.
    # Total Rank = 24 + 28 + 64 = 116.
    # This is still better than 120.
    
    # Let's stick to Schoolbook for 4x4x4 to ensure correctness and speed.
    n2b, m2b, p2b = 4, 4, 4
    rank2b = n2b * m2b * p2b
    U2b = np.zeros((n2b * m2b, rank2b))
    V2b = np.zeros((m2b * p2b, rank2b))
    W2b = np.zeros((p2b * n2b, rank2b))
    c = 0
    for i in range(n2b):
        for j in range(m2b):
            for k in range(p2b):
                U2b[i * m2b + j][c] = 1
                V2b[j * p2b + k][c] = 1
                W2b[k * n2b + i][c] = 1
                c += 1
    
    col = add_decomposition(n2b, m2b, p2b, 0, 1, 2, col, U2b, V2b, W2b)
    # j_off=1 (m indices 1..4), k_off=2 (p indices 2..5).
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
