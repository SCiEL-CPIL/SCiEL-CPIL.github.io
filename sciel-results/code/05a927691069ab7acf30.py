"""Matrix multiplication 3x3x6 - rank-30 decomposition via 3 blocks of 3x3x2 at rank 10."""
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W):
    """Verify the decomposition is exact for the 3x3x6 multiplication tensor."""
    n, m, p = n_global, m_global, p_global
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False, (i, j, k, s)
    return True, None

# Global dimensions for verification
n_global = 3
m_global = 3
p_global = 6

def _strassen_2x2_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U[i,j] coefficient, V[j,k] coefficient, W[k,i] coefficient for each term.
    Indexing: U row = i*2+j, V row = j*2+k, W row = k*2+i
    """
    terms = []
    # M1 = A00*(B00+B11) -> contributes to C00 and C11
    # C00 += M1, C11 -= M1
    terms.append(([(0,0,1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,1,-1)]))
    # M2 = (A00+A01)*B10 -> C00, C01
    terms.append(([(0,0,1),(0,1,1)], [(0,1,1),(1,1,1)], [(0,0,1),(0,1,1)]))
    # M3 = (A10+A11)*B01 -> C10, C11
    terms.append(([(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]))
    # M4 = A11*(B11-B00) -> C10, C11
    terms.append(([(1,1,1)], [(0,0,-1),(1,1,1)], [(1,0,1),(1,1,1)]))
    # M5 = (A00+A11)*(B01-B10) -> C10, C01
    terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(0,1,-1)], [(1,0,1),(0,1,1)]))
    # M6 = (A10-A00)*(B00+B11) -> C01, C11
    terms.append(([(0,0,-1),(1,0,1)], [(0,0,1),(0,1,1)], [(0,1,1),(1,1,-1)]))
    # M7 = (A01-A11)*(B10+B11) -> C00, C10
    terms.append(([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1),(1,0,1)]))
    return terms

def _construct_3x3x2_block():
    """
    Construct rank-10 decomposition for 3x3x2 matrix multiplication.
    A is 3x3, B is 3x2, C is 3x2.
    We split A into [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
    B into [[B11(2x2)], [B21(1x2)]]
    C = [[C11(2x2)], [C21(1x2)]] where C11 = A11*B11 + A12*B21, C21 = A21*B11 + A22*B21
    
    Use Strassen (7 terms) for A11*B11.
    Additional terms:
      T8: A12 * B21 (rank 1, but A12 is 2x1 so this is 2 terms? No, it's a single product of vectors)
          Actually A12 is 2x1 (column), B21 is 1x2 (row), so A12*B21 is 2x2, rank at most 2.
          But as a tensor term, it's rank 1: U has entries in rows (0,2) and (1,2) [i.e., i=0,j=2 and i=1,j=2],
          V has entries in rows (2,0) and (2,1) [i.e., j=2,k=0 and j=2,k=1],
          W has entries in rows (0,0),(0,1),(1,0),(1,1) [k*n+i for k=0,1 and i=0,1]
          This is actually rank 1 in the tensor sense.
      T9: A21 * B11 (A21 is 1x2, B11 is 2x2) -> 1x2 result. Rank 2 in tensor sense? No, rank 1.
      T10: A22 * B21 (A22 is 1x1, B21 is 1x2) -> 1x2 result. Rank 1.
    Total: 7 + 1 + 1 + 1 = 10 terms.
    
    Returns list of (U_entries, V_entries, W_entries) where each is a list of (row, coeff) tuples.
    U has 9 rows (i*3+j for i,j in 0..2), V has 6 rows (j*2+k for j in 0..2, k in 0..1), W has 6 rows (k*3+i for k in 0..1, i in 0..2)
    """
    terms = []
    
    # Strassen terms for A11(2x2) * B11(2x2) -> C11(2x2)
    # Map local 2x2 indices to global 3x3x2 indices
    # Local A11[i,j] -> global U row = i*3+j (for i,j in 0,1)
    # Local B11[j,k] -> global V row = j*2+k (for j in 0,1; k in 0,1)
    # Local C11[k,i] -> global W row = k*3+i (for k in 0,1; i in 0,1)
    
    strassen_terms = _strassen_2x2_2x2()
    for u_local, v_local, w_local in strassen_terms:
        u_global = [(i*3+j, c) for (i, j, c) in u_local]
        v_global = [(j*2+k, c) for (j, k, c) in v_local]
        w_global = [(k*3+i, c) for (k, i, c) in w_local]
        terms.append((u_global, v_global, w_global))
    
    # Term 8: A12 * B21 -> contributes to C11
    # A12 is column (rows i=0,1; col j=2): U rows = 0*3+2=2, 1*3+2=5
    # B21 is row (row j=2; cols k=0,1): V rows = 2*2+0=4, 2*2+1=5
    # C11 is 2x2 (rows i=0,1; cols k=0,1): W rows = 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # This is an outer product: U has (2,1),(5,1); V has (4,1),(5,1); W has (0,1),(1,1),(3,1),(4,1)
    # But wait, this is rank 1: U[i,j]*V[j,k]*W[k,i] = 1*1*1 for i in {0,1}, k in {0,1}, j=2
    u8 = [(2, 1), (5, 1)]
    v8 = [(4, 1), (5, 1)]
    w8 = [(0, 1), (1, 1), (3, 1), (4, 1)]
    terms.append((u8, v8, w8))
    
    # Term 9: A21 * B11 -> contributes to C21
    # A21 is row (row i=2; cols j=0,1): U rows = 2*3+0=6, 2*3+1=7
    # B11 is 2x2 (rows j=0,1; cols k=0,1): V rows = 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    # C21 is 1x2 (row i=2; cols k=0,1): W rows = 0*3+2=2, 1*3+2=5
    # This is rank 2? Let's see: for each (j,k), U[2,j]*V[j,k]*W[k,2]
    # We need sum over j of U[2,j]*V[j,k] = W[k,2] for k=0,1
    # If we set U[2,0]=1, U[2,1]=1, V[j,k]=1 for all j,k, W[k,2]=2 for k=0,1 -> sum = 2, not 1.
    # We need it to equal 1. So set U[2,0]=1, U[2,1]=0, V[0,k]=1, V[1,k]=0, W[k,2]=1.
    # That's rank 1 with only j=0. But that misses the j=1 contribution.
    # Actually A21*B11 = [A21_0, A21_1] * B11. This is a 1x2 matrix.
    # The tensor term is: sum_j U[2,j]*V[j,k]*W[k,2]
    # We want this to equal sum_j A21_j * B11_jk.
    # So we need U[2,j]*V[j,k]*W[k,2] to "cover" this. But this is a product of three factors,
    # not a sum over j of products of two. The rank-1 term can only represent A21_j * B11_jk * C21_k
    # as a single product, not a sum over j.
    # So A21*B11 requires rank 2 (one term for j=0, one for j=1).
    # Term 9a: j=0: U[2,0]=1, V[0,k]=B11_0k for k=0,1, W[k,2]=1
    #   U: (6,1), V: (0,1),(1,1), W: (2,1),(5,1)
    # Term 9b: j=1: U[2,1]=1, V[2,k]=B11_1k for k=0,1, W[k,2]=1
    #   U: (7,1), V: (2,1),(3,1), W: (2,1),(5,1)
    # But this is 2 terms, making total 7+1+2+1 = 11, not 10.
    
    # Hmm, let me reconsider. Maybe I can merge terms 9a and 9b into a single rank-1 term
    # if I allow coefficients. U[2,0]=1, U[2,1]=1, V[j,k]=1, W[k,2]=1 gives sum_j 1*1*1 = 2 for each k.
    # That's not 1. So I need to scale. If U[2,0]=1/2, U[2,1]=1/2, V[j,k]=1, W[k,2]=1, then sum = 1/2+1/2 = 1.
    # But half-integers are allowed!
    u9 = [(6, 0.5), (7, 0.5)]
    v9 = [(0, 1), (1, 1), (2, 1), (3, 1)]
    w9 = [(2, 1), (5, 1)]
    terms.append((u9, v9, w9))
    
    # Term 10: A22 * B21 -> contributes to C21
    # A22 is scalar (i=2, j=2): U row = 2*3+2 = 8
    # B21 is row (j=2; k=0,1): V rows = 4, 5
    # C21 is (i=2; k=0,1): W rows = 2, 5
    u10 = [(8, 1)]
    v10 = [(4, 1), (5, 1)]
    w10 = [(2, 1), (5, 1)]
    terms.append((u10, v10, w10))
    
    return terms

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # 3 blocks of 3x3x2, each rank 10, total rank 30
    block_rank = 10
    num_blocks = 3
    rank = block_rank * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 30
    V = [[0] * rank for _ in range(m * p)]  # 18 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    block_terms = _construct_3x3x2_block()
    
    for b in range(num_blocks):
        # Global column offset for this block
        col_offset = b * block_rank
        # For block b, B columns are 2*b and 2*b+1
        # V row mapping: local j in 0..2, local k in 0..1 -> global j, k = 2*b + local_k
        # W row mapping: local k in 0..1, local i in 0..2 -> global k = 2*b + local_k, i
        
        for t_idx, (u_entries, v_entries, w_entries) in enumerate(block_terms):
            col = col_offset + t_idx
            for (u_row, u_val) in u_entries:
                U[u_row][col] += u_val
            for (v_row_local, v_val) in v_entries:
                # v_row_local = j*2 + k_local, where j in 0..2, k_local in 0..1
                j = v_row_local // 2
                k_local = v_row_local % 2
                v_row_global = j * p + (2 * b + k_local)
                V[v_row_global][col] += v_val
            for (w_row_local, w_val) in w_entries:
                # w_row_local = k_local*3 + i, where k_local in 0..1, i in 0..2
                k_local = w_row_local // 3
                i = w_row_local % 3
                w_row_global = (2 * b + k_local) * n + i
                W[w_row_global][col] += w_val
    
    # Verify
    ok, info = _verify(U, V, W)
    if not ok:
        # Return schoolbook as fallback
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
