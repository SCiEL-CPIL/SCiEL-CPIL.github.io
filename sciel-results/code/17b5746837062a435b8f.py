import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U row r corresponds to A[r//2, r%2]
    V row r corresponds to B[r//2, r%2]
    W row r corresponds to C[r//2, r%2] -> Note: W index is k*n+i.
    For 2x2, W index r corresponds to C[r//2, r%2]?
    Let's check the convention: W[k*n+i].
    If n=2, W index is k*2+i.
    r=0 -> k=0, i=0 -> C[0,0]
    r=1 -> k=0, i=1 -> C[0,1]
    r=2 -> k=1, i=0 -> C[1,0]
    r=3 -> k=1, i=1 -> C[1,1]
    So W row r corresponds to C[r//2, r%2].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (A parts)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: A11+A22 -> [1, 0, 0, 1]
    U[:, 0] = [1, 0, 0, 1]
    # M2: A21+A22 -> [0, 0, 1, 1]
    U[:, 1] = [0, 0, 1, 1]
    # M3: A11 -> [1, 0, 0, 0]
    U[:, 2] = [1, 0, 0, 0]
    # M4: A22 -> [0, 0, 0, 1]
    U[:, 3] = [0, 0, 0, 1]
    # M5: A11+A12 -> [1, 1, 0, 0]
    U[:, 4] = [1, 1, 0, 0]
    # M6: A21-A11 -> [-1, 0, 1, 0]
    U[:, 5] = [-1, 0, 1, 0]
    # M7: A12-A22 -> [0, 1, 0, -1]
    U[:, 6] = [0, 1, 0, -1]
    
    # V vectors (B parts)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: B11+B22 -> [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    V[:, 1] = [1, 0, 0, 0]
    # M3: B12-B22 -> [0, 1, 0, -1]
    V[:, 2] = [0, 1, 0, -1]
    # M4: B21-B11 -> [0, 0, 1, -1]
    V[:, 3] = [0, 0, 1, -1]
    # M5: B22 -> [0, 0, 0, 1]
    V[:, 4] = [0, 0, 0, 1]
    # M6: B11+B12 -> [1, 1, 0, 0]
    V[:, 5] = [1, 1, 0, 0]
    # M7: B21+B22 -> [0, 0, 1, 1]
    V[:, 6] = [0, 0, 1, 1]
    
    # W vectors (C parts)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C11: M1+M4-M5+M7 -> [1, 0, 0, 0] for M1, [0,0,0,0] for M2, ...
    # We need W[r, c] such that sum_c U[ai, c] V[bj, c] W[ck, c] = delta_{ai,bi} delta_{bj,ck}?
    # No, the tensor T[i,j,k] = 1.
    # The term is U[i,j,c] V[j,k,c] W[k,i,c].
    # So W[k,i,c] is the coefficient for C[i,k] (or C[k,i] depending on convention).
    # Convention: W[k*n+i, c].
    # So W row r corresponds to C[r//n, r%n] = C[k,i].
    # For 2x2, r=0->C[0,0], r=1->C[0,1], r=2->C[1,0], r=3->C[1,1].
    
    # C11 (r=0): M1(1) + M4(1) - M5(1) + M7(1)
    W[0, 0] = 1; W[0, 1] = 0; W[0, 2] = 0; W[0, 3] = 1; W[0, 4] = -1; W[0, 5] = 0; W[0, 6] = 1
    # C12 (r=1): M3(1) + M5(1)
    W[1, 0] = 0; W[1, 1] = 0; W[1, 2] = 1; W[1, 3] = 0; W[1, 4] = 1; W[1, 5] = 0; W[1, 6] = 0
    # C21 (r=2): M2(1) + M4(1)
    W[2, 0] = 0; W[2, 1] = 1; W[2, 2] = 0; W[2, 3] = 1; W[2, 4] = 0; W[2, 5] = 0; W[2, 6] = 0
    # C22 (r=3): M1(1) - M2(1) + M3(1) + M6(1)
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 3] = 0; W[3, 4] = 0; W[3, 5] = 1; W[3, 6] = 0
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m x p schoolbook decomposition.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
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

def _embed(U, V, W, i0, j0, k0, U_s, V_s, W_s, n_global, m_global, p_global, col_offset):
    """
    Embeds a local decomposition (U_s, V_s, W_s) for a block at (i0, j0, k0)
    into the global matrices.
    """
    n_local = U_s.shape[0] // m_local  # This is tricky, we need to know local dims
    # Actually, we know the local dims from the shape of U_s, V_s, W_s?
    # U_s is (n_l * m_l, rank_l)
    # V_s is (m_l * p_l, rank_l)
    # W_s is (p_l * n_l, rank_l)
    
    # We can infer n_l, m_l, p_l from the global context and offsets?
    # Or we can pass them.
    # For Strassen, n_l=2, m_l=2, p_l=2.
    # For Schoolbook, we pass n_l, m_l, p_l.
    
    # Let's just iterate over all rows in U_s, V_s, W_s.
    # We need to map local row index to global row index.
    
    # For U_s: row r corresponds to local (li, lj).
    # We need to know m_l to compute li, lj.
    # For Strassen, m_l=2.
    # For Schoolbook, we need to pass m_l.
    
    # Let's modify the signature to include local dims.
    pass

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # Total rank
    # 4 Strassen blocks (2x2x2) -> 4 * 7 = 28
    # 1 Schoolbook block (2x4x1) -> 2 * 4 * 1 = 8
    # 1 Schoolbook block (1x4x5) -> 1 * 4 * 5 = 20
    total_rank = 28 + 8 + 20
    # total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # 1. Embed 4 Strassen blocks for the 2x4x4 part (top-left 2 rows of A, all 4 cols of B, first 4 cols of C)
    U_s, V_s, W_s = _strassen_222()
    
    # Block 1: i0=0, j0=0, k0=0
    # A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Local m_l=2, n_l=2, p_l=2
    for r in range(4):
        li, lj = divmod(r, 2)
        gi, gj = i0 + li, j0 + lj
        U[gi * m + gj, col_offset : col_offset + 7] = U_s[r, :]
        
        li, lj = divmod(r, 2) # Same r for V? No, V index is (lj, lk)
        # V row r in local corresponds to (lj, lk) with lj, lk in 0..1
        # But U row r corresponds to (li, lj).
        # We need to map correctly.
        # U_s row r -> A[li, lj]
        # V_s row r -> B[lj, lk]
        # W_s row r -> C[lk, li]
        
        # So for U, r -> (li, lj)
        # For V, r -> (lj, lk)
        # For W, r -> (lk, li)
        
        # This implies the mapping is consistent if we iterate r 0..3.
        
    # Let's do the embedding properly.
    def embed_strassen(i0, j0, k0, col_off):
        nonlocal U, V, W
        for r in range(4):
            li, lj = divmod(r, 2)
            # U
            gi, gj = i0 + li, j0 + lj
            U[gi * m + gj, col_off : col_off + 7] = U_s[r, :]
            
            # V
            # V row r corresponds to B[lj, lk]
            # We need to find lk such that the local index is r?
            # In _strassen_222, V row r is defined for the term r.
            # The term r uses specific A and B components.
            # The index r in V_s corresponds to the B-component used in term r.
            # But the row index in V_s is the index of the B element.
            # So V_s[r, c] is the coefficient for B_element_r in term c.
            # So we just copy V_s[r, :] to the global V row corresponding to local B index r.
            # Local B index r corresponds to (lj, lk) = divmod(r, 2).
            gj, gk = j0 + lj, k0 + lk
            # Wait, divmod(r, 2) gives (row, col) of B.
            # So local B row is lj, local B col is lk.
            # Global B row is j0 + lj, Global B col is k0 + lk.
            V[(j0 + lj) * p + (k0 + lk), col_off : col_off + 7] = V_s[r, :]
            
            # W
            # W row r corresponds to C[lk, li].
            # Local C index r corresponds to (lk, li) = divmod(r, 2).
            # So local C row is lk, local C col is li.
            # Global C row is k0 + lk, Global C col is i0 + li.
            # W index is k*n + i.
            # Global W row index: (k0 + lk) * n + (i0 + li).
            W[(k0 + lk) * n + (i0 + li), col_off : col_off + 7] = W_s[r, :]

    # We need to be careful: the loop variable r is the local index.
    # In the function above, I used r for U, V, W.
    # For U, r -> (li, lj).
    # For V, r -> (lj, lk).
    # For W, r -> (lk, li).
    # This is correct for Strassen where the term index c is independent of the element index r.
    # But here, we are copying the whole column c for all r.
    # So for each c, we update U, V, W at the specific rows.
    
    # Let's re-implement the embedding to be safe.
    def embed_block(i0, j0, k0, U_s, V_s, W_s, n_l, m_l, p_l, col_off):
        rank_l = U_s.shape[1]
        for r in range(U_s.shape[0]):
            # U
            li, lj = divmod(r, m_l)
            gi, gj = i0 + li, j0 + lj
            U[gi * m + gj, col_off : col_off + rank_l] = U_s[r, :]
            
            # V
            # V row index in local is r.
            # r corresponds to (lj, lk) in B.
            # We need to extract lj, lk from r.
            # Since V_s has shape (m_l * p_l, rank_l), and r is the row index,
            # the local B index is r.
            # local_b_row = r // p_l
            # local_b_col = r % p_l
            # BUT, in Strassen 2x2x2, m_l=2, p_l=2.
            # V row 0 -> B[0,0]. V row 1 -> B[0,1]. V row 2 -> B[1,0]. V row 3 -> B[1,1].
            # So local_b_row = r // p_l, local_b_col = r % p_l.
            lb_r, lb_c = divmod(r, p_l)
            gj, gk = j0 + lb_r, k0 + lb_c
            V[gj * p + gk, col_off : col_off + rank_l] = V_s[r, :]
            
            # W
            # W row index in local is r.
            # r corresponds to (lk, li) in C.
            # local_c_row = r // n_l
            # local_c_col = r % n_l
            lc_r, lc_c = divmod(r, n_l)
            gk, gi = k0 + lc_r, i0 + lc_c
            W[gk * n + gi, col_off : col_off + rank_l] = W_s[r, :]

    # Embed 4 Strassen blocks
    # 1. i0=0, j0=0, k0=0
    embed_block(0, 0, 0, U_s, V_s, W_s, 2, 2, 2, col_offset)
    col_offset += 7
    
    # 2. i0=0, j0=0, k0=2
    embed_block(0, 0, 2, U_s, V_s, W_s, 2, 2, 2, col_offset)
    col_offset += 7
    
    # 3. i0=0, j0=2, k0=0
    embed_block(0, 2, 0, U_s, V_s, W_s, 2, 2, 2, col_offset)
    col_offset += 7
    
    # 4. i0=0, j0=2, k0=2
    embed_block(0, 2, 2, U_s, V_s, W_s, 2, 2, 2, col_offset)
    col_offset += 7
    
    # 5. Schoolbook block for M_{2,4,1}
    # i0=0, j0=0, k0=4
    # n_l=2, m_l=4, p_l=1
    U_sb1, V_sb1, W_sb1 = _schoolbook_block(2, 4, 1)
    embed_block(0, 0, 4, U_sb1, V_sb1, W_sb1, 2, 4, 1, col_offset)
    col_offset += 8
    
    # 6. Schoolbook block for M_{1,4,5}
    # i0=2, j0=0, k0=0
    # n_l=1, m_l=4, p_l=5
    U_sb2, V_sb2, W_sb2 = _schoolbook_block(1, 4, 5)
    embed_block(2, 0, 0, U_sb2, V_sb2, W_sb2, 1, 4, 5, col_offset)
    col_offset += 20
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
