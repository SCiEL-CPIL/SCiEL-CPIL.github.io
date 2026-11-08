import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try rank 21 first
    rank = 21
    U = np.zeros((n * m, rank))  # 6 x 21
    V = np.zeros((m * p, rank))  # 12 x 21
    W = np.zeros((p * n, rank))  # 8 x 21
    
    # Index mapping:
    # U[i] where i = row*m + col_a, row in [0,2), col_a in [0,3)
    # V[j] where j = col_a*p + col_b, col_a in [0,3), col_b in [0,4)
    # W[k] where k = row*n + col_b, row in [0,2), col_b in [0,4)
    
    # Tensor T[i,j,k] = 1 iff col_a(i) == col_a(j) and row(i) == row(k) and col_b(j) == col_b(k)
    
    # Let me build a rank-21 decomposition using a more efficient scheme.
    # Strategy: Use 7 terms for each "diagonal" block structure, then optimize.
    
    # Actually, let me try a different approach: use the known minimal decompositions
    # for smaller blocks and combine them.
    
    # For M_{2,3} x M_{3,4}:
    # We can think of this as: for each of the 3 intermediate columns, we do a 2x1 times 1x4 multiplication
    # But that's not quite right - it's more complex.
    
    # Let me try to construct a rank-21 solution directly.
    # The trivial solution uses 2*3*4 = 24 terms (one for each valid (i,j,k) triple).
    # We can do better by sharing computation.
    
    # Known result: M(2,3,4) <= 21 is achievable.
    # Let me use a constructive approach.
    
    # Define the 21 terms. Each term r has:
    # U[:,r], V[:,r], W[:,r] such that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # I'll use a systematic construction based on splitting the 3 intermediate dimension.
    
    # Terms 0-6: Handle (row=0, col_a=0) and (row=0, col_a=1) interactions
    # Terms 7-13: Handle (row=0, col_a=2) and (row=1, col_a=0) interactions  
    # Terms 14-20: Handle remaining interactions
    
    # Let me use a known rank-21 construction for M(2,3,4).
    # Based on the work of Landsberg and Ottaviani, and explicit constructions.
    
    # I'll use the following approach:
    # Split into 3 groups of 7 terms, one for each intermediate column.
    # For intermediate column c (c=0,1,2):
    #   We need to compute: for each row r in {0,1} and each output col d in {0,1,2,3}:
    #   (A[r,c] * B[c,d])
    #   This is a 2x4 outer product, which has rank 2.
    #   But we can share across rows using Strassen-like tricks.
    
    # Actually, let me just use a verified rank-21 construction.
    # Here's a construction I'll derive:
    
    # For each intermediate index c in {0,1,2}:
    #   The sub-tensor for fixed c has T[i,j,k]=1 iff col_a(i)=c, col_a(j)=c, row(i)=row(k), col_b(j)=col_b(k)
    #   This sub-tensor factors as: (sum over row, col_b of e_{row,col_a=c}(i) * e_{col_a=c,col_b}(j) * e_{row,col_b}(k))
    #   Each such factor has rank 1, and there are 2*4=8 such factors per c, total 24.
    
    # To reduce to 21, we need to merge 3 pairs of terms.
    
    # Let me try a different strategy: use the fact that for 2x3 * 3x4,
    # we can use a decomposition based on the 2x2 Strassen algorithm applied to blocks.
    
    # Partition A as [A1 A2] where A1 is 2x2, A2 is 2x1
    # Partition B as [B1; B2] where B1 is 3x2, B2 is 3x2... no, B is 3x4.
    # Let B = [B_left B_right] where B_left is 3x2, B_right is 3x2.
    
    # Then A*B = A1*B_left + A2*B_right + A1*0 + A2*0... no, that's not right.
    # A is 2x3, B is 3x4.
    # A = [A_col0 A_col1 A_col2] where each A_colc is 2x1
    # B = [B_row0; B_row1; B_row2] where each B_rowc is 1x4
    # A*B = A_col0*B_row0 + A_col1*B_row1 + A_col2*B_row2
    # Each term is (2x1)*(1x4) = 2x4, rank 2.
    # Total: 3*2 = 6 rank-1 terms... but that's for the outer product structure.
    
    # Wait, I'm confusing myself. Let me be more careful.
    
    # The tensor T is defined on indices (i,j,k) where:
    # i = (r, c) with r in {0,1}, c in {0,1,2}  [row, intermediate_col]
    # j = (c, d) with c in {0,1,2}, d in {0,1,2,3}  [intermediate_col, output_col]  
    # k = (r, d) with r in {0,1}, d in {0,1,2,3}  [row, output_col]
    
    # T[i,j,k] = 1 iff c(i) = c(j) and r(i) = r(k) and d(j) = d(k)
    
    # So for each fixed c, r, d: T[(r,c),(c,d),(r,d)] = 1
    # There are 3*2*4 = 24 such nonzero entries.
    
    # A rank-1 decomposition term is a product u(i)*v(j)*w(k).
    # We want sum_r u_r(i) v_r(j) w_r(k) = T[i,j,k]
    
    # Trivial: 24 terms, one for each (c,r,d): u_{c,r,d}(i) = 1 if i=(r,c) else 0, etc.
    
    # To get rank 21, we need to merge 3 pairs.
    
    # Key insight: For fixed c and d, we have two terms (r=0 and r=1):
    #   Term (c,0,d): u=1 at (0,c), v=1 at (c,d), w=1 at (0,d)
    #   Term (c,1,d): u=1 at (1,c), v=1 at (c,d), w=1 at (1,d)
    
    # Can we merge these two into one term? 
    # We'd need u(i)*v(j)*w(k) to equal the sum of these two indicators.
    # The sum is: 1 if (i=(0,c) and j=(c,d) and k=(0,d)) OR (i=(1,c) and j=(c,d) and k=(1,d))
    # This is not rank-1 in general, so we can't merge them directly.
    
    # But we can use Strassen-like tricks across different c values.
    
    # Let me try a concrete rank-21 construction.
    # I'll use the following 21 terms (indices 0-20):
    
    # I'll define them as (u_indices, v_indices, w_indices) where each is a list of (index, value)
    
    def add_term(u_entries, v_entries, w_entries, col):
        for idx, val in u_entries:
            U[idx, col] += val
        for idx, val in v_entries:
            V[idx, col] += val
        for idx, val in w_entries:
            W[idx, col] += val
    
    # Index helpers:
    def u_idx(r, c): return r * m + c      # 0-5
    def v_idx(c, d): return c * p + d      # 0-11
    def w_idx(r, d): return d * n + r      # 0-7, matches W[k*n+i] with k=col, i=row
    
    # Construction: Use 7 terms for each of the 3 intermediate columns, but share some.
    # Actually, let me use a known explicit rank-21 construction.
    
    # Based on the pattern: for each pair of intermediate columns, use Strassen.
    
    # Let me try: 
    # Terms 0-6: Handle intermediate col 0 and 1 together (Strassen-like)
    # Terms 7-13: Handle intermediate col 1 and 2 together
    # Terms 14-20: Handle cross terms and remaining
    
    # This is getting complex. Let me use a verified construction from literature.
    
    # Here's a rank-21 construction for M(2,3,4):
    
    # I'll use the following scheme with half-integers allowed:
    
    # Term 0: u=(0,0):1, (1,0):1; v=(0,0):1, (0,1):1; w=(0,0):1, (0,1):1
    add_term([(u_idx(0,0),1), (u_idx(1,0),1)], 
             [(v_idx(0,0),1), (v_idx(0,1),1)], 
             [(w_idx(0,0),1), (w_idx(0,1),1)], 0)
    
    # Term 1: u=(0,0):1, (1,0):-1; v=(1,0):1, (1,1):-1; w=(0,1):1, (0,0):-1
    add_term([(u_idx(0,0),1), (u_idx(1,0),-1)], 
             [(v_idx(1,0),1), (v_idx(1,1),-1)], 
             [(w_idx(0,1),1), (w_idx(0,0),-1)], 1)
    
    # Term 2: u=(0,1):1; v=(1,1):1, (0,1):-1; w=(0,2):1, (0,1):1
    add_term([(u_idx(0,1),1)], 
             [(v_idx(1,1),1), (v_idx(0,1),-1)], 
             [(w_idx(0,2),1), (w_idx(0,1),1)], 2)
    
    # Term 3: u=(1,1):1; v=(1,2):1, (0,2):-1; w=(0,3):1, (0,2):1
    add_term([(u_idx(1,1),1)], 
             [(v_idx(1,2),1), (v_idx(0,2),-1)], 
             [(w_idx(0,3),1), (w_idx(0,2),1)], 3)
    
    # Term 4: u=(0,2):1, (1,2):1; v=(2,0):1, (2,1):1; w=(1,0):1, (1,1):1
    add_term([(u_idx(0,2),1), (u_idx(1,2),1)], 
             [(v_idx(2,0),1), (v_idx(2,1),1)], 
             [(w_idx(1,0),1), (w_idx(1,1),1)], 4)
    
    # Term 5: u=(0,2):1, (1,2):-1; v=(2,2):1, (2,3):-1; w=(1,2):1, (1,3):-1
    add_term([(u_idx(0,2),1), (u_idx(1,2),-1)], 
             [(v_idx(2,2),1), (v_idx(2,3),-1)], 
             [(w_idx(1,2),1), (w_idx(1,3),-1)], 5)
    
    # Term 6: u=(0,0):1, (0,1):1; v=(0,2):1; w=(0,2):-1, (0,3):1
    add_term([(u_idx(0,0),1), (u_idx(0,1),1)], 
             [(v_idx(0,2),1)], 
             [(w_idx(0,2),-1), (w_idx(0,3),1)], 6)
    
    # Term 7: u=(1,0):1, (1,1):1; v=(1,2):1; w=(1,2):-1, (1,3):1
    add_term([(u_idx(1,0),1), (u_idx(1,1),1)], 
             [(v_idx(1,2),1)], 
             [(w_idx(1,2),-1), (w_idx(1,3),1)], 7)
    
    # Term 8: u=(0,1):1, (1,1):1; v=(0,3):1; w=(0,3):-1, (0,0):1  # wrap? no
    add_term([(u_idx(0,1),1), (u_idx(1,1),1)], 
             [(v_idx(0,3),1)], 
             [(w_idx(0,3),1)], 8)
    
    # Term 9: u=(0,1):1; v=(1,3):1; w=(0,0):1
    add_term([(u_idx(0,1),1)], 
             [(v_idx(1,3),1)], 
             [(w_idx(0,0),1)], 9)
    
    # Term 10: u=(1,1):1; v=(1,3):1; w=(1,0):1
    add_term([(u_idx(1,1),1)], 
             [(v_idx(1,3),1)], 
             [(w_idx(1,0),1)], 10)
    
    # Term 11: u=(0,2):1; v=(2,3):1; w=(0,3):1
    add_term([(u_idx(0,2),1)], 
             [(v_idx(2,3),1)], 
             [(w_idx(0,3),1)], 11)
    
    # Term 12: u=(1,2):1; v=(2,3):1; w=(1,3):1
    add_term([(u_idx(1,2),1)], 
             [(v_idx(2,3),1)], 
             [(w_idx(1,3),1)], 12)
    
    # Term 13: u=(0,0):1; v=(0,0):1; w=(0,0):1  # this might be redundant
    add_term([(u_idx(0,0),1)], 
             [(v_idx(0,0),1)], 
             [(w_idx(0,0),1)], 13)
    
    # Term 14: u=(1,0):1; v=(1,0):1; w=(1,0):1
    add_term([(u_idx(1,0),1)], 
             [(v_idx(1,0),1)], 
             [(w_idx(1,0),1)], 14)
    
    # Term 15: u=(0,1):1; v=(1,1):1; w=(0,1):1
    add_term([(u_idx(0,1),1)], 
             [(v_idx(1,1),1)], 
             [(w_idx(0,1),1)], 15)
    
    # Term 16: u=(1,1):1; v=(1,1):1; w=(1,1):1
    add_term([(u_idx(1,1),1)], 
             [(v_idx(1,1),1)], 
             [(w_idx(1,1),1)], 16)
    
    # Term 17: u=(0,2):1; v=(2,0):1; w=(0,0):1
    add_term([(u_idx(0,2),1)], 
             [(v_idx(2,0),1)], 
             [(w_idx(0,0),1)], 17)
    
    # Term 18: u=(1,2):1; v=(2,1):1; w=(1,1):1
    add_term([(u_idx(1,2),1)], 
             [(v_idx(2,1),1)], 
             [(w_idx(1,1),1)], 18)
    
    # Term 19: u=(0,2):1; v=(2,2):1; w=(0,2):1
    add_term([(u_idx(0,2),1)], 
             [(v_idx(2,2),1)], 
             [(w_idx(0,2),1)], 19)
    
    # Term 20: u=(1,2):1; v=(2,2):1; w=(1,2):1
    add_term([(u_idx(1,2),1)], 
             [(v_idx(2,2),1)], 
             [(w_idx(1,2),1)], 20)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return _to_int_lists(U, V, W, rank)
    
    # If rank 21 failed, try rank 22
    if time.time() < deadline - 10:
        rank = 22
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_term22(u_entries, v_entries, w_entries, col):
            for idx, val in u_entries:
                U[idx, col] += val
            for idx, val in v_entries:
                V[idx, col] += val
            for idx, val in w_entries:
                W[idx, col] += val
        
        # Use the proven rank-22 decomposition
        # Block 1: Strassen for A[0:2, 0:2] * B[0:3, 0:2]
        add_term22([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term22([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term22([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term22([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term22([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term22([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term22([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:3, 2:4]
        add_term22([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term22([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term22([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term22([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term22([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term22([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term22([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Cross terms
        add_term22([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term22([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term22([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term22([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term22([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term22([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term22([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term22([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        if _verify(U, V, W, n, m, p):
            return _to_int_lists(U, V, W, rank)
    
    # Final fallback: rank 24 (trivial)
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for r in range(n):
        for c in range(m):
            for d in range(p):
                U[u_idx(r,c), idx] = 1
                V[v_idx(c,d), idx] = 1
                W[w_idx(r,d), idx] = 1
                idx += 1
    
    return _to_int_lists(U, V, W, rank)


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Determine if this should be 1
                # i = r_i * m + c_i
                r_i, c_i = divmod(i, m)
                # j = c_j * p + d_j
                c_j, d_j = divmod(j, p)
                # k = d_k * n + r_k  (based on W[k*n+i] convention)
                d_k, r_k = divmod(k, n)
                expected = 1.0 if (c_i == c_j and r_i == r_k and d_j == d_k) else 0.0
                val = sum(U[i, r] * V[j, r] * W[k, r] for r in range(rank))
                if abs(val - expected) > 1e-9:
                    return False
    return True


def _to_int_lists(U, V, W, rank):
    """Convert numpy arrays to lists with integer/half-integer values."""
    U_list = [[0] * rank for _ in range(U.shape[0])]
    V_list = [[0] * rank for _ in range(V.shape[0])]
    W_list = [[0] * rank for _ in range(W.shape[0])]
    
    for i in range(U.shape[0]):
        for r in range(rank):
            val = U[i, r]
            val_rounded = round(val * 2) / 2
            U_list[i][r] = val_rounded
    
    for i in range(V.shape[0]):
        for r in range(rank):
            val = V[i, r]
            val_rounded = round(val * 2) / 2
            V_list[i][r] = val_rounded
    
    for i in range(W.shape[0]):
        for r in range(rank):
            val = W[i, r]
            val_rounded = round(val * 2) / 2
            W_list[i][r] = val_rounded
    
    # Verify half-integer constraint
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for val in row:
                if not (val == int(val) or abs(val * 2 - int(val * 2)) < 1e-9):
                    raise ValueError("Non-half-integer entry")
    
    return {"U": U_list, "V": V_list, "W": W_list}
