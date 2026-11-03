import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of lists of Fractions.
    U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    n = 2
    rank = 7
    U = [[Fraction(0) for _ in range(rank)] for _ in range(4)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(4)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U (A factors)
    # A = [[a, b], [c, d]]
    # U[0, c] = coeff of a
    # U[1, c] = coeff of b
    # U[2, c] = coeff of c
    # U[3, c] = coeff of d
    
    # M1: a+d, e+h
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: c+d, e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a, f-h
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: d, g-e
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: a+b, h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: c-a, e+f
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: b-d, g+h
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Base 2x2x2 Strassen
    U_base, V_base, W_base = _strassen_2x2()
    
    # We want to multiply (4x4) @ (4x8) -> (4x8)
    # This can be seen as two independent 4x4 @ 4x4 multiplications.
    # Or rather, one 4x4 matrix A and one 4x8 matrix B.
    # We can split B into B1 (4x4) and B2 (4x4).
    # C = A @ B1 + A @ B2.
    
    # First, let's build a 4x4x4 Strassen decomposition using the 2x2x2 base.
    # A 4x4 matrix is 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = [[C00, C01], [C10, C11]] where each is 2x2.
    
    # Strassen on blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product of sums/differences of 2x2 blocks.
    # Each block product requires rank 7 (Strassen 2x2x2).
    # So total rank for 4x4x4 is 7 * 7 = 49.
    
    # Now for 4x4x8:
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C1 = A @ B1 (4x4)
    # C2 = A @ B2 (4x4)
    # Total rank = 49 + 49 = 98.
    
    total_rank = 98
    
    # Initialize U, V, W
    # U: (n*m) x rank = 16 x 98
    # V: (m*p) x rank = 32 x 98
    # W: (p*n) x rank = 32 x 98
    
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(16)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(32)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(32)]
    
    # Helper to add a 2x2x2 decomposition into the main tensors
    # offset_u: row offset in U for A's 2x2 block
    # offset_v: row offset in V for B's 2x2 block
    # offset_w: row offset in W for C's 2x2 block
    # base_rank_offset: column offset for the 7 rank terms
    
    def add_2x2_block(offset_u, offset_v, offset_w, base_rank_offset, sign_u, sign_v, sign_w):
        for r in range(7):
            col = base_rank_offset + r
            # U factors
            for i in range(2):
                for j in range(2):
                    val = U_base[i*2+j][r]
                    if val != 0:
                        row_u = offset_u + i*2 + j
                        U[row_u][col] += sign_u * val
                        
            # V factors
            for j in range(2):
                for k in range(2):
                    val = V_base[j*2+k][r]
                    if val != 0:
                        row_v = offset_v + j*2 + k
                        V[row_v][col] += sign_v * val
                        
            # W factors
            for k in range(2):
                for i in range(2):
                    val = W_base[k*2+i][r]
                    if val != 0:
                        row_w = offset_w + k*2 + i
                        W[row_w][col] += sign_w * val
    
    # A is 4x4. Rows 0-3, Cols 0-3.
    # A blocks:
    # A00: rows 0-1, cols 0-1 -> offset_u = 0
    # A01: rows 0-1, cols 2-3 -> offset_u = 0 (but different cols in A, wait)
    # U index is i*m + j. m=4.
    # A[i, j].
    # A00 is A[0:2, 0:2]. Indices in U: i*4 + j for i in 0,1; j in 0,1.
    # So rows 0,1,4,5.
    # A01 is A[0:2, 2:4]. Indices in U: i*4 + j for i in 0,1; j in 2,3.
    # So rows 2,3,6,7.
    
    # This "block" approach in U is tricky because U is flattened A.
    # Let's think in terms of A entries.
    # A is 4x4.
    # B is 4x8.
    # C is 4x8.
    
    # Let's define the 7 products for the 4x4x4 part 1 (B1 = B[:, 0:4])
    # and 7 products for the 4x4x4 part 2 (B2 = B[:, 4:8]).
    
    # Part 1: B1 (4x4)
    # B1 blocks:
    # B1_00: B[0:2, 0:2]
    # B1_01: B[0:2, 2:4]
    # B1_10: B[2:4, 0:2]
    # B1_11: B[2:4, 2:4]
    
    # A blocks:
    # A_00: A[0:2, 0:2]
    # A_01: A[0:2, 2:4]
    # A_10: A[2:4, 0:2]
    # A_11: A[2:4, 2:4]
    
    # C1 blocks:
    # C1_00: C[0:2, 0:2]
    # C1_01: C[0:2, 2:4]
    # C1_10: C[2:4, 0:2]
    # C1_11: C[2:4, 2:4]
    
    # We need to map the 2x2 block indices to the flattened U, V, W indices.
    
    # U[i, j] is at U[i*4+j]
    # V[j, k] is at V[j*8+k]
    # W[k, i] is at W[k*4+i]
    
    # Let's create a helper to get the U indices for a 2x2 block of A at (r, c)
    # where r is block row (0 or 1) and c is block col (0 or 1).
    # Block (r, c) corresponds to A rows 2r to 2r+1, cols 2c to 2c+1.
    # U index for A[i, j] is i*4 + j.
    
    def get_u_indices(block_r, block_c):
        indices = []
        for i in range(2):
            for j in range(2):
                idx_i = block_r * 2 + i
                idx_j = block_c * 2 + j
                indices.append(idx_i * 4 + idx_j)
        return indices
        
    def get_v_indices(block_r, block_c, part):
        # part 0: cols 0-3, part 1: cols 4-7
        # B is 4x8.
        # Block (r, c) in 4x4 part corresponds to B rows 2r to 2r+1, cols 2c to 2c+1 + part*4.
        indices = []
        for j in range(2):
            for k in range(2):
                idx_j = block_r * 2 + j
                idx_k = block_c * 2 + k + part * 4
                indices.append(idx_j * 8 + idx_k)
        return indices
        
    def get_w_indices(block_r, block_c, part):
        # C is 4x8.
        # Block (r, c) in 4x4 part corresponds to C rows 2r to 2r+1, cols 2c to 2c+1 + part*4.
        # W index is k*4 + i.
        indices = []
        for k in range(2):
            for i in range(2):
                idx_k = block_r * 2 + k # Wait, C block row is i?
                # C[i, j]. i is row, j is col.
                # Block (r, c) -> C rows 2r..2r+1, cols 2c..2c+1.
                # So i is in 2r..2r+1, j is in 2c..2c+1 + part*4.
                # W index is j*4 + i.
                idx_i = block_r * 2 + i
                idx_j = block_c * 2 + k + part * 4
                indices.append(idx_j * 4 + idx_i)
        return indices

    # Strassen coefficients for 4x4 blocks
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each part (0, 1):
    for part in range(2):
        base_rank = part * 49
        
        for m_idx in range(7):
            rank_offset = base_rank + m_idx * 7
            
            # Determine A blocks and signs
            # m_idx 0: M1 -> A00(1), A11(1)
            # m_idx 1: M2 -> A10(1), A11(1)
            # m_idx 2: M3 -> A00(1)
            # m_idx 3: M4 -> A11(1)
            # m_idx 4: M5 -> A00(1), A01(1)
            # m_idx 5: M6 -> A10(1), A00(-1)
            # m_idx 6: M7 -> A01(1), A11(-1)
            
            if m_idx == 0:
                a_blocks = [(0, 0, 1), (1, 1, 1)]
                b_blocks = [(0, 0, 1), (1, 1, 1)]
            elif m_idx == 1:
                a_blocks = [(1, 0, 1), (1, 1, 1)]
                b_blocks = [(0, 0, 1)]
            elif m_idx == 2:
                a_blocks = [(0, 0, 1)]
                b_blocks = [(1, 0, 1), (1, 1, -1)]
            elif m_idx == 3:
                a_blocks = [(1, 1, 1)]
                b_blocks = [(0, 1, 1), (0, 0, -1)]
            elif m_idx == 4:
                a_blocks = [(0, 0, 1), (0, 1, 1)]
                b_blocks = [(1, 1, 1)]
            elif m_idx == 5:
                a_blocks = [(1, 0, 1), (0, 0, -1)]
                b_blocks = [(0, 0, 1), (0, 1, 1)]
            elif m_idx == 6:
                a_blocks = [(0, 1, 1), (1, 1, -1)]
                b_blocks = [(1, 0, 1), (1, 1, 1)]
            
            # For each A block, add U contributions
            for ar, ac, sign in a_blocks:
                u_indices = get_u_indices(ar, ac)
                # The 2x2 block corresponds to 4 entries in A.
                # We need to map the 2x2 local indices to the 4 U indices.
                # Local (li, lj) -> Global (2*ar+li, 2*ac+lj) -> U[ (2*ar+li)*4 + (2*ac+lj) ]
                # U_base[li*2+lj, col]
                for li in range(2):
                    for lj in range(2):
                        u_global_idx = u_indices[li*2+lj]
                        u_base_val = U_base[li*2+lj][m_idx] # Wait, U_base col is r (0..6)
                        # In add_2x2_block, we used col = base_rank_offset + r.
                        # Here m_idx is the Strassen term index (0..6).
                        # So U_base[li*2+lj][m_idx] is correct.
                        if u_base_val != 0:
                            U[u_global_idx][rank_offset + m_idx] += sign * u_base_val
            
            # For each B block, add V contributions
            for br, bc, sign in b_blocks:
                v_indices = get_v_indices(br, bc, part)
                for lj in range(2): # Local row in B block
                    for lk in range(2): # Local col in B block
                        v_global_idx = v_indices[lj*2+lk]
                        v_base_val = V_base[lj*2+lk][m_idx]
                        if v_base_val != 0:
                            V[v_global_idx][rank_offset + m_idx] += sign * v_base_val
            
        # Now W contributions
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Map C block to M terms
        # C block (0,0) gets M1(+1), M4(+1), M5(-1), M7(+1)
        # C block (0,1) gets M3(+1), M5(+1)
        # C block (1,0) gets M2(+1), M4(+1)
        # C block (1,1) gets M1(+1), M2(-1), M3(+1), M6(+1)
        
        w_block_coeffs = {
            (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
            (0, 1): [(2, 1), (4, 1)],
            (1, 0): [(1, 1), (3, 1)],
            (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
        }
        
        for cr, cc in w_block_coeffs:
            coeffs = w_block_coeffs[(cr, cc)]
            w_indices = get_w_indices(cr, cc, part)
            for m_idx, sign in coeffs:
                for lk in range(2): # Local col in C block (k)
                    for li in range(2): # Local row in C block (i)
                        w_global_idx = w_indices[lk*2+li]
                        w_base_val = W_base[lk*2+li][m_idx]
                        if w_base_val != 0:
                            W[w_global_idx][base_rank + m_idx * 7 + m_idx] += sign * w_base_val
                            # Wait, rank offset logic:
                            # base_rank = part * 49
                            # rank_offset = base_rank + m_idx * 7
                            # But U and V used rank_offset + m_idx.
                            # This is wrong.
                            # U and V should use the SAME column index for a given (part, m_idx, r).
                            # Let's fix the column indexing.
    
    # RESTART WITH CORRECT INDEXING
    U = [[Fraction(0) for _ in range(total_rank)] for _ in range(16)]
    V = [[Fraction(0) for _ in range(total_rank)] for _ in range(32)]
    W = [[Fraction(0) for _ in range(total_rank)] for _ in range(32)]
    
    # Column index for part p, Strassen term s, base term r
    # col = p * 49 + s * 7 + r
    
    for part in range(2):
        for s in range(7):
            # U and V contributions for this (part, s)
            # s determines which blocks of A and B are involved.
            
            # Get A blocks and signs for term s
            if s == 0:
                a_blocks = [(0, 0, 1), (1, 1, 1)]
                b_blocks = [(0, 0, 1), (1, 1, 1)]
            elif s == 1:
                a_blocks = [(1, 0, 1), (1, 1, 1)]
                b_blocks = [(0, 0, 1)]
            elif s == 2:
                a_blocks = [(0, 0, 1)]
                b_blocks = [(1, 0, 1), (1, 1, -1)]
            elif s == 3:
                a_blocks = [(1, 1, 1)]
                b_blocks = [(0, 1, 1), (0, 0, -1)]
            elif s == 4:
                a_blocks = [(0, 0, 1), (0, 1, 1)]
                b_blocks = [(1, 1, 1)]
            elif s == 5:
                a_blocks = [(1, 0, 1), (0, 0, -1)]
                b_blocks = [(0, 0, 1), (0, 1, 1)]
            elif s == 6:
                a_blocks = [(0, 1, 1), (1, 1, -1)]
                b_blocks = [(1, 0, 1), (1, 1, 1)]
            
            # U
            for ar, ac, sign_a in a_blocks:
                for li in range(2):
                    for lj in range(2):
                        u_global_idx = (2*ar+li)*4 + (2*ac+lj)
                        for r in range(7):
                            col = part * 49 + s * 7 + r
                            u_base_val = U_base[li*2+lj][r]
                            if u_base_val != 0:
                                U[u_global_idx][col] += sign_a * u_base_val
            
            # V
            for br, bc, sign_b in b_blocks:
                for lj in range(2):
                    for lk in range(2):
                        v_global_idx = (2*br+lj)*8 + (2*bc+lk + part*4)
                        for r in range(7):
                            col = part * 49 + s * 7 + r
                            v_base_val = V_base[lj*2+lk][r]
                            if v_base_val != 0:
                                V[v_global_idx][col] += sign_b * v_base_val
            
            # W
            # Determine which C blocks this term s contributes to
            # s=0 (M1) -> C00(+1), C11(+1)
            # s=1 (M2) -> C10(+1), C11(-1)
            # s=2 (M3) -> C01(+1), C11(+1)
            # s=3 (M4) -> C00(+1), C10(+1)
            # s=4 (M5) -> C00(-1), C01(+1)
            # s=5 (M6) -> C11(+1)
            # s=6 (M7) -> C00(+1)
            
            w_targets = {
                0: [(0, 0, 1), (1, 1, 1)],
                1: [(1, 0, 1), (1, 1, -1)],
                2: [(0, 1, 1), (1, 1, 1)],
                3: [(0, 0, 1), (1, 0, 1)],
                4: [(0, 0, -1), (0, 1, 1)],
                5: [(1, 1, 1)],
                6: [(0, 0, 1)]
            }
            
            for cr, cc, sign_w in w_targets[s]:
                for lk in range(2):
                    for li in range(2):
                        w_global_idx = (2*cc+lk + part*4)*4 + (2*cr+li)
                        for r in range(7):
                            col = part * 49 + s * 7 + r
                            w_base_val = W_base[lk*2+li][r]
                            if w_base_val != 0:
                                W[w_global_idx][col] += sign_w * w_base_val

    # Convert to lists of floats (or ints if possible)
    # The task says "integer/half-integer entries". Fractions are exact.
    # We can return lists of floats or ints.
    # Let's return lists of ints if all are integers, else floats.
    
    def to_native(val):
        if val.denominator == 1:
            return val.numerator
        else:
            return float(val)
            
    U_final = [[to_native(v) for v in row] for row in U]
    V_final = [[to_native(v) for v in row] for row in V]
    W_final = [[to_native(v) for v in row] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
