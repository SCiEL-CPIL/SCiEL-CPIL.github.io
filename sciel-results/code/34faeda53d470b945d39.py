"""Rank-56 decomposition for 2x4 x 4x8 matrix multiplication using Strassen."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    # A is n x m = 2 x 4
    # B is m x p = 4 x 8
    # C is n x p = 2 x 8
    
    # We split A into A1 (2x2) and A2 (2x2)
    # We split B into B1 (2x8) and B2 (2x8)
    # C = A1*B1 + A2*B2
    
    # Each product A_i * B_j is computed using a rank-28 decomposition
    # derived from Strassen's 2x2x2 decomposition applied to the 2x2 blocks of B.
    # B1 is 2x8, split into 4 blocks of 2x2: B1_1, B1_2, B1_3, B1_4
    # Similarly for B2.
    
    # Strassen multipliers for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 x 2x8:
    # A is fixed 2x2.
    # B is 2x8, split into 4 blocks of 2x2.
    # For each block q (0..3), we compute 7 multipliers.
    # Total 28 multipliers per 2x2 x 2x8 block.
    
    # Total rank = 28 * 2 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Indices:
    # A rows: i (0..1), cols: j (0..3). Index in U: i*m + j = i*4 + j
    # B rows: j (0..3), cols: k (0..7). Index in V: j*p + k = j*8 + k
    # C rows: k (0..7), cols: i (0..1). Index in W: k*n + i = k*2 + i
    
    # Let's define the Strassen linear forms for A (2x2) and B (2x2).
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Multipliers:
    # L1 = A00 + A11
    # L2 = A10 + A11
    # L3 = A00
    # L4 = A11
    # L5 = A00 + A01
    # L6 = A10 - A00
    # L7 = A01 - A11
    
    # R1 = B00 + B11
    # R2 = B00
    # R3 = B01 - B11
    # R4 = B10 - B00
    # R5 = B11
    # R6 = B00 + B01
    # R7 = B10 + B11
    
    # Output combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We have two A blocks: A1 (cols 0,1 of A) and A2 (cols 2,3 of A).
    # A1 = [[A00, A01], [A10, A11]] where A00 is A[0,0], etc.
    # A2 = [[A02, A03], [A12, A13]]
    
    # We have two B blocks: B1 (rows 0,1 of B) and B2 (rows 2,3 of B).
    # B1 is 2x8. B2 is 2x8.
    
    # Let's iterate over the two A blocks and two B blocks.
    # For each pair (a_block, b_block), we add 28 rank-1 tensors.
    
    # a_block 0: A1. Indices in A: 0,1 (row 0), 4,5 (row 1).
    # A1[0,0] = A[0,0] -> U index 0
    # A1[0,1] = A[0,1] -> U index 1
    # A1[1,0] = A[1,0] -> U index 4
    # A1[1,1] = A[1,1] -> U index 5
    
    # a_block 1: A2. Indices in A: 2,3 (row 0), 6,7 (row 1).
    # A2[0,0] = A[0,2] -> U index 2
    # A2[0,1] = A[0,3] -> U index 3
    # A2[1,0] = A[1,2] -> U index 6
    # A2[1,1] = A[1,3] -> U index 7
    
    # b_block 0: B1. Rows 0,1 of B.
    # B1 is 2x8.
    # B1[0,0] = B[0,0] -> V index 0
    # B1[0,1] = B[0,1] -> V index 1
    # ...
    # B1[0,7] = B[0,7] -> V index 7
    # B1[1,0] = B[1,0] -> V index 8
    # ...
    # B1[1,7] = B[1,7] -> V index 15
    
    # b_block 1: B2. Rows 2,3 of B.
    # B2[0,0] = B[2,0] -> V index 16
    # ...
    # B2[1,7] = B[3,7] -> V index 31
    
    # Output C is 2x8.
    # C[0,0] depends on A1*B1 and A2*B2.
    # A1*B1 gives C[0,0..1] (2x2 block).
    # A2*B2 gives C[0,2..3] (2x2 block).
    # Wait, A is 2x4, B is 4x8.
    # C[0,k] = sum_j A[0,j] B[j,k].
    # A[0,j] for j=0,1 is A1[0,j]. A[0,j] for j=2,3 is A2[0,j-2].
    # B[j,k] for j=0,1 is B1[j,k]. B[j,k] for j=2,3 is B2[j-2,k].
    # So C[0,k] = sum_j' A1[0,j'] B1[j',k] + sum_j' A2[0,j'] B2[j',k].
    # This means the (0,0) block of C (cols 0,1) is A1*B1_0 + A2*B2_0?
    # No. B1 is 2x8. B1[:, k] is a column.
    # Let's be precise.
    # C[0, k] = A1[0,0] B[0,k] + A1[0,1] B[1,k] + A2[0,0] B[2,k] + A2[0,1] B[3,k].
    # Here B[0,k] is B1[0,k], B[1,k] is B1[1,k], B[2,k] is B2[0,k], B[3,k] is B2[1,k].
    # So C[0, :] = (A1[0,0] B1[0,:] + A1[0,1] B1[1,:]) + (A2[0,0] B2[0,:] + A2[0,1] B2[1,:]).
    # This is the first row of (A1 B1) plus the first row of (A2 B2).
    # So the full C matrix is the concatenation of A1 B1 and A2 B2?
    # No, it's the sum.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x8. A2 B2 is 2x8.
    # So C[i,k] = (A1 B1)[i,k] + (A2 B2)[i,k].
    
    # So for each k (0..7), the output C[i,k] gets contributions from both A1B1 and A2B2.
    # This means the W matrix entries will have coefficients from both sets of multipliers.
    
    # Let's define the multipliers for A1 B1.
    # These use U indices 0,1,4,5.
    # And V indices 0..15.
    # And produce contributions to C[i,k] for all i,k.
    
    # Similarly for A2 B2.
    # U indices 2,3,6,7.
    # V indices 16..31.
    # Contributions to C[i,k].
    
    # Let's build the rank-1 tensors.
    # We have 28 multipliers for A1B1 and 28 for A2B2.
    # Total 56.
    
    # For A1B1:
    # Multiplier 0: L1(A1) * R1(B1_block_q)
    # ...
    
    # Let's map the indices.
    # For a given A block (a_idx) and B block (b_idx):
    # a_idx 0: U indices [0, 1, 4, 5] corresponding to A00, A01, A10, A11.
    # a_idx 1: U indices [2, 3, 6, 7] corresponding to A00, A01, A10, A11.
    
    # b_idx 0: B1. V indices 0..15.
    # B1[0, k] -> V[k]
    # B1[1, k] -> V[8+k]
    # So B1_block_q (2x2) for q=0..3:
    # B1[0, 2q]   -> V[2q]
    # B1[0, 2q+1] -> V[2q+1]
    # B1[1, 2q]   -> V[8+2q]
    # B1[1, 2q+1] -> V[8+2q+1]
    
    # b_idx 1: B2. V indices 16..31.
    # B2[0, k] -> V[16+k]
    # B2[1, k] -> V[24+k]
    # So B2_block_q (2x2) for q=0..3:
    # B2[0, 2q]   -> V[16+2q]
    # B2[0, 2q+1] -> V[16+2q+1]
    # B2[1, 2q]   -> V[24+2q]
    # B2[1, 2q+1] -> V[24+2q+1]
    
    # Output C indices:
    # C[0, 2q]   -> W[2q*2 + 0] = W[4q]
    # C[0, 2q+1] -> W[4q+1]
    # C[1, 2q]   -> W[4q+2]
    # C[1, 2q+1] -> W[4q+3]
    
    # Let's define the linear forms.
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11 (local to the block)
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11 (local to the block)
    
    # L forms:
    # L1: A00 + A11
    # L2: A10 + A11
    # L3: A00
    # L4: A11
    # L5: A00 + A01
    # L6: A10 - A00
    # L7: A01 - A11
    
    # R forms:
    # R1: B00 + B11
    # R2: B00
    # R3: B01 - B11
    # R4: B10 - B00
    # R5: B11
    # R6: B00 + B01
    # R7: B10 + B11
    
    # C outputs:
    # C00: M1 + M4 - M5 + M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M3 + M6
    
    # We will iterate over a_idx (0,1), b_idx (0,1), q (0..3), and mult (0..6).
    # Total 2*2*4*7 = 112? No, 2*2*28 = 112?
    # Wait, rank is 56.
    # 2 A blocks * 2 B blocks * 4 q blocks * 7 multipliers = 112?
    # No. The decomposition is for 2x4 x 4x8.
    # A1 is 2x2, B1 is 2x8.
    # A1 B1 is computed with rank 28.
    # A2 B2 is computed with rank 28.
    # Total 56.
    
    # So for each (a_idx, b_idx) pair, we have 28 multipliers.
    # But wait, A1 and A2 are disjoint parts of A.
    # B1 and B2 are disjoint parts of B.
    # The products A1 B1 and A2 B2 are independent?
    # C = A1 B1 + A2 B2.
    # Yes, the tensors for A1 B1 and A2 B2 are added.
    # So we can just construct the rank-28 decomposition for A1 B1 and add it to U,V,W.
    # Then construct the rank-28 decomposition for A2 B2 and add it.
    
    # Let's write a helper to add a rank-28 decomposition for a 2x2 x 2x8 product.
    
    def add_2x2_x_2x8(u_base, v_base, q_offset, col_offset):
        # u_base: list of 4 U indices for A00, A01, A10, A11
        # v_base: function to get V index for B[r, c] where r in 0,1 and c in 0..7
        # q_offset: not needed if v_base handles it?
        # Actually, v_base should take (r, k) where k is 0..7.
        # And return the V index.
        # col_offset: starting column in U,V,W for this set of 28 multipliers.
        
        # L forms
        Ls = [
            {0: 1, 3: 1},      # L1: A00 + A11
            {2: 1, 3: 1},      # L2: A10 + A11
            {0: 1},            # L3: A00
            {3: 1},            # L4: A11
            {0: 1, 1: 1},      # L5: A00 + A01
            {2: 1, 0: -1},     # L6: A10 - A00
            {1: 1, 3: -1}      # L7: A01 - A11
        ]
        
        # R forms
        Rs = [
            {0: 1, 3: 1},      # R1: B00 + B11
            {0: 1},            # R2: B00
            {1: 1, 3: -1},     # R3: B01 - B11
            {2: 1, 0: -1},     # R4: B10 - B00
            {3: 1},            # R5: B11
            {0: 1, 1: 1},      # R6: B00 + B01
            {2: 1, 3: 1}       # R7: B10 + B11
        ]
        
        # C output coefficients
        # C00, C01, C10, C11
        # M1..M7
        Cs = [
            {0: 1, 3: 1, 4: -1, 6: 1},   # C00: M1 + M4 - M5 + M7
            {2: 1, 4: 1},                 # C01: M3 + M5
            {1: 1, 3: 1},                 # C10: M2 + M4
            {0: 1, 1: -1, 2: 1, 5: 1}    # C11: M1 - M2 + M3 + M6
        ]
        
        for q in range(4):
            # B block q is 2x2.
            # B00 = B[0, 2q]
            # B01 = B[0, 2q+1]
            # B10 = B[1, 2q]
            # B11 = B[1, 2q+1]
            
            # V indices for this block
            v00 = v_base(0, 2*q)
            v01 = v_base(0, 2*q+1)
            v10 = v_base(1, 2*q)
            v11 = v_base(1, 2*q+1)
            
            # W indices for this block
            w00 = q_offset + 4*q + 0  # C[0, 2q]
            w01 = q_offset + 4*q + 1  # C[0, 2q+1]
            w10 = q_offset + 4*q + 2  # C[1, 2q]
            w11 = q_offset + 4*q + 3  # C[1, 2q+1]
            
            w_indices = [w00, w01, w10, w11]
            
            for mult in range(7):
                col = col_offset + q * 7 + mult
                L = Ls[mult]
                R = Rs[mult]
                
                # Update U
                for a_loc, coeff in L.items():
                    U[u_base[a_loc]][col] += coeff
                
                # Update V
                # R keys: 0=B00, 1=B01, 2=B10, 3=B11
                for b_loc, coeff in R.items():
                    if b_loc == 0:
                        V[v00][col] += coeff
                    elif b_loc == 1:
                        V[v01][col] += coeff
                    elif b_loc == 2:
                        V[v10][col] += coeff
                    elif b_loc == 3:
                        V[v11][col] += coeff
                
                # Update W
                # C keys: 0=C00, 1=C01, 2=C10, 3=C11
                for c_loc, coeff in Cs[mult].items():
                    W[w_indices[c_loc]][col] += coeff
    
    # A1 B1
    # A1 indices: 0,1,4,5
    u_indices_1 = [0, 1, 4, 5]
    # B1 V indices:
    def v_idx_b1(r, k):
        return r * 8 + k
    add_2x2_x_2x8(u_indices_1, v_idx_b1, 0, 0)
    
    # A2 B2
    # A2 indices: 2,3,6,7
    u_indices_2 = [2, 3, 6, 7]
    # B2 V indices:
    def v_idx_b2(r, k):
        return (2 + r) * 8 + k
    add_2x2_x_2x8(u_indices_2, v_idx_b2, 0, 28)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
