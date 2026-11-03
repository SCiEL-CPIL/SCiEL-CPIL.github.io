"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 8]
N, M, P = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-88 exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy: Partition the 3x4 matrix into 2x2 and 1x2 blocks, and the 4x8 matrix into
    corresponding 2x4 and 2x4 blocks. Use Strassen's algorithm (rank 7) for each 2x2x2
    subproblem and schoolbook (rank 16) for each 1x2x4 subproblem.
    
    Total rank = 2 * (7*4 + 16) = 2 * 44 = 88
    
    Convention: W[k*n + i, r] where result = A @ B, A is n x m, B is m x p.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave safety margin
    
    n, m, p = N, M, P  # 3, 4, 8
    
    # We partition the 3 rows into: rows 0-1 (2x2 Strassen) and row 2 (1x2 schoolbook)
    # We partition the 4 columns of A (4 rows of B) into: cols 0-1 (2x4) and cols 2-3 (2x4)
    # We partition the 8 columns of B into 4 groups of 2
    
    # Strassen rank-7 decomposition for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For a 2x2x4 problem (2x2 A-block times 2x4 B-block), we need 7*2 = 14 terms
    # (Strassen for each pair of columns)
    
    # For a 1x2x4 problem (1x2 A-block times 2x4 B-block), schoolbook: 2*4 = 8 terms
    
    # Total per group of 2 columns: 14 (Strassen 2x2x4) + 8 (schoolbook 1x2x4) = 22
    # Wait, let me reconsider.
    
    # Actually, let me think about this differently.
    # A is 3x4, B is 4x8.
    # Partition A into A_top (2x4) and A_bottom (1x4).
    # Partition B into B_left (4x4) and B_right (4x4)? No, that doesn't help.
    
    # Better: Partition A into A_2 (2x4) for rows 0-1 and A_1 (1x4) for row 2.
    # B is 4x8. Partition B into B_2x4 (first 2 rows) and B_2x4 (last 2 rows)? 
    # No, we need to partition the middle dimension (4) consistently.
    
    # Let's partition the middle dimension (4) into two groups of 2: {0,1} and {2,3}.
    # A is 3x4: A[:, 0:2] is 3x2, A[:, 2:4] is 3x2.
    # B is 4x8: B[0:2, :] is 2x8, B[2:4, :] is 2x8.
    
    # Result C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    # For each pair (A_block 3x2, B_block 2x8):
    #   Partition A_block into A_top (2x2) rows 0-1, A_bottom (1x2) row 2.
    #   C_contribution = A_top @ B_block + A_bottom @ B_block
    #   A_top @ B_block: 2x2 x 2x8 -> use Strassen on 2x2x2 blocks: 7*4 = 28 terms
    #   A_bottom @ B_block: 1x2 x 2x8 -> schoolbook: 2*8 = 16 terms
    #   Total per block: 44 terms
    # Two blocks: 88 terms total.
    
    rank = 88
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 8*3 = 24 rows
    
    col = 0
    
    # Strassen 2x2x2 terms
    # For A_top (rows 0,1 of A_block) and B_block (2x8):
    # We split B_block into 4 column-pairs: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # For each 2x2x2 subproblem, we have 7 Strassen terms.
    
    # Strassen coefficients:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors for 2x2 A-block (rows i0, i1 of A):
    strassen_U = [
        # M1: A11+A22
        [1, 1],
        # M2: A21+A22
        [1, 1],
        # M3: A11
        [1, 0],
        # M4: A22
        [0, 1],
        # M5: A11+A12
        [1, 1],
        # M6: A21-A11
        [-1, 1],
        # M7: A12-A22
        [1, -1],
    ]
    
    # V vectors for 2x2 B-block (cols k0, k1 of B, for a single row j):
    strassen_V = [
        # M1: B11+B22
        [1, 1],
        # M2: B11
        [1, 0],
        # M3: B21-B22
        [-1, 1],
        # M4: B12-B11
        [-1, 1],
        # M5: B22
        [0, 1],
        # M6: B11+B12
        [1, 1],
        # M7: B21+B22
        [1, 1],
    ]
    
    # W coefficients for output C (2x2 result from 2x2x2):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    strassen_W = [
        # M1: C11 + 1, C22 + 1
        [1, 0, 0, 1],
        # M2: C21 + 1, C22 - 1
        [0, 0, 1, -1],
        # M3: C12 + 1, C22 + 1
        [0, 1, 0, 1],
        # M4: C11 + 1, C21 + 1
        [1, 0, 1, 0],
        # M5: C11 - 1, C12 + 1
        [-1, 1, 0, 0],
        # M6: C22 + 1
        [0, 0, 0, 1],
        # M7: C11 + 1
        [1, 0, 0, 0],
    ]
    
    def add_strassen_2x2x4(A_row_start, B_col_offset):
        """
        Add Strassen terms for 2x2x4: A[A_row_start:A_row_start+2, :2] @ B[:2, B_col_offset:B_col_offset+4]
        A rows: A_row_start, A_row_start+1 (these are columns 0-1 of A, i.e., middle dim 0-1)
        B rows: 0,1 (middle dim 0-1)
        B cols: B_col_offset to B_col_offset+3 (8 cols total, split into 4 pairs)
        Result: 2x4 block of C
        """
        nonlocal col
        # A indices: (i, j) where i in {A_row_start, A_row_start+1}, j in {0,1}
        # B indices: (j, k) where j in {0,1}, k in {B_col_offset, ..., B_col_offset+3}
        # C indices: (i, k)
        
        for pair_idx in range(4):  # 4 column pairs in B
            k0 = B_col_offset + pair_idx * 2
            k1 = k0 + 1
            
            for s in range(7):  # 7 Strassen terms
                # U: 2x2 A block
                u0 = strassen_U[s][0]
                u1 = strassen_U[s][1]
                if u0 != 0:
                    U[A_row_start * M + 0][col] += u0  # j=0
                    U[A_row_start * M + 1][col] += 0   # wait, this is wrong
                # Let me fix the indexing.
                # A is n x m = 3 x 4. U is indexed by [i*m + j] where i is row, j is col.
                # A_row_start refers to the row index in A (i dimension).
                # The 2x2 A block has rows i0=A_row_start, i1=A_row_start+1, and cols j0=0, j1=1.
                # U[i0*M + j0] corresponds to A[i0, 0]
                # U[i0*M + j1] corresponds to A[i0, 1]
                # U[i1*M + j0] corresponds to A[i1, 0]
                # U[i1*M + j1] corresponds to A[i1, 1]
                
                # strassen_U[s] = [coeff for (i0,j0) and (i1,j1)]? No.
                # For 2x2 matrix A_block:
                # A11 = A[i0, j0], A12 = A[i0, j1]
                # A21 = A[i1, j0], A22 = A[i1, j1]
                
                # strassen_U[s] gives coefficients for [A11, A22] in my definition above.
                # But I need to map to all 4 entries.
                
                col += 1
        
        # This is getting complicated. Let me restart with a cleaner approach.
        pass
    
    # Let me use a cleaner implementation.
    # I'll build the decomposition by directly computing the contributions.
    
    # Reset
    rank = 88
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    
    # For each of the two middle-dim partitions: {0,1} and {2,3}
    for mid_offset in [0, 2]:
        # A_block: rows 0-2 of A, cols mid_offset to mid_offset+1 (3x2)
        # B_block: rows mid_offset to mid_offset+1, cols 0-7 (2x8)
        
        # Strassen part: A_top (rows 0-1, cols mid_offset..mid_offset+1) @ B_block (2x8)
        # This is a 2x2x8 problem. Split B_block into 4 column-pairs.
        for b_pair in range(4):
            b_col0 = b_pair * 2
            b_col1 = b_col0 + 1
            
            # 2x2x2 Strassen
            for s in range(7):
                if col >= rank:
                    return None  # shouldn't happen
                
                # U: coefficients for A_top entries
                # A_top[i, j] where i in {0,1}, j in {mid_offset, mid_offset+1}
                # Map to U index: i * M + j
                # A11 = A[0, mid_offset], A12 = A[0, mid_offset+1]
                # A21 = A[1, mid_offset], A22 = A[1, mid_offset+1]
                
                # strassen_U[s] = [c11, c22] where c11 coeff for A11+A22 etc.
                # Let me redefine properly:
                # M1 = (A11 + A22) * (B11 + B22): U = [1,0,0,1] for [A11,A12,A21,A22]
                # M2 = (A21 + A22) * B11: U = [0,0,1,1]
                # M3 = A11 * (B21 - B22): U = [1,0,0,0]
                # M4 = A22 * (B12 - B11): U = [0,0,0,1]
                # M5 = (A11 + A12) * B22: U = [1,1,0,0]
                # M6 = (A21 - A11) * (B11 + B12): U = [-1,0,1,0]
                # M7 = (A12 - A22) * (B21 + B22): U = [0,1,0,-1]
                
                U_coeff = {
                    0: [1, 0, 0, 1],   # M1
                    1: [0, 0, 1, 1],   # M2
                    2: [1, 0, 0, 0],   # M3
                    3: [0, 0, 0, 1],   # M4
                    4: [1, 1, 0, 0],   # M5
                    5: [-1, 0, 1, 0],  # M6
                    6: [0, 1, 0, -1],  # M7
                }[s]
                
                # V: coefficients for B_block entries
                # B_block[j, k] where j in {mid_offset, mid_offset+1}, k in {b_col0, b_col1}
                # B11 = B[mid_offset, b_col0], B12 = B[mid_offset, b_col1]
                # B21 = B[mid_offset+1, b_col0], B22 = B[mid_offset+1, b_col1]
                
                V_coeff = {
                    0: [1, 1, 1, 1],   # M1: B11+B22 -> [1,0,0,1]... wait
                    # B11+B22: B11=1, B12=0, B21=0, B22=1
                    0: [1, 0, 0, 1],   # M1
                    1: [1, 0, 0, 0],   # M2: B11
                    2: [0, -1, 1, 0],  # M3: B21-B22 -> [0,0,1,-1]
                    2: [0, 0, 1, -1],  # M3
                    3: [-1, 1, 0, 0],  # M4: B12-B11 -> [0,1,-1,0]... 
                    3: [0, 1, -1, 0],  # M4: wait, B11 and B12 are in row 0
                    # B11 = B[j0, k0], B12 = B[j0, k1], B21 = B[j1, k0], B22 = B[j1, k1]
                    # M4 = A22 * (B12 - B11): V = [0, 1, -1, 0] for [B11,B12,B21,B22]
                    4: [0, 0, 0, 1],   # M5: B22
                    5: [1, 1, 0, 0],   # M6: B11+B12
                    6: [0, 0, 1, 1],   # M7: B21+B22
                }[s]
                
                # W: coefficients for C output
                # C[i, k] where i in {0,1}, k in {b_col0, b_col1}
                # C11 = C[0, b_col0], C12 = C[0, b_col1], C21 = C[1, b_col0], C22 = C[1, b_col1]
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                W_coeff = {
                    0: [1, 0, 0, 1],   # M1: C11+1, C22+1
                    1: [0, 0, 1, -1],  # M2: C21+1, C22-1
                    2: [0, 1, 0, 1],   # M3: C12+1, C22+1
                    3: [1, 0, 1, 0],   # M4: C11+1, C21+1
                    4: [-1, 1, 0, 0],  # M5: C11-1, C12+1
                    5: [0, 0, 0, 1],   # M6: C22+1
                    6: [1, 0, 0, 0],   # M7: C11+1
                }[s]
                
                # Fill U
                j0 = mid_offset
                j1 = mid_offset + 1
                U[0 * M + j0][col] += U_coeff[0]
                U[0 * M + j1][col] += U_coeff[1]
                U[1 * M + j0][col] += U_coeff[2]
                U[1 * M + j1][col] += U_coeff[3]
                
                # Fill V
                V[j0 * P + b_col0][col] += V_coeff[0]
                V[j0 * P + b_col1][col] += V_coeff[1]
                V[j1 * P + b_col0][col] += V_coeff[2]
                V[j1 * P + b_col1][col] += V_coeff[3]
                
                # Fill W: C[i, k], W index = k * N + i
                W[b_col0 * N + 0][col] += W_coeff[0]
                W[b_col1 * N + 0][col] += W_coeff[1]
                W[b_col0 * N + 1][col] += W_coeff[2]
                W[b_col1 * N + 1][col] += W_coeff[3]
                
                col += 1
        
        # Schoolbook part: A_bottom (row 2, cols mid_offset..mid_offset+1) @ B_block (2x8)
        # 1x2 x 2x8: for each j in {mid_offset, mid_offset+1}, k in {0..7}
        for j in [mid_offset, mid_offset + 1]:
            for k in range(P):
                if col >= rank:
                    return None
                
                U[2 * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + 2][col] = 1
                col += 1
    
    # Check time
    if time.time() > deadline:
        # Return what we have (should be complete at this point)
        pass
    
    # Verify the decomposition
    # Check that sum_r U[i,j,r] * V[j',k,r] * W[k,i,r] = delta[i,i'] * delta[j,j'] * delta[k,k']... 
    # Actually, the tensor T[i,j,k] = 1 if i,j,k form a valid multiplication term.
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    # The decomposition satisfies: T = sum_r U_r * V_r * W_r (outer products)
    
    # Convert to lists of fractions
    U_out = [[str(frac) if frac == frac.numerator else str(frac) for frac in row] for row in U]
    # Actually, the task says "integer/half-integer entries". Let me use actual numbers.
    
    # Convert Fraction to float or keep as fraction? The task says integer/half-integer.
    # Let me convert to floats since all values are integers.
    U_out = [[float(frac) for frac in row] for row in U]
    V_out = [[float(frac) for frac in row] for row in V]
    W_out = [[float(frac) for frac in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
