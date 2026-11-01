"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Construct a decomposition explicitly using Strassen for the 2x2 part
    # and Schoolbook for the 1x2 part, then merge them into a single set of factor matrices.
    
    # The matrix A is 2x3. We can split the columns of A into A1 (2x2) and A2 (2x1).
    # The matrix B is 3x7. We split the rows of B into B1 (2x7) and B2 (1x7).
    # C = A * B = [A1 A2] * [B1; B2] = A1*B1 + A2*B2.
    
    # Part 1: A1 (2x2) * B1 (2x7). 
    # We can use Strassen's algorithm for the 2x2 * 2x2 blocks of B1.
    # B1 is 2x7, so we can view it as 3 blocks of 2x2 and one 2x1.
    # Let's just use a general bilinear algorithm for 2x2 * 2x7.
    # Actually, simpler: Use Strassen for 2x2*2x2 (rank 7) and schoolbook for 2x2*2x1 (rank 4).
    # Total rank for A1*B1 = 7*3 + 4 = 25? No, Strassen is for 2x2*2x2.
    # Let's stick to a simpler, guaranteed improvement:
    # The schoolbook rank is n*m*p = 2*3*7 = 42.
    
    # Let's try to construct a rank 39 decomposition as hinted in the knowledge:
    # "Replaces the trivial schoolbook decomposition (rank 42) with a hybrid Strassen/Schoolbook 
    # decomposition that exploits the structure of the 2x3x7 dimensions by splitting the inner 
    # dimension and using Strassen for 2x2x2 blocks, reducing the rank to 39."
    
    # Construction:
    # A is 2x3. B is 3x7.
    # Split B into B_top (2x7) and B_bot (1x7).
    # Split A into A_left (2x2) and A_right (2x1).
    # C = A_left * B_top + A_right * B_bot.
    
    # Term 1: A_left (2x2) * B_top (2x7).
    # We can use Strassen for the 2x2 * 2x2 parts of B_top.
    # B_top has 7 columns. Let's split B_top into B_t1 (2x2), B_t2 (2x2), B_t3 (2x2), B_t4 (2x1).
    # For each 2x2 block, Strassen uses 7 multiplications.
    # For the 2x1 block, schoolbook uses 2 multiplications.
    # Total rank for this part = 7 + 7 + 7 + 2 = 23.
    
    # Term 2: A_right (2x1) * B_bot (1x7).
    # This is an outer product of a vector and a vector? No, it's (2x1) * (1x7) = 2x7.
    # This is rank 1? No, the bilinear complexity of (2x1) * (1x7) is 2*1*7 = 14?
    # Wait, the tensor is T_{i,j,k} where i in 0..1, j in 2, k in 0..6.
    # The term is A_right[:, j] * B_bot[j, :].
    # This is a sum of 7 terms? No, j is fixed to the index 2 (the 3rd column/row).
    # So it is a single outer product: A_right (2x1) * B_bot (1x7).
    # The rank of the outer product tensor (2x1x7) is 2*1*7 = 14? No.
    # The tensor is T_{i, 2, k} = A[i, 2] * B[2, k].
    # This is a rank 1 tensor? No, it's a matrix.
    # In the decomposition sum_c U_c V_c W_c, each term is a rank 1 tensor.
    # The tensor for (2x1) * (1x7) has rank 2*1*7 = 14?
    # Let's check: T_{i, 2, k} = A_{i, 2} B_{2, k}.
    # We can write this as sum_{l=0}^1 sum_{m=0}^6 ... no.
    # The bilinear complexity of multiplying a (n x 1) matrix by a (1 x p) matrix is n*p.
    # So rank is 2*7 = 14.
    
    # Total rank = 23 + 14 = 37.
    # This is better than 42.
    
    # Let's verify the Strassen part.
    # Strassen for 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the U, V, W matrices.
    # U has shape (n*m, rank) = (6, rank).
    # V has shape (m*p, rank) = (21, rank).
    # W has shape (p*n, rank) = (14, rank).
    
    # Indices:
    # U row index: i*m + j = i*3 + j.
    # V row index: j*p + k = j*7 + k.
    # W row index: k*n + i = k*2 + i.
    
    # Let's build the decomposition.
    
    terms = []
    
    # Part 1: A_left (cols 0,1) * B_top (rows 0,1)
    # B_top columns 0,1 -> B_t1
    # B_top columns 2,3 -> B_t2
    # B_top columns 4,5 -> B_t3
    # B_top columns 6   -> B_t4
    
    # Strassen for B_t1 (cols 0,1)
    # M1: (A00+A11) * (B00+B11)
    # A indices: (0,0), (1,1). B indices: (0,0), (1,1).
    # U: U[0, c] += 1, U[3, c] += 1 (A00, A11)
    # V: V[0, c] += 1, V[7, c] += 1 (B00, B11)
    # W: W[0, c] += 1, W[1, c] += 1, W[2, c] += 1, W[3, c] += 1? 
    # Wait, W[k*n+i].
    # C00 = A00*B00 + A01*B10 + A10*B01 + A11*B11.
    # In Strassen, C00 is combination of Ms.
    # We need to add the contributions to the final C.
    # Each M contributes to specific C entries.
    
    # Let's define a helper to add a Strassen term.
    # This is getting complex to code generically. 
    # Given the time limit and the need for a robust solution, 
    # I will implement the specific rank 37 construction.
    
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        terms.append((u_idx, v_idx, w_idx, val))
        
    # Part 1: Strassen for 2x2 * 2x2 blocks
    # Block 1: A(0:2, 0:2) * B(0:2, 0:2)
    # M1
    u1 = [0, 3] # A00, A11
    v1 = [0, 7] # B00, B11
    w1 = [0, 1, 2, 3] # C00, C01, C10, C11? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # This requires tracking coefficients.
    # Let's just use a dense representation of the tensor and decompose it?
    # No, that's what the previous program tried and failed to optimize.
    
    # Alternative: Use the known rank 39 or 37 from literature?
    # For 2x3 x 3x7, the schoolbook is 42.
    # A simple improvement:
    # Split B into B1 (2x7) and B2 (1x7).
    # Split A into A1 (2x2) and A2 (2x1).
    # A1*B1: Use Strassen for the 2x2*2x2 parts.
    # B1 has 7 cols. 3 full 2x2 blocks and 1 2x1 block.
    # Rank for 2x2*2x2 is 7.
    # Rank for 2x2*2x1 is 4 (schoolbook: 2*2*1=4).
    # Total for A1*B1 = 7*3 + 4 = 25.
    # A2*B2: (2x1)*(1x7). Rank is 14 (schoolbook: 2*1*7=14).
    # Total rank = 25 + 14 = 39.
    
    # Let's implement this rank 39 construction.
    
    U = np.zeros((6, 39))
    V = np.zeros((21, 39))
    W = np.zeros((14, 39))
    
    col = 0
    
    # Helper to add a scalar multiplication term (schoolbook style)
    def add_schoolbook(i, j, k, coeff=1.0):
        nonlocal col
        U[i*3 + j, col] = coeff
        V[j*7 + k, col] = 1.0
        W[k*2 + i, col] = 1.0
        col += 1
        
    # Helper to add a Strassen term for 2x2*2x2
    # A is rows 0,1 cols 0,1. B is rows 0,1 cols c0, c0+1.
    def add_strassen_block(c0):
        nonlocal col
        # M1 = (A00+A11)(B0,c0 + B1,c0+1)
        U[0, col] = 1.0; U[3, col] = 1.0
        V[c0, col] = 1.0; V[7+c0+1, col] = 1.0
        # M1 contributes to C00 (+1), C11 (+1)
        # We need to accumulate these into W.
        # But W is fixed size. We can't just add to W here because
        # the final C is sum of all terms.
        # We must add the contribution of M1 to the specific C entries.
        # C00 is W[0], C11 is W[3].
        # But M1 is a single term in the sum.
        # So we add 1 to W[0] and 1 to W[3] for this column?
        # No, W[k*n+i] is the coefficient for the output.
        # The term is U * V * W.
        # If we set U, V, W for this column, it adds to the dot product.
        # So for M1, we need to add to C00 and C11.
        # C00 corresponds to i=0, k=0 -> W[0].
        # C11 corresponds to i=1, k=1 -> W[3].
        # So:
        W[0, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B0,c0
        U[3, col] = 1.0; U[4, col] = 1.0 # A11 is idx 3? No.
        # A indices:
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        # So A10 is U[3], A11 is U[4].
        U[3, col] = 1.0; U[4, col] = 1.0
        V[c0, col] = 1.0
        # M2 contributes to C10 (-1), C11 (-1)
        # C10 is i=1, k=0 -> W[2].
        # C11 is i=1, k=1 -> W[3].
        W[2, col] = -1.0
        W[3, col] = -1.0
        col += 1
        
        # M3 = A00(B1,c0+1 - B0,c0+1)
        U[0, col] = 1.0
        V[7+c0+1, col] = 1.0; V[c0+1, col] = -1.0
        # M3 contributes to C01 (+1), C11 (+1)
        # C01 is i=0, k=1 -> W[1].
        # C11 is i=1, k=1 -> W[3].
        W[1, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # M4 = A11(B0,c0 - B1,c0)
        U[4, col] = 1.0
        V[c0, col] = 1.0; V[7+c0, col] = -1.0
        # M4 contributes to C00 (+1), C10 (+1)
        # C00 is W[0].
        # C10 is W[2].
        W[0, col] = 1.0
        W[2, col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B1,c0+1
        U[0, col] = 1.0; U[1, col] = 1.0
        V[7+c0+1, col] = 1.0
        # M5 contributes to C01 (+1), C00 (-1)
        # C01 is W[1].
        # C00 is W[0].
        W[1, col] = 1.0
        W[0, col] = -1.0
        col += 1
        
        # M6 = (A10-A00)(B0,c0 + B1,c0)
        U[3, col] = 1.0; U[0, col] = -1.0
        V[c0, col] = 1.0; V[7+c0, col] = 1.0
        # M6 contributes to C11 (+1)
        # C11 is W[3].
        W[3, col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B1,c0+1 + B0,c0+1)
        U[1, col] = 1.0; U[4, col] = -1.0
        V[7+c0+1, col] = 1.0; V[c0+1, col] = 1.0
        # M7 contributes to C00 (+1)
        # C00 is W[0].
        W[0, col] = 1.0
        col += 1

    # Apply Strassen to B_top cols 0,1
    add_strassen_block(0)
    # Apply Strassen to B_top cols 2,3
    add_strassen_block(2)
    # Apply Strassen to B_top cols 4,5
    add_strassen_block(4)
    
    # Now handle the remaining column 6 of B_top with Schoolbook for A_left (2x2)
    # A_left is cols 0,1. B_top row 0,1 col 6.
    # Terms: (0,0,6), (0,1,6), (1,0,6), (1,1,6)
    add_schoolbook(0, 0, 6)
    add_schoolbook(0, 1, 6)
    add_schoolbook(1, 0, 6)
    add_schoolbook(1, 1, 6)
    
    # Now handle A_right (col 2) * B_bot (row 2)
    # A_right is 2x1 (rows 0,1, col 2).
    # B_bot is 1x7 (row 2, cols 0..6).
    # Terms: (0,2,k) and (1,2,k) for k in 0..6.
    for k in range(7):
        add_schoolbook(0, 2, k)
        add_schoolbook(1, 2, k)
        
    # Total rank used: 7*3 + 4 + 14 = 21 + 4 + 14 = 39.
    # Check if col == 39.
    
    if col != 39:
        # Fallback to schoolbook if construction failed
        U = np.zeros((6, 42))
        V = np.zeros((21, 42))
        W = np.zeros((14, 42))
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    U[i*3+j, i*9+j*3+k] = 1.0
                    V[j*7+k, i*9+j*3+k] = 1.0
                    W[k*2+i, i*9+j*3+k] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.dot(u_val, v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback
        U = np.zeros((6, 42))
        V = np.zeros((21, 42))
        W = np.zeros((14, 42))
        for i in range(2):
            for j in range(3):
                for k in range(7):
                    U[i*3+j, i*9+j*3+k] = 1.0
                    V[j*7+k, i*9+j*3+k] = 1.0
                    W[k*2+i, i*9+j*3+k] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
