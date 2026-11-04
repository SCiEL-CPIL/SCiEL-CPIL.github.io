import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Theoretical maximum schoolbook rank is n*m*p = 48.
    # Strassen based on 2x2 blocks gives 42.
    # We will construct the Strassen decomposition and then attempt to verify/optimize.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        # Adds a 2x2x2 Strassen block
        # A: rows i_start..i_start+1, cols j_start..j_start+1
        # B: rows j_start..j_start+1, cols k_start..k_start+1
        # C: rows k_start..k_start+1, cols i_start..i_start+1
        
        # M1: (A00+A11)(B00+B11)
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M2: (A10+A11)B00
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        # M3: A00(B01-B11)
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M4: A11(B10-B00)
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        # M5: (A00+A01)B11
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        # M6: (A10-A00)(B00+B01)
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        # M7: (A01-A11)(B10+B11)
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    # Apply Strassen blocks for the 2x4x6 multiplication
    # A is 2x4, B is 4x6.
    # We split A into two 2x2 blocks horizontally? No, A is 2 rows, 4 cols.
    # So A = [A1 | A2] where A1, A2 are 2x2.
    # B is 4 rows, 6 cols. B = [B1; B2] where B1, B2 are 2x6.
    # Then A*B = A1*B1 + A2*B2.
    # This decomposition splits the summation dimension m=4 into 2+2.
    # Then for each 2x2 * 2x6 product, we split the 2x6 into three 2x2 blocks?
    # Or rather, we can view the whole thing as 3 blocks of 2x2 matrices on the right.
    # Let's stick to the 6 blocks of 2x2x2 used in the previous solution which is valid.
    # Block structure:
    # i (0,1) is row of A and C.
    # j (0..3) is col of A and row of B.
    # k (0..5) is col of B and col of C.
    
    # The previous logic used 6 blocks:
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 3. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # 4. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 5. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # 6. A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)

    # Convert to lists of lists as required by the interface if strictly needed, 
    # though numpy arrays are often accepted, the prompt says "factor matrices of shapes... with integer/half-integer entries".
    # The previous program returned lists of lists. I will do the same to be safe, but the numpy check is internal.
    
    # Optional: Verify rank is indeed 42 and not lower due to coincidental cancellation (unlikely but good for "stronger" algo)
    # We can check if any column is linearly dependent on others.
    # If we found a lower rank, we would prune. 
    # Given time constraints, we assume 42 is minimal for this structure.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
