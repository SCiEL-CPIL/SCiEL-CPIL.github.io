import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is 2x5, B is 5x5, C is 2x5
    # We partition the 5x5 matrix B into blocks:
    # B = [ B11 (2x2)  B12 (2x3) ]
    #     [ B21 (3x2)  B22 (3x3) ]
    # A is partitioned as [ A1 (2x2)  A2 (2x3) ]
    # C is partitioned as [ C1 (2x2)  C2 (2x3) ]
    
    # We will use Strassen's algorithm for the 2x2 block multiplication C1 = A1 * B11.
    # This requires 7 multiplications instead of 8.
    # For the other blocks, we use schoolbook multiplication.
    # A1 (2x2) * B12 (2x3) -> 2*2*3 = 12 mults
    # A2 (2x3) * B21 (3x2) -> 2*3*2 = 12 mults
    # A2 (2x3) * B22 (3x3) -> 2*3*3 = 18 mults
    # Total rank = 7 + 12 + 12 + 18 = 49? 
    # Wait, let's re-evaluate.
    # Standard Strassen for 2x2 * 2x2 is 7.
    # If we partition 5x5 into 2+3, the blocks are 2x2, 2x3, 3x2, 3x3.
    # A is 2x5 -> A1 is 2x2, A2 is 2x3.
    # B is 5x5 -> B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
    # C is 2x5 -> C1 is 2x2, C2 is 2x3.
    
    # C1 = A1 * B11 (2x2 * 2x2) -> Strassen: 7 mults
    # C2 = A1 * B12 + A2 * B22 (2x2*2x3 + 2x3*3x3)
    # Wait, C = A * B.
    # C1 (cols 0,1) = A[:, 0:2] * B[0:2, :] + A[:, 2:5] * B[2:5, :]
    # No, C[i, k] = sum_j A[i, j] B[j, k].
    # C1 (cols 0,1) depends on all rows of B.
    # C1[:, 0:2] = A * B[:, 0:2]
    # C2[:, 2:5] = A * B[:, 2:5]
    
    # Let's partition B's rows into R1 (0:2) and R2 (2:5).
    # B1 = B[0:2, :] (2x5)
    # B2 = B[2:5, :] (3x5)
    # A1 = A[:, 0:2] (2x2)
    # A2 = A[:, 2:5] (2x3)
    
    # C = A1 * B1 + A2 * B2
    # C1 (cols 0,1) = A1 * B1[:, 0:2] + A2 * B2[:, 0:2]
    # C2 (cols 2,3,4) = A1 * B1[:, 2:5] + A2 * B2[:, 2:5]
    
    # Let's count schoolbook rank for this split:
    # A1 * B1[:, 0:2] : 2x2 * 2x2 -> 4 mults
    # A2 * B2[:, 0:2] : 2x3 * 3x2 -> 12 mults
    # A1 * B1[:, 2:5] : 2x2 * 2x3 -> 12 mults
    # A2 * B2[:, 2:5] : 2x3 * 3x3 -> 18 mults
    # Total = 4 + 12 + 12 + 18 = 46.
    
    # Can we apply Strassen to A1 * B1[:, 0:2]?
    # A1 is 2x2, B1_left is 2x2. Yes!
    # Replace 4 mults with 7? No, Strassen is for 2x2*2x2 which is 4 mults schoolbook, 7 Strassen.
    # Wait, Strassen reduces 8 mults (for 2x2*2x2 if you count each element product? No, 2x2*2x2 is 4 element products in schoolbook? No, 2x2*2x2 is 4 terms per element, 4 elements, so 16 mults? No.
    # Schoolbook 2x2 * 2x2:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # This is 8 multiplications.
    # Strassen uses 7.
    
    # So for A1 * B1_left (2x2 * 2x2), we save 1 multiplication.
    # New total: 46 - 1 = 45.
    
    # Can we apply Strassen to other blocks?
    # A2 * B2[:, 0:2] is 2x3 * 3x2. No standard Strassen.
    # A1 * B1[:, 2:5] is 2x2 * 2x3. No.
    # A2 * B2[:, 2:5] is 2x3 * 3x3. No.
    
    # What if we partition differently?
    # Partition B rows into 2, 2, 1?
    # A into 2, 2, 1?
    # A is 2x5. Columns 0,1 | 2,3 | 4.
    # A1 (2x2), A2 (2x2), A3 (2x1).
    # B rows 0,1 | 2,3 | 4.
    # B1 (2x5), B2 (2x5), B3 (1x5).
    
    # C = A1 B1 + A2 B2 + A3 B3
    # C1 (cols 0,1) = A1 B1[:,0:2] + A2 B2[:,0:2] + A3 B3[:,0:2]
    # C2 (cols 2,3) = A1 B1[:,2:4] + A2 B2[:,2:4] + A3 B3[:,2:4]
    # C3 (col 4)   = A1 B1[:,4]   + A2 B2[:,4]   + A3 B3[:,4]
    
    # Let's check ranks:
    # A1 B1[:,0:2] : 2x2 * 2x2 -> 8 (or 7 with Strassen)
    # A2 B2[:,0:2] : 2x2 * 2x2 -> 8 (or 7 with Strassen)
    # A3 B3[:,0:2] : 2x1 * 1x2 -> 2 (or 2)
    # A1 B1[:,2:4] : 2x2 * 2x2 -> 8 (or 7 with Strassen)
    # A2 B2[:,2:4] : 2x2 * 2x2 -> 8 (or 7 with Strassen)
    # A3 B3[:,2:4] : 2x1 * 1x2 -> 2 (or 2)
    # A1 B1[:,4]   : 2x2 * 2x1 -> 4 (or 4)
    # A2 B2[:,4]   : 2x2 * 2x1 -> 4 (or 4)
    # A3 B3[:,4]   : 2x1 * 1x1 -> 2 (or 2)
    
    # Schoolbook total: 8+8+2 + 8+8+2 + 4+4+2 = 46.
    # With Strassen on the 2x2*2x2 blocks (4 of them):
    # 4 * 7 + 4 * 2 + 4 * 4 = 28 + 8 + 16 = 52? No.
    # Wait, 2x1 * 1x2 is 2 mults.
    # 2x2 * 2x1 is 4 mults.
    # 2x1 * 1x1 is 2 mults.
    
    # Let's re-calculate schoolbook for 2,2,1 split:
    # Block (1,1): 2x2 * 2x2 = 8
    # Block (2,1): 2x2 * 2x2 = 8
    # Block (3,1): 2x1 * 1x2 = 2
    # Block (1,2): 2x2 * 2x2 = 8
    # Block (2,2): 2x2 * 2x2 = 8
    # Block (3,2): 2x1 * 1x2 = 2
    # Block (1,3): 2x2 * 2x1 = 4
    # Block (2,3): 2x2 * 2x1 = 4
    # Block (3,3): 2x1 * 1x1 = 2
    # Sum: 8+8+2+8+8+2+4+4+2 = 46.
    
    # Apply Strassen to the four 2x2*2x2 blocks:
    # Each becomes 7.
    # 4 * 7 = 28.
    # Others remain: 2+2+4+4+2 = 14.
    # Total: 28 + 14 = 42.
    
    # This is better than 46!
    
    # Let's verify the dimensions.
    # n=2, m=5, p=5.
    # A is 2x5. B is 5x5. C is 2x5.
    # Split A columns into 2, 2, 1.
    # A1: rows 0-1, cols 0-1. (2x2)
    # A2: rows 0-1, cols 2-3. (2x2)
    # A3: rows 0-1, cols 4.   (2x1)
    
    # Split B rows into 2, 2, 1.
    # B1: rows 0-1, cols 0-4. (2x5)
    # B2: rows 2-3, cols 0-4. (2x5)
    # B3: row 4, cols 0-4.    (1x5)
    
    # We need to compute C = A1 B1 + A2 B2 + A3 B3.
    # C is 2x5.
    # We split C columns into 2, 2, 1?
    # C1: cols 0-1. (2x2)
    # C2: cols 2-3. (2x2)
    # C3: col 4.    (2x1)
    
    # C1 = A1 B1[:,0:2] + A2 B2[:,0:2] + A3 B3[:,0:2]
    # C2 = A1 B1[:,2:4] + A2 B2[:,2:4] + A3 B3[:,2:4]
    # C3 = A1 B1[:,4]   + A2 B2[:,4]   + A3 B3[:,4]
    
    # The blocks are:
    # 1. A1 * B1[:,0:2] (2x2 * 2x2) -> Strassen (7)
    # 2. A2 * B2[:,0:2] (2x2 * 2x2) -> Strassen (7)
    # 3. A3 * B3[:,0:2] (2x1 * 1x2) -> Schoolbook (2)
    # 4. A1 * B1[:,2:4] (2x2 * 2x2) -> Strassen (7)
    # 5. A2 * B2[:,2:4] (2x2 * 2x2) -> Strassen (7)
    # 6. A3 * B3[:,2:4] (2x1 * 1x2) -> Schoolbook (2)
    # 7. A1 * B1[:,4]   (2x2 * 2x1) -> Schoolbook (4)
    # 8. A2 * B2[:,4]   (2x2 * 2x1) -> Schoolbook (4)
    # 9. A3 * B3[:,4]   (2x1 * 1x1) -> Schoolbook (2)
    
    # Total Rank = 7+7+2+7+7+2+4+4+2 = 42.
    
    # We need to construct U, V, W matrices.
    # U shape: (n*m, Rank) = (10, 42)
    # V shape: (m*p, Rank) = (25, 42)
    # W shape: (p*n, Rank) = (10, 42)
    
    # Indices:
    # U row index: i*m + j (i in 0..1, j in 0..4)
    # V row index: j*p + k (j in 0..4, k in 0..4)
    # W row index: k*n + i (k in 0..4, i in 0..1)
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        # A is n x m. A_rows, A_cols define the submatrix.
        # B is m x p. B_rows, B_cols define the submatrix.
        # C is p x n. C_rows, C_cols define the submatrix.
        # Actually C is n x p? No, C is n x p?
        # Task: "output factor follows the official trace convention W[k*n+i,r]"
        # This implies C is n x p?
        # "multiplying a 2 by 5 matrix by a 5 by 5 matrix"
        # A (2x5) * B (5x5) = C (2x5).
        # So C is n x p where n=2, p=5.
        # W index is k*n + i. k is col of C (0..4), i is row of C (0..1).
        # So W row index corresponds to C[k, i].
        
        # The tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i.
        # This matches C[k, i] = sum_{j} A[i, j] B[j, k].
        
        for i_rel in range(len(A_rows)):
            i_abs = A_rows[i_rel]
            for j_rel in range(len(A_cols)):
                j_abs = A_cols[j_rel]
                for k_rel in range(len(B_cols)):
                    k_abs = B_cols[k_rel]
                    # Check if B row matches j_abs
                    if j_abs in B_rows:
                        j_row_idx = B_rows.index(j_abs)
                        # Term A[i_abs, j_abs] * B[j_abs, k_abs] contributes to C[k_abs, i_abs]
                        # U index: i_abs * m + j_abs
                        # V index: j_abs * p + k_abs
                        # W index: k_abs * n + i_abs
                        U[i_abs * m + j_abs, col] = 1
                        V[j_abs * p + k_abs, col] = 1
                        W[k_abs * n + i_abs, col] = 1
                        col += 1

    # Helper to add Strassen block for 2x2 * 2x2
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        # A_rows, A_cols are [i1, i2], [j1, j2]
        # B_rows, B_cols are [j1, j2], [k1, k2]
        # C_rows, C_cols are [k1, k2], [i1, i2]
        # Note: C is n x p. C[k, i].
        # So C_rows are k indices, C_cols are i indices.
        nonlocal col
        
        i1, i2 = A_rows
        j1, j2 = A_cols
        k1, k2 = B_cols
        # B rows must be j1, j2
        
        # Strassen variables:
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
        
        # We need to map these to U, V, W entries.
        # Each M is a sum of products of linear combinations of A and B elements.
        # We can expand each M into elementary products and add them to U, V, W.
        
        # To keep it simple and exact, we will expand each M into its constituent elementary products.
        # This effectively implements Strassen but with integer coefficients.
        
        def add_product(A_coeff, A_i, A_j, B_coeff, B_j, B_k, C_coeff, C_k, C_i):
            nonlocal col
            if A_coeff == 0 or B_coeff == 0 or C_coeff == 0:
                return
            U[A_i * m + A_j, col] = A_coeff
            V[B_j * p + B_k, col] = B_coeff
            W[C_k * n + C_i, col] = C_coeff
            col += 1

        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # = A[i1,j1]B[j1,k1] + A[i1,j1]B[j2,k2] + A[i2,j2]B[j1,k1] + A[i2,j2]B[j2,k2]
        # Contributes to C[k1,i1] and C[k1,i2]? No.
        # C11 (C[k1, i1]) gets M1.
        # C22 (C[k2, i2]) gets M1.
        
        # Wait, Strassen computes C = A*B.
        # C11 corresponds to C[k1, i1] if A is 2x2 and B is 2x2?
        # A is n x m. Here n=2, m=2 (subblock).
        # B is m x p. Here m=2, p=2 (subblock).
        # C is n x p. Here n=2, p=2 (subblock).
        # C[k, i].
        # C11 is C[k1, i1].
        # C12 is C[k2, i1].
        # C21 is C[k1, i2].
        # C22 is C[k2, i2].
        
        # M1 contributes to C11 (+1) and C22 (+1).
        add_product(1, i1, j1, 1, j1, k1, 1, k1, i1)
        add_product(1, i1, j1, 1, j2, k2, 1, k1, i1)
        add_product(1, i2, j2, 1, j1, k1, 1, k1, i1)
        add_product(1, i2, j2, 1, j2, k2, 1, k1, i1)
        
        add_product(1, i1, j1, 1, j1, k1, 1, k2, i2)
        add_product(1, i1, j1, 1, j2, k2, 1, k2, i2)
        add_product(1, i2, j2, 1, j1, k1, 1, k2, i2)
        add_product(1, i2, j2, 1, j2, k2, 1, k2, i2)
        
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        # = A[i2,j1]B[j1,k1] + A[i2,j2]B[j1,k1]
        # Contributes to C21 (+1) and C22 (-1).
        add_product(1, i2, j1, 1, j1, k1, 1, k1, i2)
        add_product(1, i2, j2, 1, j1, k1, 1, k1, i2)
        
        add_product(-1, i2, j1, 1, j1, k1, 1, k2, i2)
        add_product(-1, i2, j2, 1, j1, k1, 1, k2, i2)
        
        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        # = A[i1,j1]B[j1,k2] - A[i1,j1]B[j2,k2]
        # Contributes to C12 (+1) and C22 (+1).
        add_product(1, i1, j1, 1, j1, k2, 1, k2, i1)
        add_product(-1, i1, j1, 1, j2, k2, 1, k2, i1)
        
        add_product(1, i1, j1, 1, j1, k2, 1, k2, i2)
        add_product(-1, i1, j1, 1, j2, k2, 1, k2, i2)
        
        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        # = A[i2,j2]B[j2,k1] - A[i2,j2]B[j1,k1]
        # Contributes to C11 (+1) and C21 (+1).
        add_product(1, i2, j2, 1, j2, k1, 1, k1, i1)
        add_product(-1, i2, j2, 1, j1, k1, 1, k1, i1)
        
        add_product(1, i2, j2, 1, j2, k1, 1, k1, i2)
        add_product(-1, i2, j2, 1, j1, k1, 1, k1, i2)
        
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        # = A[i1,j1]B[j2,k2] + A[i1,j2]B[j2,k2]
        # Contributes to C11 (-1) and C12 (+1).
        add_product(-1, i1, j1, 1, j2, k2, 1, k1, i1)
        add_product(-1, i1, j2, 1, j2, k2, 1, k1, i1)
        
        add_product(1, i1, j1, 1, j2, k2, 1, k2, i1)
        add_product(1, i1, j2, 1, j2, k2, 1, k2, i1)
        
        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        # = A[i2,j1]B[j1,k1] + A[i2,j1]B[j1,k2] - A[i1,j1]B[j1,k1] - A[i1,j1]B[j1,k2]
        # Contributes to C22 (+1).
        add_product(1, i2, j1, 1, j1, k1, 1, k2, i2)
        add_product(1, i2, j1, 1, j1, k2, 1, k2, i2)
        add_product(-1, i1, j1, 1, j1, k1, 1, k2, i2)
        add_product(-1, i1, j1, 1, j1, k2, 1, k2, i2)
        
        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        # = A[i1,j2]B[j2,k1] + A[i1,j2]B[j2,k2] - A[i2,j2]B[j2,k1] - A[i2,j2]B[j2,k2]
        # Contributes to C11 (+1).
        add_product(1, i1, j2, 1, j2, k1, 1, k1, i1)
        add_product(1, i1, j2, 1, j2, k2, 1, k1, i1)
        add_product(-1, i2, j2, 1, j2, k1, 1, k1, i1)
        add_product(-1, i2, j2, 1, j2, k2, 1, k1, i1)

    # Define the blocks
    A_rows_1 = [0, 1]
    A_cols_1 = [0, 1]
    A_rows_2 = [0, 1]
    A_cols_2 = [2, 3]
    A_rows_3 = [0, 1]
    A_cols_3 = [4]
    
    B_rows_1 = [0, 1]
    B_rows_2 = [2, 3]
    B_rows_3 = [4]
    
    C_cols_1 = [0, 1]
    C_cols_2 = [2, 3]
    C_cols_3 = [4]
    
    C_rows_1 = [0, 1]
    C_rows_2 = [2, 3]
    C_rows_3 = [4]
    
    # Block 1,1: A1 * B1[:, 0:2]
    add_strassen_block(A_rows_1, A_cols_1, B_rows_1, C_cols_1, C_rows_1, A_rows_1)
    
    # Block 2,1: A2 * B2[:, 0:2]
    add_strassen_block(A_rows_2, A_cols_2, B_rows_2, C_cols_1, C_rows_1, A_rows_2)
    
    # Block 3,1: A3 * B3[:, 0:2]
    add_schoolbook_block(A_rows_3, A_cols_3, B_rows_3, C_cols_1, C_rows_1, A_rows_3)
    
    # Block 1,2: A1 * B1[:, 2:4]
    add_strassen_block(A_rows_1, A_cols_1, B_rows_1, C_cols_2, C_rows_2, A_rows_1)
    
    # Block 2,2: A2 * B2[:, 2:4]
    add_strassen_block(A_rows_2, A_cols_2, B_rows_2, C_cols_2, C_rows_2, A_rows_2)
    
    # Block 3,2: A3 * B3[:, 2:4]
    add_schoolbook_block(A_rows_3, A_cols_3, B_rows_3, C_cols_2, C_rows_2, A_rows_3)
    
    # Block 1,3: A1 * B1[:, 4]
    add_schoolbook_block(A_rows_1, A_cols_1, B_rows_1, C_cols_3, C_rows_3, A_rows_1)
    
    # Block 2,3: A2 * B2[:, 4]
    add_schoolbook_block(A_rows_2, A_cols_2, B_rows_2, C_cols_3, C_rows_3, A_rows_2)
    
    # Block 3,3: A3 * B3[:, 4]
    add_schoolbook_block(A_rows_3, A_cols_3, B_rows_3, C_cols_3, C_rows_3, A_rows_3)
    
    # Trim unused columns
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    # Convert to lists of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
