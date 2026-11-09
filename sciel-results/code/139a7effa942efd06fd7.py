"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x4 * 4x5.
    Uses a block decomposition strategy:
    - Split the 4x5 right matrix into a 4x4 block and a 4x1 slice.
    - Use Strassen's rank-7 algorithm for the 2x4 * 4x4 part (viewed as four 2x2 * 2x2 blocks).
    - Use schoolbook for the 2x4 * 4x1 part.
    This gives rank 36 (7*4 + 4*1 = 32 + 4 = 36).
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130.0

    # Target tensor T[i, j, k] = 1 for all i in [0,2), j in [0,4), k in [0,5)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # --- Strassen's rank-7 decomposition for 2x2 * 2x2 ---
    # For matrices A (2x2), B (2x2), C = A*B (2x2):
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

    # For the 2x4 * 4x4 part:
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A1 (2x2), A2 (2x2) horizontally.
    # Split B into B1 (2x4), B2 (2x4) vertically? No, B is 4x4.
    # Let's index B as B[j, k] where j in [0,4), k in [0,4).
    # Split j into j1 in [0,2), j2 in [2,4).
    # Split k into k1 in [0,2), k2 in [2,4).
    # Then C[i, k] = sum_{j=0}^{3} A[i, j] B[j, k]
    # = sum_{j1=0}^{1} A[i, j1] B[j1, k] + sum_{j2=0}^{1} A[i, 2+j2] B[2+j2, k]
    # This is not directly four 2x2*2x2 blocks because the output C is 2x4, not 2x2.
    # Actually, we can view the 2x4 * 4x4 multiplication as computing a 2x4 result.
    # We can split the result C into C1 (2x2) and C2 (2x2) horizontally.
    # C1[i, k1] = sum_{j=0}^{3} A[i, j] B[j, k1]
    # C2[i, k2] = sum_{j=0}^{3} A[i, j] B[j, 2+k2]
    # This doesn't factor nicely into independent 2x2*2x2 problems because A is shared.
    
    # Alternative approach: View the 2x4 * 4x4 as two separate 2x2 * 2x4 multiplications?
    # A1 (2x2) * B (2x4) -> C1 (2x4)
    # A2 (2x2) * B (2x4) -> C2 (2x4)
    # This doesn't help with rank reduction directly.
    
    # Let's reconsider the block structure.
    # The standard way to apply Strassen to larger matrices is to split them into blocks.
    # For 2x4 * 4x4, we can split the middle dimension (4) into two 2s.
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B is 4x4. Split B into B11, B12 (top rows) and B21, B22 (bottom rows), each 2x2?
    # No, B is 4x4. If we split the rows of B into two groups of 2, and columns into two groups of 2:
    # B = [[B11, B12], [B21, B22]] where each Bi is 2x2.
    # Then C = A1*B_top + A2*B_bottom
    # C = A1 * [B11, B12] + A2 * [B21, B22]
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # This is two 2x4 results, each being a sum of two 2x2*2x2 products.
    # This doesn't reduce the rank below schoolbook for this specific split.
    
    # Let's try a different split.
    # What if we view the 2x4 * 4x5 as a 2x4 * 4x4 + 2x4 * 4x1?
    # The 2x4 * 4x1 part has rank 8 (schoolbook).
    # The 2x4 * 4x4 part: Can we do better than 16?
    # The rank of 2x4*4x4 is at least 2*4 + 4*4 - 4 = 16? No, that's not a lower bound.
    # Actually, the rank of matrix multiplication m x n * n x p is at least max(m, n, p) * something.
    # For 2x2*2x2, rank is 7.
    # For 2x4*4x4, we can split the 4x4 into four 2x2 blocks?
    # Let A be 2x4, B be 4x4.
    # Split A into A1, A2 (2x2 each).
    # Split B into B1, B2 (2x4 each, top and bottom).
    # C = A1*B1 + A2*B2.
    # This is just linearity, no rank reduction.
    
    # Let's look at the inspiration again: "Using Strassen's rank-7 decomposition for four 2×2×2 sub-blocks plus schoolbook for the remaining 2×4×1 slice reduces rank from 40 to 36"
    # How are there four 2x2x2 sub-blocks in a 2x4*4x4 multiplication?
    # Maybe the 2x4*4x4 is viewed as a block matrix multiplication?
    # If we view the 2x4 matrix A as a 1x2 block of 2x2 matrices: A = [A11, A12].
    # And the 4x4 matrix B as a 2x2 block of 2x2 matrices: B = [[B11, B12], [B21, B22]].
    # Then C = A * B = [A11*B11 + A12*B21, A11*B12 + A12*B22].
    # This is two 2x2 matrix multiplications (sums of two products).
    # This doesn't directly map to four independent 2x2*2x2 Strassen decompositions.
    
    # Perhaps the inspiration means something else. Let's re-read: "four 2×2×2 sub-blocks".
    # Maybe the 2x4*4x5 is split into 2x2*2x2 blocks differently?
    # If we split the 2x4 into two 2x2 blocks horizontally, and the 4x5 into... 
    # Wait, the inner dimension is 4.
    # If we split the inner dimension 4 into two 2s, we have:
    # A = [A1, A2] (2x2 each)
    # B = [B1; B2] (2x5 each, stacked vertically)
    # C = A1*B1 + A2*B2
    # This is a sum of two 2x2 * 2x5 multiplications.
    # A 2x2 * 2x5 multiplication can be viewed as two 2x2 * 2x2 multiplications plus a 2x2 * 2x1?
    # No, 2x2 * 2x5 = 2x2 * [2x2, 2x2, 2x1]? No, 5 is not divisible by 2.
    # 2x2 * 2x5: Split the 5 into 2+2+1.
    # So 2x2 * 2x5 = 2x2*2x2 + 2x2*2x2 + 2x2*2x1.
    # So A1*B1 (where B1 is 2x5) = A1*B1a (2x2*2x2) + A1*B1b (2x2*2x2) + A1*B1c (2x2*2x1).
    # Similarly for A2*B2.
    # Total: 2 * (2 Strassen + 1 schoolbook 2x2*2x1) = 4 Strassen + 2 schoolbook 2x2*2x1.
    # Rank: 4*7 + 2*2 = 28 + 4 = 32.
    # Plus the remaining 2x4*4x1 part? No, we already accounted for the full 2x4*4x5.
    # Wait, the split was A=[A1,A2] (2x4) and B=[B1;B2] (4x5).
    # So B1 is 2x5, B2 is 2x5.
    # So we have two 2x2*2x5 multiplications.
    # Each 2x2*2x5 is decomposed into two 2x2*2x2 and one 2x2*2x1.
    # So total rank = 2 * (2*7 + 1*2) = 2 * (14 + 2) = 32.
    # This is better than 36!
    
    # Let's verify this logic.
    # C = A1*B1 + A2*B2
    # B1 is 2x5. Split B1 into B11 (2x2), B12 (2x2), B13 (2x1).
    # B1 = [B11, B12, B13]
    # A1*B1 = A1*B11 + A1*B12 + A1*B13
    # A1*B11 is 2x2*2x2 -> Strassen rank 7.
    # A1*B12 is 2x2*2x2 -> Strassen rank 7.
    # A1*B13 is 2x2*2x1 -> Schoolbook rank 2.
    # So A1*B1 has rank 7+7+2 = 16.
    # Similarly A2*B2 has rank 16.
    # Total rank = 16 + 16 = 32.
    
    # Can we do even better?
    # What if we use a different split for the 2x2*2x1?
    # 2x2*2x1 is just a vector-matrix product. Rank is 2 (schoolbook).
    # Is there a lower rank? No, the rank of 2x2*2x1 is 2.
    
    # So rank 32 is achievable.
    # Let's implement this.
    
    # We need to construct U, V, W for the full 2x4*4x5 multiplication.
    # U shape: (8, rank)
    # V shape: (20, rank)
    # W shape: (10, rank)
    
    # Let's define the Strassen decomposition for 2x2*2x2.
    # U, V, W for Strassen 2x2*2x2:
    # U indices: i*2+j for i in [0,2), j in [0,2).
    # V indices: j*2+k for j in [0,2), k in [0,2).
    # W indices: k*2+i for k in [0,2), i in [0,2).
    
    def get_strassen_222():
        """
        Returns U, V, W for 2x2*2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[3, 0] = 1  # C11, C22
        W[1, 0] = 0; W[2, 0] = 0
        
        # M2 = (A21 + A22) * B11
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1; W[3, 1] = 1  # C21, C22
        
        # M3 = A11 * (B12 - B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1; V[3, 2] = -1  # B12, -B22
        W[1, 2] = 1; W[3, 2] = 1  # C12, C22
        
        # M4 = A22 * (B21 - B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1; V[0, 3] = -1  # B21, -B11
        W[0, 3] = 1; W[2, 3] = 1  # C11, C21
        
        # M5 = (A11 + A12) * B22
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1; W[1, 4] = 1  # -C11, C12
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[3, 5] = 1  # C22
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    def get_schoolbook_221():
        """
        Returns U, V, W for 2x2*2x1 schoolbook decomposition.
        U: (4, 2), V: (2, 2), W: (2, 2)
        """
        U = np.zeros((4, 2))
        V = np.zeros((2, 2))
        W = np.zeros((2, 2))
        
        # C[i, 0] = A[i, 0]*B[0, 0] + A[i, 1]*B[1, 0]
        # Term 1: i=0, j=0, k=0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: i=0, j=1, k=0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: i=1, j=0, k=0
        U[2, 0] = 1; V[0, 0] = 1; W[1, 0] = 1
        # Term 4: i=1, j=1, k=0
        U[3, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # Now, let's assemble the full decomposition.
    # We have two main parts: A1*B1 and A2*B2.
    # A1 is rows 0-1 of A, columns 0-1. (2x2)
    # A2 is rows 0-1 of A, columns 2-3. (2x2)
    # B1 is rows 0-1 of B, columns 0-4. (2x5)
    # B2 is rows 2-3 of B, columns 0-4. (2x5)
    
    # B1 is split into B11 (2x2), B12 (2x2), B13 (2x1).
    # B11: rows 0-1, cols 0-1.
    # B12: rows 0-1, cols 2-3.
    # B13: rows 0-1, col 4.
    
    # Similarly for B2.
    
    # Total rank = 2 * (7 + 7 + 2) = 32.
    
    U_full = np.zeros((8, 32))
    V_full = np.zeros((20, 32))
    W_full = np.zeros((10, 32))
    
    col_offset = 0
    
    # Part 1: A1 * B1
    # A1 indices in U: 0-3 (i*2+j for i in 0-1, j in 0-1)
    # B1 indices in V: 0-9 (j*5+k for j in 0-1, k in 0-4)
    # C1 indices in W: 0-9 (k*2+i for k in 0-4, i in 0-1)
    
    # Subpart 1.1: A1 * B11 (2x2 * 2x2)
    U_s, V_s, W_s = get_strassen_222()
    # Map indices:
    # U_s: i*2+j -> i*2+j (same)
    # V_s: j*2+k -> j*5+k (j in 0-1, k in 0-1)
    # W_s: k*2+i -> k*2+i (k in 0-1, i in 0-1)
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2+j, col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V_full[j*5+k, col_offset + r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W_full[k*2+i, col_offset + r] = W_s[k*2+i, r]
    
    col_offset += 7
    
    # Subpart 1.2: A1 * B12 (2x2 * 2x2)
    # B12 is rows 0-1, cols 2-3.
    # V indices: j*5 + (2+k) for j in 0-1, k in 0-1.
    # W indices: (2+k)*2 + i for k in 0-1, i in 0-1.
    
    U_s, V_s, W_s = get_strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2+j, col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V_full[j*5 + (2+k), col_offset + r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W_full[(2+k)*2 + i, col_offset + r] = W_s[k*2+i, r]
    
    col_offset += 7
    
    # Subpart 1.3: A1 * B13 (2x2 * 2x1)
    # B13 is rows 0-1, col 4.
    # V indices: j*5 + 4 for j in 0-1.
    # W indices: 4*2 + i for i in 0-1.
    
    U_s, V_s, W_s = get_schoolbook_221()
    for r in range(2):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2+j, col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            if V_s[j, r] != 0:
                V_full[j*5 + 4, col_offset + r] = V_s[j, r]
        for i in range(2):
            if W_s[i, r] != 0:
                W_full[4*2 + i, col_offset + r] = W_s[i, r]
    
    col_offset += 2
    
    # Part 2: A2 * B2
    # A2 indices in U: 4-7 (i*2 + (2+j) for i in 0-1, j in 0-1)
    # B2 indices in V: 10-19 (j*5+k for j in 2-3, k in 0-4)
    # C2 indices in W: 0-9 (k*2+i for k in 0-4, i in 0-1)
    
    # Subpart 2.1: A2 * B21 (2x2 * 2x2)
    # B21 is rows 2-3, cols 0-1.
    # V indices: (2+j)*5 + k for j in 0-1, k in 0-1.
    # W indices: k*2 + i for k in 0-1, i in 0-1.
    
    U_s, V_s, W_s = get_strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2 + (2+j), col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V_full[(2+j)*5 + k, col_offset + r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W_full[k*2 + i, col_offset + r] = W_s[k*2+i, r]
    
    col_offset += 7
    
    # Subpart 2.2: A2 * B22 (2x2 * 2x2)
    # B22 is rows 2-3, cols 2-3.
    # V indices: (2+j)*5 + (2+k) for j in 0-1, k in 0-1.
    # W indices: (2+k)*2 + i for k in 0-1, i in 0-1.
    
    U_s, V_s, W_s = get_strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2 + (2+j), col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, r] != 0:
                    V_full[(2+j)*5 + (2+k), col_offset + r] = V_s[j*2+k, r]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, r] != 0:
                    W_full[(2+k)*2 + i, col_offset + r] = W_s[k*2+i, r]
    
    col_offset += 7
    
    # Subpart 2.3: A2 * B23 (2x2 * 2x1)
    # B23 is rows 2-3, col 4.
    # V indices: (2+j)*5 + 4 for j in 0-1.
    # W indices: 4*2 + i for i in 0-1.
    
    U_s, V_s, W_s = get_schoolbook_221()
    for r in range(2):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, r] != 0:
                    U_full[i*2 + (2+j), col_offset + r] = U_s[i*2+j, r]
        for j in range(2):
            if V_s[j, r] != 0:
                V_full[(2+j)*5 + 4, col_offset + r] = V_s[j, r]
        for i in range(2):
            if W_s[i, r] != 0:
                W_full[4*2 + i, col_offset + r] = W_s[i, r]
    
    col_offset += 2
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        U_t = U.reshape(2, 4, rank)
        V_t = V.reshape(4, 5, rank)
        W_t = W.reshape(5, 2, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_target = np.ones((2, 4, 5))
        return np.allclose(T, T_target, atol=1e-6)
    
    if verify(U_full, V_full, W_full, 32):
        U_list = [[float(x) for x in row] for row in U_full]
        V_list = [[float(x) for x in row] for row in V_full]
        W_list = [[float(x) for x in row] for row in W_full]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = 40
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(10)]
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
