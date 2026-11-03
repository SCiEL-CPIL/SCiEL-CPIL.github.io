"""Program to find a rank-minimal decomposition for 5x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build rank-49 decomposition for 4x4 using recursive Strassen
    # Strassen 2x2: rank 7
    # 4x4 = two levels of Strassen: rank 7^2 = 49
    
    def build_strassen_2x2():
        """Build U, V, W for 2x2 matrix multiplication (rank 7).
        U: (2*2, 7) = (4, 7), indexed by (i*2+j)
        V: (2*2, 7) = (4, 7), indexed by (j*2+k)
        W: (2*2, 7) = (4, 7), indexed by (k*2+i)
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # P1 = (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1  # C11 += P1, C22 += P1
        
        # P2 = (A21+A22)B11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1  # C21 += P2, C22 -= P2
        
        # P3 = A11(B12-B22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1  # C12 += P3, C22 += P3
        
        # P4 = A22(B21-B11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1  # C11 += P4, C21 += P4
        
        # P5 = (A11+A12)B22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[0,4] = -1  # C12 += P5, C11 -= P5
        
        # P6 = (A21-A11)(B11+B12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1  # C22 += P6
        
        # P7 = (A12-A22)(B21+B22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1  # C11 += P7
        
        return U, V, W
    
    def expand_strassen(U_small, V_small, W_small, size_small):
        """Expand a size_small x size_small Strassen decomposition to size_small*2.
        U_small: (size_small^2, rank)
        V_small: (size_small^2, rank)
        W_small: (size_small^2, rank)
        Returns U, V, W of shape ((2*size_small)^2, rank^2)
        """
        s = size_small
        new_size = 2 * s
        rank_small = U_small.shape[1]
        rank_new = rank_small * rank_small
        
        U = np.zeros((new_size * new_size, rank_new), dtype=float)
        V = np.zeros((new_size * new_size, rank_new), dtype=float)
        W = np.zeros((new_size * new_size, rank_new), dtype=float)
        
        for r1 in range(rank_small):
            for r2 in range(rank_small):
                r = r1 * rank_small + r2
                # U: for each (i,j) in the 2x2 block structure
                # U_small[i*s+j, r1] tells us contribution to A-block(i,j)
                # We need to map to the 4x4 index
                for i1 in range(s):
                    for j1 in range(s):
                        val = U_small[i1 * s + j1, r1]
                        if val != 0:
                            # This contributes to A[i1, j1] in the small problem
                            # In the 2x2 block structure, A[i1,j1] is part of block
                            # Actually, we need to think of this differently.
                            pass
        
        # Let me redo this. The recursive Strassen works as follows:
        # For 4x4, we split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        #
        # Strassen gives 7 products P1..P7 of 2x2 matrices
        # Each 2x2 product is computed using the 2x2 Strassen (rank 7)
        #
        # So for each of the 7 Strassen products, we have 7 sub-products
        # Total rank = 7 * 7 = 49
        #
        # U for 4x4: index (i*4+j, r) where r = r1*7 + r2
        # U[i*4+j, r1*7+r2] = U_2x2[i_small*2+j_small, r1] * U_2x2[i_sub*2+j_sub, r2]
        # where i = i_small*2 + i_sub, j = j_small*2 + j_sub
        
        U = np.zeros((new_size * new_size, rank_new), dtype=float)
        V = np.zeros((new_size * new_size, rank_new), dtype=float)
        W = np.zeros((new_size * new_size, rank_new), dtype=float)
        
        for r1 in range(rank_small):
            for r2 in range(rank_small):
                r = r1 * rank_small + r2
                for i1 in range(s):
                    for j1 in range(s):
                        u1 = U_small[i1 * s + j1, r1]
                        if u1 == 0:
                            continue
                        for i2 in range(s):
                            for j2 in range(s):
                                u2 = U_small[i2 * s + j2, r2]
                                if u2 == 0:
                                    continue
                                i = i1 * s + i2
                                j = j1 * s + j2
                                U[i * new_size + j, r] = u1 * u2
        
        for r1 in range(rank_small):
            for r2 in range(rank_small):
                r = r1 * rank_small + r2
                for j1 in range(s):
                    for k1 in range(s):
                        v1 = V_small[j1 * s + k1, r1]
                        if v1 == 0:
                            continue
                        for j2 in range(s):
                            for k2 in range(s):
                                v2 = V_small[j2 * s + k2, r2]
                                if v2 == 0:
                                    continue
                                j = j1 * s + j2
                                k = k1 * s + k2
                                V[j * new_size + k, r] = v1 * v2
        
        for r1 in range(rank_small):
            for r2 in range(rank_small):
                r = r1 * rank_small + r2
                for k1 in range(s):
                    for i1 in range(s):
                        w1 = W_small[k1 * s + i1, r1]
                        if w1 == 0:
                            continue
                        for k2 in range(s):
                            for i2 in range(s):
                                w2 = W_small[k2 * s + i2, r2]
                                if w2 == 0:
                                    continue
                                k = k1 * s + k2
                                i = i1 * s + i2
                                W[k * new_size + i, r] = w1 * w2
        
        return U, V, W
    
    # Build 2x2 Strassen
    U2, V2, W2 = build_strassen_2x2()
    
    # Expand to 4x4
    U4, V4, W4 = expand_strassen(U2, V2, W2, 2)
    
    # Now extend to 5x5
    # The 4x4 decomposition covers indices 0-3
    # We need to add terms for index 4
    
    rank_4x4 = U4.shape[1]  # 49
    
    # Additional rank-1 terms for the 5th dimension
    # Terms needed:
    # 1. For i in 0-3, j in 0-3: A[i,j]*B[j,4] -> C[i,4] : 16 terms
    # 2. For j in 0-3, k in 0-3: A[4,j]*B[j,k] -> C[4,k] : 16 terms  
    # 3. For j in 0-3: A[4,j]*B[j,4] -> C[4,4] : 4 terms
    # 4. A[4,4]*B[4,4] -> C[4,4] : 1 term
    # Total additional: 37 terms
    
    # But wait - can we be smarter? Let me just add these as rank-1 terms.
    # Actually, some of these might be redundant or combinable, but let's start simple.
    
    num_extra = 16 + 16 + 4 + 1  # 37
    total_rank = rank_4x4 + num_extra
    
    U5 = np.zeros((25, total_rank), dtype=float)
    V5 = np.zeros((25, total_rank), dtype=float)
    W5 = np.zeros((25, total_rank), dtype=float)
    
    # Copy the 4x4 part
    U5[:16, :rank_4x4] = U4
    V5[:16, :rank_4x4] = V4
    W5[:16, :rank_4x4] = W4
    
    col = rank_4x4
    
    # Type 1: A[i,j]*B[j,4] -> C[i,4] for i,j in 0-3
    for i in range(4):
        for j in range(4):
            U5[i * 5 + j, col] = 1
            V5[j * 5 + 4, col] = 1
            W5[4 * 5 + i, col] = 1
            col += 1
    
    # Type 2: A[4,j]*B[j,k] -> C[4,k] for j,k in 0-3
    for j in range(4):
        for k in range(4):
            U5[4 * 5 + j, col] = 1
            V5[j * 5 + k, col] = 1
            W5[k * 5 + 4, col] = 1
            col += 1
    
    # Type 3: A[4,j]*B[j,4] -> C[4,4] for j in 0-3
    for j in range(4):
        U5[4 * 5 + j, col] = 1
        V5[j * 5 + 4, col] = 1
        W5[4 * 5 + 4, col] = 1
        col += 1
    
    # Type 4: A[4,4]*B[4,4] -> C[4,4]
    U5[4 * 5 + 4, col] = 1
    V5[4 * 5 + 4, col] = 1
    W5[4 * 5 + 4, col] = 1
    col += 1
    
    # Convert to lists for the output
    U_list = [[U5[i, j] for j in range(total_rank)] for i in range(25)]
    V_list = [[V5[i, j] for j in range(total_rank)] for i in range(25)]
    W_list = [[W5[i, j] for j in range(total_rank)] for i in range(25)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
