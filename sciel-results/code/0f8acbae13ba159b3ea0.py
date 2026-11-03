"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 4, 4, 6  # 4x4 * 4x6 -> 4x6
    
    # We use Kronecker product: (2x2x2) ⊗ (2x2x3)
    # 2x2x2 tensor (Strassen) has rank 7
    # 2x2x3 tensor has rank 5
    # Total rank = 7 * 5 = 35
    
    # First, define the 2x2x2 Strassen decomposition
    # For multiplying 2x2 * 2x2 -> 2x2
    # U1 shape: (2*2, 7) = (4, 7)
    # V1 shape: (2*2, 7) = (4, 7)
    # W1 shape: (2*2, 7) = (4, 7)
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indexing: A[i,j] -> index i*2+j, B[i,j] -> index i*2+j, C[i,j] -> index i*2+j
    # For the tensor: T[i,j,k] = 1 if output C[i,k] uses A[i,j]*B[j,k]
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 where i is row of A, j is col of A/row of B, k is col of B
    
    # U1[i*2+j, c] = coefficient of A[i,j] in term c
    # V1[j*2+k, c] = coefficient of B[j,k] in term c
    # W1[k*2+i, c] = coefficient of C[i,k] in term c (note: W[k*n+i] convention)
    
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # Term 0: M1 = (A11 + A22)(B11 + B22)
    U1[0, 0] = 1; U1[3, 0] = 1  # A11, A22
    V1[0, 0] = 1; V1[3, 0] = 1  # B11, B22
    W1[0, 0] = 1; W1[3, 0] = 1  # C11, C22 (W[k*2+i]: C11->W[0], C22->W[3])
    
    # Term 1: M2 = (A21 + A22)B11
    U1[2, 1] = 1; U1[3, 1] = 1  # A21, A22
    V1[0, 1] = 1  # B11
    W1[2, 1] = 1  # C21 (W[k*2+i] for C[2,1]: k=1,i=2 -> W[1*2+2]=W[4])
    # Wait, let me re-examine the W convention: W[k*n+i, r]
    # For n=2: W[k*2+i, r]
    # C[0,0] -> k=0,i=0 -> W[0]
    # C[0,1] -> k=1,i=0 -> W[1]
    # C[1,0] -> k=0,i=1 -> W[2]
    # C[1,1] -> k=1,i=1 -> W[3]
    
    # Let me redo this carefully.
    # C[i,k] where i is row (0,1), k is col (0,1)
    # W index = k*n + i = k*2 + i
    # C[0,0]: k=0,i=0 -> W[0]
    # C[0,1]: k=1,i=0 -> W[1]
    # C[1,0]: k=0,i=1 -> W[2]
    # C[1,1]: k=1,i=1 -> W[3]
    
    # Strassen:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me rebuild:
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # Term 0: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # A indices: A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    # B indices: B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1  # C[0,0]
    W1[3, 0] = 1  # C[1,1]
    
    # Term 1: M2 = (A[1,0] + A[1,1]) * B[0,0]
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1  # C[1,0]
    
    # Term 2: M3 = A[0,0] * (B[1,0] - B[1,1])
    U1[0, 2] = 1
    V1[2, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1  # C[0,1]
    
    # Term 3: M4 = A[1,1] * (B[0,0] - B[0,1])
    U1[3, 3] = 1
    V1[0, 3] = 1; V1[1, 3] = -1
    W1[0, 3] = 1  # C[0,0]
    W1[2, 3] = 1  # C[1,0]
    
    # Term 4: M5 = (A[0,0] + A[0,1]) * B[1,1]
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1  # C[0,0]
    W1[1, 4] = 1   # C[0,1]
    
    # Term 5: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1  # C[1,1]
    
    # Term 6: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1  # C[0,0]
    
    # Now define the 2x2x3 tensor decomposition
    # Multiplying 2x2 * 2x3 -> 2x3
    # U2 shape: (2*2, 5) = (4, 5)
    # V2 shape: (2*3, 5) = (6, 5)
    # W2 shape: (3*2, 5) = (6, 5)
    
    # For 2x2 * 2x3:
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2
    
    # Known rank-5 decomposition for 2x2x3:
    # This is less standard. Let me use a known result.
    # Actually, the rank of 2x2x3 is 5.
    
    # One decomposition:
    # M1 = A[0,0] * (B[0,0] + B[0,1] + B[0,2])
    # M2 = A[1,1] * (B[1,0] + B[1,1] + B[1,2])
    # M3 = (A[0,0] + A[1,0]) * (B[1,0] - B[0,0])
    # M4 = (A[0,1] + A[1,1]) * (B[1,1] - B[0,1])
    # M5 = (A[0,0] + A[0,1]) * (B[1,2] - B[0,2])
    
    # C[0,0] = M1 - M3
    # C[0,1] = M1 - M4
    # C[0,2] = M1 - M5
    # C[1,0] = M2 + M3
    # C[1,1] = M2 + M4
    # C[1,2] = M2 + M5
    
    # Let me verify:
    # C[0,0] = A[0,0]*B[0,0]
    # M1 - M3 = A[0,0]*(B[0,0]+B[0,1]+B[0,2]) - (A[0,0]+A[1,0])*(B[1,0]-B[0,0])
    # This doesn't look right. Let me use a different known decomposition.
    
    # Actually, let me just use the schoolbook for 2x2x3 which has rank 6,
    # and see if I can find a rank-5. But for now, let me try rank 6.
    # Then total rank would be 7*6 = 42.
    
    # But the strategy says rank 5 for 2x2x3. Let me look up the correct one.
    
    # Known rank-5 for 2x2x3:
    # Using the fact that it's related to the secant variety.
    
    # Let me use this decomposition (from literature):
    # M1 = A[0,0]*(B[0,0] + B[1,0]) + A[1,0]*(B[0,1] + B[1,1])
    # This is getting complex. Let me just use schoolbook rank 6 for now.
    
    # Actually, I recall that 2x2x3 has rank 5. Here's a construction:
    # Think of B as two 2x1 vectors side by side with a third column.
    
    # Let me use a simpler approach: since we need integer/half-integer entries,
    # and the Kronecker product of rank 7 and rank 6 gives rank 42,
    # let me first verify the 2x2x2 Strassen works, then use schoolbook for 2x2x3.
    
    # For 2x2x3 schoolbook (rank 6):
    U2 = np.zeros((4, 6), dtype=float)
    V2 = np.zeros((6, 6), dtype=float)
    W2 = np.zeros((6, 6), dtype=float)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i*2+j, col] = 1
                V2[j*3+k, col] = 1
                W2[k*2+i, col] = 1  # W[k*n+i] for n=2
                col += 1
    
    # Now form the Kronecker product
    # The 4x4x6 tensor = (2x2x2) ⊗ (2x2x3)
    # But wait, we need to be careful about the dimensions.
    
    # 4x4 * 4x6:
    # A is 4x4, B is 4x6, C is 4x6
    # We can view A as a 2x2 block matrix of 2x2 blocks
    # B as a 2x2 block matrix of 2x3 blocks
    # C as a 2x2 block matrix of 2x3 blocks
    
    # So the tensor is indeed (2x2x2) ⊗ (2x2x3)
    
    # U has shape (4*4, rank) = (16, 35) if using rank 5 for 2x2x3, or (16, 42) if rank 6
    # Let me use rank 6 for 2x2x3 for now, giving total rank 42.
    
    rank1 = 7
    rank2 = 6
    total_rank = rank1 * rank2  # 42
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((24, total_rank), dtype=float)
    W = np.zeros((24, total_rank), dtype=float)
    
    # Kronecker product of factors
    # U = U1 ⊗ U2 (with appropriate reshaping)
    # The indices: for 4x4, index = (i1*2+i2)*(4) + (j1*2+j2) where i1,j1 are block indices, i2,j2 are within-block
    # Actually, let me think about this differently.
    
    # For the Kronecker product of tensors:
    # If T1 has indices (a1, b1, c1) and T2 has indices (a2, b2, c2),
    # then T1⊗T2 has indices (a1*a2, b1*b2, c1*c2)
    
    # For 2x2x2: a1=2 (rows of A1), b1=2 (cols of A1/rows of B1), c1=2 (cols of B1)
    # For 2x2x3: a2=2 (rows of A2), b2=2 (cols of A2/rows of B2), c2=3 (cols of B2)
    
    # Combined: a=4, b=4, c=6
    # A index: a1*a2 + a2 = a1*2 + a2 (row of 4x4 A)
    # B index: b1*b2 + b2 = b1*2 + b2 (col of 4x4 A / row of 4x6 B)
    # C index: c1*c2 + c2 = c1*3 + c2 (col of 4x6 B / col of 4x6 C)
    
    # U1 index: a1*2 + b1 (for 2x2 A1)
    # U2 index: a2*2 + b2 (for 2x2 A2)
    # U index: (a1*2+b1)*4 + (a2*2+b2) = (a1*2+b1)*4 + a2*2 + b2
    
    # Let me just compute it directly:
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            # U: U1[i1, r1] * U2[i2, r2] where i1 is index in U1, i2 in U2
            # U1 index i1 = a1*2 + b1, U2 index i2 = a2*2 + b2
            # U index = (a1*2+b1)*4 + (a2*2+b2)
            for i1 in range(4):
                for i2 in range(4):
                    u_val = U1[i1, r1] * U2[i2, r2]
                    if u_val != 0:
                        # i1 = a1*2+b1, i2 = a2*2+b2
                        # U index = i1*4 + i2
                        U[i1*4 + i2, r] = u_val
            
            # V: V1[j1, r1] * V2[j2, r2]
            # V1 index j1 = b1*2 + c1, V2 index j2 = b2*3 + c2
            # V index = j1*6 + j2 = (b1*2+c1)*6 + (b2*3+c2)
            for j1 in range(4):
                for j2 in range(6):
                    v_val = V1[j1, r1] * V2[j2, r2]
                    if v_val != 0:
                        V[j1*6 + j2, r] = v_val
            
            # W: W1[k1, r1] * W2[k2, r2]
            # W1 index k1 = c1*2 + i1, W2 index k2 = c2*2 + i2
            # W index = k1*6 + k2 = (c1*2+i1)*6 + (c2*2+i2)
            for k1 in range(4):
                for k2 in range(6):
                    w_val = W1[k1, r1] * W2[k2, r2]
                    if w_val != 0:
                        W[k1*6 + k2, r] = w_val
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
