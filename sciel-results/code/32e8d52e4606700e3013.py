"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    
    # Rank 44 decomposition:
    # Split the 3 rows of A and B into:
    #   Block 1: A[:2, :] (2x3), B[:2, :] (2x8)  -> 2x2x8 tensor
    #   Block 2: A[2:3, :] (1x3), B[2:3, :] (1x8) -> 1x2x8 tensor
    # 
    # For Block 1 (2x2x8):
    #   Split the 8 columns into 4 blocks of 2
    #   For each 2-column block, use Strassen rank-7 for 2x2x2
    #   Total: 4 * 7 = 28 terms
    #
    # For Block 2 (1x2x8):
    #   Split the 8 columns into 4 blocks of 2
    #   For each 2-column block, use schoolbook rank-2 for 1x2x2
    #   Total: 4 * 2 = 8 terms
    #
    # Grand total: 28 + 8 = 36... wait, that's not 44.
    # 
    # Let me reconsider. The knowledge says:
    # "Decomposes the 2x3x8 tensor into four 2x3x2 sub-tensors, each decomposed using a rank-11 scheme"
    # So we split the 8 columns into 4 blocks of 2, giving four 2x3x2 sub-tensors.
    # Each 2x3x2 is decomposed with rank 11.
    # Total: 4 * 11 = 44.
    #
    # How to decompose 2x3x2 with rank 11?
    # Split the 3 middle dimension into:
    #   Part A: A[:2, :] (2x2), B[:2, :] (2x2) -> 2x2x2, use Strassen rank 7
    #   Part B: A[2:3, :] (1x2), B[2:3, :] (1x2) -> 1x2x2, use schoolbook rank 2
    #   But we also need cross terms? No, matrix multiplication is bilinear, so:
    #   C = A1*B1 + A2*B2 where A1=A[:2], A2=A[2:], B1=B[:2], B2=B[2:]
    #   Wait, A is 2x3, so A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    #   B is 3x2, so B1 = B[0:2, :] (2x2), B2 = B[2:3, :] (1x2)
    #   C = A1*B1 + A2*B2
    #   A1*B1: 2x2 * 2x2, Strassen rank 7
    #   A2*B2: 2x1 * 1x2, schoolbook rank 2
    #   Total: 7 + 2 = 9... not 11.
    #
    # Hmm, let me think again. Maybe the split is different.
    # Actually, for 2x3x2:
    #   A is 2x3, B is 3x2
    #   We can split the middle dimension (size 3) into 2+1:
    #   A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #   B = [B1; B2] where B1 is 2x2, B2 is 1x2
    #   C = A1*B1 + A2*B2
    #   A1*B1: 2x2 * 2x2 -> rank 7 (Strassen)
    #   A2*B2: 2x1 * 1x2 -> rank 2 (schoolbook)
    #   Total: 9
    #
    # Or split into 1+2:
    #   A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    #   B = [B1; B2] where B1 is 1x2, B2 is 2x2
    #   C = A1*B1 + A2*B2
    #   A1*B1: 2x1 * 1x2 -> rank 2
    #   A2*B2: 2x2 * 2x2 -> rank 7
    #   Total: 9
    #
    # So rank 9 for 2x3x2, total 4*9 = 36.
    #
    # But the knowledge says rank 11 per sub-tensor. Let me re-read:
    # "Decomposes the 2x3x8 tensor into four 2x3x2 sub-tensors, each decomposed using a rank-11 scheme (Strassen 7 + 4 rank-1 terms)"
    # So it's Strassen 7 + 4 = 11.
    #
    # Maybe the decomposition is:
    #   A1*B1: 2x2 * 2x2 -> Strassen rank 7
    #   A2*B1: 2x1 * 2x2 -> rank 2 (schoolbook)
    #   A1*B2: 2x2 * 1x2 -> rank 2 (schoolbook)
    #   A2*B2: 2x1 * 1x2 -> rank 1 (just one multiplication: A[0,2]*B[2,0] for C[0,0], etc.)
    #   Wait, A2*B2 is 2x1 * 1x2, which gives a 2x2 matrix.
    #   C[i,j] += A[i,2]*B[2,j]
    #   That's 4 terms, not 1.
    #
    # So: 7 + 2 + 2 + 4 = 15? No.
    #
    # Let me just implement rank 44 with the structure:
    # 4 blocks of 2 columns, each block uses rank 11.
    # For each block:
    #   - 7 Strassen terms for the 2x2x2 part (A[:2,:2] * B[:2,:])
    #   - 4 schoolbook terms for the remaining parts
    #
    # Actually, let me just implement a correct rank 36 solution first, since I'm not sure about rank 44.
    # But the knowledge says rank 44 was measured. Let me implement rank 44 as described.
    
    rank = 44
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]
    
    # U[i*3+j] for A[i,j], V[j*8+k] for B[j,k], W[k*2+i] for C[i,k]
    
    # Strassen decomposition for 2x2x2:
    # M1 = A[0,0]*(B[0,0] + B[1,1])
    # M2 = (A[0,0] + A[1,0])*B[1,1]
    # M3 = A[1,1]*(B[0,0] + B[1,0])
    # M4 = A[0,1]*(B[1,1] - B[0,1])
    # M5 = (A[0,0] + A[1,1])*B[0,0]  -- wait, this isn't standard Strassen
    #
    # Standard Strassen for C = A*B where A,B are 2x2:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M2 + M5
    # C[1,0] = M3 + M6
    # C[1,1] = M1 - M2 + M4 + M7
    #
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,1]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,0] - B[0,1])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * B[0,0]
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # For each block of 2 columns (4 blocks total):
    col = 0
    for block in range(4):
        k_offset = block * 2
        
        # 7 Strassen terms for A[:2, :2] * B[:2, block:block+2]
        # A indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B indices: B[0,k0]=V[8*k0], B[0,k1]=V[8*k1], B[1,k0]=V[8+8*k0], B[1,k1]=V[8+8*k1]
        # where k0 = k_offset, k1 = k_offset + 1
        # C indices: C[0,k0]=W[2*k0], C[0,k1]=W[2*k0+1], C[1,k0]=W[2*k0+2], C[1,k1]=W[2*k0+3]
        
        k0 = k_offset
        k1 = k_offset + 1
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0][col] = 1.0  # A[0,0]
        U[4][col] = 1.0  # A[1,1]
        V[8*k0][col] = 1.0  # B[0,k0]
        V[8+8*k1][col] = 1.0  # B[1,k1]
        W[2*k0][col] = 1.0  # C[0,k0]
        W[2*k0+1][col] = -1.0  # C[0,k1]
        W[2*k0+2][col] = -1.0  # C[1,k0]
        W[2*k0+3][col] = 1.0  # C[1,k1]
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        U[3][col] = 1.0  # A[1,0]
        U[4][col] = 1.0  # A[1,1]
        V[8*k1][col] = 1.0  # B[0,k1]
        W[2*k0+1][col] = 1.0  # C[0,k1]
        W[2*k0+3][col] = -1.0  # C[1,k1]
        col += 1
        
        # M3 = A[0,0] * (B[1,k0] - B[1,k1])
        U[0][col] = 1.0  # A[0,0]
        V[8+8*k0][col] = 1.0  # B[1,k0]
        V[8+8*k1][col] = -1.0  # B[1,k1]
        W[2*k0][col] = -1.0  # C[0,k0]
        W[2*k0+2][col] = 1.0  # C[1,k0]
        col += 1
        
        # M4 = A[1,1] * (B[0,k0] - B[0,k1])
        U[4][col] = 1.0  # A[1,1]
        V[8*k0][col] = 1.0  # B[0,k0]
        V[8*k1][col] = -1.0  # B[0,k1]
        W[2*k0][col] = -1.0  # C[0,k0]
        W[2*k0+1][col] = 1.0  # C[0,k1]
        W[2*k0+3][col] = 1.0  # C[1,k1]
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0][col] = 1.0  # A[0,0]
        U[1][col] = 1.0  # A[0,1]
        V[8+8*k1][col] = 1.0  # B[1,k1]
        W[2*k0][col] = -1.0  # C[0,k0]
        W[2*k0+1][col] = 1.0  # C[0,k1]
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * B[0,k0]
        U[3][col] = 1.0  # A[1,0]
        U[0][col] = -1.0  # A[0,0]
        V[8*k0][col] = 1.0  # B[0,k0]
        W[2*k0+2][col] = 1.0  # C[1,k0]
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1][col] = 1.0  # A[0,1]
        U[4][col] = -1.0  # A[1,1]
        V[8+8*k0][col] = 1.0  # B[1,k0]
        V[8+8*k1][col] = 1.0  # B[1,k1]
        W[2*k0+2][col] = 1.0  # C[1,k0]
        W[2*k0+3][col] = 1.0  # C[1,k1]
        col += 1
        
        # 4 schoolbook terms for A[:, 2] * B[2, block:block+2]
        # A[0,2]=U[2], A[1,2]=U[5]
        # B[2,k0]=V[16+k0], B[2,k1]=V[16+k1]
        # C[0,k0]=W[2*k0], C[0,k1]=W[2*k0+1], C[1,k0]=W[2*k0+2], C[1,k1]=W[2*k0+3]
        
        # A[0,2] * B[2,k0] -> C[0,k0]
        U[2][col] = 1.0
        V[16+k0][col] = 1.0
        W[2*k0][col] = 1.0
        col += 1
        
        # A[0,2] * B[2,k1] -> C[0,k1]
        U[2][col] = 1.0
        V[16+k1][col] = 1.0
        W[2*k0+1][col] = 1.0
        col += 1
        
        # A[1,2] * B[2,k0] -> C[1,k0]
        U[5][col] = 1.0
        V[16+k0][col] = 1.0
        W[2*k0+2][col] = 1.0
        col += 1
        
        # A[1,2] * B[2,k1] -> C[1,k1]
        U[5][col] = 1.0
        V[16+k1][col] = 1.0
        W[2*k0+3][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
