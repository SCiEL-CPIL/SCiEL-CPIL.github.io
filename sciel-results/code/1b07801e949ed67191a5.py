"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Try to construct a rank-49 decomposition
    # Using 7 Strassen-like 2x2x2 blocks but with overlap optimization
    # Actually, let's try rank 49: 7 blocks of 7 terms each
    
    # Better approach: use the structure that n=2
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can group columns in pairs and use Strassen on 2x2x2
    # For 7 columns: 3 pairs + 1 single
    # 3 Strassen blocks (21 terms) + 1 single column (8 terms) = 29? No that's wrong
    
    # Let me reconsider: for 2x4x7, split into:
    # Columns 0-3: 2x4x4 = 2x(2x2x2) = 2 Strassen blocks = 14 terms
    # Columns 4-5: 2x4x2 = 1x(2x2x2) = 1 Strassen block = 7 terms  
    # Column 6: 2x4x1 = 8 terms
    # Total: 14 + 7 + 8 = 29... but that's not right either
    
    # Wait, the current code uses 6 Strassen blocks = 42 + 8 = 50
    # Let me recount: add_strassen does 7 terms per call
    # 6 calls * 7 = 42, plus 8 = 50. Yes.
    
    # The issue: for 2x4x4, we need 2 Strassen blocks (one for cols 0-1, one for 2-3) = 14
    # For 2x4x2, we need 1 Strassen block = 7
    # For 2x4x1, we need 8
    # Total should be 14 + 7 + 8 = 29, but code does 50
    
    # I see the error in my understanding. Let me re-examine.
    # The code calls add_strassen 6 times for the first 5 columns?
    # add_strassen(0,0,0), (0,0,2), (0,2,0), (0,2,2) -> 4 blocks for cols 0-3
    # add_strassen(0,0,4), (0,2,4) -> 2 blocks for cols 4-5
    # That's 6 blocks * 7 = 42, + 8 = 50
    
    # But 4 blocks for 2x4x4 means: each block handles 2x2x2, so 4 blocks = 2x4x4. Correct.
    # 2 blocks for 2x4x2: each 2x2x2, so 2 blocks = 2x4x2. Correct.
    
    # So the math checks out. Can we do better than 50?
    
    # Known result: rank(M_{2,4,7}) - let's try to find it
    # Lower bound: rank >= m*p = 28? No, that's not a valid lower bound.
    # The valid lower bound is rank >= max(n*m, m*p, p*n) is NOT standard.
    # Actually for matrix multiplication, rank >= m*min(n,p) is not right either.
    
    # Let me just try to construct rank 49 directly.
    # Use 6 Strassen blocks (42 terms) + 7 terms for last column (instead of 8)
    # Can we do the last column (2x4x1) in 7 terms instead of 8?
    
    # 2x4x1: C[0,6] = sum_j A[0,j]*B[j,6], C[1,6] = sum_j A[1,j]*B[j,6]
    # This is rank-8 tensor in general, but maybe we can share with adjacent blocks.
    
    # Alternative: use a different block structure.
    # What if we do 2x4x7 as overlapping blocks?
    
    # Let me try: rank 49 using 7 blocks of 7 terms
    # Block structure: treat as 2x4x7 with 7 "strips"
    
    # Actually, let me try a simpler optimization:
    # Combine the last Strassen block with the direct terms
    
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Standard Strassen 2x2x2
        # M1 = (A11+A22)(B11+B22)
        U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M2 = (A21+A22)B11
        U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
        col += 1
        # M3 = A11(B12-B22)
        U[i0*m+j0, col] += 1
        V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
        W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M4 = A22(B21-B11)
        U[(i0+1)*m+(j0+1), col] += 1
        V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+(i0+1), col] += 1; W[k0*n+i0, col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
        V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
        W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
        V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1
        col += 1
    
    # 4 Strassen blocks for columns 0-3
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # 1 Strassen block for columns 4-5
    add_strassen(0, 0, 4)
    
    # For column 6, use 7 terms instead of 8
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6] + A[0,2]*B[2,6] + A[0,3]*B[3,6]
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6] + A[1,2]*B[2,6] + A[1,3]*B[3,6]
    # We can use 7 terms: group (A[0,0]+A[1,1]), (A[1,0]+A[0,1]), etc.
    # But we need both C[0,6] and C[1,6]
    
    # Let's use:
    # T1 = (A[0,0]+A[1,1]) * B[0,6]
    # T2 = (A[1,0]-A[0,0]) * B[0,6]  
    # T3 = A[1,1] * B[1,6]
    # T4 = A[0,2] * B[2,6]
    # T5 = A[1,2] * B[2,6]
    # T6 = A[0,3] * B[3,6]
    # T7 = A[1,3] * B[3,6]
    # This gives C[0,6] = T1 + T3 + T4 + T6 - T2... let me verify
    
    # Actually, simpler: just use 8 terms for now, total rank 50
    # Let me try rank 49 by merging one term
    
    # Use 6 Strassen blocks = 42, then 7 for last column = 49
    # For last column (k=6), 2x4x1 in 7 terms:
    # This is a 2x4x1 tensor. Rank is 8 in general for distinct entries.
    # But we can use 7 if we allow one linear combination.
    
    # Let me just output rank 50 for safety
    # Actually let me try rank 49 with a direct 7-term last column
    
    # 2x4x1 in 7 terms:
    # U terms: (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,1,0,0), (0,0,1,1), (1,0,0,1)
    # That's 7 U vectors. Need corresponding V and W.
    # C[0,6] = sum_j A[0,j]B[j,6]
    # C[1,6] = sum_j A[1,j]B[j,6]
    
    # Term 1: A[0,0]*B[0,6] -> W[6,0]
    # Term 2: A[1,0]*B[0,6] -> W[6,1]  
    # Term 3: A[0,1]*B[1,6] -> W[6,0]
    # Term 4: A[1,1]*B[1,6] -> W[6,1]
    # Term 5: A[0,2]*B[2,6] -> W[6,0]
    # Term 6: A[1,2]*B[2,6] -> W[6,1]
    # Term 7: (A[0,3]+A[1,3])*B[3,6] -> contributes to both W[6,0] and W[6,1]
    # But we need C[0,6] += A[0,3]*B[3,6] and C[1,6] += A[1,3]*B[3,6]
    # With term 7: U=(0,0,0,1,0,0,0,1), V=(0,0,0,1), W=(1,1)
    # Then C[0,6] gets A[0,3]*B[3,6]*1 + A[1,3]*B[3,6]*1 - A[1,3]*B[3,6] (from correction)
    # This doesn't work cleanly.
    
    # Let me just stick with rank 50 which is verified
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    # col should be 49 now (42 + 7)... wait, 42 + 8 = 50
    # But rank is 49, so we overflow. Let me fix.
    
    # Restart with rank 50
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
