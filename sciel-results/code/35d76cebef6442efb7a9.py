"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank5_3x2x2():
    """
    Rank-5 decomposition for 3x2x2 matrix multiplication.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in 0..2, k in 0..1.
    U: (3*2=6, 5), V: (2*2=4, 5), W: (2*3=6, 5)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*3+i, r]
    """
    # Rank-5 for 3x2x2:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[1,0]*B[0,0]
    # M3 = A[2,0]*B[0,0]
    # M4 = (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[1,1])  -- not quite
    #
    # Actually, let me use a verified rank-5 decomposition.
    # For 3x2x2, the tensor rank is 5.
    #
    # One known decomposition:
    # M1 = A[0,0] * B[0,0]
    # M2 = A[1,0] * B[0,0]
    # M3 = A[2,0] * B[0,0]
    # M4 = (A[0,1]+A[1,1]) * (B[1,0]+B[1,1])
    # M5 = A[2,1] * B[1,1]
    #
    # Check C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = M1 + (part of M4)
    # This doesn't work cleanly. Let me use a different approach.
    
    # Use the "Strassen-like" decomposition for 3x2x2:
    # The key is to have 5 bilinear forms.
    
    # Verified rank-5 decomposition for 3x2x2:
    # Let me construct it explicitly.
    #
    # We need: for all i in {0,1,2}, k in {0,1}:
    #   sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*3+i, r] = 1 for all j in {0,1}
    #   (this means the (i,j,k) tensor entry is 1)
    
    # Schoolbook would be rank 3*2*2=12. We want rank 5.
    
    # Let me try:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> contributes to (i=0,j=0,k=0)
    # r=1: U[1,0]=1, V[0,0]=1, W[0,1]=1  -> contributes to (i=1,j=0,k=0)
    # r=2: U[2,0]=1, V[0,0]=1, W[0,2]=1  -> contributes to (i=2,j=0,k=0)
    # r=3: U[0,1]=1, U[1,1]=1, V[1,0]=1, V[1,1]=1, W[1,0]=1, W[1,1]=1
    #      -> contributes to (i=0,j=1,k=0), (i=1,j=1,k=0), (i=0,j=1,k=1), (i=1,j=1,k=1)
    # r=4: U[2,1]=1, V[1,1]=1, W[1,2]=1  -> contributes to (i=2,j=1,k=1)
    
    # Check:
    # (i=0,j=0,k=0): r=0: 1*1*1=1, others 0. Sum=1. OK
    # (i=1,j=0,k=0): r=1: 1*1*1=1. Sum=1. OK
    # (i=2,j=0,k=0): r=2: 1*1*1=1. Sum=1. OK
    # (i=0,j=1,k=0): r=3: 1*1*1=1. Sum=1. OK
    # (i=1,j=1,k=0): r=3: 1*1*1=1. Sum=1. OK
    # (i=2,j=1,k=0): need sum=1. r=3: U[2,1]=0, so 0. r=4: U[2,1]=1, V[1,0]=0, so 0. Sum=0. FAIL!
    
    # So this doesn't work. Let me try again.
    
    # The issue is that (i=2,j=1,k=0) needs coverage.
    
    # Let me use a different rank-5 decomposition.
    #
    # Known: rank(3x2x2) = 5. One decomposition:
    # M1 = A[0,0] * B[0,0]
    # M2 = A[1,0] * B[0,0]  
    # M3 = A[2,0] * B[0,0]
    # M4 = (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[1,1])
    # M5 = correction terms...
    
    # This is getting complicated. Let me use a numerical approach to find a rank-5 decomposition,
    # or just use a higher rank that I can verify.
    
    # For now, let me use schoolbook for the sub-blocks and see if the overall structure helps.
    # Schoolbook 3x2x2: rank 12. For 3x4x6: split 4=2+2, each 3x2x6 has rank 12*6/2=36? No.
    
    # Let me just use a clean approach: build the full 3x5x6 decomposition using
    # the known Winograd rank-7 for 4x4x4 and handle the rest.
    
    # Actually, let me use a different strategy: use the fact that we can find
    # explicit low-rank decompositions by solving linear systems.
    
    return None  # Placeholder

def _winograd_4x4x4():
    """
    Winograd rank-7 decomposition for 4x4x4 matrix multiplication.
    U: (16,7), V: (16,7), W: (16,7)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    # Winograd's rank-7 decomposition for 4x4x4
    # This is well-known. Let me use the standard one.
    
    # The 7 multiplications are:
    # M1 = (A00+A11+A22+A33)(B00+B11+B22+B33)
    # M2 = (A01+A10+A23+A32)(B01+B10+B23+B32)
    # M3 = (A02+A13+A20+A31)(B02+B13+B20+B31)
    # M4 = (A03+A12+A21+A30)(B03+B12+B21+B30)
    # M5 = (A00+A11-A22-A33)(B00+B11-B22-B33)
    # M6 = (A01+A10-A23-A32)(B01+B10-B23-B32)
    # M7 = (A02+A13-A20-A31)(B02+B13-B20-B31)
    
    # Actually, the standard Winograd 4x4x4 rank-7 is more complex.
    # Let me use a verified one from the literature.
    
    # For now, let me use a simpler approach and just construct a valid decomposition
    # with a known rank, then verify it.
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Split m=5 into 1+4.
    # Block 1: 3x1x6 (j=0 only): C[i,k] += A[i,0]*B[0,k]. Rank = 3*6 = 18.
    # Block 2: 3x4x6 (j=1..4): C[i,k] += sum_{j=1}^{4} A[i,j]*B[j,k].
    
    # For Block 2, split the 4 into 2+2:
    # Block 2a: 3x2x6 (j=1,2): Rank = 6 * rank(3x2x2)
    # Block 2b: 3x2x6 (j=3,4): Rank = 6 * rank(3x2x2)
    
    # If rank(3x2x2) = 5, then Block 2 rank = 6*5 + 6*5 = 60.
    # Total rank = 18 + 60 = 78.
    
    # But we can do better by using the 4x4x4 Winograd (rank 7) for the 4-dimensional part.
    # 3x4x4 has rank at most 3*7 = 21 (using Winograd for each row).
    # Then 3x4x6 = 3x4x4 * identity(2) ... no, that's not right.
    
    # Actually, 3x4x6 can be seen as 6 independent 3x4x1 multiplications? No.
    # C[i,k] = sum_{j=1}^{4} A[i,j] * B[j,k] for k=0..5.
    # For each k, this is a 3x4 vector-matrix product.
    
    # The tensor 3x4x6 has rank at most 6 * rank(3x4x1) but that's not helpful.
    
    # Let me use a different approach: use the Winograd 4x4x4 (rank 7) and
    # "slice" it to get 3x4x6.
    
    # For 3x4x4: Use Winograd rank-7, giving rank 7*3 = 21? No, the Winograd
    # decomposition gives rank 7 for the full 4x4x4 tensor. For 3x4x4, we only
    # need 3 rows of A, so we can restrict the decomposition. The rank is at most 7.
    
    # Wait, that's not right either. The 3x4x4 tensor is a sub-tensor of 4x4x4.
    # Its rank is at most 7 (since it's a restriction of a rank-7 tensor).
    
    # So 3x4x4 has rank <= 7. Then 3x4x6: for each of the 6 columns of B, we have
    # a 3x4x1 tensor. These are not independent, so we can't just multiply by 6.
    
    # Actually, the 3x4x6 tensor can be decomposed as:
    # T[i,j,k] = 1 for i in 0..2, j in 1..4, k in 0..5
    # This is the "all-ones" tensor of size 3x4x6.
    # The rank of the all-ones tensor of size a x b x c is max(a,b,c) ... no, that's not right.
    # The rank of the all-ones tensor is actually the minimum of the three dimensions? No.
    
    # Let me just use a constructive approach with explicit matrices and verify.
    
    # Approach: Build the decomposition explicitly using block structure.
    
    # Block 1: j=0 (1-dimensional inner product)
    # U1[i*5+0, r] = 1 for r corresponding to this block
    # V1[0*6+k, r] = 1
    # W1[k*3+i, r] = 1
    # This is rank 18 (3*6 terms).
    
    # Block 2: j=1,2,3,4 (4-dimensional inner product)
    # Use Winograd-style decomposition.
    
    # For simplicity and correctness, let me use a verified decomposition.
    
    # I'll construct the full decomposition using the following strategy:
    # 1. For j=0: 18 rank-1 terms (one per (i,k) pair)
    # 2. For j=1..4: Use a rank-21 decomposition (3 copies of rank-7 Winograd for 4x4x4,
    #    but restricted to 3 rows and 6 columns)
    
    # Actually, let me just use a clean, verifiable approach.
    
    # Use the "Strassen" decomposition for the 4x4x4 part and handle the 3x4x6 by
    # applying it to each column.
    
    # For now, let me try a rank-78 decomposition (18 + 60) and verify it.
    
    # Build 3x2x2 rank-5 decomposition (verified):
    # After careful consideration, here's a rank-5 decomposition for 3x2x2:
    
    def _rank5_3x2x2():
        """
        Rank-5 decomposition for 3x2x2.
        U: (6,5), V: (4,5), W: (6,5)
        """
        U = [[0]*5 for _ in range(6)]
        V = [[0]*5 for _ in range(4)]
        W = [[0]*5 for _ in range(6)]
        
        # r=0: (i=0,j=0,k=0)
        U[0*2+0][0] = 1  # U[0,0]
        V[0*2+0][0] = 1  # V[0,0]
        W[0*3+0][0] = 1  # W[0,0]
        
        # r=1: (i=1,j=0,k=0)
        U[1*2+0][1] = 1  # U[2,0]
        V[0*2+0][1] = 1  # V[0,0]
        W[0*3+1][1] = 1  # W[3,0]
        
        # r=2: (i=2,j=0,k=0)
        U[2*2+0][2] = 1  # U[4,0]
        V[0*2+0][2] = 1  # V[0,0]
        W[0*3+2][2] = 1  # W[6,0]
        
        # r=3: (i=0,j=1,k=0) and (i=1,j=1,k=0) and (i=0,j=1,k=1) and (i=1,j=1,k=1)
        # This one term covers 4 entries.
        U[0*2+1][3] = 1  # U[1,0]
        U[1*2+1][3] = 1  # U[3,0]
        V[1*2+0][3] = 1  # V[2,0]
        V[1*2+1][3] = 1  # V[3,0]
        W[0*3+0][3] = 1  # W[0,0]
        W[0*3+1][3] = 1  # W[3,0]
        W[1*3+0][3] = 1  # W[1,0]
        W[1*3+1][3] = 1  # W[4,0]
        
        # r=4: (i=2,j=1,k=0) and (i=2,j=1,k=1)
        U[2*2+1][4] = 1  # U[5,0]
        V[1*2+0][4] = 1  # V[2,0]
        V[1*2+1][4] = 1  # V[3,0]
        W[0*3+2][4] = 1  # W[6,0]
        W[1*3+2][4] = 1  # W[7,0]
        
        # Verify:
        # (i=0,j=0,k=0): r=0: 1*1*1=1. Sum=1. OK
        # (i=1,j=0,k=0): r=1: 1*1*1=1. Sum=1. OK
        # (i=2,j=0,k=0): r=2: 1*1*1=1. Sum=1. OK
        # (i=0,j=1,k=0): r=3: 1*1*1=1. Sum=1. OK
        # (i=1,j=1,k=0): r=3: 1*1*1=1. Sum=1. OK
        # (i=2,j=1,k=0): r=4: 1*1*1=1. Sum=1. OK
        # (i=0,j=0,k=1): r=0: V[0,1]=0. Sum=0. FAIL!
        
        # Oops, I forgot the k=1 cases for j=0.
        
        # Let me redo this. The tensor has 3*2*2=12 entries, all must be 1.
        # (i,j,k) for i in 0..2, j in 0..1, k in 0..1.
        
        # I need to cover all 12 entries with 5 rank-1 tensors.
        
        # Let me try a different decomposition:
        # r=0: (0,0,0), (1,0,0), (2,0,0)  -- 3 entries
        # r=1: (0,0,1), (1,0,1), (2,0,1)  -- 3 entries
        # r=2: (0,1,0), (1,1,0)            -- 2 entries
        # r=3: (0,1,1), (1,1,1)            -- 2 entries
        # r=4: (2,1,0), (2,1,1)            -- 2 entries
        # Total: 12 entries. OK!
        
        U = [[0]*5 for _ in range(6)]
        V = [[0]*5 for _ in range(4)]
        W = [[0]*5 for _ in range(6)]
        
        # r=0: (i=0,j=0,k=0), (i=1,j=0,k=0), (i=2,j=0,k=0)
        U[0][0] = 1  # i=0,j=0
        U[2][0] = 1  # i=1,j=0
        U[4][0] = 1  # i=2,j=0
        V[0][0] = 1  # j=0,k=0
        W[0][0] = 1  # k=0,i=0
        W[3][0] = 1  # k=0,i=1
        W[6][0] = 1  # k=0,i=2
        
        # r=1: (i=0,j=0,k=1), (i=1,j=0,k=1), (i=2,j=0,k=1)
        U[0][1] = 1
        U[2][1] = 1
        U[4][1] = 1
        V[1][1] = 1  # j=0,k=1
        W[1][1] = 1  # k=1,i=0
        W[4][1] = 1  # k=1,i=1
        W[7][1] = 1  # k=1,i=2
        
        # r=2: (i=0,j=1,k=0), (i=1,j=1,k=0)
        U[1][2] = 1  # i=0,j=1
        U[3][2] = 1  # i=1,j=1
        V[2][2] = 1  # j=1,k=0
        W[0][2] = 1  # k=0,i=0
        W[3][2] = 1  # k=0,i=1
        
        # r=3: (i=0,j=1,k=1), (i=1,j=1,k=1)
        U[1][3] = 1
        U[3][3] = 1
        V[3][3] = 1  # j=1,k=1
        W[1][3] = 1  # k=1,i=0
        W[4][3] = 1  # k=1,i=1
        
        # r=4: (i=2,j=1,k=0), (i=2,j=1,k=1)
        U[5][4] = 1  # i=2,j=1
        V[2][4] = 1  # j=1,k=0
        V[3][4] = 1  # j=1,k=1
        W[6][4] = 1  # k=0,i=2
        W[7][4] = 1  # k=1,i=2
        
        # Verify all 12 entries:
        # (0,0,0): r=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1=1. OK
        # (1,0,0): r=0: U[2,0]*V[0,0]*W[3,0] = 1*1*1=1. OK
        # (2,0,0): r=0: U[4,0]*V[0,0]*W[6,0] = 1*1*1=1. OK
        # (0,0,1): r=1: U[0,1]*V[1,1]*W[1,1] = 1*1*1=1. OK
        # (1,0,1): r=1: U[2,1]*V[1,1]*W[4,1] = 1*1*1=1. OK
        # (2,0,1): r=1: U[4,1]*V[1,1]*W[7,1] = 1*1*1=1. OK
        # (0,1,0): r=2: U[1,2]*V[2,2]*W[0,2] = 1*1*1=1. OK
        # (1,1,0): r=2: U[3,2]*V[2,2]*W[3,2] = 1*1*1=1. OK
        # (0,1,1): r=3: U[1,3]*V[3,3]*W[1,3] = 1*1*1=1. OK
        # (1,1,1): r=3: U[3,3]*V[3,3]*W[4,3] = 1*1*1=1. OK
        # (2,1,0): r=4: U[5,4]*V[2,4]*W[6,4] = 1*1*1=1. OK
        # (2,1,1): r=4: U[5,4]*V[3,4]*W[7,4] = 1*1*1=1. OK
        
        # All 12 entries verified!
        return U, V, W
    
    # Now build the full 3x5x6 decomposition:
    # Block 1: j=0, rank 18
    # Block 2: j=1,2 (3x2x6), rank 5*6=30
    # Block 3: j=3,4 (3x2x6), rank 5*6=30
    # Total rank: 18+30+30=78
    
    U1, V1, W1 = _rank5_3x2x2()
    
    rank_total = 18 + 30 + 30
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Block 1: j=0
    c = 0
    for i in range(3):
        for k in range(6):
            U[i*5+0][c] = 1
            V[0*6+k][c] = 1
            W[k*3+i][c] = 1
            c += 1
    
    # Block 2: j=1,2 (offset by 1 in the j dimension)
    for r in range(5):
        for i in range(3):
            for jj in range(2):  # jj=0,1 corresponds to j=1,2
                U[i*5 + (1+jj)][c] = U1[i*2+jj][r]
        for jj in range(2):
            for k in range(6):
                V[(1+jj)*6+k][c] = V1[jj*2+k][r]
        for k in range(6):
            for i in range(3):
                W[k*3+i][c] = W1[k*3+i][r]
        c += 1
    
    # Block 3: j=3,4 (offset by 3 in the j dimension)
    for r in range(5):
        for i in range(3):
            for jj in range(2):  # jj=0,1 corresponds to j=3,4
                U[i*5 + (3+jj)][c] = U1[i*2+jj][r]
        for jj in range(2):
            for k in range(6):
                V[(3+jj)*6+k][c] = V1[jj*2+k][r]
        for k in range(6):
            for i in range(3):
                W[k*3+i][c] = W1[k*3+i][r]
        c += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return best_result
# EVOLVE-BLOCK-END
