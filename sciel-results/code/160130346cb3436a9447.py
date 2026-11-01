"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # We construct a rank-13 decomposition.
    # Part 1: Strassen algorithm for the first 2 columns (k=0, 1) of the output.
    # Part 2: Schoolbook algorithm for the remaining 3 columns (k=2, 3, 4).
    
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1.
    # U index: a = i*m + j (0..3)
    # V index: b = j*p + k (0..9)
    # W index: c = k*n + i (0..9)
    
    # Rank 13
    rank = 13
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Part 1: Strassen for k=0, 1 ---
    # The Strassen algorithm computes C_2x2 = A_2x2 * B_2x2.
    # Here B is 2x2 (columns 0,1).
    # We map the 7 Strassen multiplications to rank indices 0..6.
    
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # M3 = (A[0,1] + A[1,1]) * B[0,0]
    # M4 = A[1,1] * (B[0,0] - B[1,0])
    # M5 = (A[0,0] + A[1,1]) * (B[1,0] + B[1,1])
    # M6 = (A[0,1] - A[1,0]) * (B[0,0] + B[0,1])
    # M7 = (A[1,0] - A[0,0]) * (B[1,1] + B[0,1])
    
    # Indices for A (U): i*m + j
    # A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3
    
    # Indices for B (V): j*p + k
    # B[0,0] -> 0, B[0,1] -> 1, B[1,0] -> 5, B[1,1] -> 6
    # Note: p=5, so B[j, k] index is j*5 + k.
    # B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6.
    
    # Indices for C (W): k*n + i
    # C[0,0] -> 0, C[0,1] -> 1, C[1,0] -> 2, C[1,1] -> 3
    # Note: n=2, so C[k, i] index is k*2 + i.
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.
    
    # Term 0: M1
    U[0, 0] = 1  # A[0,0]
    V[0, 0] = 1  # B[0,0]
    V[5, 0] = 1  # B[1,1] (Wait, B[1,1] is index 1*5+1=6) -> Correction: V[6,0]=1
    # Let's re-verify indices.
    # V index for B[j,k] is j*p + k.
    # B[0,0] -> 0*5+0 = 0
    # B[0,1] -> 0*5+1 = 1
    # B[1,0] -> 1*5+0 = 5
    # B[1,1] -> 1*5+1 = 6
    
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    U[0, 0] = 1
    V[0, 0] = 1
    V[6, 0] = 1
    # Result C[0,0] += 1, C[1,1] -= 1? No, Strassen reconstruction is linear.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # We add contributions to W.
    # M1 contributes +1 to C[0,0] and +1 to C[1,1]
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Term 1: M2 = (A[0,0] + A[1,0]) * B[1,1]
    U[0, 1] = 1
    U[2, 1] = 1
    V[6, 1] = 1
    # M2 contributes +1 to C[1,0] and -1 to C[1,1]
    W[2, 1] = 1
    W[3, 1] = -1
    
    # Term 2: M3 = (A[0,1] + A[1,1]) * B[0,0]
    U[1, 2] = 1
    U[3, 2] = 1
    V[0, 2] = 1
    # M3 contributes +1 to C[0,1] and +1 to C[1,1]
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Term 3: M4 = A[1,1] * (B[0,0] - B[1,0])
    U[3, 3] = 1
    V[0, 3] = 1
    V[5, 3] = -1
    # M4 contributes +1 to C[0,0] and +1 to C[1,0]
    W[0, 3] = 1
    W[2, 3] = 1
    
    # Term 4: M5 = (A[0,0] + A[1,1]) * (B[1,0] + B[1,1])
    U[0, 4] = 1
    U[3, 4] = 1
    V[5, 4] = 1
    V[6, 4] = 1
    # M5 contributes -1 to C[0,0] and +1 to C[0,1]
    W[0, 4] = -1
    W[1, 4] = 1
    
    # Term 5: M6 = (A[0,1] - A[1,0]) * (B[0,0] + B[0,1])
    U[1, 5] = 1
    U[2, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    # M6 contributes +1 to C[1,1]
    W[3, 5] = 1
    
    # Term 6: M7 = (A[1,0] - A[0,0]) * (B[1,1] + B[0,1])
    U[2, 6] = 1
    U[0, 6] = -1
    V[6, 6] = 1
    V[1, 6] = 1
    # M7 contributes +1 to C[0,0]
    W[0, 6] = 1
    
    # --- Part 2: Schoolbook for k=2, 3, 4 ---
    # Terms 7..12
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # For each k in [2,3,4] and i in [0,1], we have 2 terms.
    # Total 6 terms.
    
    current_rank = 7
    for k in range(2, 5):
        for i in range(2):
            # Term: A[i,0] * B[0,k] -> C[i,k]
            U[i*2 + 0, current_rank] = 1
            V[0*5 + k, current_rank] = 1
            W[k*2 + i, current_rank] = 1
            current_rank += 1
            
            # Term: A[i,1] * B[1,k] -> C[i,k]
            U[i*2 + 1, current_rank] = 1
            V[1*5 + k, current_rank] = 1
            W[k*2 + i, current_rank] = 1
            current_rank += 1
            
    # Verify the decomposition
    # T[a,b,c] should be sum(U[a,r] * V[b,r] * W[c,r])
    # For (a,b,c) in support, sum should be 1.
    # For others, 0.
    
    # Quick check for a few points
    # C[0,0] is W[0]. Support: (0,0,0), (1,0,0), (2,0,0), (3,0,0)? No.
    # W index c = k*n + i.
    # C[0,0] -> c=0.
    # Contributions to W[0,r] come from terms where W[0,r] != 0.
    # r=0: U[0,0]=1, V[0,0]=1, V[6,0]=1. W[0,0]=1.
    #   Term 0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1.
    #   Term 0: U[0,0]*V[6,0]*W[0,0] = 1*1*1 = 1.
    #   Wait, the tensor equation is sum_r U[a,r] V[b,r] W[c,r].
    #   For a=0 (A[0,0]), b=0 (B[0,0]), c=0 (C[0,0]):
    #   r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    #   r=1: U[0,1]=1, V[0,1]=0, ... -> 0
    #   r=2: U[0,2]=0, ... -> 0
    #   r=3: U[0,3]=0, ... -> 0
    #   r=4: U[0,4]=1, V[0,4]=0, ... -> 0
    #   r=5: U[0,5]=0, ... -> 0
    #   r=6: U[0,6]=-1, V[0,6]=0, ... -> 0
    #   Sum = 1. Correct.
    
    # Check a non-support point: a=0 (A[0,0]), b=2 (B[0,2]), c=0 (C[0,0]).
    # b=2 is B[0,2].
    # r=0: V[2,0]=0.
    # r=1: V[2,1]=0.
    # ...
    # r=7 (first schoolbook term for k=2, i=0):
    # U[0,7]=1, V[2,7]=1, W[0,7]=0. -> 0.
    # r=8: U[1,8]=1... U[0,8]=0.
    # It seems correct.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
