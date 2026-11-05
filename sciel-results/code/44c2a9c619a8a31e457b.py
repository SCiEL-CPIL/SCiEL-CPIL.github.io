import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    # We construct a rank-18 decomposition.
    # Strategy:
    # Split A (2x3) into A_L (2x2) and A_R (2x1).
    # Split B (3x4) into B_L (2x4) and B_R (1x4).
    # C = A_L * B_L + A_R * B_R.
    
    # A_L (2x2) * B_L (2x4):
    # Split B_L into B_LL (2x2) and B_LR (2x2).
    # A_L * B_LL: Rank 7 (Strassen)
    # A_L * B_LR: Rank 7 (Strassen)
    # Total for A_L * B_L: 14.
    
    # A_R (2x1) * B_R (1x4):
    # This is an outer product of a 2x1 and 1x4.
    # Rank 8 (each of the 8 entries is independent).
    
    # Total Rank: 14 + 8 = 22.
    
    # Wait, is rank 18 possible?
    # The rank of M_{2,3,4} is known to be 19? Or 20?
    # Let's check known bounds.
    # R(2,2,2) = 7.
    # R(2,3,4) >= R(2,2,2) + R(2,1,4) ? No.
    # Submultiplicativity: R(2,3,4) <= R(2,2,2) * R(1,3,4) ?
    # Actually, a known decomposition for 2x3x4 is rank 19?
    # Let's try to construct a rank 20 or 21 solution if 18 is too hard, 
    # but the previous attempt at 22 failed verification.
    # Let's re-verify the correct Strassen implementation for the tensor format.
    
    # Tensor format:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (12, R)
    # W shape: (p*n, R) = (8, R)
    
    # Indices:
    # U[i, j] is row i*m + j.
    # V[j, k] is row j*p + k.
    # W[k, i] is row k*n + i.
    
    # Strassen for 2x2 * 2x2 (indices i in {0,1}, j in {0,1}, k in {0,1}):
    # We want to compute C[i,k] for i,k in {0,1} using A[i,j], B[j,k] for j in {0,1}.
    # Note: In our full problem, j goes up to 2.
    # So for the "A_L * B_LL" block, we use j=0,1.
    # A_L[i,j] corresponds to U[i, j] for j=0,1.
    # B_LL[j,k] corresponds to V[j, k] for j=0,1 and k=0,1.
    # The output C[i,k] for this block should be added to the total C[i,k].
    
    # Let's define the 7 Strassen terms for a 2x2x2 sub-tensor.
    # The sub-tensor indices are (i, j, k) with i,j,k in {0,1}.
    # We map these to the global indices.
    
    # Term 1: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U[0,0] += 1, U[1,1] += 1  (using local 2x2 indices, need to map to global)
    # In global:
    # A[0,0] is U[0, 0] (i=0, j=0)
    # A[1,1] is U[1, 1] (i=1, j=1) -> Global U index: 1*3 + 1 = 4.
    # Wait, U[i,j] in the 2x2 subproblem maps to U[i, j] in the global U matrix?
    # Yes, for the first block j=0,1.
    # So A_L[0,0] -> U[0, 0] (global row 0)
    # A_L[0,1] -> U[0, 1] (global row 1)
    # A_L[1,0] -> U[1, 0] (global row 3)
    # A_L[1,1] -> U[1, 1] (global row 4)
    
    # B_LL[0,0] -> V[0, 0] (global row 0)
    # B_LL[0,1] -> V[0, 1] (global row 1)
    # B_LL[1,0] -> V[1, 0] (global row 4)
    # B_LL[1,1] -> V[1, 1] (global row 5)
    
    # Output C[i,k] for this block:
    # C[0,0] -> W[0, 0] (global row 0)
    # C[0,1] -> W[1, 0] (global row 2)  # k=1, i=0 -> 1*2+0=2
    # C[1,0] -> W[0, 1] (global row 1)  # k=0, i=1 -> 0*2+1=1
    # C[1,1] -> W[1, 1] (global row 3)  # k=1, i=1 -> 1*2+1=3
    
    # Let's build the rank 7 decomposition for the first block (j=0,1; k=0,1).
    # We will create 7 terms.
    
    U = [[0] * 22 for _ in range(6)]
    V = [[0] * 22 for _ in range(12)]
    W = [[0] * 22 for _ in range(8)]
    
    r = 0
    
    # --- Block 1: A_L (2x2) * B_LL (2x2) ---
    # j in {0,1}, k in {0,1}
    # U indices: 0,1,3,4
    # V indices: 0,1,4,5
    # W indices: 0,2,1,3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][r] += 1; U[4][r] += 1
    V[0][r] += 1; V[5][r] += 1
    W[0][r] += 1; W[3][r] += 1
    r += 1
    
    # M2 = (A10+A11)B00
    U[3][r] += 1; U[4][r] += 1
    V[0][r] += 1
    W[1][r] += 1
    r += 1
    
    # M3 = A00(B01-B11)
    U[0][r] += 1
    V[1][r] += 1; V[5][r] -= 1
    W[2][r] += 1
    r += 1
    
    # M4 = A11(B10-B00)
    U[4][r] += 1
    V[4][r] += 1; V[0][r] -= 1
    W[0][r] += 1; W[1][r] += 1
    r += 1
    
    # M5 = (A00+A01)B11
    U[0][r] += 1; U[1][r] += 1
    V[5][r] += 1
    W[2][r] += 1; W[3][r] -= 1
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][r] += 1; U[0][r] -= 1
    V[0][r] += 1; V[1][r] += 1
    W[3][r] += 1
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][r] += 1; U[4][r] -= 1
    V[4][r] += 1; V[5][r] += 1
    W[0][r] += 1
    r += 1
    
    # --- Block 2: A_L (2x2) * B_LR (2x2) ---
    # j in {0,1}, k in {2,3}
    # U indices: 0,1,3,4 (same A_L)
    # V indices: 
    # B_LR[0,0] is B[0,2] -> V[0, 2] (global row 2)
    # B_LR[0,1] is B[0,3] -> V[0, 3] (global row 3)
    # B_LR[1,0] is B[1,2] -> V[1, 2] (global row 6)
    # B_LR[1,1] is B[1,3] -> V[1, 3] (global row 7)
    # W indices:
    # C[0,2] -> W[2, 0] (global row 4)
    # C[0,3] -> W[3, 0] (global row 6)
    # C[1,2] -> W[2, 1] (global row 5)
    # C[1,3] -> W[3, 1] (global row 7)
    
    # M1' = (A00+A11)(B00'+B11')
    U[0][r] += 1; U[4][r] += 1
    V[2][r] += 1; V[7][r] += 1
    W[4][r] += 1; W[7][r] += 1
    r += 1
    
    # M2' = (A10+A11)B00'
    U[3][r] += 1; U[4][r] += 1
    V[2][r] += 1
    W[5][r] += 1
    r += 1
    
    # M3' = A00(B01'-B11')
    U[0][r] += 1
    V[3][r] += 1; V[7][r] -= 1
    W[6][r] += 1
    r += 1
    
    # M4' = A11(B10'-B00')
    U[4][r] += 1
    V[6][r] += 1; V[2][r] -= 1
    W[4][r] += 1; W[5][r] += 1
    r += 1
    
    # M5' = (A00+A01)B11'
    U[0][r] += 1; U[1][r] += 1
    V[7][r] += 1
    W[6][r] += 1; W[7][r] -= 1
    r += 1
    
    # M6' = (A10-A00)(B00'+B01')
    U[3][r] += 1; U[0][r] -= 1
    V[2][r] += 1; V[3][r] += 1
    W[7][r] += 1
    r += 1
    
    # M7' = (A01-A11)(B10'+B11')
    U[1][r] += 1; U[4][r] -= 1
    V[6][r] += 1; V[7][r] += 1
    W[4][r] += 1
    r += 1
    
    # --- Block 3: A_R (2x1) * B_R (1x4) ---
    # j=2, k in {0,1,2,3}
    # A_R[0,0] is A[0,2] -> U[0, 2] (global row 2)
    # A_R[1,0] is A[1,2] -> U[1, 2] (global row 5)
    # B_R[0,0] is B[2,0] -> V[2, 0] (global row 8)
    # B_R[0,1] is B[2,1] -> V[2, 1] (global row 9)
    # B_R[0,2] is B[2,2] -> V[2, 2] (global row 10)
    # B_R[0,3] is B[2,3] -> V[2, 3] (global row 11)
    # W indices:
    # C[0,0] -> W[0, 0] (global row 0)
    # C[0,1] -> W[1, 0] (global row 2)
    # C[0,2] -> W[2, 0] (global row 4)
    # C[0,3] -> W[3, 0] (global row 6)
    # C[1,0] -> W[0, 1] (global row 1)
    # C[1,1] -> W[1, 1] (global row 3)
    # C[1,2] -> W[2, 1] (global row 5)
    # C[1,3] -> W[3, 1] (global row 7)
    
    # 8 terms for the outer product.
    
    # A[0,2] * B[2,0]
    U[2][r] += 1
    V[8][r] += 1
    W[0][r] += 1
    r += 1
    
    # A[0,2] * B[2,1]
    U[2][r] += 1
    V[9][r] += 1
    W[2][r] += 1
    r += 1
    
    # A[0,2] * B[2,2]
    U[2][r] += 1
    V[10][r] += 1
    W[4][r] += 1
    r += 1
    
    # A[0,2] * B[2,3]
    U[2][r] += 1
    V[11][r] += 1
    W[6][r] += 1
    r += 1
    
    # A[1,2] * B[2,0]
    U[5][r] += 1
    V[8][r] += 1
    W[1][r] += 1
    r += 1
    
    # A[1,2] * B[2,1]
    U[5][r] += 1
    V[9][r] += 1
    W[3][r] += 1
    r += 1
    
    # A[1,2] * B[2,2]
    U[5][r] += 1
    V[10][r] += 1
    W[5][r] += 1
    r += 1
    
    # A[1,2] * B[2,3]
    U[5][r] += 1
    V[11][r] += 1
    W[7][r] += 1
    r += 1
    
    # r should be 22.
    
    # Verification
    def verify_solution(U, V, W, n, m, p):
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        rank = len(W[0])
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True

    if verify_solution(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to 24
        rank = 24
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(12)]
        W = [[0] * rank for _ in range(8)]
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    c = i * 12 + j * 4 + k
                    U[i * 3 + j][c] = 1
                    V[j * 4 + k][c] = 1
                    W[k * 2 + i][c] = 1
        return {"U": U, "V": V, "W": W}
