import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # ---------- Trivial (schoolbook) decomposition as guaranteed fallback ----------
    U0 = [[0] * trivial_rank for _ in range(n * m)]
    V0 = [[0] * trivial_rank for _ in range(m * p)]
    W0 = [[0] * trivial_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best_result = {"U": U0, "V": V0, "W": W0}
    best_rank = trivial_rank

    # ---------- Fast verification (numpy, vectorised) ----------
    def check(U_r, V_r, W_r):
        Ua = np.asarray(U_r, dtype=np.int64)
        Va = np.asarray(V_r, dtype=np.int64)
        Wa = np.asarray(W_r, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i]
                    if int(np.sum(prod)) != 1:
                        return False
        return True

    # ---------- Winograd-style construction for 2x3x7 ----------
    # We use a scheme that exploits the small dimensions.
    # The key insight: we can use Winograd convolution/FMA-like tricks.
    
    # For 2x3x7, we can split into overlapping 2x2x2 blocks and use Strassen,
    # plus handle the remainder. But a better approach is to use a direct
    # Winograd-style scheme.
    
    # Let's try a direct construction based on the structure:
    # A is 2x3, B is 3x7, C is 2x7.
    # C[i][k] = sum_{j=0}^{2} A[i][j] * B[j][k]
    
    # One approach: use the fact that we have 2 rows in A.
    # We can combine the two rows of A to reduce operations.
    
    # Let's try a systematic Winograd approach:
    # For each column k of B, we compute C[:,k] = A @ B[:,k]
    # A @ v for v in R^3: result is in R^2
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # C0k = a00*B0k + a01*B1k + a02*B2k
    # C1k = a10*B0k + a11*B1k + a12*B2k
    
    # This is essentially 2 dot products of length 3.
    # Schoolbook: 2*3 = 6 mults per column, 7 columns = 42 total.
    
    # Can we do better? Let's think about it as a bilinear map.
    # The tensor has 2*3*7 = 42 entries.
    
    # Let's try to use Strassen-like tricks on sub-blocks.
    # Split A into 2x2 + 2x1, B into 2x7 + 1x7.
    # A = [A22 | A21], B = [B27; B17]
    # C = A22 @ B27 + A21 @ B17
    
    # A22 @ B27: 2x2 @ 2x7 -> use Strassen for the 2x2 part? 
    # Actually, for 2x2 @ 2x7, we can use Strassen to compute the 2x2 block
    # times each 2x1 column of B27.
    # But B27 is 2x7, so we have 7 columns of 2x1.
    # Strassen for 2x2 @ 2x1: 
    #   Let A = [[a,b],[c,d]], v = [x,y]
    #   A@v = [ax+by, cx+dy]
    #   Schoolbook: 4 mults.
    #   Can we do it in 3? 
    #   (a+c)(x+y) + ... no, that's for 2x2x2.
    #   For 2x2x1, the rank is 4 (it's just two independent dot products of length 2).
    #   Actually, 2x2x1 matrix multiplication has rank 4 (each output is a dot product).
    
    # So Strassen doesn't help for 2x2x1.
    
    # Let's try a different split. What if we group columns of B?
    # B is 3x7. Split into 3x3 + 3x3 + 3x1.
    # C = A @ B1 + A @ B2 + A @ B3
    # where B1, B2 are 3x3, B3 is 3x1.
    
    # A @ B1: 2x3 @ 3x3 -> 2x3. Schoolbook: 2*3*3 = 18.
    # Can we do 2x3 @ 3x3 better? 
    # This is equivalent to computing 3 dot products of length 3, but with 2 rows of A.
    # Actually, C[:,j] = A @ B1[:,j] for j=0,1,2.
    # Each is a 2x3 @ 3x1 = 2x1.
    # 2x3 @ 3x1: 6 mults each, 3 columns = 18.
    
    # Hmm, this doesn't seem to help directly.
    
    # Let's try the Winograd approach more carefully.
    # Winograd's trick for vector-matrix multiplication:
    # For a fixed A (2x3), multiplying A @ B (3x7) can be seen as
    # applying a linear map to each column of B.
    
    # The key is to precompute combinations of rows of A.
    # Let's define:
    #   m0 = a00 + a10
    #   m1 = a01 + a11  
    #   m2 = a02 + a12
    #   m3 = a00 - a10
    #   m4 = a01 - a11
    #   m5 = a02 - a12
    
    # Then:
    #   C0k = a00*B0k + a01*B1k + a02*B2k
    #   C1k = a10*B0k + a11*B1k + a12*B2k
    #   C0k + C1k = m0*B0k + m1*B1k + m2*B2k
    #   C0k - C1k = m3*B0k + m4*B1k + m5*B2k
    
    # So if we compute:
    #   s_k = m0*B0k + m1*B1k + m2*B2k  (3 mults + 2 adds)
    #   d_k = m3*B0k + m4*B1k + m5*B2k  (3 mults + 2 adds)
    # Then:
    #   C0k = (s_k + d_k) / 2
    #   C1k = (s_k - d_k) / 2
    
    # Total: 6 mults per column, 7 columns = 42 mults. Same as schoolbook!
    # But wait, we're using half-integers (1/2). The task allows half-integers.
    # However, this doesn't reduce the rank.
    
    # Let's think differently. The rank of the tensor is what we want to minimize.
    # The trivial rank is 42. Can we do better?
    
    # For 2x3x7, the known lower bounds are not tight. Let's try to construct
    # a decomposition with rank < 42.
    
    # Idea: use the fact that we have 2 rows. We can share some multiplications.
    
    # Let's try a direct rank-39 construction (as in the current program's hybrid).
    # The current program achieves rank 39. Can we do better?
    
    # Let's try rank 36 or lower.
    
    # Actually, let's just use the hybrid Strassen/Schoolbook approach but optimize it.
    # The current hybrid uses:
    #   Block A: 2x2x2 Strassen (rank 7)
    #   Block B: 2x1x2 schoolbook (rank 4)  
    #   Block C: 1x2x3 schoolbook (rank 6)
    #   Total: 7 + 4 + 6 = 17... wait, that's not 39.
    
    # Let me re-examine. The current program's hybrid:
    # Block A: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, Strassen rank 7
    # Block B: i in {0,1}, j in {0,1}, k = 2 -> 2x1x2, schoolbook rank 4
    # Block C: i in {0,1}, j = 2, k in {0,1,2} -> 1x2x3, schoolbook rank 6
    # Total: 7 + 4 + 6 = 17? But the program says achieved_rank is computed from base.
    
    # Wait, I think I misread. Let me trace through:
    # base starts at 0.
    # Block A: strassen_block uses 7 columns, base becomes 7.
    # Block B: 2*2 = 4 iterations, base becomes 11.
    # Block C: 2*3 = 6 iterations, base becomes 17.
    # So achieved_rank = 17? But the problem says trivial is 42 and the score is 1.0.
    
    # Oh wait, I see the issue. The Strassen block in the current program is wrong.
    # Let me check: strassen_block adds terms to U_b, V_b, W_b.
    # But the mapping might be incorrect.
    
    # Actually, looking more carefully, the current program's Strassen implementation
    # seems to have a bug in the term definitions. Let me just construct a clean solution.
    
    # Let me try a clean Winograd-style approach.
    
    # For 2x3x7, let's use the following scheme:
    # We'll use a combination of Strassen for 2x2x2 blocks and schoolbook for the rest.
    
    # Split the inner dimension j into {0,1} and {2}.
    # Split the output dimension k into {0,1,2,3,4,5,6}.
    
    # For j in {0,1}, k in {0,1}: 2x2x2 block, use Strassen (rank 7)
    # For j in {0,1}, k in {2,3,4,5,6}: 2x2x5 block, schoolbook (rank 20)
    # For j = 2, k in {0,1,2,3,4,5,6}: 2x1x7 block, schoolbook (rank 14)
    # Total: 7 + 20 + 14 = 41. Still close to 42.
    
    # Better: use Strassen for multiple 2x2x2 blocks.
    # Split k into pairs: {0,1}, {2,3}, {4,5}, and {6}.
    # For j in {0,1}, k in {0,1}: Strassen (7)
    # For j in {0,1}, k in {2,3}: Strassen (7)
    # For j in {0,1}, k in {4,5}: Strassen (7)
    # For j in {0,1}, k = 6: schoolbook 2x1x1 (2)
    # For j = 2, k in {0,1,2,3,4,5,6}: schoolbook 2x1x7 (14)
    # Total: 7+7+7+2+14 = 37.
    
    # Can we do better? What if we use Strassen for the j=2 part too?
    # j=2 is a single row, so we can't use 2x2x2 Strassen directly.
    
    # Alternative: split i instead of j.
    # A is 2x3, so i in {0,1}.
    # For i=0: A[0,:] is 1x3, multiply by B (3x7) -> 1x7. Schoolbook: 3*7=21.
    # For i=1: A[1,:] is 1x3, multiply by B (3x7) -> 1x7. Schoolbook: 3*7=21.
    # Total: 42. No improvement.
    
    # Let's go with the rank 37 approach.
    
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    base = 0
    
    # Strassen 2x2x2 helper
    def strassen_2x2x2(U_mat, V_mat, W_mat, base):
        """
        U_mat: 4xR, rows indexed by (i,j) for i,j in {0,1}, global row = i*m+j
        V_mat: 4xR, rows indexed by (j,k) for j,k in {0,1}, global row = j*p+k  
        W_mat: 4xR, rows indexed by (k,i) for k,i in {0,1}, global row = k*n+i
        """
        # Strassen's 7 products for 2x2x2
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A@B
        # M1 = (a00+a11)(b01+b10)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a10)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # Local indices
        a00, a01, a10, a11 = 0, 1, 2, 3
        b00, b01, b10, b11 = 0, 1, 2, 3
        c00, c01, c10, c11 = 0, 1, 2, 3
        
        terms = [
            # M1
            (a00, b01, c00, 1), (a11, b01, c00, 1),
            (a00, b10, c00, 1), (a11, b10, c00, 1),
            # M2
            (a10, b00, c10, 1), (a11, b00, c10, 1),
            # M3
            (a00, b10, c01, 1), (a00, b11, c01, -1),
            # M4
            (a11, b01, c00, 1), (a11, b00, c00, -1),
            (a11, b01, c11, 1), (a11, b00, c11, -1),
            # M5
            (a00, b11, c00, -1), (a10, b11, c00, -1),
            (a00, b11, c01, 1), (a10, b11, c01, 1),
            (a00, b11, c11, 1), (a10, b11, c11, 1),
            # M6
            (a10, b00, c11, 1), (a10, b01, c11, 1),
            (a00, b00, c11, -1), (a00, b01, c11, -1),
            # M7
            (a01, b10, c00, 1), (a11, b10, c00, -1),
            (a01, b11, c00, -1), (a11, b11, c00, 1),
        ]
        
        for (ur, vr, wr, s) in terms:
            U_mat[ur][base] += s
            V_mat[vr][base] += s
            W_mat[wr][base] += s
        base += 1
        return base
    
    # Block 1: j in {0,1}, k in {0,1}, i in {0,1} -> 2x2x2 Strassen
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    base = strassen_2x2x2(U_loc, V_loc, W_loc, 0)
    for a in range(4):
        i, j = a // 2, a % 2
        for t in range(7):
            U[i*m+j][base+t] = U_loc[a][t]
    for b in range(4):
        j, k = b // 2, b % 2
        for t in range(7):
            V[j*p+k][base+t] = V_loc[b][t]
    for c in range(4):
        k, i = c // 2, c % 2
        for t in range(7):
            W[k*n+i][base+t] = W_loc[c][t]
    base += 7
    
    # Block 2: j in {0,1}, k in {2,3}, i in {0,1} -> 2x2x2 Strassen
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    base = strassen_2x2x2(U_loc, V_loc, W_loc, 0)
    for a in range(4):
        i, j = a // 2, a % 2
        for t in range(7):
            U[i*m+j][base+t] = U_loc[a][t]
    for b in range(4):
        j, k = b // 2, b % 2
        k_global = k + 2
        for t in range(7):
            V[j*p+k_global][base+t] = V_loc[b][t]
    for c in range(4):
        k, i = c // 2, c % 2
        k_global = k + 2
        for t in range(7):
            W[k_global*n+i][base+t] = W_loc[c][t]
    base += 7
    
    # Block 3: j in {0,1}, k in {4,5}, i in {0,1} -> 2x2x2 Strassen
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]
    base = strassen_2x2x2(U_loc, V_loc, W_loc, 0)
    for a in range(4):
        i, j = a // 2, a % 2
        for t in range(7):
            U[i*m+j][base+t] = U_loc[a][t]
    for b in range(4):
        j, k = b // 2, b % 2
        k_global = k + 4
        for t in range(7):
            V[j*p+k_global][base+t] = V_loc[b][t]
    for c in range(4):
        k, i = c // 2, c % 2
        k_global = k + 4
        for t in range(7):
            W[k_global*n+i][base+t] = W_loc[c][t]
    base += 7
    
    # Block 4: j in {0,1}, k = 6, i in {0,1} -> 2x1x1 schoolbook (rank 2)
    for i in range(2):
        for j in range(2):
            U[i*m+j][base] = 1
            V[j*p+6][base] = 1
            W[6*n+i][base] = 1
            base += 1
    
    # Block 5: j = 2, k in {0,1,2,3,4,5,6}, i in {0,1} -> 1x7x2 schoolbook (rank 14)
    for i in range(2):
        for k in range(7):
            U[i*m+2][base] = 1
            V[2*p+k][base] = 1
            W[k*n+i][base] = 1
            base += 1
    
    achieved_rank = base
    print("achieved_rank", achieved_rank)
    
    if check(U, V, W):
        U = [row[:achieved_rank] for row in U]
        V = [row[:achieved_rank] for row in V]
        W = [row[:achieved_rank] for row in W]
        best_rank = achieved_rank
        best_result = {"U": U, "V": V, "W": W}
        print("hybrid valid, rank", best_rank)
    else:
        print("hybrid invalid, falling back to trivial rank 42")
        best_result = {"U": U0, "V": V0, "W": W0}
        best_rank = trivial_rank
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
