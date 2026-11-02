"""
Matrix multiplication 4x4x5 decomposition.

Strategy: Use the known rank-64 construction for 4x4x5 matrix multiplication.
This uses a block structure where we split the 4x4x5 problem into:
- Two 2x2x2 blocks (each rank 7 via Strassen) = 14
- Two 2x2x3 blocks (each rank 9) = 18
- Two 2x3x2 blocks (each rank 9) = 18  
- Two 2x3x3 blocks (each rank 13) = 26
Wait, that's 76, too high.

Actually, the known result is that 4x4x5 has rank at most 64.
Let me use a different approach: split A into 2x2 blocks, B into 2x2 blocks (with B being 4x5, so split into 2x2 and 2x3).

Better approach: Use the fact that we can compute 4x4x5 as:
- Split into 4 sub-problems based on block structure
- Use Strassen for 2x2x2 parts (rank 7 each)
- Use rank-9 for 2x2x3 parts

Let me construct it properly.

For 4x4x5:
A is 4x4, B is 4x5, C is 4x5.

Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
Split B into 2x2 and 2x3: B = [[B00, B01], [B10, B11]] where B00, B01 are 2x2 and 2x3? No...

Actually B is 4x5. Split rows into 2+2, columns into 2+3:
B = [[B00, B01], [B10, B11]] where B00, B10 are 2x2, B01, B11 are 2x3

C is 4x5, split similarly: C = [[C00, C01], [C10, C11]] where C00, C10 are 2x2, C01, C11 are 2x3

C00 = A00*B00 + A01*B10  (2x2 = 2x2*2x2 + 2x2*2x2)
C01 = A00*B01 + A01*B11  (2x3 = 2x2*2x3 + 2x2*2x3)
C10 = A10*B00 + A11*B10  (2x2 = 2x2*2x2 + 2x2*2x2)
C11 = A10*B01 + A11*B11  (2x3 = 2x2*2x3 + 2x2*2x3)

For 2x2x2: rank 7 (Strassen)
For 2x2x3: rank 9 (known optimal)

We need to compute:
- C00: sum of two 2x2x2 products -> can share structure
- C01: sum of two 2x2x3 products
- C10: sum of two 2x2x2 products
- C11: sum of two 2x2x3 products

Using Strassen-like combination for the 2x2x2 parts:
C00 and C10 both involve 2x2x2 products. We can use a Strassen-like scheme.

Actually, let me just use the direct approach with rank 64.

The rank-64 construction for 4x4x5:
- Use 7-term Strassen for the 4x4x4 part (rank 49)
- Add 15 more terms for the 5th column (not 16, by sharing some structure)

Or better: use the known decomposition.

Let me try a cleaner approach:
- 4x4x4 part: rank 49 (Strassen on Strassen)
- 5th column: we need T[i,j,4] = 1 for all i,j in 0..3
  This is a 4x4x1 tensor of all ones.
  Rank of 4x4x1 all-ones tensor: we can do it in rank 4!
  Because sum_r U[i,j,r]*V[j,4,r]*W[4,i,r] = 1
  Let U[i,j,r] = 1 if r == i, V[j,4,r] = 1, W[4,i,r] = 1 if r == i
  Wait: sum over r of U[i,j,r]*V[j,4,r]*W[4,i,r]
  If we set U[i,j,r] = delta(i,r), V[j,4,r] = 1, W[4,i,r] = delta(i,r)
  Then sum_r delta(i,r)*1*delta(i,r) = delta(i,i) = 1. Yes!
  But wait, V[j,4,r] = 1 for all j,r, and W[4,i,r] = delta(i,r), U[i,j,r] = delta(i,r)
  sum_r delta(i,r) * 1 * delta(i,r) = 1 (since only r=i contributes: 1*1*1=1)
  
  So we need only 4 terms for the 5th column!
  Total rank = 49 + 4 = 53.

Wait, but we also need to make sure the 5th column terms don't interfere with the 4x4x4 part.
The 4x4x4 part uses W[k*4+i, r] for k=0..3, i=0..3.
The 5th column uses W[4*4+i, r] = W[16+i, r] for i=0..3.
These are different rows of W, so no interference. 

But wait, U and V also have entries for the 5th column terms.
For the 5th column: U[i*4+j, r] = delta(i, r-49) for r=49..52
V[j*5+4, r] = 1 for r=49..52
W[16+i, r] = delta(i, r-49) for r=49..52

Let me verify: for the 5th column, T[i,j,4] should be 1.
sum_r U[i*4+j, r] * V[j*5+4, r] * W[4*4+i, r]
= sum_r U[i*4+j, r] * V[j*5+4, r] * W[16+i, r]

For r < 49: V[j*5+4, r] = 0 (since the 4x4x4 part only fills V[j*5+k, r] for k=0..3)
For r >= 49: U[i*4+j, r] = delta(i, r-49), V[j*5+4, r] = 1, W[16+i, r] = delta(i, r-49)
So sum = sum_{r=49}^{52} delta(i, r-49) * 1 * delta(i, r-49) = delta(i, i) = 1. Correct!

And for the 4x4x4 part (k=0..3), the 5th column terms don't contribute because W[k*4+i, r] for k<4 doesn't include W[16+i, r].

So total rank = 49 + 4 = 53.

Let me implement this.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    deadline = time.time() + 130  # Safety margin
    
    # Strassen 2x2x2 decomposition
    # U_2[i*2+j, r]: coefficient of A[i,j] in term r
    # V_2[j*2+k, r]: coefficient of B[j,k] in term r
    # W_2[k*2+i, r]: coefficient of C[k,i] in term r
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # A00
        [0, 0, 0, 0, 0, 1, 1],  # A01
        [1, 1, 0, 0, 0, 0, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1], # A11
    ], dtype=float)
    
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 1, 1, 1],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    
    W_2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # C00: k=0, i=0
        [0, 0, 1, 0, 0, 0, 0],   # C01: k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],   # C10: k=1, i=0
        [0, 0, 0, 0, 1, 0, 0],   # C11: k=1, i=1
    ], dtype=float)
    
    # Build 4x4x4 decomposition using Strassen on Strassen
    # A is 4x4, viewed as 2x2 blocks of 2x2
    # B is 4x4 (for the 4x4x4 part), viewed as 2x2 blocks of 2x2
    # C is 4x4, viewed as 2x2 blocks of 2x2
    
    # Strassen outer coefficients
    A_block_coeffs = [
        [(0,0,1), (1,1,1)],   # M1
        [(1,0,1), (1,1,1)],   # M2
        [(0,0,1)],             # M3
        [(1,1,1)],             # M4
        [(0,0,1), (0,1,1)],   # M5
        [(1,0,1), (0,0,-1)],  # M6
        [(0,1,1), (1,1,-1)],  # M7
    ]
    
    B_block_coeffs = [
        [(0,0,1), (1,1,1)],   # M1
        [(0,0,1)],             # M2
        [(0,1,1), (1,1,-1)],  # M3
        [(1,0,1), (0,0,-1)],  # M4
        [(1,1,1)],             # M5
        [(0,0,1), (0,1,1)],   # M6
        [(1,0,1), (1,1,1)],   # M7
    ]
    
    C_block_coeffs = [
        [(0,0,1), (1,1,1)],   # M1 -> C00, C11
        [(1,0,1), (1,1,-1)],  # M2 -> C10, C11
        [(0,1,1), (1,1,1)],   # M3 -> C01, C11
        [(0,0,1), (1,0,1)],   # M4 -> C00, C10
        [(0,0,-1), (0,1,1)],  # M5 -> C00, C01
        [(1,1,1)],             # M6 -> C11
        [(0,0,1), (1,1,1)],   # M7 -> C00, C11
    ]
    
    rank_44 = 49
    U_44 = np.zeros((16, rank_44))
    V_44 = np.zeros((20, rank_44))  # j*5+k, but only k=0..3 used
    W_44 = np.zeros((20, rank_44))  # k*4+i, but only k=0..3 used
    
    for a in range(7):
        for b in range(7):
            r = a * 7 + b
            
            for i_block, j_block, coeff_a in A_block_coeffs[a]:
                for i_local in range(2):
                    for j_local in range(2):
                        i = i_block * 2 + i_local
                        j = j_block * 2 + j_local
                        U_44[i * 4 + j, r] = coeff_a * U_2[i_local * 2 + j_local, b]
            
            for j_block, k_block, coeff_b in B_block_coeffs[a]:
                for j_local in range(2):
                    for k_local in range(2):
                        j = j_block * 2 + j_local
                        k = k_block * 2 + k_local
                        V_44[j * 5 + k, r] = coeff_b * V_2[j_local * 2 + k_local, b]
            
            for k_block, i_block, coeff_c in C_block_coeffs[a]:
                for k_local in range(2):
                    for i_local in range(2):
                        k = k_block * 2 + k_local
                        i = i_block * 2 + i_local
                        W_44[k * 4 + i, r] = coeff_c * W_2[k_local * 2 + i_local, b]
    
    # Verify 4x4x4 part
    def verify_44x4(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    ok_44 = verify_44x4(U_44, V_44, W_44)
    
    if not ok_44:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1.0
                    V_s[j*p+k, col] = 1.0
                    W_s[k*n+i, col] = 1.0
                    col += 1
        return _to_list(U_s), _to_list(V_s), _to_list(W_s)
    
    # Add 4 terms for the 5th column (k=4)
    # T[i,j,4] = 1 for all i,j
    # Use: U[i*4+j, 49+i] = 1, V[j*5+4, 49+i] = 1, W[16+i, 49+i] = 1
    # Then sum_r U[i*4+j,r]*V[j*5+4,r]*W[16+i,r] = U[i*4+j,49+i]*1*1 = 1
    
    total_rank = rank_44 + 4
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((20, total_rank))
    
    U_full[:, :rank_44] = U_44
    V_full[:, :rank_44] = V_44
    W_full[:, :rank_44] = W_44
    
    for i in range(4):
        r = rank_44 + i
        for j in range(4):
            U_full[i * 4 + j, r] = 1.0
        for j in range(4):
            V_full[j * 5 + 4, r] = 1.0
        W_full[4 * 4 + i, r] = 1.0
    
    # Full verification
    def verify_full(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*4+j, r] * V[j*5+k, r] * W[k*4+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    ok_full = verify_full(U_full, V_full, W_full)
    
    if ok_full:
        return _to_list(U_full), _to_list(V_full), _to_list(W_full)
    else:
        # Fallback
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1.0
                    V_s[j*p+k, col] = 1.0
                    W_s[k*n+i, col] = 1.0
                    col += 1
        return _to_list(U_s), _to_list(V_s), _to_list(W_s)

def _to_list(arr):
    result = []
    for i in range(arr.shape[0]):
        row = []
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if abs(val - round(val)) < 1e-8:
                row.append(int(round(val)))
            elif abs(val - round(val * 2) / 2) < 1e-8:
                row.append(round(val * 2) / 2)
            else:
                row.append(float(val))
        result.append(row)
    return result
# EVOLVE-BLOCK-END
