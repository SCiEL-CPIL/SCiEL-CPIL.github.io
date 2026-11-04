"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U shape: (4, 7)  indexed by (i,j)
    V shape: (4, 7)  indexed by (j,k)
    W shape: (4, 7)  indexed by (k,i)
    
    Standard Strassen 7-term decomposition:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U[i*m+j, c] where i,j in {0,1}, m=2
    # Row index: i*2+j
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A: rows 0,1 contribute; B: cols 0,1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # Term 2: M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10 (k=0,i=1)
    W[3, 1] = -1  # C11 (k=1,i=1)
    
    # Term 3: M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    W[1, 2] = 1  # C01 (k=1,i=0)
    W[3, 2] = 1  # C11 (k=1,i=1)
    
    # Term 4: M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=0,i=1)
    
    # Term 5: M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1  # C00 (k=0,i=0)
    W[3, 4] = 1  # C11 (k=1,i=1)
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11 (k=1,i=1)
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00 (k=0,i=0)
    
    return U, V, W


def _strassen_2x2x1():
    """
    Returns U, V, W for 2x2x1 matrix-vector multiplication.
    A is 2x2, B is 2x1 (vector), C is 2x1.
    Rank 4 decomposition.
    
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    
    This is just the 4 terms of schoolbook, but let's see if we can do better.
    Actually for 2x2x1, the tensor is:
    C_i = sum_j A_ij * B_j
    
    The multiplication tensor has entries T[i,j,k] = 1 if i==k (output row i, input col j, vector comp k).
    Wait, let me think more carefully.
    
    For matrix-vector: C = A * v where A is 2x2, v is 2x1, C is 2x1.
    C_i = sum_j A_ij * v_j
    
    The tensor T[i, j, k] where i is output row, j is A column (which is v index), k is... 
    Actually the tensor is T[i, j, v_k] = 1 if j == v_k, i.e., the A entry (i,j) multiplies v_j to give C_i.
    
    So T[i, j, k] = 1 if j == k.
    
    This is a rank-4 tensor (it's just the identity in the j-k indices times the A indices).
    Actually: T[i,j,k] = delta_{j,k}. This can be decomposed as sum_k (e_k in A-indices for column k) * (e_k in B-indices) * (e_k in C-indices for all rows).
    
    Wait: T[i,j,k] = delta_{j,k}. 
    Term for k=0: A has column 0 active (rows 0,1), B has component 0, C has components 0,1.
    Term for k=1: A has column 1 active (rows 0,1), B has component 1, C has components 0,1.
    
    So rank 4:
    c=0: U[0,0]=1, U[1,0]=1 (A col 0, both rows); V[0,0]=1 (B comp 0); W[0,0]=1, W[1,0]=1 (C both rows)
    c=1: U[2,1]=1, U[3,1]=1 (A col 1, both rows); V[1,1]=1 (B comp 1); W[2,1]=1, W[3,1]=1 (C both rows)
    c=2: U[0,2]=1, U[1,2]=1; V[0,2]=1; W[0,2]=1, W[1,2]=1  -- wait this is same as c=0
    
    Hmm, let me reconsider. The tensor T[i,j,k] = delta_{j,k} for i in {0,1}, j in {0,1}, k in {0,1}.
    
    T[0,0,0]=1, T[0,1,1]=1
    T[1,0,0]=1, T[1,1,1]=1
    
    Rank: We can write this as sum of 4 rank-1 tensors:
    (i=0,j=0,k=0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    (i=0,j=1,k=1): U[1,1]=1, V[1,1]=1, W[1,1]=1
    (i=1,j=0,k=0): U[0,2]=1, V[0,2]=1, W[2,2]=1
    (i=1,j=1,k=1): U[1,3]=1, V[1,3]=1, W[3,3]=1
    
    But can we do rank < 4? The tensor is:
    T = e_0^i e_0^j e_0^k + e_0^i e_1^j e_1^k + e_1^i e_0^j e_0^k + e_1^i e_1^j e_1^k
    = (e_0^i + e_1^i) e_0^j e_0^k + (e_0^i + e_1^i) e_1^j e_1^k
    
    So rank 2! 
    c=0: U[0,0]=1, U[1,0]=1; V[0,0]=1; W[0,0]=1, W[1,0]=1
    c=1: U[0,1]=1, U[1,1]=1; V[1,1]=1; W[2,1]=1, W[3,1]=1
    
    Wait, W index is k*n+i = k*2+i. So:
    c=0: W[0*2+0, 0]=W[0,0]=1, W[0*2+1,0]=W[1,0]=1
    c=1: W[1*2+0, 1]=W[2,1]=1, W[1*2+1, 1]=W[3,1]=1
    
    Yes! Rank 2 for 2x2x1.
    """
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((4, 2))
    
    # c=0: A col 0 (rows 0,1), B comp 0, C rows 0,1 (k=0)
    U[0, 0] = 1
    U[1, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1  # k=0, i=0
    W[1, 0] = 1  # k=0, i=1
    
    # c=1: A col 1 (rows 0,1), B comp 1, C rows 0,1 (k=1)
    U[2, 1] = 1
    U[3, 1] = 1
    V[1, 1] = 1
    W[2, 1] = 1  # k=1, i=0
    W[3, 1] = 1  # k=1, i=1
    
    return U, V, W


def construct():
    """
    Construct a rank-22 decomposition for 4x4x5 matrix multiplication.
    
    Strategy: Block-diagonal decomposition.
    Split the 4x4x5 problem into:
    - Top-left 2x2x2 block (A[0:2,0:2] * B[0:2,0:2]) -> rank 7
    - Top-right 2x2x3 block (A[0:2,0:2] * B[0:2,2:5]) -> need to handle
    - Bottom-left 2x2x2 block (A[2:4,0:2] * B[0:2,0:2]) -> rank 7  
    - Bottom-right 2x2x3 block (A[2:4,0:2] * B[0:2,2:5]) -> need to handle
    
    Actually, let me think differently. The task is A(4x4) * B(4x5) = C(4x5).
    
    Better approach: Think of B as 4x5 = [B_left(4x2) | B_right(4x3)]
    Then C = [A*B_left(4x2) | A*B_right(4x3)]
    
    For A*B_left: 4x4 * 4x2 = 4x2. Split into four 2x2x2 blocks:
    - A[0:2,0:2]*B_left[0:2,:] -> 2x2 output
    - A[0:2,2:4]*B_left[2:4,:] -> 2x2 output
    - A[2:4,0:2]*B_left[0:2,:] -> 2x2 output
    - A[2:4,2:4]*B_left[2:4,:] -> 2x2 output
    Each is 2x2x2, rank 7 each. Total: 4*7 = 28 for the left half.
    
    For A*B_right: 4x4 * 4x3 = 4x3. This is trickier.
    
    Alternative: Use the structure more carefully.
    
    Let me try a different decomposition. Consider:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    where i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3,4}.
    
    Block decomposition:
    Split i into {0,1} and {2,3}, j into {0,1} and {2,3}.
    
    C[0:2, k] = A[0:2,0:2]*B[0:2,k] + A[0:2,2:4]*B[2:4,k]
    C[2:4, k] = A[2:4,0:2]*B[0:2,k] + A[2:4,2:4]*B[2:4,k]
    
    For the first term A[0:2,0:2]*B[0:2,k]:
    This is a 2x2 matrix times a 2x5 matrix = 2x5 matrix.
    We can decompose this as: for each k, it's 2x2x1, but we want to share structure.
    
    Actually, A[0:2,0:2]*B[0:2,:] is 2x2 * 2x5 = 2x5.
    The tensor for this: T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1,2,3,4}.
    T[i,j,k] = 1 (all entries are 1 since it's just multiplication).
    Wait no. The tensor is T[i, j, k] = 1 for all valid i,j,k.
    This is just the all-ones tensor of shape 2x2x5, which has rank min(2,2,5)=2? No.
    
    The multiplication tensor for (2x2)*(2x5)->(2x5):
    C[i,k] = sum_j A[i,j] * B[j,k]
    Tensor T[i, j, k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,4}.
    
    This tensor can be decomposed as:
    sum_j (e_j in A-col) * (e_j in B-row) * (all-ones in C)
    = sum_{j=0}^{1} (A-col j) * (B-row j) * (C all rows, all cols)
    
    So rank 2! Each term:
    c=0: U[i,0] = 1 if j=0 (i.e., U[i,0]=1 for both i), V[j,0]=1 if j=0 (V[0,0]=1), W[k,n+i,0]=1 for all k,i
    c=1: U[i,1] = 1 if j=1, V[1,1]=1, W[k*2+i,1]=1 for all k,i
    
    Wait, but this doesn't capture the correct structure. Let me re-examine.
    
    For 2x2 * 2x5 -> 2x5:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Term 0: A[i,0]*B[0,k] for all i,k
    U[i,0] = 1 (for i=0,1)
    V[0,0] = 1
    W[k*2+i, 0] = 1 (for all k=0..4, i=0,1)
    
    Term 1: A[i,1]*B[1,k] for all i,k
    U[i,1] = 1 (for i=0,1)
    V[1,1] = 1
    W[k*2+i, 1] = 1 (for all k=0..4, i=0,1)
    
    Yes, rank 2 for each 2x2x5 block!
    
    So the full decomposition:
    C[0:2, 0:5] = A[0:2,0:2]*B[0:2,0:5] + A[0:2,2:4]*B[2:4,0:5]
    C[2:4, 0:5] = A[2:4,0:2]*B[0:2,0:5] + A[2:4,2:4]*B[2:4,0:5]
    
    Each of these 4 terms is a 2x2x5 multiplication, rank 2 each.
    Total rank: 4 * 2 = 8.
    
    But wait, is this correct? Let me verify.
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    If I split j into {0,1} and {2,3}:
    C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    
    For i in {0,1} (top half of C):
    C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    
    First sum: A[i,j] for i in {0,1}, j in {0,1} is a 2x2 block. B[j,k] for j in {0,1} is a 2x5 block.
    This is (2x2)*(2x5) -> 2x5, rank 2 as shown.
    
    Second sum: A[i,j] for i in {0,1}, j in {2,3} is a 2x2 block. B[j,k] for j in {2,3} is a 2x5 block.
    Also (2x2)*(2x5) -> 2x5, rank 2.
    
    Similarly for i in {2,3}.
    
    Total rank: 4 * 2 = 8.
    
    Let me construct this.
    
    The full problem: n=4, m=4, p=5.
    U shape: (n*m, rank) = (16, rank)
    V shape: (m*p, rank) = (20, rank)
    W shape: (p*n, rank) = (20, rank)
    
    U index: i*m+j = i*4+j
    V index: j*p+k = j*5+k
    W index: k*n+i = k*4+i
    
    Block 1: i in {0,1}, j in {0,1}
    A indices: i*4+j for i in {0,1}, j in {0,1} -> {0,1,4,5}
    B indices: j*5+k for j in {0,1}, k in {0,1,2,3,4} -> {0,1,2,3,4, 5,6,7,8,9}
    C indices: k*4+i for k in {0,1,2,3,4}, i in {0,1} -> {0,1, 4,5, 8,9, 12,13, 16,17}
    
    Term 1a (j=0): 
    U[i*4+0, c] = 1 for i in {0,1}
    V[0*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {0,1}
    
    Term 1b (j=1):
    U[i*4+1, c] = 1 for i in {0,1}
    V[1*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {0,1}
    
    Block 2: i in {0,1}, j in {2,3}
    A indices: i*4+j for i in {0,1}, j in {2,3} -> {2,3,6,7}
    B indices: j*5+k for j in {2,3}, k in {0,1,2,3,4} -> {10,11,12,13,14, 15,16,17,18,19}
    C indices: k*4+i for k in {0,1,2,3,4}, i in {0,1} -> {0,1, 4,5, 8,9, 12,13, 16,17}
    
    Term 2a (j=2):
    U[i*4+2, c] = 1 for i in {0,1}
    V[2*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {0,1}
    
    Term 2b (j=3):
    U[i*4+3, c] = 1 for i in {0,1}
    V[3*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {0,1}
    
    Block 3: i in {2,3}, j in {0,1}
    A indices: i*4+j for i in {2,3}, j in {0,1} -> {8,9,12,13}
    B indices: j*5+k for j in {0,1}, k in {0,1,2,3,4} -> {0,1,2,3,4, 5,6,7,8,9}
    C indices: k*4+i for k in {0,1,2,3,4}, i in {2,3} -> {2,3, 6,7, 10,11, 14,15, 18,19}
    
    Term 3a (j=0):
    U[i*4+0, c] = 1 for i in {2,3}
    V[0*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {2,3}
    
    Term 3b (j=1):
    U[i*4+1, c] = 1 for i in {2,3}
    V[1*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {2,3}
    
    Block 4: i in {2,3}, j in {2,3}
    A indices: i*4+j for i in {2,3}, j in {2,3} -> {10,11,14,15}
    B indices: j*5+k for j in {2,3}, k in {0,1,2,3,4} -> {10,11,12,13,14, 15,16,17,18,19}
    C indices: k*4+i for k in {0,1,2,3,4}, i in {2,3} -> {2,3, 6,7, 10,11, 14,15, 18,19}
    
    Term 4a (j=2):
    U[i*4+2, c] = 1 for i in {2,3}
    V[2*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {2,3}
    
    Term 4b (j=3):
    U[i*4+3, c] = 1 for i in {2,3}
    V[3*5+k, c] = 1 for k in {0,1,2,3,4}
    W[k*4+i, c] = 1 for k in {0,1,2,3,4}, i in {2,3}
    
    Total rank: 8.
    """
    n, m, p = 4, 4, 5
    rank = 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i_block in [0, 2]:  # i ranges: 0-1, 2-3
        for j_block in [0, 2]:  # j ranges: 0-1, 2-3
            for j_offset in [0, 1]:  # j = j_block, j_block+1
                j = j_block + j_offset
                for i in range(i_block, i_block + 2):
                    U[i * m + j, c] = 1
                for k in range(p):
                    V[j * p + k, c] = 1
                    W[k * n + i_block, c] = 1
                    W[k * n + i_block + 1, c] = 1
                c += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
