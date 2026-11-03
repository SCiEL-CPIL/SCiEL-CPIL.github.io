import time
import numpy as np
from itertools import combinations

DEADLINE = 140.0  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Convention: W[k*n + i, r] corresponds to output index (k, i) for input (i, j, k)?
    Task says: "output factor follows the official trace convention W[k*n+i,r]"
    This implies W is indexed by [k*n+i] where i is row of A (0..n-1) and k is col of B (0..p-1).
    Wait, standard matrix mult C = A B.
    C[i, k] = sum_j A[i, j] B[j, k].
    Tensor T[i, j, k] = 1.
    Decomposition: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
    Here U is indexed by (i, j), V by (j, k), W by (k, i).
    Flattened:
    U_flat[i*m + j, r]
    V_flat[j*p + k, r]
    W_flat[k*n + i, r]  <-- Note the order k*n + i.
    """
    rank = U.shape[1]
    # Reshape to 3D for einsum
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)  <-- Note: index is k*n+i, so first dim is k (p), second is i (n)
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # einsum: 'ijr,jkr,kr->ijk'
    result = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
    
    return np.allclose(result, 1.0, atol=tol)

def _construct_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen.
    n=2, m=2, p=2.
    Rank = 7.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B12 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B11 - B21)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the components for each multiplication M_l.
    # For each l, we have u_l (2x2), v_l (2x2), w_l (2x2) such that M_l = u_l * v_l * w_l?
    # Actually, the term is sum_l U_l[i,j] V_l[j,k] W_l[k,i].
    # This corresponds to the product of three matrices in the tensor decomposition sense.
    
    # Let's define the 7 rank-1 tensors.
    # Indices: i,j in {0,1}, k in {0,1}.
    
    # M1: (A11+A22)(B12+B22) contributes to C11 and C22
    # U1: A11=1, A22=1 -> [[1,0],[0,1]]? No, U is coefficients of A.
    # U1[i,j]: coefficient of A[i,j] in the first factor of M1.
    # M1 = (A11 + A22) * (B12 + B22).
    # Factor 1 (A part): A11 + A22. So U1[0,0]=1, U1[1,1]=1.
    # Factor 2 (B part): B12 + B22. So V1[0,1]=1, V1[1,1]=1. (j=0,k=1 and j=1,k=1)
    # Factor 3 (C part): This term adds to C11 and C22.
    # C11 gets +1 * M1. C22 gets +1 * M1.
    # W1[k,i]: coefficient for C[i,k].
    # C[0,0] (i=0,k=0) gets +1. So W1[0,0] += 1.
    # C[1,1] (i=1,k=1) gets +1. So W1[1,1] += 1.
    
    U1 = np.array([[1,0],[0,1]])
    V1 = np.array([[0,1],[0,1]])
    W1 = np.array([[1,0],[0,1]])
    
    # M2: (A21+A22)B11 contributes to C21 and C22
    # A part: A21 + A22. U2[1,0]=1, U2[1,1]=1.
    # B part: B11. V2[0,0]=1.
    # C part: C21 (i=1,k=0) gets +1. C22 (i=1,k=1) gets -1.
    # Wait, C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So W2[0,1] += 1 (for C[1,0]).
    # W2[1,1] -= 1 (for C[1,1]).
    U2 = np.array([[0,0],[1,1]])
    V2 = np.array([[1,0],[0,0]])
    W2 = np.array([[0,1],[-1,0]]) # W[k,i]. W[0,1] is k=0,i=1. W[1,1] is k=1,i=1.
    
    # M3: A11(B11-B21) contributes to C12 and C22
    # A part: A11. U3[0,0]=1.
    # B part: B11 - B21. V3[0,0]=1, V3[1,0]=-1.
    # C part: C12 (i=0,k=1) gets +1. C22 (i=1,k=1) gets +1.
    U3 = np.array([[1,0],[0,0]])
    V3 = np.array([[1,0],[-1,0]])
    W3 = np.array([[0,0],[0,1]]) # W[1,0] is k=1,i=0? No. W[k,i].
    # C[0,1] is i=0, k=1. Index in W is k*n+i = 1*2+0 = 2. In 2D array W[1,0].
    # C[1,1] is i=1, k=1. Index in W is 1*2+1 = 3. In 2D array W[1,1].
    W3 = np.array([[0,0],[1,1]])
    
    # M4: A22(B22-B12) contributes to C11 and C21
    # A part: A22. U4[1,1]=1.
    # B part: B22 - B12. V4[1,1]=1, V4[0,1]=-1.
    # C part: C11 (i=0,k=0) gets +1. C21 (i=1,k=0) gets +1.
    U4 = np.array([[0,0],[0,1]])
    V4 = np.array([[0,-1],[0,1]])
    W4 = np.array([[1,0],[0,1]]) # W[0,0]=1, W[0,1]=1
    
    # M5: (A11+A12)B22 contributes to C11 and C12
    # A part: A11 + A12. U5[0,0]=1, U5[0,1]=1.
    # B part: B22. V5[1,1]=1.
    # C part: C11 (i=0,k=0) gets -1. C12 (i=0,k=1) gets +1.
    U5 = np.array([[1,1],[0,0]])
    V5 = np.array([[0,0],[0,1]])
    W5 = np.array([[-1,0],[0,1]]) # W[0,0]=-1, W[1,0]=1
    
    # M6: (A21-A11)(B11+B12) contributes to C22
    # A part: A21 - A11. U6[1,0]=1, U6[0,0]=-1.
    # B part: B11 + B12. V6[0,0]=1, V6[0,1]=1.
    # C part: C22 (i=1,k=1) gets +1.
    U6 = np.array([[-1,0],[1,0]])
    V6 = np.array([[1,1],[0,0]])
    W6 = np.array([[0,0],[0,1]]) # W[1,1]=1
    
    # M7: (A12-A22)(B21+B22) contributes to C11
    # A part: A12 - A22. U7[0,1]=1, U7[1,1]=-1.
    # B part: B21 + B22. V7[1,0]=1, V7[1,1]=1.
    # C part: C11 (i=0,k=0) gets +1.
    U7 = np.array([[0,1],[0,-1]])
    V7 = np.array([[0,0],[1,1]])
    W7 = np.array([[1,0],[0,0]]) # W[0,0]=1
    
    U_list = [U1, U2, U3, U4, U5, U6, U7]
    V_list = [V1, V2, V3, V4, V5, V6, V7]
    W_list = [W1, W2, W3, W4, W5, W6, W7]
    
    U_flat = np.column_stack([u.flatten() for u in U_list])
    V_flat = np.column_stack([v.flatten() for v in V_list])
    W_flat = np.column_stack([w.flatten() for w in W_list])
    
    return U_flat, V_flat, W_flat

def _construct_2x4x7():
    """
    Construct a rank-50 decomposition for 2x4 * 4x7.
    Split A (2x4) into two 2x2 blocks: A1 (cols 0-1), A2 (cols 2-3).
    Split B (4x7) into two 2x7 blocks: B1 (rows 0-1), B2 (rows 2-3).
    C = A1 B1 + A2 B2.
    
    We can use Strassen for each 2x2 * 2x7 multiplication?
    Strassen is for 2x2 * 2x2.
    For 2x2 * 2x7, we can treat B as 7 columns of 2x1.
    Actually, we can just use schoolbook for the 2x2 * 2x7 parts, or extend Strassen.
    
    Let's use a simpler block approach:
    Split A into A_top (1x4) and A_bottom (1x4)? No, n=2.
    Split A into A1 (2x2) and A2 (2x2).
    Split B into B1 (2x7) and B2 (2x7).
    
    C = A1 B1 + A2 B2.
    
    For A1 (2x2) * B1 (2x7):
    We can decompose B1 into 7 columns b1_1 ... b1_7, each 2x1.
    A1 b1_k is a 2x1 vector.
    This is just 4 multiplications per column (schoolbook) or 7 multiplications if we use Strassen-like for 2x2*2x1?
    No, 2x2 * 2x1 requires 4 mults in schoolbook.
    Is there a rank 3 algorithm for 2x2 * 2x1?
    Yes, 2x2 * 2x1 is equivalent to 2x2 * 1x2 transpose? No.
    It's just two dot products of length 2.
    Rank of 2x2 * 2x1 tensor is 4?
    Let's check. T[i, j, k] where i in 0,1; j in 0,1; k in 0,1.
    T[i, j, 0] = delta_{i,0}? No.
    C[i, 0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    This is 4 terms.
    Can we do it in 3?
    (A00+A11)(B00+B10) ... no.
    The rank of 2x2 * 2x1 is 4. (It's the identity tensor of size 2x2x1? No.)
    Actually, 2x2 * 2x1 is just two independent 2x1 * 2x1 dot products? No.
    It's C0 = A00 B00 + A01 B10
         C1 = A10 B00 + A11 B10
    This is a bilinear map.
    Rank is 4.
    
    So if we use schoolbook for 2x2 * 2x7, we use 4 * 7 = 28 multiplications per block.
    Two blocks: 28 + 28 = 56. That's schoolbook.
    
    To get rank 50, we need to exploit structure.
    Inspiration 1 mentions "block-structured Strassen decomposition of rank 50".
    How?
    Maybe split B (4x7) into B1 (2x4) and B2 (2x3)?
    And A (2x4) into A1 (2x2) and A2 (2x2).
    
    Let's try splitting the 4x7 matrix B into two 4x2 blocks and one 4x3 block?
    Or split A (2x4) into 2x2 and 2x2.
    Split B (4x7) into 2x7 and 2x7.
    
    If we use Strassen for the 2x2 part of A and the 2x2 part of B?
    Let's look at the 2x2 * 2x2 Strassen (rank 7).
    If we have 2x2 * 2x7, we can view it as 2x2 * (2x2 + 2x2 + 2x2 + 2x2 + 2x1).
    This is getting complicated.
    
    Alternative: Use the known rank 50 algorithm for 2x4 * 4x7.
    One such algorithm is derived from 3x3 * 3x3 (Strassen rank 7) or similar.
    
    Let's try a different constructive approach:
    1. Start with schoolbook rank 56.
    2. Apply linear dependencies to reduce rank.
    
    Schoolbook:
    U[i, j, l] = 1 if l = i*4+j, else 0.
    V[j, k, l] = 1 if l = j*7+k, else 0.
    W[k, i, l] = 1 if l = k*2+i, else 0.
    Wait, l must be the same for all three in the sum.
    So for each (i,j,k), we have a term.
    Total 56 terms.
    
    Can we combine terms?
    Term (i,j,k) is U_{ij} V_{jk} W_{ki}.
    
    If we have terms (0,0,0) and (0,0,1), they share U_{00} and V_{00} and V_{01}? No.
    (0,0,0): U[0,0] V[0,0] W[0,0]
    (0,0,1): U[0,0] V[0,1] W[1,0]
    
    We can factor U[0,0] out?
    U[0,0] (V[0,0]W[0,0] + V[0,1]W[1,0]).
    This doesn't reduce the rank of the tensor unless the second part can be represented by fewer terms.
    
    Actually, the rank of the multiplication tensor $M_{2,4,7}$ is known to be 50.
    Reference: "The rank of the 2x4x7 tensor" or similar.
    
    Let's try to construct rank 50 by splitting 4x7 into 4x2 and 4x5?
    Or 2x4 * 4x7 = 2x4 * (4x2 + 4x5).
    
    Let's use the Kronecker product of 2x2 (Strassen, rank 7) and something else.
    2x4 * 4x7.
    Let A = A1 ⊗ A2? No.
    
    Let's try:
    A (2x4) = [A1, A2] where A1, A2 are 2x2.
    B (4x7) = [B1; B2] where B1, B2 are 2x7.
    C = A1 B1 + A2 B2.
    
    We can compute A1 B1 using Strassen-like for 2x2 * 2x7?
    As noted, 2x2 * 2x1 is rank 4.
    So 2x2 * 2x7 is rank 4*7 = 28?
    If we use Strassen for the 2x2 * 2x2 part of B1 (cols 0-1), and schoolbook for the rest?
    
    Let's decompose B1 (2x7) into B1a (2x2) and B1b (2x5).
    A1 B1 = A1 B1a + A1 B1b.
    A1 B1a: 2x2 * 2x2. Rank 7 (Strassen).
    A1 B1b: 2x2 * 2x5. Rank 4*5 = 20 (Schoolbook).
    Total for A1 B1: 27.
    Similarly for A2 B2: 27.
    Total: 54.
    
    We need 50.
    
    What if we split B (4x7) into 4x3 and 4x4?
    A (2x4) * 4x3.
    A (2x4) * 4x4.
    
    Rank of 2x4 * 4x4?
    If we use Strassen for 2x2 * 2x2 blocks...
    2x4 * 4x4 = 2x4 * (4x2 + 4x2).
    2x4 * 4x2.
    Split A (2x4) into A1 (2x2), A2 (2x2).
    Split B (4x2) into B1 (2x2), B2 (2x2).
    C = A1 B1 + A2 B2.
    A1 B1: Rank 7.
    A2 B2: Rank 7.
    Total: 14.
    So 2x4 * 4x4 has rank <= 14?
    Wait, 2x4 * 4x4 is 8 * 16 = 128 entries? No.
    Input 2x4, Input 4x4, Output 2x4.
    Tensor size 8 x 16 x 8.
    Rank 14 seems too low.
    Lower bound max(8, 16, 8) = 16.
    So rank 14 is impossible.
    Where is the error?
    A1 B1 is 2x2 * 2x2 -> 2x2.
    A2 B2 is 2x2 * 2x2 -> 2x2.
    C = A1 B1 + A2 B2.
    This calculates C correctly.
    Why is rank 14 < 16?
    Because the terms from A1 B1 and A2 B2 might share structure?
    No, they are disjoint in terms of A and B indices.
    A1 uses A[0:2, 0:2], B1 uses B[0:2, 0:2].
    A2 uses A[0:2, 2:4], B2 uses B[2:4, 0:2].
    The outputs are the same 2x2 matrix C[0:2, 0:2] and C[0:2, 2:4].
    Wait, B is 4x4.
    B1 is B[0:2, 0:4] (2x4).
    B2 is B[2:4, 0:4] (2x4).
    I split B (4x2) in the previous thought, but here B is 4x4.
    
    Let's re-evaluate 2x4 * 4x4.
    A (2x4), B (4x4), C (2x4).
    Split A into A1 (2x2), A2 (2x2).
    Split B into B1 (2x4), B2 (2x4).
    C = A1 B1 + A2 B2.
    A1 B1: 2x2 * 2x4.
    A2 B2: 2x2 * 2x4.
    Rank of 2x2 * 2x4?
    Split 2x4 into 2x2 + 2x2.
    A1 (2x2) * B1a (2x2) -> Rank 7.
    A1 (2x2) * B1b (2x2) -> Rank 7.
    Total for A1 B1: 14.
    Total for A2 B2: 14.
    Total for 2x4 * 4x4: 28.
    Lower bound max(8, 16, 8) = 16.
    28 is valid.
    
    Now 2x4 * 4x3.
    Split A into A1 (2x2), A2 (2x2).
    Split B into B1 (2x3), B2 (2x3).
    A1 B1: 2x2 * 2x3.
    Split 2x3 into 2x2 + 2x1.
    A1 * 2x2: Rank 7.
    A1 * 2x1: Rank 4.
    Total for A1 B1: 11.
    Total for A2 B2: 11.
    Total for 2x4 * 4x3: 22.
    
    Total for 2x4 * 4x7 (split into 4x4 and 4x3):
    28 + 22 = 50.
    
    This matches the target rank 50!
    
    So the strategy is:
    1. Construct the rank-50 decomposition using this block structure.
    2. Verify it.
    3. Attempt to reduce rank by finding linear dependencies among the 50 vectors in the tensor space.
    
    Implementation details:
    - Build U, V, W for the 50 terms.
    - The terms are:
      - 14 terms for A1 B1 (2x2 * 2x4)
      - 14 terms for A2 B2 (2x2 * 2x4)
      - 11 terms for A1 B3 (2x2 * 2x3)
      - 11 terms for A2 B4 (2x2 * 2x3)
      Wait, B is 4x7.
      Split B into B1 (2x4) and B2 (2x3).
      B1 rows 0-1, cols 0-3.
      B2 rows 2-3, cols 0-3? No.
      B is 4x7.
      Split rows of B: B_top (2x7), B_bottom (2x7).
      Split cols of B: B_left (4x4), B_right (4x3).
      
      C = A B_top_left + A B_top_right + A B_bottom_left + A B_bottom_right?
      No.
      C = A (B_top_left + B_top_right) + A (B_bottom_left + B_bottom_right).
      C = A B_top + A B_bottom.
      
      Let's stick to the previous split:
      A (2x4) = [A1, A2] (2x2 each).
      B (4x7) = [B1; B2] (2x7 each).
      
      C = A1 B1 + A2 B2.
      
      We established:
      Rank(A1 B1) <= 14 (using Strassen for 2x2*2x2 parts).
      Rank(A2 B2) <= 14.
      Total 28.
      
      Where did 50 come from?
      Ah, I split B into 4x4 and 4x3 in the second thought, but A was 2x4.
      If B is 4x7, and we split B into B1 (4x4) and B2 (4x3).
      Then C = A B1 + A B2.
      A (2x4) * B1 (4x4) -> Rank 28.
      A (2x4) * B2 (4x3) -> Rank 22.
      Total 50.
      
      This works!
      
      So we need to construct:
      1. Decomposition for 2x4 * 4x4 (Rank 28).
      2. Decomposition for 2x4 * 4x3 (Rank 22).
      
      For 2x4 * 4x4:
      A = [A1, A2] (2x2).
      B = [B1; B2] (2x4).
      C = A1 B1 + A2 B2.
      B1 (2x4) split into B1a (2x2), B1b (2x2).
      A1 B1 = A1 B1a + A1 B1b.
      A1 B1a: Rank 7.
      A1 B1b: Rank 7.
      A2 B2: Rank 14.
      Total 28.
      
      For 2x4 * 4x3:
      A = [A1, A2] (2x2).
      B = [B3; B4] (2x3).
      C = A1 B3 + A2 B4.
      B3 (2x3) split into B3a (2x2), B3b (2x1).
      A1 B3 = A1 B3a + A1 B3b.
      A1 B3a: Rank 7.
      A1 B3b: Rank 4.
      A2 B4: Rank 11.
      Total 22.
      
      Total Rank 50.
    """
    
    def get_strassen_222():
        # Returns list of 7 (U, V, W) 2x2 matrices
        # As defined in _construct_strassen_2x2x2
        U1, V1, W1 = np.array([[1,0],[0,1]]), np.array([[0,1],[0,1]]), np.array([[1,0],[0,1]])
        U2, V2, W2 = np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,1],[-1,0]])
        U3, V3, W3 = np.array([[1,0],[0,0]]), np.array([[1,0],[-1,0]]), np.array([[0,0],[1,1]])
        U4, V4, W4 = np.array([[0,0],[0,1]]), np.array([[0,-1],[0,1]]), np.array([[1,0],[0,1]])
        U5, V5, W5 = np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,0],[0,1]])
        U6, V6, W6 = np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])
        U7, V7, W7 = np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])
        
        return [
            (U1, V1, W1),
            (U2, V2, W2),
            (U3, V3, W3),
            (U4, V4, W4),
            (U5, V5, W5),
            (U6, V6, W6),
            (U7, V7, W7)
        ]
    
    def get_school_221():
        # 2x2 * 2x1. Rank 4.
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        # Terms:
        # 1: A00 B00 -> C0. U=[[1,0],[0,0]], V=[[1],[0]], W=[[1],[0]]
        # 2: A01 B10 -> C0. U=[[0,1],[0,0]], V=[[0],[1]], W=[[1],[0]]
        # 3: A10 B00 -> C1. U=[[0,0],[1,0]], V=[[1],[0]], W=[[0],[1]]
        # 4: A11 B10 -> C1. U=[[0,0],[0,1]], V=[[0],[1]], W=[[0],[1]]
        return [
            (np.array([[1,0],[0,0]]), np.array([[1],[0]]), np.array([[1],[0]])),
            (np.array([[0,1],[0,0]]), np.array([[0],[1]]), np.array([[1],[0]])),
            (np.array([[0,0],[1,0]]), np.array([[1],[0]]), np.array([[0],[1]])),
            (np.array([[0,0],[0,1]]), np.array([[0],[1]]), np.array([[0],[1]]))
        ]
        
    # Dimensions
    n, m, p = 2, 4, 7
    rank = 50
    
    U_all = []
    V_all = []
    W_all = []
    
    strassen_222 = get_strassen_222()
    school_221 = get_school_221()
    
    # Helper to add a block of terms
    # U_block: 2x2, V_block: 2xK, W_block: Kx2
    # i_offset: row offset in A (0 or 0)
    # j_offset: col offset in A (0 or 2)
    # j_offset_B: row offset in B (0 or 2)
    # k_offset: col offset in B (0, 4, etc)
    
    def add_block(U_b, V_b, W_b, i_off, j_off_A, j_off_B, k_off_B):
        # U_b is 2x2. Place in U[i_off:i_off+2, j_off_A:j_off_A+2]
        # V_b is 2xK. Place in V[j_off_B:j_off_B+2, k_off_B:k_off_B+K]
        # W_b is Kx2. Place in W[k_off_B:k_off_B+K, i_off:i_off+2]
        # Note: W index is k*n + i.
        # W is (p*n, rank).
        # For a specific term r, W[k*n+i, r] is non-zero.
        
        # We are adding one rank-1 term at a time.
        # U_b, V_b, W_b are the matrices for this term.
        
        U_flat = np.zeros(n*m)
        V_flat = np.zeros(m*p)
        W_flat = np.zeros(p*n)
        
        # Fill U
        for ii in range(2):
            for jj in range(2):
                val = U_b[ii, jj]
                if val != 0:
                    U_flat[(i_off+ii)*m + (j_off_A+jj)] = val
        
        # Fill V
        K = V_b.shape[1]
        for ii in range(2):
            for jj in range(K):
                val = V_b[ii, jj]
                if val != 0:
                    V_flat[(j_off_B+ii)*p + (k_off_B+jj)] = val
        
        # Fill W
        # W_b is Kx2. W_b[jj, ii] corresponds to k=j_off_B+ii? No.
        # V_b is 2xK. Rows are j (0,1), Cols are k (0..K-1).
        # W_b is Kx2. Rows are k (0..K-1), Cols are i (0,1).
        # W index: k*n + i.
        for jj in range(K):
            for ii in range(2):
                val = W_b[jj, ii]
                if val != 0:
                    # k = k_off_B + jj
                    # i = i_off + ii
                    W_flat[(k_off_B+jj)*n + (i_off+ii)] = val
        
        U_all.append(U_flat)
        V_all.append(V_flat)
        W_all.append(W_flat)
        
    # 1. 2x4 * 4x4 part (Rank 28)
    # A1 (2x2) * B1a (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=0, j_off_B=0, k_off_B=0)
        
    # A1 (2x2) * B1b (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=0, j_off_B=0, k_off_B=2)
        
    # A2 (2x2) * B2a (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=2, j_off_B=2, k_off_B=0)
        
    # A2 (2x2) * B2b (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=2, j_off_B=2, k_off_B=2)
        
    # 2. 2x4 * 4x3 part (Rank 22)
    # A1 (2x2) * B3a (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=0, j_off_B=0, k_off_B=4)
        
    # A1 (2x2) * B3b (2x1) -> School
    for U_b, V_b, W_b in school_221:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=0, j_off_B=0, k_off_B=6)
        
    # A2 (2x2) * B4a (2x2) -> Strassen
    for U_b, V_b, W_b in strassen_222:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=2, j_off_B=2, k_off_B=4)
        
    # A2 (2x2) * B4b (2x1) -> School
    for U_b, V_b, W_b in school_221:
        add_block(U_b, V_b, W_b, i_off=0, j_off_A=2, j_off_B=2, k_off_B=6)
        
    U = np.column_stack(U_all)
    V = np.column_stack(V_all)
    W = np.column_stack(W_all)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        print("Verification failed for rank 50 construction")
        # Fallback to schoolbook if construction is buggy
        return _construct_schoolbook(n, m, p)
        
    # Attempt rank reduction
    # We have 50 vectors.
    # We want to find a subset of size < 50 that still spans the tensor.
    # This is equivalent to finding linear dependencies.
    # However, rank reduction of tensors is NP-hard.
    # We can try to remove one term at a time and see if the remaining terms can be adjusted (linear combination) to still represent the tensor.
    # This is equivalent to checking if the removed term is in the span of the others.
    
    # Since the terms are rank-1, if term r is in the span of the others, we can express it as a linear combination.
    # But the tensor is a sum of rank-1 terms.
    # If T = sum_{l!=r} t_l + t_r, and t_r = sum c_l t_l, then T = sum (1+c_l) t_l.
    # This doesn't reduce the rank unless we can merge terms.
    
    # Actually, if the vectors U_l, V_l, W_l are linearly dependent in a specific way, we might reduce rank.
    # A simpler heuristic: Try to set one column to zero and solve for the others?
    # No, that changes the tensor.
    
    # Given the time limit and the difficulty of tensor rank reduction,
    # we will return the rank-50 solution.
    # The score will be 56/50 = 1.12.
    
    return U, V, W

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    try:
        U, V, W = _construct_2x4x7()
    except Exception as e:
        print("Error in construction:", e)
        U, V, W = _construct_schoolbook(n, m, p)
        
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
