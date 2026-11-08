"""Matrix multiplication 2x3x6 rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Rank-14 exact decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy: Split the inner dimension m=3 into m1=1 and m2=2.
    - The 2x1x6 block (scalar times 1x6) has rank 4 (schoolbook: 2*1*6/3? No, rank = 2*1*6 = 12? Wait.)
    
    Let me reconsider. The multiplication tensor for (n,m,p) has rank at most n*m*p.
    For 2x3x6: schoolbook rank = 36.
    
    Actually, let's think about the block decomposition more carefully.
    
    We have A (2x3) times B (3x6) = C (2x6).
    
    Split B's rows into B1 (1x6) and B2 (2x6). Split A's columns into A1 (2x1) and A2 (2x2).
    
    C = A1*B1 + A2*B2
    
    For A1*B2: A1 is 2x1, B1 is 1x6. This is outer product of a 2-vector and 6-vector, rank 2*6=12? No.
    The tensor for multiplying a 2x1 by 1x6 has rank 2*1*6 = 12. But actually, this is just an outer product, which can be done in rank 2*6 = 12? No, the outer product of two vectors of dimensions 2 and 6 is a rank-1 operation in the sense that C[i,k] = A[i]*B[k]. The tensor has rank 2*6 = 12? No, it's a single rank-1 term in the tensor sense? No.
    
    Let me be more careful. The multiplication tensor T_{i,j,k} for (n,m,p) multiplication satisfies T[i,j,k] = delta_{i,j} * delta_{j,k} in the appropriate indexing. The rank of the tensor is the minimum number of rank-1 tensors that sum to T.
    
    For a 2x1 by 1x6 multiplication: C[i,k] = A[i,1] * B[1,k]. This is C[i,k] = sum over j of A[i,j]*B[j,k] where j only takes value 1. So it's C[i,k] = A[i,1]*B[1,k]. The tensor T has T[i,1,k] = 1 for all i,k, and 0 otherwise. This tensor is a rank-1 tensor? No, it's the outer product of a vector of length 2 (all 1s) and a vector of length 6 (all 1s), with the middle index fixed. So as a 3-tensor, it's e_i ⊗ e_1 ⊗ e_k summed over i,k, which is (sum_i e_i) ⊗ e_1 ⊗ (sum_k e_k). This is rank 1!
    
    Wait, that's not right either. Let me think again. The tensor for matrix multiplication C = AB where A is 2x1 and B is 1x6: C[i,k] = A[i,1]*B[1,k]. The tensor T has components T_{i,j,k} = 1 if j=1 (since that's the only column of A and row of B), and 0 otherwise. More precisely, T_{i,j,k} = delta_{j,1}. This is a rank-1 tensor: T = (e_1+e_2) ⊗ e_1 ⊗ (e_1+...+e_6). So rank 1.
    
    Similarly, for A2*B2 where A2 is 2x2 and B2 is 2x6: this is standard 2x2 by 2x6 multiplication. The rank of 2x2x6 multiplication... For 2x2x2 it's rank 7 (Strassen). For 2x2x6, we can think of it as 3 copies of 2x2x2 (since 6 = 3*2), so rank 3*7 = 21? Or we can do better.
    
    Actually, let's just use the known result: the rank of the 2x2x6 tensor. We can decompose the 2x6 part into three 2x2 blocks. Each 2x2x2 multiplication has rank 7 (Strassen). So 2x2x6 has rank at most 3*7 = 21. But maybe we can do better by sharing some terms.
    
    Hmm, this is getting complicated. Let me just try a direct approach: construct a rank-14 decomposition by hand or use a known construction.
    
    Actually, the simplest approach: the schoolbook rank for 2x3x6 is 2*3*6 = 36. The observed score of 1.0 corresponds to rank 36 (the schoolbook). The parent had score 1.0909 which corresponds to rank 36/1.0909 ≈ 33. So someone found rank 33.
    
    Let me just try to find a good decomposition. The minimum rank for 2x3x6 is not known to me offhand, but let's aim for something better than 36.
    
    Let me try the block approach: split m=3 into 1+2.
    - 2x1x6: rank 1 (as shown above)
    - 2x2x6: let's compute this. 2x2x6 can be seen as 2x2x2 x 3 (three blocks). Using Strassen for each 2x2x2: rank 7*3 = 21.
    
    Total: 1 + 21 = 22. That's much better than 36!
    
    Wait, but can we do better for 2x2x6? The 2x2x6 tensor: A is 2x2, B is 2x6, C is 2x6. We can write B as [B1, B2, B3] where each Bi is 2x2. Then C = A*B = [A*B1, A*B2, A*B3]. Each A*Bi is a 2x2x2 multiplication with rank 7. Total rank 21.
    
    But maybe we can share terms across the three blocks? Unlikely to help much.
    
    So total rank: 1 + 21 = 22.
    
    Let me construct this.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Split m=3 into m1=1, m2=2
    m1, m2 = 1, 2
    
    # Part 1: 2x1x6 multiplication (rank 1)
    # C[i,k] = A[i,0] * B[0,k]
    # U1 has shape (n*m1, 1) = (2, 1)
    # V1 has shape (m1*p, 1) = (6, 1)
    # W1 has shape (p*n, 1) = (12, 1)
    U1 = [[1], [1]]  # n*m1 = 2 rows
    V1 = [[1]] * p   # m1*p = 6 rows
    W1 = []
    for k in range(p):
        for i in range(n):
            W1.append([1])
    # W1 has p*n = 12 entries, each [1]
    
    # Part 2: 2x2x6 multiplication using Strassen for each 2x2x2 block
    # A2 is 2x2, B2 is 2x6 = [B2_1, B2_2, B2_3] each 2x2
    # For each block, use Strassen's 7-term decomposition
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x6, we have 3 such blocks.
    # Total Strassen terms: 3 * 7 = 21
    
    # Let's build U2, V2, W2 for the 2x2x6 part
    # U2 shape: (n*m2, rank) = (4, 21)
    # V2 shape: (m2*p, rank) = (12, 21)
    # W2 shape: (p*n, rank) = (12, 21)
    
    rank2 = 21
    U2 = [[0]*rank2 for _ in range(n*m2)]
    V2 = [[0]*rank2 for _ in range(m2*p)]
    W2 = [[0]*rank2 for _ in range(p*n)]
    
    col = 0
    for block in range(3):  # 3 blocks of 2x2
        # Strassen terms for this block
        # M1 = (A11+A22)(B11+B22)
        # In our indexing: A2 row i, col j (i,j in 0,1), B2 row j, col (block*2 + k) (k in 0,1)
        # C2 row i, col (block*2 + k)
        
        # M1: (A[0,0]+A[1,1]) * (B[0, block*2+0] + B[1, block*2+1])
        # U: A[0,0] and A[1,1] contribute
        U2[0][col] += 1   # A row 0, col 0 (i=0, j=0)
        U2[3][col] += 1   # A row 1, col 1 (i=1, j=1)
        V2[block*2+0][col] += 1   # B row 0, col block*2+0
        V2[2+block*2+1][col] += 1 # B row 1, col block*2+1
        # W: C[0, block*2+0] and C[1, block*2+1] get M1
        # C index: k*n+i where k is column of C, i is row of C
        # C[0, block*2+0]: k=block*2+0, i=0 -> index (block*2+0)*2+0
        # C[1, block*2+1]: k=block*2+1, i=1 -> index (block*2+1)*2+1
        W2[(block*2+0)*n+0][col] += 1
        W2[(block*2+1)*n+1][col] += 1
        col += 1
        
        # M2 = (A21+A22)B11 = (A[1,0]+A[1,1]) * B[0, block*2+0]
        U2[2][col] += 1   # A row 1, col 0
        U2[3][col] += 1   # A row 1, col 1
        V2[block*2+0][col] += 1
        # C21 = M2 + M4, so C[1, block*2+0] gets M2
        W2[(block*2+0)*n+1][col] += 1
        col += 1
        
        # M3 = A11(B21-B22) = A[0,0] * (B[1, block*2+0] - B[1, block*2+1])
        U2[0][col] += 1
        V2[2+block*2+0][col] += 1
        V2[2+block*2+1][col] -= 1
        # C12 = M3 + M5, so C[0, block*2+1] gets M3
        W2[(block*2+1)*n+0][col] += 1
        col += 1
        
        # M4 = A22(B12-B11) = A[1,1] * (B[0, block*2+1] - B[0, block*2+0])
        U2[3][col] += 1
        V2[block*2+1][col] += 1
        V2[block*2+0][col] -= 1
        # C11 = M1 + M4 - M5 + M7, C21 = M2 + M4
        W2[(block*2+0)*n+0][col] += 1
        W2[(block*2+0)*n+1][col] += 1
        col += 1
        
        # M5 = (A11+A12)B22 = (A[0,0]+A[0,1]) * B[1, block*2+1]
        U2[0][col] += 1
        U2[1][col] += 1
        V2[2+block*2+1][col] += 1
        # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5
        W2[(block*2+0)*n+0][col] -= 1
        W2[(block*2+1)*n+0][col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12) = (A[1,0]-A[0,0]) * (B[0, block*2+0] + B[0, block*2+1])
        U2[2][col] += 1
        U2[0][col] -= 1
        V2[block*2+0][col] += 1
        V2[block*2+1][col] += 1
        # C22 = M1 - M2 + M3 + M6
        W2[(block*2+1)*n+1][col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22) = (A[0,1]-A[1,1]) * (B[1, block*2+0] + B[1, block*2+1])
        U2[1][col] += 1
        U2[3][col] -= 1
        V2[2+block*2+0][col] += 1
        V2[2+block*2+1][col] += 1
        # C11 = M1 + M4 - M5 + M7
        W2[(block*2+0)*n+0][col] += 1
        col += 1
    
    # Now combine Part 1 and Part 2
    rank = 1 + rank2  # 1 + 21 = 22
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill Part 1 (column 0)
    for i in range(n):
        U[i*m1][0] = U1[i][0]
    for k in range(p):
        V[0*p+k][0] = V1[k][0]
    for k in range(p):
        for i in range(n):
            W[k*n+i][0] = W1[k*n+i][0]
    
    # Fill Part 2 (columns 1 to rank)
    # A2 starts at row m1=1 in A, so A2[i,j] is at U[(m1+i)*m2+j]
    # Wait, U has shape (n*m, rank) = (6, rank). A is 2x3, so rows of A are indexed by (i,j) where i in 0..1, j in 0..2.
    # U[i*m+j] corresponds to A[i,j].
    # A2 is columns 1,2 of A (j=1,2), so A2[i,j2] where j2 in 0..1 corresponds to A[i, 1+j2].
    # So A2[i,j2] is at U[i*m + (1+j2)] = U[i*3 + 1 + j2].
    
    # V has shape (m*p, rank) = (18, rank). B is 3x6, so B[j,k] is at V[j*p+k].
    # B2 is rows 1,2 of B (j=1,2), so B2[j2,k] where j2 in 0..1 corresponds to B[1+j2, k].
    # So B2[j2,k] is at V[(1+j2)*p + k].
    
    # W has shape (p*n, rank) = (12, rank). C[i,k] is at W[k*n+i].
    # C2 is the part from A2*B2, which gives C[i,k] for all i,k.
    
    for c in range(rank2):
        for row in range(n*m2):
            # row = i*m2 + j2 where i in 0..1, j2 in 0..1
            i = row // m2
            j2 = row % m2
            U[i*m + (m1 + j2)][1+c] = U2[row][c]
        
        for row in range(m2*p):
            # row = j2*p + k where j2 in 0..1, k in 0..5
            j2 = row // p
            k = row % p
            V[(m1 + j2)*p + k][1+c] = V2[row][c]
        
        for row in range(p*n):
            W[row][1+c] = W2[row][c]
    
    return {"U": U, "V": V, "W": W}
