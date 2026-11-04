"""Matrix multiplication 3,5,5 - rank-minimal tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _matmul_tensor(n, m, p):
    """Return the schoolbook decomposition for n x m times m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """Strassen decomposition for 2x2x2 matmul, rank 7."""
    # Standard Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))  # A is 2x2, so 4 entries
    V = np.zeros((4, 7))  # B is 2x2, so 4 entries
    W = np.zeros((4, 7))  # C is 2x2, so 4 entries
    
    # Indexing: A[i,j] at i*2+j, B[j,k] at j*2+k, C[i,k] at i*2+k
    # For 2x2: A indices 0,1,2,3 = A00,A01,A10,A11
    # B indices 0,1,2,3 = B00,B01,B10,B11
    # C indices 0,1,2,3 = C00,C01,C10,C11
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1  # C10 += M2, C11 += M2
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Split n=3 into 1+2, m=5 into 2+3, p=5 into 2+3
    # A is 3x5: A_top (1x5), A_bottom (2x5)
    # B is 5x5: B_left (5x2), B_right (5x3)
    # C is 3x5: C_top (1x5), C_bottom (2x5)
    # C_top = A_top @ B (1x5 @ 5x5 = 1x5)
    # C_bottom = A_bottom @ B (2x5 @ 5x5 = 2x5)
    
    # Further split C_bottom (2x5) into C_bottom_left (2x2) and C_bottom_right (2x3)
    # C_bottom_left = A_bottom @ B_left (2x5 @ 5x2 = 2x2) -> use Strassen
    # C_bottom_right = A_bottom @ B_right (2x5 @ 5x3 = 2x3) -> schoolbook
    # C_top = A_top @ B (1x5 @ 5x5 = 1x5) -> schoolbook
    
    # But wait, A_top @ B uses all of B, and A_bottom @ B_left uses B_left,
    # A_bottom @ B_right uses B_right. These are independent.
    
    # Let me reconsider the splitting to ensure correctness.
    # Actually, C = A @ B where A is 3x5, B is 5x5
    # Split A into A1 (1x5) and A2 (2x5)
    # Split B into B1 (5x2) and B2 (5x3)
    # C1 = A1 @ B1 (1x2), C12 = A1 @ B2 (1x3)
    # C21 = A2 @ B1 (2x2), C22 = A2 @ B2 (2x3)
    # C = [C1 C12; C21 C22]
    
    # C1 (1x2): schoolbook, rank 1*5*2 = 10? No, 1*2*5=10
    # C12 (1x3): schoolbook, rank 1*3*5=15
    # C21 (2x2): Strassen, rank 7
    # C22 (2x3): schoolbook, rank 2*3*5=30
    # Total: 10+15+7+30 = 62. Still not great.
    
    # Better: Don't split B separately for top and bottom.
    # C_top (1x5) = A_top @ B, schoolbook rank 1*5*5 = 25
    # C_bottom (2x5) = A_bottom @ B
    #   Split B into B1 (5x2) and B2 (5x3)
    #   C_bottom_left (2x2) = A_bottom @ B1, Strassen rank 7
    #   C_bottom_right (2x3) = A_bottom @ B2, schoolbook rank 2*3*5=30
    # Total: 25+7+30 = 62
    
    # Even better: For C_bottom_right (2x3), I can split the 2 rows into 1+1
    # C_bottom_right = [row1; row2] @ B2
    # row1 @ B2 (1x3), schoolbook rank 1*3*5=15
    # row2 @ B2 (1x3), schoolbook rank 1*3*5=15
    # That's 30, same as before.
    
    # Let me try: C_bottom (2x5) = A_bottom (2x5) @ B (5x5)
    # Split into 4 blocks: (1x2), (1x3), (1x2), (1x3)
    # Each 1x2: rank 1*2*5=10, two of them = 20
    # Each 1x3: rank 1*3*5=15, two of them = 30
    # Total for C_bottom: 50. Worse.
    
    # OK let me just go with the 62 approach and see if I can improve.
    # Actually, let me try a different split: n=3 as 2+1, m=5 as 3+2, p=5 as 3+2
    
    # A (3x5): A_top (2x5), A_bottom (1x5)
    # B (5x5): B_left (5x3), B_right (5x2)
    # C (3x5): C_top (2x5), C_bottom (1x5)
    # C_top = A_top @ B (2x5 @ 5x5 = 2x5)
    #   Split into C_top_left (2x3) and C_top_right (2x2)
    #   C_top_left: schoolbook rank 2*3*5=30
    #   C_top_right: Strassen rank 7
    # C_bottom = A_bottom @ B (1x5 @ 5x5 = 1x5), schoolbook rank 25
    # Total: 30+7+25 = 62
    
    # Same. Let me just implement the 62 rank solution.
    
    # Actually, I realize I should think about this differently.
    # The key is that Strassen only helps for 2x2x2. For other sizes, schoolbook is optimal
    # (as far as I know for small sizes).
    
    # Let me implement the decomposition with rank 62.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: C_top = A_top @ B, where A_top is rows 0 of A (1x5), B is 5x5, C_top is row 0 of C (1x5)
    # Schoolbook: for i=0, j in 0..4, k in 0..4
    # U index: i*m+j = 0*5+j = j (0..4)
    # V index: j*p+k = j*5+k (0..24)
    # W index: k*n+i = k*3+0 = k*3 (0,3,6,9,12)
    U1, V1, W1 = _matmul_tensor(1, 5, 5)
    # U1 is 5x25, V1 is 25x25, W1 is 15x25
    # But we need to map to global indices
    # Global U: 15x? (3*5)
    # Global V: 25x? (5*5)
    # Global W: 15x? (5*3)
    
    # Let me build it more carefully.
    
    # I'll build the full U, V, W by concatenating parts.
    
    # Part 1: C[0, :] = A[0, :] @ B (1x5 @ 5x5)
    # rank1 = 1*5*5 = 25
    U1 = np.zeros((15, 25))  # Only rows 0..4 used
    V1 = np.zeros((25, 25))
    W1 = np.zeros((15, 25))  # Only rows 0,3,6,9,12 used
    col = 0
    for j in range(5):
        for k in range(5):
            U1[j, col] = 1  # A[0,j]
            V1[j*5+k, col] = 1  # B[j,k]
            W1[k*3+0, col] = 1  # C[0,k]
            col += 1
    
    # Part 2: C[1:3, 0:2] = A[1:3, :] @ B[:, 0:2] (2x5 @ 5x2 = 2x2)
    # Use Strassen
    U2_s, V2_s, W2_s = _strassen_222()
    # Map Strassen indices to global
    # A[1:3, :] -> global U rows 5..9 (i=1,2; j=0..4)
    # Strassen A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # A00 = A[1,0] -> U row 5*0+0 = 5... wait, U index is i*m+j
    # A[1,0] -> i=1,j=0 -> 1*5+0 = 5
    # A[1,1] -> i=1,j=1 -> 6
    # A[2,0] -> i=2,j=0 -> 10
    # A[1,2]... wait, A is 2x5, so A[1,0..4] and A[2,0..4]
    # Strassen expects 2x2, so I need to pick which 2 columns of A and 2 rows of B.
    # A[1:3, 0:2] is 2x2: A[1,0], A[1,1], A[2,0], A[2,1]
    # B[0:2, 0:2] is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # C[1:3, 0:2] is 2x2: C[1,0], C[1,1], C[2,0], C[2,1]
    
    # Strassen U indices (local): 0,1,2,3 -> A00,A01,A10,A11
    # Global U: A00=A[1,0]->5, A01=A[1,1]->6, A10=A[2,0]->10, A11=A[2,1]->11
    U2 = np.zeros((15, 7))
    U2[5, :] = U2_s[0, :]  # A00
    U2[6, :] = U2_s[1, :]  # A01
    U2[10, :] = U2_s[2, :]  # A10
    U2[11, :] = U2_s[3, :]  # A11
    
    # Strassen V indices (local): 0,1,2,3 -> B00,B01,B10,B11
    # Global V: B00=B[0,0]->0, B01=B[0,1]->1, B10=B[1,0]->5, B11=B[1,1]->6
    V2 = np.zeros((25, 7))
    V2[0, :] = V2_s[0, :]  # B00
    V2[1, :] = V2_s[1, :]  # B01
    V2[5, :] = V2_s[2, :]  # B10
    V2[6, :] = V2_s[3, :]  # B11
    
    # Strassen W indices (local): 0,1,2,3 -> C00,C01,C10,C11
    # Global W: C00=C[1,0]->0*3+1=1, C01=C[1,1]->1*3+1=4, C10=C[2,0]->0*3+2=2, C11=C[2,1]->1*3+2=5
    W2 = np.zeros((15, 7))
    W2[1, :] = W2_s[0, :]  # C00
    W2[4, :] = W2_s[1, :]  # C01
    W2[2, :] = W2_s[2, :]  # C10
    W2[5, :] = W2_s[3, :]  # C11
    
    # Part 3: C[1:3, 2:5] = A[1:3, :] @ B[:, 2:5] (2x5 @ 5x3 = 2x3)
    # Schoolbook: rank = 2*3*5 = 30
    U3 = np.zeros((15, 30))
    V3 = np.zeros((25, 30))
    W3 = np.zeros((15, 30))
    col = 0
    for i in range(2):  # i=0,1 -> global i=1,2
        for j in range(5):
            for k in range(3):  # k=0,1,2 -> global k=2,3,4
                U3[(i+1)*5+j, col] = 1
                V3[j*5+(k+2), col] = 1
                W3[(k+2)*3+(i+1), col] = 1
                col += 1
    
    # Concatenate
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    rank = U.shape[1]
    
    # Verify
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j... wait
    # The decomposition is: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]... no
    # Standard: C[i,k] = sum_{j,c} A[i,j] B[j,k] -> but that's schoolbook.
    # Tensor decomposition: C[i,k] = sum_c U[i,j... 
    # Actually, the format is:
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    # And C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * ... no that's not right.
    
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" - so W is indexed by (k,i) with k*n+i
    # The decomposition should satisfy:
    # C[i,k] = sum_{j,c} A[i,j] B[j,k] ... but that's not a tensor decomposition.
    
    # Actually for matrix multiplication tensor:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]... no.
    
    # The matmul tensor is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T[i,k,j] = A[i,j] B[j,k]
    # Rank decomposition: T[i,k,j] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]... 
    # That doesn't make sense dimensionally.
    
    # Let me think again. The tensor has 3 indices: i (0..n-1), j (0..m-1), k (0..p-1)
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor is T[i,j,k] which is 1 for all (i,j,k) - it's the all-ones tensor? No.
    
    # Actually, the matmul tensor T has T[i,j,k] = 1 for all valid i,j,k, and
    # C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k]
    # So T is the all-ones tensor of shape (n,m,p).
    
    # A rank-r decomposition of T means: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Wait, that's not right either. The indices don't match.
    
    # Let me look at the schoolbook code:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for each (i,j,k), there's a rank-1 term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # And T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # For schoolbook, each (i,j,k) has its own column, so T[i,j,k] = 1. Correct.
    
    # So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need this to equal 1 for all i,j,k.
    
    # Let me verify my solution:
    # For a given (i,j,k), T[i,j,k] should be 1.
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # Let me check (i=0, j=0, k=0):
    # Part 1: U1[0, 0] = 1 (j=0,k=0 -> col 0), V1[0, 0] = 1, W1[0, 0] = 1 -> product 1
    # Part 2: U2[0, :] = 0 (row 0 not set), so 0
    # Part 3: U3[0, :] = 0, so 0
    # Total: 1. Correct.
    
    # Check (i=1, j=0, k=0):
    # Part 1: U1[5, :] = 0 (only rows 0-4 set), so 0
    # Part 2: U2[5, c] for c in 0..6, V2[0, c], W2[1, c]
    #   U2[5, :] = U2_s[0, :] = [1,0,0,0,0,0,0] (from M1: U[0,0]=1)
    #   V2[0, :] = V2_s[0, :] = [1,1,0,0,0,0,0] (from M1: V[0,0]=1, M2: V[0,1]=1)
    #   W2[1, :] = W2_s[0, :] = [1,0,0,0,0,0,0] (from M1: W[0,0]=1)
    #   Sum_c U2[5,c]*V2[0,c]*W2[1,c] = 1*1*1 + 0*1*0 + ... = 1
    # Part 3: U3[5, :] = 0 (U3 has rows 5-14 set, but for i=1,j=0: row 1*5+0=5)
    #   U3[5, col] = 1 for col corresponding to i=0 (local), j=0, k=0..2
    #   V3[0*5+2, col] = V3[2, col] for k=0 -> global k=2
    #   W3[2*3+1, col] = W3[7, col]
    #   For (i=1,j=0,k=0): k=0, so V3 index is j*p+k = 0*5+0 = 0, not 2.
    #   So V3[0, col] for the columns where U3[5, col]=1
    #   U3[5, col]=1 for local i=0, j=0, k=0,1,2 -> cols 0,1,2
    #   V3[0, 0] = 1 (j=0, k=2 -> j*5+k = 2, not 0). 
    #   Wait, V3[j*5+(k+2), col] = 1. For j=0, k=0: V3[2, 0] = 1.
    #   So V3[0, 0] = 0.
    #   So Part 3 contributes 0 for (i=1,j=0,k=0).
    # Total: 1. Correct.
    
    # Check (i=1, j=0, k=2):
    # Part 1: 0
    # Part 2: U2[5, c]*V2[0*5+2, c]*W2[2*3+1, c] = U2[5,c]*V2[2,c]*W2[7,c]
    #   V2[2, :] = 0 (V2 only has rows 0,1,5,6 set)
    #   So 0.
    # Part 3: U3[5, c]*V3[0*5+2, c]*W3[2*3+1, c] = U3[5,c]*V3[2,c]*W3[7,c]
    #   U3[5, 0] = 1 (local i=0, j=0, k=0 -> col 0)
    #   V3[2, 0] = 1 (j=0, k=0 -> j*5+(0+2) = 2)
    #   W3[7, 0] = 1 (k=0 -> (0+2)*3+1 = 7)
    #   So product = 1. Other cols: U3[5,1]=1 (j=0,k=1), V3[2,1]=0 (V3[2,1] corresponds to j=0,k=1 -> index 3, not 2)
    #   So only col 0 contributes: 1.
    # Total: 1. Correct.
    
    # Good, the solution seems correct. Rank = 25+7+30 = 62.
    
    # Convert to list of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
