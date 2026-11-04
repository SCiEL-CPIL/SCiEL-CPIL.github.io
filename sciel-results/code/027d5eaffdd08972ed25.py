"""Strassen-based block decomposition for 2x5x6 matrix multiplication (rank 54)."""
import time

DIMENSIONS = [2, 5, 6]

# Precompute the rank-54 decomposition using Strassen's algorithm on 2x2 blocks.
# We split the 5 intermediate columns into blocks of sizes 2 and 3.
# M_{2,2,6}: split 6 into 2+2+2 -> 3 * R(M_{2,2,2}) = 3*7 = 21
# M_{2,3,6}: split 3 into 1+2, and 6 into 2+2+2
#   M_{2,1,2}: rank 4
#   M_{2,2,2}: rank 7
#   M_{2,1,6}: rank 12  -> 4+7+12 = 23? No wait.
#   Actually M_{2,3,6}: split 3=1+2, 6=2+2+2
#   M_{2,1,6}: rank 12
#   M_{2,2,6}: rank 21
#   Total: 12+21 = 33
# So M_{2,5,6} = M_{2,2,6} + M_{2,3,6} = 21 + 33 = 54.

# Let's build this carefully.
# Convention: U has shape (n*m, rank) = (10, rank)
# V has shape (m*p, rank) = (30, rank)
# W has shape (p*n, rank) = (12, rank)
# Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
# n=2, m=5, p=6

def _strassen_2x2x2():
    """
    Return (U, V, W) for M_{2,2,2} with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U indexed by [i*2+j], V by [j*2+k], W by [k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # C11 = M1 - M2 + M3 + M4
    # C12 = M5 + M4 - M6 - M7
    # C21 = M5 + M1
    # C22 = M2 - M3 + M6 + M7
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U index: i*2+j. i in {0,1}, j in {0,1}
    # V index: j*2+k. j in {0,1}, k in {0,1}
    # W index: k*2+i. k in {0,1}, i in {0,1}
    
    # M1: (A11+A22)(B11+B22) -> contributes to C11, C12, C21, C22
    # A indices: A11=U[0], A22=U[3]
    # B indices: B11=V[0], B22=V[3]
    # C indices: C11=W[0], C12=W[2], C21=W[1], C22=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C11
    W[2][0] = 1  # C12
    W[1][0] = 1  # C21
    W[3][0] = 1  # C22
    
    # M2: (A21+A22)B11 -> C11, C22
    # A21=U[1], A22=U[3], B11=V[0]
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1  # C11
    W[3][1] = 1  # C22
    
    # M3: A11(B21-B22) -> C11, C12
    # A11=U[0], B21=V[2], B22=V[3]
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1  # C11
    W[2][2] = 1  # C12
    
    # M4: A22(B12-B11) -> C11, C12
    # A22=U[3], B12=V[1], B11=V[0]
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C11
    W[2][3] = 1  # C12
    
    # M5: (A11+A12)B22 -> C12, C21
    # A11=U[0], A12=U[2], B22=V[3]
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[2][4] = 1  # C12
    W[1][4] = 1  # C21
    
    # M6: (A21-A11)(B11+B12) -> C12, C22
    # A21=U[1], A11=U[0], B11=V[0], B12=V[1]
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1  # C12
    W[3][5] = 1  # C22
    
    # M7: (A12-A22)(B21+B22) -> C12, C22
    # A12=U[2], A22=U[3], B21=V[2], B22=V[3]
    U[2][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[2][6] = 1  # C12
    W[3][6] = 1  # C22
    
    return U, V, W

def _multiply_2x1x6():
    """
    M_{2,1,6}: A is 2x1, B is 1x6, C is 2x6.
    Rank = 12 (schoolbook is optimal here since min dim is 1).
    U: (2*1, 12) = (2, 12), V: (1*6, 12) = (6, 12), W: (6*2, 12) = (12, 12)
    """
    n, m, p = 2, 1, 6
    rank = n * m * p  # 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i*m+0][col] = 1
            V[0*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    return U, V, W

def _multiply_2x2x6():
    """
    M_{2,2,6}: split 6 into 2+2+2, each M_{2,2,2} has rank 7.
    Total rank = 21.
    U: (4, 21), V: (12, 21), W: (12, 21)
    """
    # Split p=6 into three blocks of size 2.
    # For each block b in 0,1,2:
    #   A columns: all of A (2x2)
    #   B rows: block b (2x2)
    #   C columns: block b (2x2)
    # Use Strassen for each.
    
    U, V, W = [], [], []
    for b in range(3):
        Ub, Vb, Wb = _strassen_2x2x2()
        # Ub: (4, 7) -> U columns for this block
        # Vb: (4, 7) -> V rows for this block (j in 0,1; k in 0,1 of the block)
        # Wb: (4, 7) -> W for this block (k in 0,1; i in 0,1)
        
        # Append to U: same for all blocks since A is the same
        if b == 0:
            U = [row[:] for row in Ub]
            V = [row[:] for row in Vb]
            W = [row[:] for row in Wb]
        else:
            for row in U:
                pass  # U doesn't change, just extend columns
            # Actually, let's build properly.
            for i in range(4):
                U[i].extend(Ub[i])
            for j in range(4):
                V[j].extend(Vb[j])
            for k in range(4):
                W[k].extend(Wb[k])
    
    # Now U is (4, 21), V is (4, 21), W is (4, 21)
    # But V should be (m*p, rank) = (12, 21). 
    # V index: j*p+k where j in 0,1 and k in 0,1,2,3,4,5.
    # In our construction, Vb has index j*2+k_block where k_block in 0,1.
    # We need to map to the full V index: j*6 + (b*2 + k_block)
    # Similarly W index: k*n+i where k in 0..5, i in 0,1.
    # Wb has index k_block*2 + i. We need k = b*2 + k_block.
    # So W index = (b*2 + k_block)*2 + i
    
    # Rebuild V and W with proper indexing.
    U_full = [[0]*21 for _ in range(4)]
    V_full = [[0]*21 for _ in range(12)]
    W_full = [[0]*21 for _ in range(12)]
    
    for b in range(3):
        Ub, Vb, Wb = _strassen_2x2x2()
        for c in range(7):
            col = b*7 + c
            for i in range(4):
                U_full[i][col] = Ub[i][c]
            for j in range(2):
                for k_block in range(2):
                    full_k = b*2 + k_block
                    V_full[j*6 + full_k][col] = Vb[j*2 + k_block][c]
            for k_block in range(2):
                for i in range(2):
                    full_k = b*2 + k_block
                    W_full[full_k*2 + i][col] = Wb[k_block*2 + i][c]
    
    return U_full, V_full, W_full

def _multiply_2x3x6():
    """
    M_{2,3,6}: split m=3 into 1+2, and p=6 into 2+2+2.
    M_{2,1,6}: rank 12
    M_{2,2,6}: rank 21
    Total rank = 33.
    U: (6, 33), V: (18, 33), W: (12, 33)
    """
    U1, V1, W1 = _multiply_2x1x6()  # (2,12), (6,12), (12,12)
    U2, V2, W2 = _multiply_2x2x6()  # (4,21), (12,21), (12,21)
    
    rank = 33
    U_full = [[0]*rank for _ in range(6)]
    V_full = [[0]*rank for _ in range(18)]
    W_full = [[0]*rank for _ in range(12)]
    
    # First part: m=1 block (j=0)
    for c in range(12):
        for i in range(2):
            U_full[i*3+0][c] = U1[i][c]
        for k in range(6):
            V_full[0*6+k][c] = V1[k][c]
        for k in range(6):
            for i in range(2):
                W_full[k*2+i][c] = W1[k*2+i][c]
    
    # Second part: m=2 block (j=1,2)
    for c in range(21):
        col = 12 + c
        for i in range(2):
            for j_block in range(2):
                full_j = 1 + j_block
                U_full[i*3+full_j][col] = U2[i*2+j_block][c]
        for j_block in range(2):
            for k in range(6):
                full_j = 1 + j_block
                V_full[full_j*6+k][col] = V2[j_block*6+k][c]
        for k in range(6):
            for i in range(2):
                W_full[k*2+i][col] = W2[k*2+i][c]
    
    return U_full, V_full, W_full

def construct():
    # M_{2,5,6} = M_{2,2,6} + M_{2,3,6}
    # Split m=5 into 2+3.
    U2, V2, W2 = _multiply_2x2x6()  # (4,21), (12,21), (12,21)
    U3, V3, W3 = _multiply_2x3x6()  # (6,33), (18,33), (12,33)
    
    rank = 54
    U_full = [[0]*rank for _ in range(10)]
    V_full = [[0]*rank for _ in range(30)]
    W_full = [[0]*rank for _ in range(12)]
    
    # First part: m=2 block (j=0,1)
    for c in range(21):
        for i in range(2):
            for j in range(2):
                U_full[i*5+j][c] = U2[i*2+j][c]
        for j in range(2):
            for k in range(6):
                V_full[j*6+k][c] = V2[j*6+k][c]
        for k in range(6):
            for i in range(2):
                W_full[k*2+i][c] = W2[k*2+i][c]
    
    # Second part: m=3 block (j=2,3,4)
    for c in range(33):
        col = 21 + c
        for i in range(2):
            for j_block in range(3):
                full_j = 2 + j_block
                U_full[i*5+full_j][col] = U3[i*3+j_block][c]
        for j_block in range(3):
            for k in range(6):
                full_j = 2 + j_block
                V_full[full_j*6+k][col] = V3[j_block*6+k][c]
        for k in range(6):
            for i in range(2):
                W_full[k*2+i][col] = W3[k*2+i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}
