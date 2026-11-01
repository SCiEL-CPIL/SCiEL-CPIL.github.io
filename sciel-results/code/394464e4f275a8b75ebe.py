DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # We decompose the multiplication tensor for C = A * B where:
    # A is 2x3, B is 3x10, C is 2x10.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Strategy:
    # 1. Split A into A_left (2x2) and A_right (2x1).
    # 2. Split B into B_left (2x10) and B_right (1x10).
    # 3. C = A_left * B_left + A_right * B_right.
    #
    # For A_right * B_right:
    #   A_right is 2x1, B_right is 1x10.
    #   C_right[i,k] = A_right[i,0] * B_right[0,k]
    #   This is rank 20 (schoolbook for 2x1x10).
    #
    # For A_left * B_left:
    #   A_left is 2x2, B_left is 2x10.
    #   We can use Strassen-like decomposition for the 2x2 part.
    #   But B_left has 10 columns, so we need to handle each pair of columns.
    #   Actually, we can use the fact that for each pair of columns (k1, k2) in B_left,
    #   we can apply a Strassen-like trick.
    #
    # Let's use a different approach: 
    # For the 2x2 x 2x10 multiplication, we can use rank 7 per 2x2x2 block,
    # but we have 10 columns. Let's group columns in pairs: 5 pairs.
    # For each pair (k1, k2), we use 7 multiplications.
    # Total for A_left * B_left: 5 * 7 = 35.
    # Total rank: 35 + 20 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Part 1: A_right * B_right (rank 20)
    # A_right[i,0] = A[i,2] -> U[i*3 + 2]
    # B_right[0,k] = B[2,k] -> V[2*10 + k] = V[20 + k]
    # C_right[i,k] -> W[k*2 + i]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: A_left * B_left using Strassen-like decomposition
    # A_left[i,j] = A[i,j] for j in {0,1} -> U[i*3 + j]
    # B_left[j,k] = B[j,k] for j in {0,1} -> V[j*10 + k]
    # C_left[i,k] = A_left[i,0]*B_left[0,k] + A_left[i,1]*B_left[1,k]
    
    # Group columns in pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    for pair_idx in range(5):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        
        # Strassen for 2x2 matrices:
        # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # B = [[e, f], [g, h]] where e=B[0,k1], f=B[0,k2], g=B[1,k1], h=B[1,k2]
        # 
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        #
        # C[0,k1] = P1 + P4 - P5 + P7
        # C[0,k2] = P3 + P5
        # C[1,k1] = P2 + P4
        # C[1,k2] = P1 - P2 + P3 + P6
        
        # U indices for A_left:
        # a = A[0,0] -> U[0]
        # b = A[0,1] -> U[1]
        # c = A[1,0] -> U[3]
        # d = A[1,1] -> U[4]
        
        # V indices for B_left:
        # e = B[0,k1] -> V[0*10 + k1] = V[k1]
        # f = B[0,k2] -> V[0*10 + k2] = V[k2]
        # g = B[1,k1] -> V[1*10 + k1] = V[10 + k1]
        # h = B[1,k2] -> V[1*10 + k2] = V[10 + k2]
        
        # W indices for C_left:
        # C[0,k1] -> W[k1*2 + 0] = W[2*k1]
        # C[0,k2] -> W[k2*2 + 0] = W[2*k2]
        # C[1,k1] -> W[k1*2 + 1] = W[2*k1 + 1]
        # C[1,k2] -> W[k2*2 + 1] = W[2*k2 + 1]
        
        # P1 = (a+d)(e+h)
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[10 + k2][col] = 1
        W[2*k1][col] = 1; W[2*k2 + 1][col] = -1
        col += 1
        
        # P2 = (c+d)e
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[2*k1 + 1][col] = 1
        col += 1
        
        # P3 = a(f-h)
        U[0][col] = 1
        V[k2][col] = 1; V[10 + k2][col] = -1
        W[2*k2][col] = 1; W[2*k2 + 1][col] = 1
        col += 1
        
        # P4 = d(g-e)
        U[4][col] = 1
        V[10 + k1][col] = 1; V[k1][col] = -1
        W[2*k1][col] = 1; W[2*k1 + 1][col] = 1
        col += 1
        
        # P5 = (a+b)h
        U[0][col] = 1; U[1][col] = 1
        V[10 + k2][col] = 1
        W[2*k1][col] = -1; W[2*k2][col] = 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[k2][col] = 1
        W[2*k2 + 1][col] = 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[1][col] = 1; U[4][col] = -1
        V[10 + k1][col] = 1; V[10 + k2][col] = 1
        W[2*k1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
