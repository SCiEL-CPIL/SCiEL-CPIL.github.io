import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_strassen_2x2_2x2():
    """
    Return U,V,W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0,0]=1, C11: W[3,0]=1
    # But C00 = M1 - M4 + M5 - M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's just define the coefficients directly.
    # For each r in 0..6, U[:,r], V[:,r], W[:,r]
    
    # r=0 (M1):
    # A: [1, 0, 0, 1]
    # B: [1, 0, 0, 1]
    # C: W[0,0]=1, W[3,0]=1 => C[0,0]=1, C[1,1]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1 (M2): (A10+A11)B00
    # A: [0, 0, 1, 1]
    # B: [1, 0, 0, 0]
    # C: C[0,1] += 1, C[1,1] -= 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2 (M3): A00(B10-B11)
    # A: [1, 0, 0, 0]
    # B: [0, 0, 1, -1]
    # C: C[1,0] += 1, C[1,1] += 1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3 (M4): A11(B01-B10)
    # A: [0, 0, 0, 1]
    # B: [0, 1, -1, 0]
    # C: C[0,0] -= 1, C[0,1] += 1
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # r=4 (M5): (A00+A01)B11
    # A: [1, 1, 0, 0]
    # B: [0, 0, 0, 1]
    # C: C[0,0] += 1, C[1,0] += 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # r=5 (M6): (A10-A00)(B00+B10)
    # A: [-1, 0, 1, 0]
    # B: [1, 0, 1, 0]
    # C: C[1,1] += 1
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # r=6 (M7): (A01-A11)(B01+B11)
    # A: [0, 1, 0, -1]
    # B: [0, 1, 0, 1]
    # C: C[0,0] -= 1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # We can view the 2x4x6 multiplication as 2 blocks of 2x2x2 if we group correctly.
    # Actually, 2x4x6 can be seen as 2 (rows of A) x 4 (cols of A/rows of B) x 6 (cols of B).
    # We can decompose m=4 into two 2s, and p=6 into three 2s.
    # So we have a 2x2x2 grid of Strassen blocks? No, that would be 2x2x2 blocks.
    # A is 2x4, B is 4x6.
    # We can partition A into two 2x2 blocks (left and right), and B into three 2x2 blocks (top, middle, bottom? No, B is 4x6).
    # B is 4x6. Partition rows into two 2s, cols into three 2s.
    # So B is a 2x3 grid of 2x2 blocks.
    # A is a 1x2 grid of 2x2 blocks.
    # So for each pair of (A-block, B-block), we do a Strassen multiplication.
    # There are 1*2 * 2*3 = 12 such pairs.
    # Each Strassen multiplication uses 7 multiplications.
    # Total rank = 12 * 7 = 84? That's worse than schoolbook 48.
    
    # Wait, schoolbook is n*m*p = 2*4*6 = 48.
    # The previous run reduced to 42. How?
    # "Reduces rank from 48 (schoolbook) to 42 by applying Strassen's algorithm to 2x2x2 sub-blocks"
    # Let's re-read the inspiration.
    # Maybe it's not simply 12 independent Strassen blocks.
    # If we group m into 2 blocks of 2, and p into 3 blocks of 2.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2; B3] where B1, B2, B3 are 4x2? No.
    # B is 4x6.
    # Let's try to partition the indices.
    # i in 0..1, j in 0..3, k in 0..5.
    # Group j into 2 sets: J1={0,1}, J2={2,3}.
    # Group k into 3 sets: K1={0,1}, K2={2,3}, K3={4,5}.
    # For each j-group J_a and k-group K_b, we have a 2x2x2 multiplication.
    # A[:, J_a] is 2x2. B[J_a, K_b] is 2x2.
    # Wait, B is 4x6. Rows are j, cols are k.
    # So B[J_a, K_b] is indeed 2x2.
    # So for each a in {1,2} and b in {1,2,3}, we multiply A[:, J_a] (2x2) by B[J_a, K_b] (2x2).
    # This gives 6 Strassen blocks.
    # Each block has rank 7. Total rank 6 * 7 = 42.
    # This matches the 42 rank!
    
    # So the decomposition is:
    # For each a in 0..1, b in 0..2:
    #   Perform Strassen multiplication of A[:, 2a:2a+2] and B[2a:2a+2, 2b:2b+2].
    #   The result contributes to C[:, 2b:2b+2].
    
    # Let's construct U, V, W.
    # Rank = 42.
    # U: (2*4, 42) = (8, 42)
    # V: (4*6, 42) = (24, 42)
    # W: (6*2, 42) = (12, 42)
    
    U = [[0] * 42 for _ in range(8)]
    V = [[0] * 42 for _ in range(24)]
    W = [[0] * 42 for _ in range(12)]
    
    # Get Strassen U,V,W for 2x2x2.
    sU, sV, sW = _build_strassen_2x2_2x2()
    
    col = 0
    for a in range(2):  # j-group
        for b in range(3):  # k-group
            # A part: rows 0..1, cols 2a..2a+1.
            # U indices: i*m + j. i in 0..1, j in 2a..2a+1.
            # V part: rows 2a..2a+1, cols 2b..2b+1.
            # V indices: j*p + k. j in 2a..2a+1, k in 2b..2b+1.
            # W part: rows 2b..2b+1, cols 0..1.
            # W indices: k*n + i. k in 2b..2b+1, i in 0..1.
            
            for r in range(7):
                # Map sU[i_s, r] to U[i*m + (2a + j_s), col]
                # sU index: i_s * 2 + j_s
                for idx_s in range(4):
                    val = sU[idx_s][r]
                    if val != 0:
                        i_s = idx_s // 2
                        j_s = idx_s % 2
                        i = i_s
                        j = 2 * a + j_s
                        U[i * 4 + j][col] += val
                
                # Map sV[j_s, r] to V[(2a + j_s)*6 + (2b + k_s), col]
                for idx_s in range(4):
                    val = sV[idx_s][r]
                    if val != 0:
                        j_s = idx_s // 2
                        k_s = idx_s % 2
                        j = 2 * a + j_s
                        k = 2 * b + k_s
                        V[j * 6 + k][col] += val
                
                # Map sW[k_s, r] to W[(2b + k_s)*2 + i, col]
                for idx_s in range(4):
                    val = sW[idx_s][r]
                    if val != 0:
                        k_s = idx_s // 2
                        i_s = idx_s % 2
                        k = 2 * b + k_s
                        i = i_s
                        W[k * 2 + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick validation
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    # Check tensor decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    n, m, p = 2, 4, 6
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(U.shape[1]):
                    T[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Expected T[i, j, k] = 1
    if not np.allclose(T, 1.0):
        print("Error: T is not all ones.")
        print(T)
    else:
        print("Success: Rank 42 decomposition is valid.")
        print("Rank:", U.shape[1])
