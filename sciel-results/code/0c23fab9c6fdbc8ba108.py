"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial schoolbook decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def make_strassen_3x3():
        """
        Construct a rank-23 decomposition for 3x3x3 matrix multiplication.
        Uses Strassen 2x2x2 for the top-left block plus direct terms.
        Returns (U, V, W) where each is 9x23.
        """
        # 3x3 matrices indexed as:
        # A = [[a00, a01, a02],
        #      [a10, a11, a12],
        #      [a20, a21, a22]]
        # B = [[b00, b01, b02],
        #      [b10, b11, b12],
        #      [b20, b21, b22]]
        # C = A*B
        
        # U indices: i*3+j for A[i,j]
        # V indices: j*3+k for B[j,k]  
        # W indices: k*3+i for C[k,i]
        
        rank = 23
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        # Strassen 2x2x2 for top-left 2x2 block (rows 0,1; cols 0,1)
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b10-b11)
        # m4 = a11*(b01-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        
        # m1
        r = 0
        U[0][r] = 1; U[4][r] = 1  # a00, a11
        V[0][r] = 1; V[3][r] = 1  # b00, b11
        W[0][r] = 1; W[3][r] = 1; W[1][r] = -1; W[4][r] = -1  # c00, c11, subtract c01, c10
        
        # m2
        r = 1
        U[4][r] = 1; U[1][r] = 1  # a10, a11
        V[0][r] = 1  # b00
        W[3][r] = 1  # c10
        
        # m3
        r = 2
        U[0][r] = 1  # a00
        V[3][r] = 1; V[4][r] = -1  # b10-b11
        W[3][r] = 1  # c10
        
        # m4
        r = 3
        U[4][r] = 1  # a11
        V[1][r] = 1; V[0][r] = -1  # b01-b00
        W[0][r] = 1  # c01
        
        # m5
        r = 4
        U[0][r] = 1; U[1][r] = 1  # a00, a01
        V[4][r] = 1  # b11
        W[1][r] = 1  # c01
        
        # m6
        r = 5
        U[1][r] = 1; U[0][r] = -1  # a10-a00
        V[0][r] = 1; V[1][r] = 1  # b00+b01
        W[0][r] = 1  # c00
        
        # m7
        r = 6
        U[1][r] = 1; U[4][r] = -1  # a01-a11
        V[3][r] = 1; V[4][r] = 1  # b10+b11
        W[1][r] = 1  # c11
        
        # Direct terms for remaining 8 products:
        # a02*b00 -> c00, a02*b10 -> c10, a02*b20 -> c20
        # a12*b00 -> c00, a12*b10 -> c10, a12*b20 -> c20
        # a20*b00 -> c00, a20*b10 -> c10, a20*b20 -> c20
        # a21*b01 -> c01, a21*b11 -> c11, a21*b21 -> c21
        # a22*b02 -> c02, a22*b12 -> c12, a22*b22 -> c22
        # a02*b01 -> c01, a02*b11 -> c11, a02*b21 -> c21
        # a12*b01 -> c01, a12*b11 -> c11, a12*b21 -> c21
        # a20*b01 -> c01, a20*b11 -> c11, a20*b21 -> c21
        # a02*b02 -> c02, a02*b12 -> c12, a02*b22 -> c22
        # a12*b02 -> c02, a12*b12 -> c12, a12*b22 -> c22
        # a20*b02 -> c02, a20*b12 -> c12, a20*b22 -> c22
        # a21*b00 -> c00, a21*b10 -> c10, a21*b20 -> c20
        # a22*b00 -> c00, a22*b10 -> c10, a22*b20 -> c20
        # a21*b02 -> c02, a21*b12 -> c12, a21*b22 -> c22
        # a22*b01 -> c01, a22*b11 -> c11, a22*b21 -> c21
        
        # Let me be more systematic. After the 2x2 Strassen handles:
        # c00 = a00*b00 + a10*b10 + a20*b20 (partially)
        # Actually Strassen 2x2 only covers the top-left 2x2 subproduct.
        # C[0:2, 0:2] gets contributions from A[0:2,0:2]*B[0:2,0:2]
        # But we also need contributions from A[0:2,2]*B[2,0:2] and A[2,0:2]*B[0:2,0:2]
        
        # Let me redo this more carefully.
        # The full product C[i,j] = sum_k A[i,k]*B[k,j]
        
        # For the 2x2 Strassen on rows {0,1} and cols {0,1}:
        # It computes the 2x2 subproduct A[0:2,0:2]*B[0:2,0:2]
        # This contributes to C[0:2, 0:2]
        
        # Additional terms needed:
        # C[0,j] += A[0,2]*B[2,j] for j=0,1,2
        # C[1,j] += A[1,2]*B[2,j] for j=0,1,2
        # C[i,0] += A[i,2]*B[2,0] for i=0,1,2
        # C[i,1] += A[i,2]*B[2,1] for i=0,1,2
        # C[2,j] += A[2,k]*B[k,j] for k=0,1,2 and j=0,1,2
        # But C[0,0], C[0,1], C[1,0], C[1,1] already get contributions from Strassen
        # The terms A[i,2]*B[2,j] for i,j in {0,1} are additional
        
        # So additional scalar multiplications needed:
        # A[0,2]*B[2,0] -> C[0,0]
        # A[0,2]*B[2,1] -> C[0,1]
        # A[0,2]*B[2,2] -> C[0,2]
        # A[1,2]*B[2,0] -> C[1,0]
        # A[1,2]*B[2,1] -> C[1,1]
        # A[1,2]*B[2,2] -> C[1,2]
        # A[2,0]*B[0,0] -> C[2,0]
        # A[2,0]*B[0,1] -> C[2,1]
        # A[2,0]*B[0,2] -> C[2,2]
        # A[2,1]*B[1,0] -> C[2,0]
        # A[2,1]*B[1,1] -> C[2,1]
        # A[2,1]*B[1,2] -> C[2,2]
        # A[2,2]*B[2,0] -> C[2,0]
        # A[2,2]*B[2,1] -> C[2,1]
        # A[2,2]*B[2,2] -> C[2,2]
        
        # That's 15 more, total 7+15=22. But we need 23.
        # Wait, some of these can be combined.
        
        # Actually, let me just use a simpler approach: 
        # Use the known fact that rank(3x3x3) = 23 and construct it.
        
        # Alternative: use a different decomposition strategy.
        # Let me use the "Bini" algorithm or similar.
        
        # Actually, for reliability, let me just use rank 27 (schoolbook for 3x3x3)
        # and get rank 54 for 3x3x6. But that's not better than current.
        
        # Let me try a different approach: use Strassen for 2x2 blocks
        # in a block decomposition.
        
        # Partition 3x3 * 3x3 as:
        # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
        # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
        
        # C11 = A11*B11 + A12*B21  (2x2)
        # C12 = A11*B12 + A12*B22  (2x1)
        # C21 = A21*B11 + A22*B21  (1x2)
        # C22 = A21*B12 + A22*B22  (1x1)
        
        # C11: use Strassen for A11*B11 (rank 7) + 2 more for A12*B21 (rank 2)
        # C12: 2x1 product, rank 4 direct
        # C21: 1x2 product, rank 4 direct
        # C22: 1x1 product, rank 1 direct
        # Plus cross terms... this is getting complex.
        
        # For simplicity and correctness, let me just return None and use schoolbook.
        # But that gives rank 54 which is the current score.
        
        # Actually, let me try a different approach: 
        # Use the fact that we can do 3x3 * 3x6 more efficiently.
        
        # 3x3 * 3x6: partition B into two 3x3 matrices B1 and B2.
        # Then A*B = [A*B1 | A*B2]
        # If we can do 3x3*3x3 with rank 23, total rank is 46.
        
        # But I need a valid rank-23 3x3x3 decomposition.
        
        # Let me use a known valid one. The rank-23 algorithm for 3x3x3
        # is well-known. Let me construct it properly.
        
        # Here's a clean construction:
        # Use 7 Strassen multiplications for the 2x2 sub-block,
        # plus 16 additional scalar multiplications for the remaining terms.
        # Total: 23.
        
        # Let me index A[i,j] as A[i*3+j], B[i,j] as B[i*3+j], C[i,j] as C[i*3+j]
        
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # m1 = (a00+a11)*(b00+b11): contributes to c00, c11; subtract from c01, c10
        r = 0
        U[0][r]=1; U[4][r]=1
        V[0][r]=1; V[3][r]=1
        W[0][r]=1; W[4][r]=1
        W[1][r]=-1; W[3][r]=-1
        
        # m2 = (a10+a11)*b00: contributes to c10
        r = 1
        U[4][r]=1; U[1][r]=1
        V[0][r]=1
        W[3][r]=1
        
        # m3 = a00*(b10-b11): contributes to c10
        r = 2
        U[0][r]=1
        V[3][r]=1; V[4][r]=-1
        W[3][r]=1
        
        # m4 = a11*(b01-b00): contributes to c01
        r = 3
        U[4][r]=1
        V[1][r]=1; V[0][r]=-1
        W[1][r]=1
        
        # m5 = (a00+a01)*b11: contributes to c01
        r = 4
        U[0][r]=1; U[1][r]=1
        V[4][r]=1
        W[1][r]=1
        
        # m6 = (a10-a00)*(b00+b01): contributes to c00
        r = 5
        U[1][r]=1; U[0][r]=-1
        V[0][r]=1; V[1][r]=1
        W[0][r]=1
        
        # m7 = (a01-a11)*(b10+b11): contributes to c11
        r = 6
        U[1][r]=1; U[4][r]=-1
        V[3][r]=1; V[4][r]=1
        W[4][r]=1
        
        # Now additional terms for the 3rd row/col:
        # a02*b20 -> c20
        r = 7
        U[2][r]=1
        V[6][r]=1
        W[6][r]=1
        
        # a02*b21 -> c21
        r = 8
        U[2][r]=1
        V[7][r]=1
        W[7][r]=1
        
        # a02*b22 -> c22
        r = 9
        U[2][r]=1
        V[8][r]=1
        W[8][r]=1
        
        # a12*b20 -> c20
        r = 10
        U[5][r]=1
        V[6][r]=1
        W[6][r]=1
        
        # a12*b21 -> c21
        r = 11
        U[5][r]=1
        V[7][r]=1
        W[7][r]=1
        
        # a12*b22 -> c22
        r = 12
        U[5][r]=1
        V[8][r]=1
        W[8][r]=1
        
        # a20*b00 -> c00
        r = 13
        U[6][r]=1
        V[0][r]=1
        W[0][r]=1
        
        # a20*b01 -> c01
        r = 14
        U[6][r]=1
        V[1][r]=1
        W[1][r]=1
        
        # a20*b02 -> c02
        r = 15
        U[6][r]=1
        V[2][r]=1
        W[2][r]=1
        
        # a21*b10 -> c10
        r = 16
        U[7][r]=1
        V[3][r]=1
        W[3][r]=1
        
        # a21*b11 -> c11
        r = 17
        U[7][r]=1
        V[4][r]=1
        W[4][r]=1
        
        # a21*b12 -> c12
        r = 18
        U[7][r]=1
        V[5][r]=1
        W[5][r]=1
        
        # a22*b20 -> c20
        r = 19
        U[8][r]=1
        V[6][r]=1
        W[6][r]=1
        
        # a22*b21 -> c21
        r = 20
        U[8][r]=1
        V[7][r]=1
        W[7][r]=1
        
        # a22*b22 -> c22
        r = 21
        U[8][r]=1
        V[8][r]=1
        W[8][r]=1
        
        # We have 22 so far. Need one more for the missing term.
        # Wait, I'm missing a02*b00, a02*b10, etc.
        # Let me reconsider.
        
        # The full product: C[i,j] = sum_k A[i,k]*B[k,j]
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[0,2] = a00*b02 + a01*b12 + a02*b22
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # C[1,1] = a10*b01 + a11*b11 + a12*b21
        # C[1,2] = a10*b02 + a11*b12 + a12*b22
        # C[2,0] = a20*b00 + a21*b10 + a22*b20
        # C[2,1] = a20*b01 + a21*b11 + a22*b21
        # C[2,2] = a20*b02 + a21*b12 + a22*b22
        
        # The Strassen 2x2 handles:
        # C[0,0] from a00*b00 + a01*b10 (partially, via combination)
        # Actually Strassen computes the exact 2x2 product A[0:2,0:2]*B[0:2,0:2]
        # which gives C[0,0], C[0,1], C[1,0], C[1,1] from those 4 elements of A and 4 of B.
        
        # But C[0,0] also gets a02*b20
        # C[0,1] also gets a02*b21
        # C[0,2] gets a00*b02 + a01*b12 + a02*b22
        # C[1,0] also gets a12*b20
        # C[1,1] also gets a12*b21
        # C[1,2] gets a10*b02 + a11*b12 + a12*b22
        # C[2,0] gets a20*b00 + a21*b10 + a22*b20
        # C[2,1] gets a20*b01 + a21*b11 + a22*b21
        # C[2,2] gets a20*b02 + a21*b12 + a22*b22
        
        # So after Strassen, we need:
        # a02*b20, a02*b21, a02*b22 (3 terms for row 0, col 2)
        # a12*b20, a12*b21, a12*b22 (3 terms for row 1, col 2)
        # a00*b02, a01*b12 (2 terms for row 0, col 2)
        # a10*b02, a11*b12 (2 terms for row 1, col 2)
        # a20*b00, a21*b10, a22*b20 (3 terms for row 2, col 0)
        # a20*b01, a21*b11, a22*b21 (3 terms for row 2, col 1)
        # a20*b02, a21*b12, a22*b22 (3 terms for row 2, col 2)
        
        # That's 3+3+2+2+3+3+3 = 19 additional terms, total 7+19=26. Too many.
        
        # The key is that some of these can be combined.
        # For example, a00*b02 + a01*b12 can be computed with 2 multiplications,
        # but maybe we can do better.
        
        # Actually, the known rank-23 algorithm is more clever.
        # Let me just use a different approach: return None and fall back to schoolbook.
        
        return None

    def make_rank_46():
        """
        Construct a rank 46 decomposition by combining two 3x3x3 decompositions.
        """
        s = make_strassen_3x3()
        if s is None:
            return None
            
        U_s, V_s, W_s = s
        rank_s = len(U_s[0])
        rank_total = 2 * rank_s
        
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # First half (k=0,1,2)
        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][r] = U_s[i*m+j][r]
        
        for j in range(m):
            for k in range(3):
                for r in range(rank_s):
                    V[j*p+k][r] = V_s[j*3+k][r]
            
        for k in range(3):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][r] = W_s[k*3+i][r]

        # Second half (k=3,4,5)
        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][rank_s + r] = U_s[i*m+j][r]
        
        for j in range(m):
            for k in range(3, 6):
                for r in range(rank_s):
                    V[j*p+k][rank_s + r] = V_s[j*3+(k-3)][r]
            
        for k in range(3, 6):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][rank_s + r] = W_s[(k-3)*3+i][r]

        return {"U": U, "V": V, "W": W}

    # Try Rank 46
    result_46 = make_rank_46()
    if result_46 is not None and verify(result_46["U"], result_46["V"], result_46["W"]):
        return result_46

    # Fallback: Schoolbook (rank 54)
    return make_schoolbook()
# EVOLVE-BLOCK-END
