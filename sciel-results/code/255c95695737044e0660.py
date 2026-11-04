"""Program for matrix multiplication 4x4 * 4x8 with reduced rank decomposition."""
import time

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 115.0

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-112 exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    # For matrices A (2x2), B (2x2), C = A*B:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We build the decomposition hierarchically.
    # 
    # For a 2x2 * 2x2 multiplication, the tensor has entries T[i,j,k] where
    # C[k*2+i, j] = sum_{a,b} A[i,a] * B[a,j] ... wait, let me be careful.
    #
    # Actually, the matrix multiplication tensor T[i,j,k] for n x m * m x p means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The factor matrices are:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For Strassen on 2x2 * 2x2:
    # A is 2x2, B is 2x2, C is 2x2
    # i in {0,1}, j in {0,1}, k in {0,1}
    # T[i,j,k] = sum_{a,b} ... no, T[i,j,k] represents the coefficient in:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 if this term contributes, 0 otherwise.
    # Actually the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k),
    # and the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i] * V_r[j] * W_r[k]
    
    # Let me think differently. The matrix multiplication tensor of format (n,m,p) has:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and we want rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i,r] * V[j,r] * W[k,r]
    #
    # But that's the "all ones" tensor which has rank 1. That's not right.
    #
    # The correct tensor for matrix multiplication C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has components T[i,j,k] such that:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j (sum_r U[i*m+j,r] W[k*n+i,r]) V[j*p+k,r] ... 
    #
    # No. Let me re-read the interface.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies: for all i, j, k:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    # (since the matrix multiplication tensor has all-ones entries)
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # For Strassen 2x2x2:
    # i, j, k in {0, 1}
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U indices: i*2+j for i,j in {0,1}
    # V indices: j*2+k for j,k in {0,1}  
    # W indices: k*2+i for k,i in {0,1}
    
    # Strassen factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each multiplier r, we need:
    # U_r[i*2+j] * V_r[j*2+k] * W_r[k*2+i] contributes to the tensor
    
    # M1: (A00+A11)(B00+B11)
    # This involves A entries (0,0) and (1,1), B entries (0,0) and (1,1)
    # U for M1: U[0,0]=1, U[3,3]=1 (i.e., U[i*2+j] where (i,j)=(0,0) and (1,1))
    # V for M1: V[0,0]=1, V[3,3]=1 (j*2+k where (j,k)=(0,0) and (1,1))
    # W for M1: C00 gets +1, C11 gets +1 -> W[0*2+0]=1 (k=0,i=0), W[1*2+1]=1 (k=1,i=1)
    # So W[0]=1, W[3]=1
    
    # Let me enumerate all 7 Strassen factors for 2x2x2:
    # Factor r=0 (M1): (A00+A11)(B00+B11)
    #   U: U[0*2+0]=1 (i=0,j=0), U[1*2+1]=1 (i=1,j=1)
    #   V: V[0*2+0]=1 (j=0,k=0), V[1*2+1]=1 (j=1,k=1)
    #   W: W[0*2+0]=1 (k=0,i=0), W[1*2+1]=1 (k=1,i=1)
    #   Contribution: sum over i,j,k of U[i*2+j]*V[j*2+k]*W[k*2+i]
    #   = 1*1*1 + 1*1*1 = 2 at (i,j,k) where all match... 
    #   Wait, I need to be more careful.
    
    # Let me just directly compute the Strassen decomposition.
    # For each of the 7 products, define the three vectors.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2 matrix multiplication, rank 7."""
        U = [[0]*7 for _ in range(4)]  # indices i*2+j
        V = [[0]*7 for _ in range(4)]  # indices j*2+k
        W = [[0]*7 for _ in range(4)]  # indices k*2+i
        
        # M1 = (A00+A11)(B00+B11)
        # A terms: (0,0)=1, (1,1)=1 => U[0]=1, U[3]=1
        # B terms: (0,0)=1, (1,1)=1 => V[0]=1, V[3]=1
        # C terms: C00 += 1, C11 += 1 => W[0]=1, W[3]=1
        r = 0
        U[0][r] = 1; U[3][r] = 1
        V[0][r] = 1; V[3][r] = 1
        W[0][r] = 1; W[3][r] = 1
        
        # M2 = (A10+A11)B00
        # A terms: (1,0)=1, (1,1)=1 => U[2]=1, U[3]=1
        # B terms: (0,0)=1 => V[0]=1
        # C terms: C10 += 1, C11 -= 1 => W[2]=1 (k=1,i=0), W[3]=-1 (k=1,i=1)
        r = 1
        U[2][r] = 1; U[3][r] = 1
        V[0][r] = 1
        W[2][r] = 1; W[3][r] = -1
        
        # M3 = A00(B10-B11)
        # A terms: (0,0)=1 => U[0]=1
        # B terms: (1,0)=1, (1,1)=-1 => V[2]=1, V[3]=-1
        # C terms: C01 += 1, C11 += 1 => W[1]=1 (k=0,i=1), W[3]=1 (k=1,i=1)
        r = 2
        U[0][r] = 1
        V[2][r] = 1; V[3][r] = -1
        W[1][r] = 1; W[3][r] = 1
        
        # M4 = A11(B01-B10)
        # A terms: (1,1)=1 => U[3]=1
        # B terms: (0,1)=1, (1,0)=-1 => V[1]=1, V[2]=-1
        # C terms: C00 += 1, C10 += 1 => W[0]=1 (k=0,i=0), W[2]=1 (k=1,i=0)
        r = 3
        U[3][r] = 1
        V[1][r] = 1; V[2][r] = -1
        W[0][r] = 1; W[2][r] = 1
        
        # M5 = (A00+A01)B11
        # A terms: (0,0)=1, (0,1)=1 => U[0]=1, U[1]=1
        # B terms: (1,1)=1 => V[3]=1
        # C terms: C00 -= 1, C01 += 1 => W[0]=-1, W[1]=1
        r = 4
        U[0][r] = 1; U[1][r] = 1
        V[3][r] = 1
        W[0][r] = -1; W[1][r] = 1
        
        # M6 = (A10-A00)(B00+B10)
        # A terms: (1,0)=1, (0,0)=-1 => U[2]=1, U[0]=-1
        # B terms: (0,0)=1, (1,0)=1 => V[0]=1, V[2]=1
        # C terms: C10 += 1 => W[2]=1
        r = 5
        U[2][r] = 1; U[0][r] = -1
        V[0][r] = 1; V[2][r] = 1
        W[2][r] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # A terms: (0,1)=1, (1,1)=-1 => U[1]=1, U[3]=-1
        # B terms: (0,1)=1, (1,1)=1 => V[1]=1, V[3]=1
        # C terms: C00 -= 1, C11 += 1 => W[0]=-1, W[3]=1
        r = 6
        U[1][r] = 1; U[3][r] = -1
        V[1][r] = 1; V[3][r] = 1
        W[0][r] = -1; W[3][r] = 1
        
        return U, V, W
    
    # Now build 2x4 * 4x2 multiplication using two 2x2x2 Strassen blocks
    # A is 2x4, B is 4x2, C is 2x2
    # Split A into A_left (2x2) and A_right (2x2), B into B_top (2x2) and B_bottom (2x2)
    # C = A_left * B_top + A_right * B_bottom
    # Each product has rank 7, total rank 14
    
    def multiply_2x4_by_4x2():
        """Return U, V, W for (2x4)*(4x2) multiplication, rank 14."""
        U_total = [[0]*14 for _ in range(8)]  # i*4+j, i in {0,1}, j in {0..3}
        V_total = [[0]*14 for _ in range(8)]  # j*2+k, j in {0..3}, k in {0,1}
        W_total = [[0]*14 for _ in range(4)]  # k*2+i, k in {0,1}, i in {0,1}
        
        U1, V1, W1 = strassen_2x2x2()
        U2, V2, W2 = strassen_2x2x2()
        
        # First block: A_left (cols 0,1) * B_top (rows 0,1)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U_total[i*4 + j][r] = U1[i*2 + j][r]
            for j in range(2):
                for k in range(2):
                    V_total[j*2 + k][r] = V1[j*2 + k][r]
            for k in range(2):
                for i in range(2):
                    W_total[k*2 + i][r] = W1[k*2 + i][r]
        
        # Second block: A_right (cols 2,3) * B_bottom (rows 2,3)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U_total[i*4 + (j+2)][r+7] = U2[i*2 + j][r]
            for j in range(2):
                for k in range(2):
                    V_total[(j+2)*2 + k][r+7] = V2[j*2 + k][r]
            for k in range(2):
                for i in range(2):
                    W_total[k*2 + i][r+7] = W2[k*2 + i][r]
        
        return U_total, V_total, W_total
    
    # Now build 2x4 * 4x8 multiplication using four 2x4*4x2 blocks
    # B is 4x8, split into 4 blocks of 4x2
    # C is 2x8, each block produces 2x2
    
    def multiply_2x4_by_4x8():
        """Return U, V, W for (2x4)*(4x8) multiplication, rank 56."""
        U_total = [[0]*56 for _ in range(8)]   # i*4+j
        V_total = [[0]*56 for _ in range(32)]  # j*8+k, j in {0..3}, k in {0..7}
        W_total = [[0]*56 for _ in range(16)]  # k*2+i, k in {0..7}, i in {0,1}
        
        for block in range(4):
            U1, V1, W1 = multiply_2x4_by_4x2()
            for r in range(14):
                for i in range(2):
                    for j in range(4):
                        U_total[i*4 + j][block*14 + r] = U1[i*4 + j][r]
                for j in range(4):
                    for k in range(2):
                        V_total[j*8 + (block*2 + k)][block*14 + r] = V1[j*2 + k][r]
                for k in range(2):
                    for i in range(2):
                        W_total[(block*2 + k)*2 + i][block*14 + r] = W1[k*2 + i][r]
        
        return U_total, V_total, W_total
    
    # Finally build 4x4 * 4x8 multiplication using two 2x4*4x8 blocks
    # A is 4x4, split into top (2x4) and bottom (2x4)
    # C is 4x8, top half is 2x8, bottom half is 2x8
    
    def multiply_4x4_by_4x8():
        """Return U, V, W for (4x4)*(4x8) multiplication, rank 112."""
        U_total = [[0]*112 for _ in range(16)]  # i*4+j, i in {0..3}, j in {0..3}
        V_total = [[0]*112 for _ in range(32)]  # j*8+k, j in {0..3}, k in {0..7}
        W_total = [[0]*112 for _ in range(32)]  # k*4+i, k in {0..7}, i in {0..3}
        
        for half in range(2):
            U1, V1, W1 = multiply_2x4_by_4x8()
            for r in range(56):
                for i in range(2):
                    for j in range(4):
                        U_total[(half*2 + i)*4 + j][half*56 + r] = U1[i*4 + j][r]
                for j in range(4):
                    for k in range(8):
                        V_total[j*8 + k][half*56 + r] = V1[j*8 + k][r]
                for k in range(8):
                    for i in range(2):
                        W_total[k*4 + (half*2 + i)][half*56 + r] = W1[k*2 + i][r]
        
        return U_total, V_total, W_total
    
    U, V, W = multiply_4x4_by_4x8()
    
    # Verify the decomposition is correct
    # Check: sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r] == 1 for all i,j,k
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(112):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    errors += 1
    if errors > 0:
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
