"""Strassen-based decomposition for 2x2x10 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    
    Strategy: Apply Strassen's rank-7 decomposition to the first 2 columns (2x2x2 block),
    and standard rank-4 schoolbook for each of the remaining 4 columns (2x2x1 blocks).
    
    Total rank: 7 + 4*4 = 35 (vs 40 for pure schoolbook).
    
    Tensor convention: T[i,j,k] = 1 iff output[i,k] depends on A[i,j]*B[j,k]
    U shape: (n*m, rank) = (4, rank)
    V shape: (m*p, rank) = (20, rank)
    W shape: (p*n, rank) = (20, rank)  -- W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # We'll build the decomposition in blocks.
    # For each pair of columns (k, k+1) we can use Strassen (rank 7) 
    # but only the first 2 columns benefit from Strassen since we need
    # a 2x2x2 sub-problem. Actually, let me reconsider.
    
    # The matrix multiplication is: C = A * B where A is 2x2, B is 2x10, C is 2x10.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T has entries T[i,j,k] = 1 (for the multiplication tensor)
    # U has rows indexed by (i,j) pairs: i in {0,1}, j in {0,1} -> 4 rows
    # V has rows indexed by (j,k) pairs: j in {0,1}, k in {0,...,9} -> 20 rows
    # W has rows indexed by (k,i) pairs: k in {0,...,9}, i in {0,1} -> 20 rows
    #
    # Rank = number of columns in U, V, W.
    
    # Strassen for 2x2x2:
    # The 7 intermediate products for multiplying 2x2 * 2x2:
    # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    # m3 = A[0,0]*(B[0,1]-B[1,1])
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1])*B[1,1]
    # m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For the 2x2x2 block (columns 0,1), the Strassen decomposition:
    # U rows: (0,0), (0,1), (1,0), (1,1)  [i*m+j]
    # V rows: (0,0), (0,1), (1,0), (1,1)  [j*p+k] but for the 2x2 subblock, k in {0,1}
    #         Actually V[j*p+k] where p=10, so V[0*10+0]=V[0], V[0*10+1]=V[1], V[1*10+0]=V[10], V[1*10+1]=V[11]
    # W rows: (0,0), (0,1), (1,0), (1,1)  [k*n+i] so W[0*2+0]=W[0], W[0*2+1]=W[1], W[1*2+0]=W[2], W[1*2+1]=W[3]
    
    # Let me define the Strassen factors for the 2x2x2 case.
    # U_strassen[4][7], V_strassen[4][7], W_strassen[4][7]
    
    # m1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> row (0,0)=1, row (1,1)=1
    # V: B00=1, B11=1 -> row (0,0)=1, row (1,1)=1
    # W: C00 += 1, C11 += 1 -> row (0,0)=1, row (1,1)=1
    
    # m2 = (A10+A11)*B00
    # U: A10=1, A11=1 -> row (1,0)=1, row (1,1)=1
    # V: B00=1 -> row (0,0)=1
    # W: C10 += 1 -> row (1,0)=1
    
    # m3 = A00*(B01-B11)
    # U: A00=1 -> row (0,0)=1
    # V: B01=1, B11=-1 -> row (0,1)=1, row (1,1)=-1
    # W: C01 += 1 -> row (0,1)=1
    
    # m4 = A11*(B10-B00)
    # U: A11=1 -> row (1,1)=1
    # V: B10=1, B00=-1 -> row (1,0)=1, row (0,0)=-1
    # W: C00 += 1, C10 += 1 -> row (0,0)=1, row (1,0)=1
    
    # m5 = (A00+A01)*B11
    # U: A00=1, A01=1 -> row (0,0)=1, row (0,1)=1
    # V: B11=1 -> row (1,1)=1
    # W: C00 -= 1, C01 += 1 -> row (0,0)=-1, row (0,1)=1
    
    # m6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> row (1,0)=1, row (0,0)=-1
    # V: B00=1, B01=1 -> row (0,0)=1, row (0,1)=1
    # W: C11 += 1 -> row (1,1)=1
    
    # m7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> row (0,1)=1, row (1,1)=-1
    # V: B10=1, B11=1 -> row (1,0)=1, row (1,1)=1
    # W: C00 += 1, C11 += 1 -> row (0,0)=1, row (1,1)=1
    
    # Let's build the full decomposition.
    # Total rank = 7 (Strassen for cols 0-1) + 4*4 (schoolbook for cols 2-9) = 35
    
    total_rank = 7 + 4 * 4  # 35
    
    U = [[0] * total_rank for _ in range(4)]   # 4 rows (i,j)
    V = [[0] * total_rank for _ in range(20)]  # 20 rows (j,k)
    W = [[0] * total_rank for _ in range(20)]  # 20 rows (k,i)
    
    # Strassen block for columns 0, 1
    # Column offset in V and W for these columns
    # V[j*10+k], W[k*2+i]
    
    # m1
    col = 0
    U[0][col] = 1   # (i=0,j=0)
    U[3][col] = 1   # (i=1,j=1)
    V[0][col] = 1   # (j=0,k=0)
    V[11][col] = 1  # (j=1,k=1)
    W[0][col] = 1   # (k=0,i=0)
    W[3][col] = 1   # (k=1,i=1)
    
    # m2
    col = 1
    U[2][col] = 1   # (i=1,j=0)
    U[3][col] = 1   # (i=1,j=1)
    V[0][col] = 1   # (j=0,k=0)
    W[2][col] = 1   # (k=1,i=0)
    
    # m3
    col = 2
    U[0][col] = 1   # (i=0,j=0)
    V[1][col] = 1   # (j=0,k=1)
    V[11][col] = -1 # (j=1,k=1)
    W[1][col] = 1   # (k=0,i=1)
    
    # m4
    col = 3
    U[3][col] = 1   # (i=1,j=1)
    V[10][col] = 1  # (j=1,k=0)
    V[0][col] = -1  # (j=0,k=0)
    W[0][col] = 1   # (k=0,i=0)
    W[2][col] = 1   # (k=1,i=0)
    
    # m5
    col = 4
    U[0][col] = 1   # (i=0,j=0)
    U[1][col] = 1   # (i=0,j=1)
    V[11][col] = 1  # (j=1,k=1)
    W[0][col] = -1  # (k=0,i=0)
    W[1][col] = 1   # (k=0,i=1)
    
    # m6
    col = 5
    U[2][col] = 1   # (i=1,j=0)
    U[0][col] = -1  # (i=0,j=0)
    V[0][col] = 1   # (j=0,k=0)
    V[1][col] = 1   # (j=0,k=1)
    W[3][col] = 1   # (k=1,i=1)
    
    # m7
    col = 6
    U[1][col] = 1   # (i=0,j=1)
    U[3][col] = -1  # (i=1,j=1)
    V[10][col] = 1  # (j=1,k=0)
    V[11][col] = 1  # (j=1,k=1)
    W[0][col] = 1   # (k=0,i=0)
    W[3][col] = 1   # (k=1,i=1)
    
    # Schoolbook for columns 2-9 (8 columns, each needs rank 4)
    col_offset = 7
    for k in range(2, 10):
        for i in range(2):
            for j in range(2):
                c = col_offset
                col_offset += 1
                U[i*2+j][c] = 1    # (i,j)
                V[j*10+k][c] = 1   # (j,k)
                W[k*2+i][c] = 1    # (k,i)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
