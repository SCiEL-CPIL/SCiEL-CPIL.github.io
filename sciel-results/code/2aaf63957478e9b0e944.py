"""Block Strassen decomposition for 3x5x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U,V,W) for the 2x2x2 matrix multiplication tensor using Strassen (rank 7).
    Convention: U index = i*2+j (i in [0,2), j in [0,2)), V index = j*2+k, W index = k*2+i.
    Strassen 7 products:
      M1=(A11-A22)(B12+B22)
      M2=(A11+A22)B22
      M3=A11(B11-B22)
      M4=A22(B22-B11)
      M5=(A11+A12)B22
      M6=(A21-A11)(B11+B12)
      M7=(A12-A22)(B21+B22)
    C11=M1+M4-M5+M7
    C12=M3+M5
    C21=M2+M4
    C22=M1-M2+M3+M6
    """
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # A[i,j], B[j,k], C[i,k]
    # M1: (A11-A22)(B12+B22)
    # M2: (A11+A22)B22
    # M3: A11(B11-B22)
    # M4: A22(B22-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    #
    # A indices: A00=(0,0),A01=(0,1),A10=(1,0),A11=(1,1)
    # B indices: B00=(0,0),B01=(0,1),B10=(1,0),B11=(1,1)
    #
    # M1: A00 - A11 times B01 + B11
    # M2: A00 + A11 times B11
    # M3: A00 times B00 - B11
    # M4: A11 times B11 - B00
    # M5: A00 + A01 times B11
    # M6: A10 - A00 times B00 + B01
    # M7: A01 - A11 times B10 + B11
    U = [[0]*7 for _ in range(4)]  # i*2+j
    V = [[0]*7 for _ in range(4)]  # j*2+k
    W = [[0]*7 for _ in range(4)]  # k*2+i
    # A00 -> U[0], A01 -> U[1], A10 -> U[2], A11 -> U[3]
    # B00 -> V[0], B01 -> V[1], B10 -> V[2], B11 -> V[3]
    # C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3]
    # M1 (c=0): A00 - A11, B01 + B11
    U[0][0]=1; U[3][0]=-1
    V[1][0]=1; V[3][0]=1
    W[0][0]=1; W[1][0]=1; W[2][0]=-1; W[3][0]=1
    # M2 (c=1): A00 + A11, B11
    U[0][1]=1; U[3][1]=1
    V[3][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3 (c=2): A00, B00 - B11
    U[0][2]=1
    V[0][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # M4 (c=3): A11, B11 - B00
    U[3][3]=1
    V[3][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5 (c=4): A00 + A01, B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=-1
    # M6 (c=5): A10 - A00, B00 + B01
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7 (c=6): A01 - A11, B10 + B11
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1; W[3][6]=-1
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p block.
    U index = i*m+j, V index = j*p+k, W index = k*n+i.
    """
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W


def _concat(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions along the rank dimension."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    U = [row + [0]*r2 for row in U1]
    for row in U2:
        U.append([0]*r1 + list(row))
    V = [row + [0]*r2 for row in V1]
    for row in V2:
        V.append([0]*r1 + list(row))
    W = [row + [0]*r2 for row in W1]
    for row in W2:
        W.append([0]*r1 + list(row))
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor.
    Tensor T[i,j,k] = delta: product over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 if it's the identity term... 
    Actually, the decomposition means: (A*B)[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_l V[j*p+k,c]*B[j,k]) ... 
    The standard check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1 if j is the "middle" index connecting i-j and j-k, i.e., for the tensor of matrix multiplication, T_{i,j,k} = 1 for all (i,j,k) since (AB)_{ik} = sum_j A_{ij}B_{jk}.
    """
    # The matrix multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def _block_decomposition():
    """Build a block decomposition for 3x5x6 using Strassen on 2x2x2 and schoolbook elsewhere.
    
    Partition: n=3 = 2+1, m=5 = 2+3, p=6 = 2+4.
    Blocks:
      (2,2,2) -> Strassen rank 7
      (2,3,2) -> schoolbook rank 12
      (2,2,4) -> schoolbook rank 16
      (1,3,2) -> schoolbook rank 6
      (1,2,4) -> schoolbook rank 8
      (1,3,4) -> schoolbook rank 12
    Total rank = 7+12+16+6+8+12 = 61
    
    But we need to embed each block into the full tensor with correct index offsets.
    """
    n, m, p = 3, 5, 6
    
    # Define blocks as (i_start, j_start, k_start, bi, bj, bk, use_strassen)
    blocks = [
        (0, 0, 0, 2, 2, 2, True),   # 2x2x2 Strassen
        (0, 2, 0, 2, 3, 2, False),  # 2x3x2 schoolbook
        (0, 0, 2, 2, 2, 4, False),  # 2x2x4 schoolbook
        (2, 2, 0, 1, 3, 2, False),  # 1x3x2 schoolbook
        (2, 0, 2, 1, 2, 4, False),  # 1x2x4 schoolbook
        (2, 2, 2, 1, 3, 4, False),  # 1x3x4 schoolbook
    ]
    
    # For each block, we need to create local U,V,W and embed them into the global tensor.
    # Global U index: i*m + j, where i is in [0,n), j in [0,m)
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    # First, compute total rank
    total_rank = 0
    for (i0, j0, k0, bi, bj, bk, use_str) in blocks:
        if use_str:
            total_rank += 7
        else:
            total_rank += bi * bj * bk
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0, bi, bj, bk, use_str) in blocks:
        if use_str:
            u_loc, v_loc, w_loc = _strassen_2x2x2()
            # Embed: local i -> global i0+li, local j -> global j0+lj, local k -> global k0+lk
            # Local U index: li*2+lj -> global (i0+li)*m + (j0+lj)
            # Local V index: lj*2+lk -> global (j0+lj)*p + (k0+lk)
            # Local W index: lk*2+li -> global (k0+lk)*n + (i0+li)
            for li in range(2):
                for lj in range(2):
                    for c in range(7):
                        U[(i0+li)*m + (j0+lj)][col+c] = u_loc[li*2+lj][c]
            for lj in range(2):
                for lk in range(2):
                    for c in range(7):
                        V[(j0+lj)*p + (k0+lk)][col+c] = v_loc[lj*2+lk][c]
            for lk in range(2):
                for li in range(2):
                    for c in range(7):
                        W[(k0+lk)*n + (i0+li)][col+c] = w_loc[lk*2+li][c]
            col += 7
        else:
            u_loc, v_loc, w_loc = _schoolbook_block(bi, bj, bk)
            for li in range(bi):
                for lj in range(bj):
                    for c in range(bi*bj*bk):
                        U[(i0+li)*m + (j0+lj)][col+c] = u_loc[li*bj+lj][c]
            for lj in range(bj):
                for lk in range(bk):
                    for c in range(bi*bj*bk):
                        V[(j0+lj)*p + (k0+lk)][col+c] = v_loc[lj*bk+lk][c]
            for lk in range(bk):
                for li in range(bi):
                    for c in range(bi*bj*bk):
                        W[(k0+lk)*n + (i0+li)][col+c] = w_loc[lk*bi+li][c]
            col += bi*bj*bk
    
    return U, V, W


def _schoolbook_full():
    """Full schoolbook decomposition for 3x5x6, rank 90."""
    n, m, p = 3, 5, 6
    return _schoolbook_block(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try block Strassen decomposition
    try:
        U, V, W = _block_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: full schoolbook
    U, V, W = _schoolbook_full()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
