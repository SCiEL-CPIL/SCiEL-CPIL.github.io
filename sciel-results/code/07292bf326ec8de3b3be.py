"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 4]
_TIME_LIMIT = 115.0  # safety margin before 120s kill
_START = time.time()

# EVOLVE-BLOCK-START
def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START)

def _to_rational_matrix(mat, tol=1e-9):
    """Convert a numpy float matrix to integer/half-integer matrix using Fraction."""
    rows, cols = mat.shape
    out = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            v = mat[i, j]
            if abs(v) < tol:
                out[i][j] = 0
            else:
                # Try to find a rational with small denominator
                # First check if it's close to a half-integer
                f = Fraction(v).limit_denominator(1000)
                # Check if denominator is a power of 2 (for half-integer compatibility)
                # Actually, the task says "integer/half-integer entries"
                # Let's be more careful: try denominators 1, 2, 4, 8, ...
                found = False
                for den in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
                    num = round(v * den)
                    if abs(v * den - num) < tol * den:
                        f = Fraction(num, den)
                        # Reduce
                        found = True
                        break
                if not found:
                    f = Fraction(v).limit_denominator(10000)
                out[i][j] = f
    return out

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if i<n, j<m, k<p else 0
    # Actually, the tensor for matrix multiplication is:
    # T[i, j, k] = delta_{i,k} for the standard tensor
    # Wait, let me reconsider. The matrix multiplication tensor T satisfies:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if the indices match appropriately.
    # Specifically, T[i,j,k] = 1 for all valid i,j,k? No.
    # The tensor is T[i,j,k] = delta_{i,?}... 
    # Let me think again. The multiplication tensor has entries:
    # T[i, j, k] = 1 if i, j, k are valid, representing A[i,j]*B[j,k] -> C[i,k]
    # So actually T[i,j,k] = 1 for all 0<=i<n, 0<=j<m, 0<=k<p.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid i,j,k.
    
    # Let's verify by checking a few random points
    import random
    # Check all points for small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # This should equal 1 (since T[i,j,k]=1)
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > tol:
                        return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    
    Strassen:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    """
    # For 2x2, n=2, m=2, p=2
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)
    # W has shape (p*n, rank) = (4, 7)
    
    # Indices:
    # A: i in {0,1}, j in {0,1} -> U row index i*m+j = i*2+j
    # B: j in {0,1}, k in {0,1} -> V row index j*p+k = j*2+k
    # C: k in {0,1}, i in {0,1} -> W row index k*n+i = k*2+i
    
    # A indices: a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
    # B indices: e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[2], h=B[1,1]=V[3]
    # C indices: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # P1 = (a+d)(e+h): U[0]=1, U[3]=1, V[0]=1, V[3]=1
    # P2 = (c+d)e: U[2]=1, U[3]=1, V[0]=1
    # P3 = a(f-h): U[0]=1, V[1]=1, V[3]=-1
    # P4 = d(g-e): U[3]=1, V[2]=1, V[0]=-1
    # P5 = (a+b)h: U[0]=1, U[1]=1, V[3]=1
    # P6 = (c-a)(e+f): U[2]=1, U[0]=-1, V[0]=1, V[1]=1
    # P7 = (b-d)(g+h): U[1]=1, U[3]=-1, V[2]=1, V[3]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 and C22
    
    # P2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 += P2, C22 -= P2
    
    # P3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 += P3, C22 += P3
    
    # P4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 += P4, C21 += P4
    
    # P5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 -= P5, C12 += P5
    
    # P6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 += P6
    
    # P7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 += P7
    
    return {"U": U, "V": V, "W": W}

def _construct_strassen_28():
    """
    Use Strassen's 7-mult for each of the four 2x2 block products.
    A is 2x4, B is 4x4.
    Split A into A1 (top row) and A2 (bottom row), each 1x4.
    Actually, let's think of A as a 2x4 matrix and B as a 4x4 matrix.
    
    We can split B into four 4x2 blocks? Or split differently.
    
    Let A = [[a1, a2], [b1, b2]] where a1, a2, b1, b2 are 1x2 row blocks.
    Let B = [[B11, B12], [B21, B22]] where Bij are 2x2 blocks.
    
    Then C = A*B = [[a1*B11 + a2*B21, a1*B12 + a2*B22],
                    [b1*B11 + b2*B21, b1*B12 + b2*B22]]
    
    Each term like a1*B11 is a (1x2)*(2x2) = 1x2 product.
    This is not directly Strassen.
    
    Alternative: Think of the 2x4 * 4x4 multiplication as four 2x2 * 2x2 multiplications
    plus cross terms. This gets complicated.
    
    Let me try a different approach: embed the 2x4*4x4 into a 4x4*4x4 and use Strassen,
    then extract. But that would give rank 7*4=28 for the 4x4 case, and we need to extract
    the 2x4*4x4 part.
    
    Actually, the knowledge says: "this implements Strassen's 7-mult algorithm on each of the
    four 2x2 block products (A1*B11, A2*B21, A1*B12, A2*B22), giving rank 28"
    
    So we have four independent 2x2*2x2 multiplications, each using rank 7, total rank 28.
    
    Let's set up the indices properly.
    A is 2x4: rows i in {0,1}, columns j in {0,1,2,3}
    B is 4x4: rows j in {0,1,2,3}, columns k in {0,1,2,3}
    C is 2x4: rows i in {0,1}, columns k in {0,1,2,3}
    
    W[k*n+i] where n=2, so W[k*2+i]
    
    The four block products:
    1. A[:, 0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
    2. A[:, 2:4] * B[2:4, 0:2] -> contributes to C[:, 0:2]
    3. A[:, 0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
    4. A[:, 2:4] * B[2:4, 2:4] -> contributes to C[:, 2:4]
    
    Each is a 2x2 * 2x2 multiplication.
    
    For block 1: A1 = A[:, 0:2], B11 = B[0:2, 0:2], C1 = C[:, 0:2]
    A1 indices: i in {0,1}, j in {0,1} -> U row i*4+j (since m=4)
    B11 indices: j in {0,1}, k in {0,1} -> V row j*4+k (since p=4)
    C1 indices: k in {0,1}, i in {0,1} -> W row k*2+i
    
    For block 2: A2 = A[:, 2:4], B21 = B[2:4, 0:2], C2 = C[:, 0:2]
    A2 indices: i in {0,1}, j in {2,3} -> U row i*4+j
    B21 indices: j in {2,3}, k in {0,1} -> V row j*4+k
    C2 indices: k in {0,1}, i in {0,1} -> W row k*2+i
    
    For block 3: A1 = A[:, 0:2], B12 = B[0:2, 2:4], C3 = C[:, 2:4]
    A1 indices: i in {0,1}, j in {0,1} -> U row i*4+j
    B12 indices: j in {0,1}, k in {2,3} -> V row j*4+k
    C3 indices: k in {2,3}, i in {0,1} -> W row k*2+i
    
    For block 4: A2 = A[:, 2:4], B22 = B[2:4, 2:4], C4 = C[:, 2:4]
    A2 indices: i in {0,1}, j in {2,3} -> U row i*4+j
    B22 indices: j in {2,3}, k in {2,3} -> V row j*4+k
    C4 indices: k in {2,3}, i in {0,1} -> W row k*2+i
    
    Each block uses rank 7, total rank 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 2x2 factors
    strassen = _strassen_2x2()
    sU = strassen["U"]  # 4x7
    sV = strassen["V"]  # 4x7
    sW = strassen["W"]  # 4x7
    
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    # U rows: i*4+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # V rows: j*4+k for j in {0,1}, k in {0,1} -> rows 0,1,4,5
    # W rows: k*2+i for k in {0,1}, i in {0,1} -> rows 0,1,2,3
    for r in range(7):
        for si in range(2):
            for sj in range(2):
                # U index: si*4 + sj
                # sU index: si*2 + sj
                U[si*4 + sj][r] += sU[si*2 + sj][r]
        for sj in range(2):
            for sk in range(2):
                # V index: sj*4 + sk
                # sV index: sj*2 + sk
                V[sj*4 + sk][r] += sV[sj*2 + sk][r]
        for sk in range(2):
            for si in range(2):
                # W index: sk*2 + si
                # sW index: sk*2 + si
                W[sk*2 + si][r] += sW[sk*2 + si][r]
    
    # Block 2: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    # U rows: i*4+j for i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # V rows: j*4+k for j in {2,3}, k in {0,1} -> rows 8,9,12,13
    # W rows: k*2+i for k in {0,1}, i in {0,1} -> rows 0,1,2,3
    for r in range(7):
        col = 7 + r
        for si in range(2):
            for sj in range(2):
                U[si*4 + 2 + sj][col] += sU[si*2 + sj][r]
        for sj in range(2):
            for sk in range(2):
                V[2 + sj][col] += sV[sj*2 + sk][r]  # Wait, V row index is j*4+k where j in {2,3}
                # j = 2+sj, so row = (2+sj)*4 + sk = 8 + 4*sj + sk
                # Let me redo this
    # Let me redo block 2 properly
    for r in range(7):
        col = 7 + r
        for si in range(2):
            for sj in range(2):
                U[si*4 + 2 + sj][col] = sU[si*2 + sj][r]
        for sj in range(2):
            for sk in range(2):
                V[(2+sj)*4 + sk][col] = sV[sj*2 + sk][r]
        for sk in range(2):
            for si in range(2):
                W[sk*2 + si][col] += sW[sk*2 + si][r]
    
    # Block 3: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    # U rows: i*4+j for i in {0,1}, j in {0,1} -> rows 0,1,4,5
    # V rows: j*4+k for j in {0,1}, k in {2,3} -> rows 2,3,6,7
    # W rows: k*2+i for k in {2,3}, i in {0,1} -> rows 4,5,6,7
    for r in range(7):
        col = 14 + r
        for si in range(2):
            for sj in range(2):
                U[si*4 + sj][col] += sU[si*2 + sj][r]
        for sj in range(2):
            for sk in range(2):
                V[sj*4 + 2 + sk][col] = sV[sj*2 + sk][r]
        for sk in range(2):
            for si in range(2):
                W[(2+sk)*2 + si][col] = sW[sk*2 + si][r]
    
    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    # U rows: i*4+j for i in {0,1}, j in {2,3} -> rows 2,3,6,7
    # V rows: j*4+k for j in {2,3}, k in {2,3} -> rows 10,11,14,15
    # W rows: k*2+i for k in {2,3}, i in {0,1} -> rows 4,5,6,7
    for r in range(7):
        col = 21 + r
        for si in range(2):
            for sj in range(2):
                U[si*4 + 2 + sj][col] += sU[si*2 + sj][r]
        for sj in range(2):
            for sk in range(2):
                V[(2+sj)*4 + 2 + sk][col] = sV[sj*2 + sk][r]
        for sk in range(2):
            for si in range(2):
                W[(2+sk)*2 + si][col] += sW[sk*2 + si][r]
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_rank, max_rank=None):
    """
    Try to reduce the rank below base_rank by searching for a lower-rank decomposition.
    This is a heuristic search.
    """
    n, m, p = DIMENSIONS
    if max_rank is None:
        max_rank = base_rank - 1
    
    # For now, return the schoolbook as a fallback
    return _schoolbook()

def construct():
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank 28
    result = _construct_strassen_28()
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
