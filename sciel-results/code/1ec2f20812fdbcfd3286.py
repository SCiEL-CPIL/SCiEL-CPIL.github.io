"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_DEADLINE = 140.0
_T0 = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _T0)

# EVOLVE-BLOCK-START
def _strassen_2x2_2():
    """
    Strassen 2x2 * 2x2 decomposition, rank 7.
    Returns U (4,7), V (4,7), W (4,7).
    U indexed by [i,j] -> i*2+j
    V indexed by [j,k] -> j*2+k
    W indexed by [k,i] -> k*2+i
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # P1 = (a+d)(e+h): u[0,0]=1, u[3,0]=1; v[0,0]=1, v[3,0]=1; w[0,0]=1, w[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (c+d)e: u[2,1]=1, u[3,1]=1; v[0,1]=1; w[1,1]=1, w[3,1]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # P3 = a(f-h): u[0,2]=1; v[1,2]=1, v[3,2]=-1; w[2,2]=1, w[3,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4 = d(g-e): u[3,3]=1; v[2,3]=1, v[0,3]=-1; w[0,3]=1, w[1,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (a+b)h: u[0,4]=1, u[1,4]=1; v[3,4]=1; w[0,4]=-1, w[2,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (c-a)(e+f): u[2,5]=1, u[0,5]=-1; v[0,5]=1, v[1,5]=1; w[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (b-d)(g+h): u[1,6]=1, u[3,6]=-1; v[2,6]=1, v[3,6]=1; w[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_decomposition(n, m, p, blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    blocks: list of (i0, j0, k0) - top-left corner of each 2x2x2 block
    """
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0 + di < n and j0 + dj < m and k0 + dk < p:
                        covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_str, V_str, W_str = _strassen_2x2_2()
    
    col = 0
    for (i0, j0, k0) in blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = U_str[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = V_str[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = W_str[dk*2+di, c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x8 matrix.
    
    Uses Strassen 2x2x2 blocks where possible, schoolbook for remainder.
    """
    n, m, p = DIMENSIONS
    
    # Find non-overlapping 2x2x2 blocks
    # For 3x3x8: we can place blocks at (0,0,k) for k=0,2,4,6
    # Each covers i in {0,1}, j in {0,1}, k in {k0,k0+1}
    # 4 blocks cover 4*8=32 entries, remainder = 72-32 = 40
    # Total rank = 4*7 + 40 = 68
    
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    U, V, W = _build_decomposition(n, m, p, blocks)
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
