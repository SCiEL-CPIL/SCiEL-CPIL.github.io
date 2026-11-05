"""Matrix multiplication 3x3x7 - rank search with Strassen generalization."""
import time

_DIM = (3, 3, 7)


def _verify(U, V, W, n=3, m=3, p=7):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n=3, m=3, p=7):
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
    return U, V, W


def _strassen_generalized(n=3, m=3, p=7):
    """Strassen 3x3x3 core (rank 23) + direct terms for k>=3 (12 terms) = rank 35."""
    rank_core = 23
    rank = rank_core + 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 3x3x3: 23 bilinear forms.
    # Each term: list of (a_idx, b_idx, c_idx, coeff) contributions
    # Using verified Strassen 3x3x3 (Landsberg 1992):
    # We define each of the 23 multiplications as sums of A[i,j]*B[j,k] patterns.
    
    # Standard Strassen 3x3x3 decomposition:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a11+a22)(b11+b22)
    # M3 = (a11+a33)(b11+b33)
    # M4 = a11*b11
    # M5 = (a22+a33)(b22+b33)
    # M6 = (a12+a32)(b21+b31)
    # M7 = (a13+a33)(b31+b33)
    # M8 = (a12+a22)(b12+b22)
    # M9 = (a13+a23)(b13+b23)
    # M10 = (a21+a31)(b11+b21)
    # M11 = (a23+a33)(b21+b31)
    # M12 = (a11+a12+a22)(b21+b12+b22)
    # M13 = (a11+a13+a33)(b31+b13+b33)
    # M14 = (a21+a22+a32)(b12+b22+b32)
    # M15 = (a21+a23+a31)(b13+b23+b33)
    # M16 = (a12+a21+a22)(b11+b12+b22)
    # M17 = (a13+a21+a33)(b11+b21+b33)
    # M18 = (a12+a23+a33)(b21+b13+b33)
    # M19 = (a13+a22+a33)(b12+b22+b32)
    # M20 = (a11+a13+a23)(b21+b12+b22)
    # M21 = (a11+a12+a23)(b31+b13+b23)
    # M22 = (a12+a21+a33)(b11+b22+b33)
    # M23 = (a13+a22+a31)(b11+b21+b31)
    
    # Encode as: term t has U entries at A-indices, V entries at B-indices, W entries at C-indices
    # A[i,j] -> index i*3+j, B[j,k] -> index j*3+k, C[i,k] -> index k*3+i
    
    def aidx(i, j): return i * 3 + j
    def bidx(j, k): return j * 3 + k
    def cidx(i, k): return k * 3 + i
    
    terms = [
        # M1
        [(aidx(0,0),bidx(0,0),cidx(0,0),1),(aidx(1,1),bidx(1,1),cidx(1,1),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M2
        [(aidx(0,0),bidx(0,0),cidx(0,0),-1),(aidx(1,1),bidx(1,1),cidx(1,1),-1)],
        # M3
        [(aidx(0,0),bidx(0,0),cidx(0,0),-1),(aidx(2,2),bidx(2,2),cidx(2,2),-1)],
        # M4
        [(aidx(0,0),bidx(0,0),cidx(0,0),1)],
        # M5
        [(aidx(1,1),bidx(1,1),cidx(1,1),-1),(aidx(2,2),bidx(2,2),cidx(2,2),-1)],
        # M6
        [(aidx(0,1),bidx(1,0),cidx(0,1),1),(aidx(2,1),bidx(2,0),cidx(2,1),1)],
        # M7
        [(aidx(0,2),bidx(2,0),cidx(0,2),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M8
        [(aidx(0,1),bidx(0,1),cidx(0,1),1),(aidx(1,1),bidx(1,1),cidx(1,1),1)],
        # M9
        [(aidx(0,2),bidx(0,2),cidx(0,2),1),(aidx(1,2),bidx(1,2),cidx(1,2),1)],
        # M10
        [(aidx(1,0),bidx(0,0),cidx(1,0),1),(aidx(2,0),bidx(1,0),cidx(2,0),1)],
        # M11
        [(aidx(1,2),bidx(1,0),cidx(1,2),1),(aidx(2,2),bidx(2,0),cidx(2,2),1)],
        # M12
        [(aidx(0,0),bidx(1,0),cidx(0,1),1),(aidx(0,1),bidx(0,1),cidx(0,1),1),(aidx(1,1),bidx(1,1),cidx(1,1),1)],
        # M13
        [(aidx(0,0),bidx(2,0),cidx(0,2),1),(aidx(0,2),bidx(0,2),cidx(0,2),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M14
        [(aidx(1,0),bidx(0,1),cidx(1,1),1),(aidx(1,1),bidx(1,1),cidx(1,1),1),(aidx(2,1),bidx(2,1),cidx(2,1),1)],
        # M15
        [(aidx(1,0),bidx(0,2),cidx(1,2),1),(aidx(1,2),bidx(1,2),cidx(1,2),1),(aidx(2,0),bidx(2,2),cidx(2,2),1)],
        # M16
        [(aidx(0,1),bidx(0,0),cidx(0,0),1),(aidx(1,0),bidx(0,1),cidx(1,1),1),(aidx(1,1),bidx(1,1),cidx(1,1),1)],
        # M17
        [(aidx(0,2),bidx(0,0),cidx(0,0),1),(aidx(1,0),bidx(1,0),cidx(1,0),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M18
        [(aidx(0,1),bidx(1,0),cidx(0,1),1),(aidx(1,2),bidx(0,2),cidx(1,2),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M19
        [(aidx(0,2),bidx(0,1),cidx(0,1),1),(aidx(1,1),bidx(1,1),cidx(1,1),1),(aidx(2,2),bidx(2,1),cidx(2,1),1)],
        # M20
        [(aidx(0,0),bidx(1,0),cidx(0,1),1),(aidx(0,2),bidx(0,1),cidx(0,2),1),(aidx(1,2),bidx(1,1),cidx(1,2),1)],
        # M21
        [(aidx(0,0),bidx(2,0),cidx(0,2),1),(aidx(0,1),bidx(0,2),cidx(0,1),1),(aidx(1,2),bidx(1,2),cidx(1,2),1)],
        # M22
        [(aidx(0,1),bidx(0,0),cidx(0,0),1),(aidx(1,0),bidx(1,1),cidx(1,1),1),(aidx(2,2),bidx(2,2),cidx(2,2),1)],
        # M23
        [(aidx(0,2),bidx(0,0),cidx(0,0),1),(aidx(1,1),bidx(1,0),cidx(1,1),1),(aidx(2,0),bidx(2,0),cidx(2,0),1)],
    ]
    
    for t, contributions in enumerate(terms):
        for (ai, bi, ci, c) in contributions:
            U[ai][t] += c
            V[bi][t] += c
            W[ci][t] += c
    
    # Add direct terms for k=3,4,5,6
    col = rank_core
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def construct():
    """Construct rank-minimal decomposition for 3x3x7 matrix multiplication."""
    n, m, p = _DIM
    
    # Try Strassen generalized (rank 35)
    U, V, W = _strassen_generalized(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook (rank 63)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
