import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_terms():
    """Return the 7 Strassen bilinear forms for 2x2x2 matmul.
    Convention: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    """
    terms = []
    # m1 = (A00+A11)(B00+B11) -> C00 += m1, C11 += m1
    a = [[1,0],[0,1]]
    b = [[1,0],[0,1]]
    c = [[1,0],[0,1]]  # C[i,k], so W[k,i] = c[i,k]
    terms.append((a,b,c))
    # m2 = (A10+A11)B00 -> C10 += m2, C11 -= m2
    a = [[0,0],[1,1]]
    b = [[1,0],[0,0]]
    c = [[0,0],[1,-1]]  # C[1,0]+=1, C[1,1]-=1
    terms.append((a,b,c))
    # m3 = A00(B01-B11) -> C01 += m3
    a = [[1,0],[0,0]]
    b = [[0,1],[0,-1]]
    c = [[0,1],[0,0]]  # C[0,1]+=1
    terms.append((a,b,c))
    # m4 = A11(B10-B00) -> C00 += m4, C10 += m4
    a = [[0,0],[0,1]]
    b = [[-1,0],[1,0]]
    c = [[1,0],[1,0]]  # C[0,0]+=1, C[1,0]+=1
    terms.append((a,b,c))
    # m5 = (A00+A01)B11 -> C00 -= m5, C01 += m5
    a = [[1,1],[0,0]]
    b = [[0,0],[0,1]]
    c = [[-1,1],[0,0]]  # C[0,0]-=1, C[0,1]+=1
    terms.append((a,b,c))
    # m6 = (A10-A00)(B00+B10) -> C11 += m6
    a = [[-1,0],[1,0]]
    b = [[1,0],[1,0]]
    c = [[0,0],[0,1]]  # C[1,1]+=1
    terms.append((a,b,c))
    # m7 = (A01-A11)(B10+B11) -> C00 += m7
    a = [[0,1],[0,-1]]
    b = [[0,0],[1,1]]
    c = [[1,0],[0,0]]  # C[0,0]+=1
    terms.append((a,b,c))
    return terms

def _add_block(U, V, W, n, m, p, i0, j0, k0, sa, sb, sc, col):
    if sa == 2 and sb == 2 and sc == 2:
        for (a, b, c) in _strassen_terms():
            for i in range(2):
                for j in range(2):
                    if a[i][j] != 0:
                        U[(i0+i)*m + (j0+j)][col] += a[i][j]
            for j in range(2):
                for k in range(2):
                    if b[j][k] != 0:
                        V[(j0+j)*p + (k0+k)][col] += b[j][k]
            for i in range(2):
                for k in range(2):
                    if c[i][k] != 0:
                        W[(k0+k)*n + (i0+i)][col] += c[i][k]
            col += 1
    else:
        for i in range(sa):
            for j in range(sb):
                for k in range(sc):
                    U[(i0+i)*m + (j0+j)][col] += 1
                    V[(j0+j)*p + (k0+k)][col] += 1
                    W[(k0+k)*n + (i0+i)][col] += 1
                    col += 1
    return col

def _block_decomp(n, m, p):
    blocks = [(0, 2), (2, 2), (4, 1)]
    U = [[0] * 200 for _ in range(n * m)]
    V = [[0] * 200 for _ in range(m * p)]
    W = [[0] * 200 for _ in range(p * n)]
    col = 0
    for (i0, sa) in blocks:
        for (j0, sb) in blocks:
            for (k0, sc) in blocks:
                col = _add_block(U, V, W, n, m, p, i0, j0, k0, sa, sb, sc, col)
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}, rank

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i]))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    try:
        result, rank = _block_decomp(n, m, p)
        if time.time() < deadline:
            if _verify(result, n, m, p):
                return result
    except Exception:
        pass
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
