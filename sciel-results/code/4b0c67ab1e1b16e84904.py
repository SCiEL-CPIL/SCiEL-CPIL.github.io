"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 150.0
MARGIN = 15.0

def _strassen_222():
    """Return U,V,W (each 4x7) for 2x2x2 matmul (Strassen). U[i*2+j], V[j*2+k], W[k*2+i]."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # r=0 (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[1,0]=1; V[2,0]=1; V[3,0]=1
    W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    # r=1 (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    # r=2 A00(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1
    # r=3 A11(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[2,3]=1
    # r=4 (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1
    # r=5 (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[0,5]=1
    # r=6 (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=-1
    return U, V, W

def _block(n, m, p, U, V, W, j0, k0, col, sU, sV, sW):
    """Place a 2x2x2 Strassen block for A[i,j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i,k0:k0+2]."""
    for r in range(7):
        for ii in range(2):
            U[ii*m + (j0+ii), col] = sU[ii*2 + ii, r]
        for jj in range(2):
            V[(j0+jj)*p + (k0+jj), col] = sV[jj*2 + jj, r]
        for kk in range(2):
            W[(k0+kk)*n + kk, col] = sW[kk*2 + kk, r]
        col += 1
    return col

def _sb(n, m, p, U, V, W, j, k0, k1, col):
    """Schoolbook for A[:,j] (2x1) * B[j,k0:k1] (1xw) -> C[:,k0:k1]."""
    w = k1 - k0
    for ii in range(n):
        for kk in range(w):
            U[ii*m + j, col] = 1
            V[j*p + (k0+kk), col] = 1
            W[(k0+kk)*n + ii, col] = 1
            col += 1
    return col

def _build_block(n, m, p, sU, sV, sW):
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    # A1 (j=0,1) * B1 (j=0,1, k=0..5): three Strassen blocks
    for b in range(3):
        col = _block(n, m, p, U, V, W, 0, b*2, col, sU, sV, sW)
    # A2a (j=2) * B2a (j=2, k=0..5): schoolbook
    col = _sb(n, m, p, U, V, W, 2, 0, 6, col)
    # A2b (j=3,4) * B2b (j=3,4, k=0..5): three Strassen blocks
    for b in range(3):
        col = _block(n, m, p, U, V, W, 3, b*2, col, sU, sV, sW)
    assert col == rank, f"col {col} != {rank}"
    return U, V, W

def _verify(U, V, W, n=2, m=5, p=6, tol=1e-9):
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(Ua.shape[1]):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compress(U, V, W, tol=1e-9):
    """Drop zero columns, then linearly dependent columns (sum of triple products = 0)."""
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    n, m, p = Ua.shape[0]//m, Ua.shape[0]//n if False else 5, 6
    n, m, p = 2, 5, 6
    R = Ua.shape[1]
    keep = list(range(R))
    # drop zero columns
    nz = [r for r in range(R) if (Ua[:,r].any() and Va[:,r].any() and Wa[:,r].any())]
    # greedily remove linearly dependent columns
    basis = []
    basis_coeffs = []
    for r in nz:
        t = Ua[:,r] * Va[:,r] * Wa[:,r]
        t = t - np.dot(basis_coeffs, basis)
        if np.linalg.norm(t) > tol:
            basis.append(t)
            basis_coeffs.append(t)
        # else dependent -> drop
    keep = [r for r in nz]  # placeholder
    # rebuild keep from basis membership
    keep = []
    basis = []
    basis_coeffs = []
    for r in nz:
        t = Ua[:,r] * Va[:,r] * Wa[:,r]
        t = t - np.dot(basis_coeffs, basis)
        if np.linalg.norm(t) > tol:
            basis.append(t)
            basis_coeffs.append(t)
            keep.append(r)
    if len(keep) < R:
        Ua = Ua[:, keep]; Va = Va[:, keep]; Wa = Wa[:, keep]
    return Ua.tolist(), Va.tolist(), Wa.tolist()

def construct():
    t0 = time.time()
    sU, sV, sW = _strassen_222()
    U, V, W = _build_block(2, 5, 6, sU, sV, sW)
    U, V, W = _compress(U, V, W)
    # Systematic term-elimination search: try removing a set of columns that can be
    # re-expressed by linear combination; keep best valid lower rank.
    best = (len(U[0]), U, V, W)
    # verify current
    if _verify(U, V, W):
        best = (len(U[0]), U, V, W)
    # Attempt a few targeted eliminations using linear dependence over the full set
    # (already done in _compress). Try a second pass with tighter tolerance.
    U2, V2, W2 = _compress(U, V, W, tol=1e-12)
    if _verify(U2, V2, W2):
        if len(U2[0]) < best[0]:
            best = (len(U2[0]), U2, V2, W2)
    # Time guard: return best found so far with margin.
    if time.time() - t0 > DEADLINE - MARGIN:
        pass
    return {"U": best[1], "V": best[2], "W": best[3]}
# EVOLVE-BLOCK-END
