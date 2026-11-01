"""Rewritten: structured decomposition for 2x4 by 4x8 with optimization."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_2x2():
    """Return U,V,W (each list of 4x7) for 2x2 by 2x2 with rank 7 (Strassen).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    # Strassen 7 products for (a b; c d)(e f; g h)
    # m1 = (a+b)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+d)(e+h)
    # m6 = (b-c)(g+h)
    # m7 = (a-c)(e+f)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m5
    # C22 = m1-m2+m3+m6
    U = np.zeros((4,7))
    V = np.zeros((4,7))
    W = np.zeros((4,7))
    # indices: A: (0,0)=a (0,1)=b (1,0)=c (1,1)=d -> U index i*2+j
    # B: (0,0)=e (0,1)=f (1,0)=g (1,1)=h -> V index j*2+k
    # C: (0,0) (0,1) (1,0) (1,1) -> W index k*2+i
    # m1 = (a+b)(e+h): U: a,b ; V: e,h
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[3,0]=1
    # m2 = (c+d)(e): U: c,d ; V: e
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    # m3 = a(f-h): U: a ; V: f, -h
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    # m4 = d(g-e): U: d ; V: g, -e
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    # m5 = (a+d)(e+h): U: a,d ; V: e,h
    U[0,4]=1; U[3,4]=1; V[0,4]=1; V[3,4]=1
    # m6 = (b-c)(g+h): U: b, -c ; V: g,h
    U[1,5]=1; U[2,5]=-1; V[2,5]=1; V[3,5]=1
    # m7 = (a-c)(e+f): U: a, -c ; V: e,f
    U[0,6]=1; U[2,6]=-1; V[0,6]=1; V[1,6]=1
    # W: C11 = m1+m4-m5+m7 -> W[0,0]
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    # C12 = m3+m5 -> W[1,0] (k=1,i=0 -> index 1*2+0=2? wait W index = k*2+i)
    # C12 is i=0,k=1 -> index 1*2+0 = 2
    W[2,2]=1; W[2,4]=1
    # C21 is i=1,k=0 -> index 0*2+1 = 1
    W[1,1]=1; W[1,4]=1
    # C22 is i=1,k=1 -> index 1*2+1 = 3
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    return U, V, W

def _build_structured():
    """Build rank-28 structured decomposition for 2x4 by 4x8.
    Split middle 4 into two 2's: A=[A1|A2] (2x2 each), B=[B1;B2] (2x4 each).
    C = A1*B1 + A2*B2. Each A_i*B_i is 2x2 by 2x4.
    Split B_i (2x4) into two 2x2: B_i = [B_i1 | B_i2].
    So A_i*B_i = A_i*B_i1 + A_i*B_i2, each 2x2 by 2x2 (rank 7).
    Total 4 blocks * 7 = 28.
    U: 2x4 = 8 rows, V: 4x8 = 32 rows, W: 8x2 = 16 rows, rank 28.
    """
    sU, sV, sW = _strassen_2x2_2x2()
    rank = 28
    U = np.zeros((8, rank))   # 2x4
    V = np.zeros((32, rank))  # 4x8
    W = np.zeros((16, rank))  # 8x2
    # A is 2x4: rows i in 0..1, cols j in 0..3. U index = i*4 + j.
    # B is 4x8: rows j in 0..3, cols k in 0..7. V index = j*8 + k.
    # C is 2x8: rows i in 0..1, cols k in 0..7. W index = k*2 + i.
    # Block (ii, kk): ii in 0..1 (which half of middle), kk in 0..1 (which half of right).
    # A_ii: rows i, cols 2*ii + a  (a in 0..1)
    # B_ii,kk: rows 2*ii + a, cols 2*kk + b  (a in 0..1, b in 0..1)
    # C_ii,kk: rows i, cols 2*kk + b
    for ii in range(2):
        for kk in range(2):
            base = (ii*2 + kk) * 7
            for c in range(7):
                # U: A_ii entries at (i, 2*ii + a)
                for (urow, uval) in enumerate(sU[:,c]):
                    if uval != 0:
                        i = urow // 2
                        a = urow % 2
                        U[i*4 + (2*ii + a), base + c] = uval
                # V: B_ii,kk entries at (2*ii + a, 2*kk + b)
                for (vrow, vval) in enumerate(sV[:,c]):
                    if vval != 0:
                        a = vrow // 2
                        b = vrow % 2
                        V[(2*ii + a)*8 + (2*kk + b), base + c] = vval
                # W: C_ii,kk entries at (i, 2*kk + b)
                for (wrow, wval) in enumerate(sW[:,c]):
                    if wval != 0:
                        k = wrow // 2
                        i = wrow % 2
                        W[(2*kk + b)*2 + i, base + c] = wval
    return U, V, W

def _verify(U, V, W):
    """Check that U * V^T * W^T reconstructs the multiplication tensor.
    Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] = 1 if we compute C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # U index: i*m + j ; V index: j*p + k ; W index: k*n + i
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Build T from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    # Expected: T[i,j,k] = 1
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _try_reduce_rank(U, V, W, deadline):
    """Attempt to reduce rank by finding linear dependencies among columns.
    Returns (U,V,W,rank) possibly reduced."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The decomposition is valid if for all i,j,k: sum_c U[i*m+j,c]V[j*p+k,c]W[k*n+i,c] = 1
    # If columns c are linearly dependent in a certain way, we can merge.
    # Simple approach: check if any column is a linear combination that can be removed.
    # This is complex; instead we try random linear combinations to find lower-rank equivalent.
    # For now, return as-is.
    return U, V, W, rank

def construct():
    start = time.time()
    deadline = start + 130  # safety margin
    U, V, W = _build_structured()
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n*m*p
        Us = [[0]*rank for _ in range(n*m)]
        Vs = [[0]*rank for _ in range(m*p)]
        Ws = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j][col]=1
                    Vs[j*p+k][col]=1
                    Ws[k*n+i][col]=1
                    col+=1
        return {"U": Us, "V": Vs, "W": Ws}
    rank = U.shape[1]
    # Try to reduce rank within time
    if time.time() < deadline:
        U, V, W, rank = _try_reduce_rank(U, V, W, deadline)
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
