"""Matrix multiplication 3x4x6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U,V,W) each 4x7 for 2x2x2 Strassen, schoolbook index convention.
    U[i*2+j], V[j*2+k], W[k*2+i]."""
    import numpy as np
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # Strassen products for a=[[a,b],[c,d]], b=[[e,f],[g,h]]
    # M1=(a-d)(e+h)  M2=(a+b)e  M3=d(f-h)  M4=a(g+h)
    # M5=(a+d)(h-g)  M6=c(g+h)  M7=(b-c)(e+f)
    # p11=M5+M4-M2+M6 ; p12=M1+M2 ; p21=M3+M5 ; p22=M1-M4+M7+M2
    # U columns: which (i,j) of A participates; V: (j,k) of B; W: (k,i) of C
    def setc(col, us, vs, ws):
        for (i, j), v in us:
            U[i * 2 + j, col] += v
        for (j, k), v in vs:
            V[j * 2 + k, col] += v
        for (k, i), v in ws:
            W[k * 2 + i, col] += v
    # M1=(a-d)(e+h): A: (0,0)+1,(1,1)-1 ; B: (0,0)+1,(0,1)+1 ; C: (0,0)+1
    setc(0, [( (0,0),1),((1,1),-1)], [((0,0),1),((0,1),1)], [((0,0),1)])
    # M2=(a+b)e: A:(0,0)+1,(0,1)+1 ; B:(0,0)+1 ; C:(0,1)+1
    setc(1, [((0,0),1),((0,1),1)], [((0,0),1)], [((0,1),1)])
    # M3=d(f-h): A:(1,1)+1 ; B:(0,1)+1,(1,1)-1 ; C:(1,0)+1
    setc(2, [((1,1),1)], [((0,1),1),((1,1),-1)], [((1,0),1)])
    # M4=a(g+h): A:(0,0)+1 ; B:(1,0)+1,(1,1)+1 ; C:(1,1)+1
    setc(3, [((0,0),1)], [((1,0),1),((1,1),1)], [((1,1),1)])
    # M5=(a+d)(h-g): A:(0,0)+1,(1,1)+1 ; B:(0,1)-1,(1,1)+1 ; C:(0,0)+1
    setc(4, [((0,0),1),((1,1),1)], [((0,1),-1),((1,1),1)], [((0,0),1)])
    # M6=c(g+h): A:(1,0)+1 ; B:(1,0)+1,(1,1)+1 ; C:(0,1)+1
    setc(5, [((1,0),1)], [((1,0),1),((1,1),1)], [((0,1),1)])
    # M7=(b-c)(e+f): A:(0,1)+1,(1,0)-1 ; B:(0,0)+1,(0,1)+1 ; C:(1,1)+1
    setc(6, [((0,1),1),((1,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)])
    return U, V, W

def _verify_block(Ub, Vb, Wb, n, m, p):
    """Check that sum_c Ub[i*m+j,c]*Vb[j*p+k,c] equals Wb[k*n+i,c] structure."""
    import numpy as np
    U = np.array(Ub, dtype=float)
    V = np.array(Vb, dtype=float)
    W = np.array(Wb, dtype=float)
    # For each (i,j,k), the scalar tensor entry T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]
    # must equal sum_c W[k*n+i,c] ... but W is a column vector per c.
    # Actually decomposition: T = sum_c U_c (outer) V_c (outer) W_c where
    # U_c[i*m+j]=U[i*m+j,c], V_c[j*p+k]=V[j*p+k,c], W_c[k*n+i]=W[k*n+i,c].
    # So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
    # We verify by random A,B.
    np.random.seed(0)
    A = np.random.randint(-3, 4, size=(n, m))
    B = np.random.randint(-3, 4, size=(m, p))
    C = A @ B
    C2 = np.zeros((n, p))
    for c in range(U.shape[1]):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        C2 += np.trace(uc @ vc @ wc.T)  # W[k*n+i] -> wc[k,i]; (uc@vc)[i,k]; need [i,k]*wc[k,i]
    return np.array_equal(C, C2)

def _build_block(n, m, p):
    """Build a block decomposition for dims n,m,p. Use Strassen if all are 2."""
    import numpy as np
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_222()
        assert _verify_block(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        return U, V, W
    # schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    import numpy as np
    n, m, p = DIMENSIONS
    # Splits: n=3 -> [1,2]; m=4 -> [2,2]; p=6 -> [2,2,2]
    ns = [1, 2]
    ms = [2, 2]
    ps = [2, 2, 2]
    # Accumulate global matrices
    U_parts = []
    V_parts = []
    W_parts = []
    # Track row offsets for each factor
    for a in range(len(ns)):
        for b in range(len(ms)):
            for c in range(len(ps)):
                ni, mi, pi = ns[a], ms[b], ps[c]
                Ub, Vb, Wb = _build_block(ni, mi, pi)
                # U rows: global row = (i_in_block + n_offset) * m + (j_in_block + m_offset)
                n_off = sum(ns[:a])
                m_off = sum(ms[:b])
                p_off = sum(ps[:c])
                Urow = np.zeros((n * m, Ub.shape[1]), dtype=int)
                for li in range(ni):
                    for lj in range(mi):
                        Urow[(n_off + li) * m + (m_off + lj), :] = Ub[li * mi + lj, :]
                Vrow = np.zeros((m * p, Vb.shape[1]), dtype=int)
                for lj in range(mi):
                    for lk in range(pi):
                        Vrow[(m_off + lj) * p + (p_off + lk), :] = Vb[lj * pi + lk, :]
                Wrow = np.zeros((p * n, Wb.shape[1]), dtype=int)
                for lk in range(pi):
                    for li in range(ni):
                        Wrow[(p_off + lk) * n + (n_off + li), :] = Wb[lk * ni + li, :]
                U_parts.append(Urow)
                V_parts.append(Vrow)
                W_parts.append(Wrow)
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    # Final verification
    np.random.seed(1)
    A = np.random.randint(-3, 4, size=(n, m))
    B = np.random.randint(-3, 4, size=(m, p))
    C = A @ B
    C2 = np.zeros((n, p))
    for col in range(U.shape[1]):
        uc = U[:, col].reshape(n, m)
        vc = V[:, col].reshape(m, p)
        wc = W[:, col].reshape(p, n)
        C2 += np.trace(uc @ vc @ wc.T)
    assert np.array_equal(C, C2), "decomposition failed"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
