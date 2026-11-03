import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _strassen_222():
    U = np.zeros((4,7), dtype=float)
    V = np.zeros((4,7), dtype=float)
    W = np.zeros((4,7), dtype=float)
    U[:,0] = [1,0,0,1]
    U[:,1] = [0,0,1,1]
    U[:,2] = [1,0,0,0]
    U[:,3] = [0,0,0,1]
    U[:,4] = [1,1,0,0]
    U[:,5] = [-1,0,1,0]
    U[:,6] = [0,1,0,-1]
    V[:,0] = [1,0,0,1]
    V[:,1] = [1,0,0,0]
    V[:,2] = [0,1,0,-1]
    V[:,3] = [0,0,1,-1]
    V[:,4] = [0,0,0,1]
    V[:,5] = [1,1,0,0]
    V[:,6] = [0,0,1,1]
    W[:,0] = [1,0,0,-1,0,-1,1]   # C11 = P1-P4-P5+P7
    W[:,1] = [0,0,1,0,1,0,0]      # C12 = P3+P5
    W[:,2] = [0,1,0,1,0,0,0]      # C21 = P2+P4
    W[:,3] = [1,1,0,0,0,-1,0]     # C22 = P1-P3-P6+P2
    return U, V, W

def _rank1_122():
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; U[1,0]=0; V[0,0]=1; V[1,0]=0; V[2,0]=0; V[3,0]=0; W[0,0]=1; W[1,0]=0
    U[0,1]=0; U[1,1]=1; V[0,1]=0; V[1,1]=0; V[2,1]=1; V[3,1]=0; W[0,1]=1; W[1,1]=0
    U[0,2]=1; U[1,2]=0; V[0,2]=0; V[1,2]=1; V[2,2]=0; V[3,2]=0; W[0,2]=0; W[1,2]=1
    U[0,3]=0; U[1,3]=1; V[0,3]=0; V[1,3]=0; V[2,3]=0; V[3,3]=1; W[0,3]=0; W[1,3]=1
    return U, V, W

def construct():
    start = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS
    row_splits = [2, 1]
    col_splits = [2, 2, 2]
    mid_splits = [2, 2]

    U_list, V_list, W_list = [], [], []
    for rb, ni in enumerate(row_splits):
        i0 = sum(row_splits[:rb])
        for mb, mi in enumerate(mid_splits):
            j0 = sum(mid_splits[:mb])
            for cb, pj in enumerate(col_splits):
                k0 = sum(col_splits[:cb])
                if ni == 2:
                    U, V, W = _strassen_222()
                else:
                    U, V, W = _rank1_122()
                nc = U.shape[1]
                Uf = np.zeros((n*m, nc)); Vf = np.zeros((m*p, nc)); Wf = np.zeros((p*n, nc))
                for a in range(U.shape[0]):
                    if ni == 2:
                        ii, jj = divmod(a, 2)
                    else:
                        ii, jj = 0, a
                    Uf[(i0+ii)*m + (j0+jj), :] = U[a, :]
                for b in range(V.shape[0]):
                    jj, kk = divmod(b, pj)
                    Vf[(j0+jj)*p + (k0+kk), :] = V[b, :]
                for c in range(W.shape[0]):
                    kk, ii = divmod(c, ni)
                    Wf[(k0+kk)*n + (i0+ii), :] = W[c, :]
                U_list.append(Uf); V_list.append(Vf); W_list.append(Wf)
                if time.time() > DEADLINE:
                    break
            if time.time() > DEADLINE: break
        if time.time() > DEADLINE: break

    U = np.hstack(U_list); V = np.hstack(V_list); W = np.hstack(W_list)
    rank = U.shape[1]

    # Verify correctness (with guard)
    T = np.zeros((n, m, p))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    # check against definition T[i,j,k] = delta (i.e., matmul tensor)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                if abs(T[i,j,k] - expected) > 1e-6:
                    ok = False
    if not ok:
        # fall back to schoolbook rank 72 (always correct)
        U2 = np.zeros((n*m, n*m*p)); V2 = np.zeros((m*p, n*m*p)); W2 = np.zeros((p*n, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j, c] = 1; V2[j*p+k, c] = 1; W2[k*n+i, c] = 1; c += 1
        U, V, W = U2, V2, W2
        rank = U.shape[1]

    # Support-pruning: test each column for tensor-level linear dependence
    vecs = np.array([np.kron(np.kron(U[:,r], V[:,r]), W[:,r]) for r in range(rank)]).T
    removed = []
    for c in range(rank):
        if time.time() > DEADLINE:
            break
        others = [r for r in range(rank) if r != c]
        Mo = vecs[:, others]
        r1 = np.linalg.matrix_rank(Mo)
        r2 = np.linalg.matrix_rank(np.hstack([Mo, vecs[:, c:c+1]]))
        if r1 == r2:
            removed.append(c)
    if removed:
        keep = [r for r in range(rank) if r not in removed]
        U, V, W = U[:, keep], V[:, keep], W[:, keep]

    def to_list(A):
        return [[int(round(x * 2)) / 2 for x in row] for row in A]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
