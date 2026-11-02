import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2  # 6 blocks of 2 columns each

    U_strassen = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1},
    ]
    V_strassen = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
        {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1},
    ]
    W_strassen = [
        {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
        {0: -1, 1: 1}, {3: 1}, {0: 1},
    ]

    def lb2g(lb, t):
        return [0*p+2*t, 0*p+2*t+1, 1*p+2*t, 1*p+2*t+1][lb]
    def lc2g(lc, t):
        return [(2*t)*n+0, (2*t+1)*n+0, (2*t)*n+1, (2*t+1)*n+1][lc]

    rank = num_blocks * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    for t in range(num_blocks):
        for s in range(7):
            col = t*7+s
            for ai, v in U_strassen[s].items(): U[ai, col] = v
            for lb, v in V_strassen[s].items(): V[lb2g(lb,t), col] = v
            for lc, v in W_strassen[s].items(): W[lc2g(lc,t), col] += v

    # Try to reduce rank by finding linearly dependent terms
    terms = np.array([np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).flatten() for r in range(rank)])
    try:
        Q, R = np.linalg.qr(terms.T, mode='complete')
        diag_R = np.abs(np.diag(R))
        keep = diag_R > 1e-10
        num_keep = int(np.sum(keep))
        if num_keep < rank:
            keep_idx = np.where(keep)[0]
            U = U[:, keep_idx]
            V = V[:, keep_idx]
            W = W[:, keep_idx]
            rank = num_keep
    except:
        pass

    # Verify
    rng = np.random.default_rng(42)
    A_t = rng.standard_normal((n,m))
    B_t = rng.standard_normal((m,p))
    C_exp = A_t @ B_t
    C_comp = np.zeros((n,p))
    for r in range(rank):
        pa = np.sum(U[:,r].reshape(n,m) * A_t)
        pb = np.sum(V[:,r].reshape(m,p) * B_t)
        C_comp += pa*pb*W[:,r].reshape(p,n).T
    if not np.allclose(C_comp, C_exp, atol=1e-10):
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p+j*p+k
                    U[i*m+j,idx]=1; V[j*p+k,idx]=1; W[k*n+i,idx]=1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
