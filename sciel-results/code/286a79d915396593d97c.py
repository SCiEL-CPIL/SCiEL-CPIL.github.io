import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _try_rank30():
    n, m, p = DIMENSIONS
    # Build rank-30 via tensor product factorization and validate.
    # M_{2,2,5} = M_{2,2,1} x M_{1,1,5}: rank 2*5=10 per block, 2 blocks => 30.
    U = []
    V = []
    W = []
    # For each block b in {0,1}, local l in {0,1}, j = 2b + l
    # M_{2,2,1} rank-2: terms for (i,l) identity-like
    # C[i,k] = sum_l A[i,l] B[l,k]; rank-1 terms: for each (i,l) pair? That's 4 per block = 8 total, too many.
    # Use tensor product: M_{2,2,1} has rank 2 with:
    #   term1: u=[1,0], v=[1,0], w=[1] -> A[0,0]*B[0,k]  (but w must be length p=5)
    # This is getting too uncertain; abort rank-30 attempt.
    return None

def _validate(res):
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    rank = len(U[0])
    # Reconstruct: C[i,k] = sum_c U[i*m+? ... ]
    # Convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank), W[k*n+i, r]
    # C[i,k] = sum_r U[i*m + j ... ] -- need the decomposition form:
    # C[i,k] = sum_{r} W[k*n + i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])? 
    # Standard: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    A = np.random.RandomState(0).rand(n, m)
    B = np.random.RandomState(1).rand(m, p)
    C_true = A @ B
    C_rec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                for j in range(m):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            C_rec[i, k] = s
    return np.allclose(C_rec, C_true, atol=1e-6)

def construct():
    start = time.time()
    deadline = start + 110.0
    r30 = _try_rank30()
    if r30 is not None and time.time() < deadline and _validate(r30):
        return r30
    return _schoolbook()
