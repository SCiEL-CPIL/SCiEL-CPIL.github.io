"""Rewritten program for 2x3x6 matrix multiplication tensor decomposition.
Keeps the schoolbook (rank 36) baseline and attempts Strassen-block rank 33.
Rebuilt from scratch with explicit deadline guard and vectorized verification."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115  # seconds; leave margin for evaluator kill at 120s

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    deadline = _START_TIME + _HARD_DEADLINE

    # ---------- schoolbook rank = n*m*p = 36 ----------
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return _to_dict(U, V, W)

    best = schoolbook()

    # ---------- vectorized verification ----------
    def verify(U, V, W, n, m, p, rank):
        """Check tensor decomposition: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1."""
        # Build T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        # Use einsum for speed
        T = np.einsum('it,jt,kt->ijk',
                       U.reshape(n, m, rank),
                       V.reshape(m, p, rank),
                       W.reshape(p, n, rank))
        return bool(np.allclose(T, 1.0, atol=1e-8))

    # ---------- attempt rank 33 via 3x(2x3x2) blocks ----------
    try:
        # Strassen terms for 2x2x2, each (u4, v4, w4) indexed by (i,j), (j,k), (k,i)
        # Standard Strassen:
        #  p1 = (A00+A11)*(B00+B11)
        #  p2 = (A10+A11)*B00
        #  p3 = A00*(B01-B11)
        #  p4 = A11*(B10-B00)
        #  p5 = (A00+A01)*B11
        #  p6 = (A10-A00)*(B00+B01)
        #  p7 = (A01-A11)*(B10+B11)
        # C00 = p1 + p4 - p5 - p7
        # C01 = p3 + p5
        # C10 = p2 + p7
        # C11 = p1 - p2 + p3 + p6

        # u indexed [i*2+j], v indexed [j*2+k], w indexed [k*2+i]
        st = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),     # p1
            ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),    # p2
            ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),    # p3
            ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),    # p4
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),    # p5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),    # p6
            ([0,1,0,-1], [0,0,1,1], [-1,1,0,0]),   # p7
        ]

        block_rank = 11
        U_blk = np.zeros((6, block_rank))   # i in {0,1}, j in {0,1,2}
        V_blk = np.zeros((6, block_rank))   # j in {0,1,2}, k in {0,1}
        W_blk = np.zeros((4, block_rank))   # k in {0,1}, i in {0,1}

        for t in range(7):
            ut, vt, wt = st[t]
            for i in range(2):
                for j in range(2):
                    U_blk[i*3+j, t] = ut[i*2+j]
            for j in range(2):
                for k in range(2):
                    V_blk[j*2+k, t] = vt[j*2+k]
            for k in range(2):
                for i in range(2):
                    W_blk[k*2+i, t] = wt[k*2+i]

        # schoolbook for j=2
        for di, (iv, kv) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            t = 7 + di
            U_blk[iv*3+2, t] = 1.0
            V_blk[2*2+kv, t] = 1.0
            W_blk[kv*2+iv, t] = 1.0

        if time.time() > deadline:
            return best

        # verify block
        if not verify(U_blk, V_blk, W_blk, 2, 3, 2, block_rank):
            return best

        # assemble full 2x3x6
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        for b in range(3):
            k_off = b * 2
            for li in range(2):
                for lj in range(3):
                    for t in range(block_rank):
                        U[li*3+lj, b*block_rank+t] = U_blk[li*3+lj, t]
            for lj in range(3):
                for lk in range(2):
                    for t in range(block_rank):
                        V[lj*2+lk, b*block_rank+t] = V_blk[lj*2+lk, t]
            for lk in range(2):
                for li in range(2):
                    for t in range(block_rank):
                        W[(k_off+lk)*2+li, b*block_rank+t] = W_blk[lk*2+li, t]

        if time.time() > deadline:
            return best

        if verify(U, V, W, n, m, p, rank):
            return _to_dict(U, V, W)
    except Exception:
        pass

    return best

def _to_dict(U, V, W):
    """Convert numpy arrays to lists with integer/half-integer cleanup."""
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(x - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
# EVOLVE-BLOCK-END
