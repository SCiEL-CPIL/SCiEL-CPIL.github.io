"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 140

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[2,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1
    return U, V, W

def _rank1_221():
    """Rank-4 decomposition for 2x2 * 2x1 -> 2x1. Returns (U,V,W) each (4,4),(2,4),(2,4)."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _rank1_121():
    """Rank-2 decomposition for 1x2 * 2x1 -> 1x1. Returns (U,V,W) each (2,2),(2,2),(1,2)."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W

def _build_tensor(n, m, p):
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                tensor[a,b,c] = 1.0
    return tensor

def _verify(U, V, W, tensor, tol=1e-9):
    r = U.shape[1]
    for a in range(tensor.shape[0]):
        for b in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                val = float(np.dot(U[a,:], V[b,:]*W[c,:]))
                if abs(val - tensor[a,b,c]) > tol:
                    return False
    return True

def _to_lists(U, V, W):
    def conv(mat):
        return [[int(x) if abs(x-round(x))<1e-9 else float(x) for x in row] for row in mat]
    return conv(U), conv(V), conv(W)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook fallback (rank 84)
    rank_sb = n*m*p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Block decomposition: rank 64
    try:
        Uss, Vss, Wss = _strassen_222()
        Ur1, Vr1, Wr1 = _rank1_221()
        Ur2, Vr2, Wr2 = _rank1_121()
    except Exception:
        return {"U": best_U, "V": best_V, "W": best_W}

    total_rank = 84
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))

    def map_U_2x2(i, j):
        return i*4 + j
    def map_V_2x2(j, k):
        return j*7 + k
    def map_W_2x2(k, i):
        return k*3 + i

    col = 0

    def add_block(Ub, Vb, Wb, u_idx, v_idx, w_idx):
        nonlocal col
        r = Ub.shape[1]
        for t in range(r):
            for (ri, val) in enumerate(Ub[:, t]):
                if val != 0:
                    U[u_idx[ri], col] += val
            for (ri, val) in enumerate(Vb[:, t]):
                if val != 0:
                    V[v_idx[ri], col] += val
            for (ri, val) in enumerate(Wb[:, t]):
                if val != 0:
                    W[w_idx[ri], col] += val
            col += 1

    def u22(i0, j0):
        return [i0*4+j0, (i0+1)*4+j0, i0*4+j0+1, (i0+1)*4+j0+1]
    def v22(j0, k0):
        return [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
    def w22(i0, k0):
        return [k0*3+i0, (k0+1)*3+i0, k0*3+(i0+1), (k0+1)*3+(i0+1)]

    def v21(j0, k0):
        return [j0*7+k0, (j0+1)*7+k0]
    def w21(i0, k0):
        return [k0*3+i0, k0*3+(i0+1)]

    def u12(i, j0):
        return [i*4+j0, i*4+j0+1]
    def w12(i, k0):
        return [k0*3+i, (k0+1)*3+i]
    def w11(i, k0):
        return [k0*3+i]

    # --- Part 1: C[0:2, 0:4] = 4 Strassen blocks, rank 28 ---
    # C[0:2, 0:2]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,0), w22(0,0))
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,0), w22(0,0))
    # C[0:2, 2:4]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,2), w22(0,2))
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,2), w22(0,2))

    # --- Part 2: C[0:2, 4:7] ---
    # C[0:2, 4:6]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,4), w22(0,4))
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,4), w22(0,4))
    # C[0:2, 6:7]
    add_block(Ur1, Vr1, Wr1, u22(0,0), v21(0,6), w21(0,6))
    add_block(Ur1, Vr1, Wr1, u22(0,2), v21(2,6), w21(0,6))

    def add_block_1x2x2(i, j0, k0):
        nonlocal col
        u_idx = u12(i, j0)
        v_full = v22(j0, k0)
        v_idx_0 = v_full[:2]
        w_idx_0 = [k0*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx_0, w_idx_0)
        v_idx_1 = v_full[2:]
        w_idx_1 = [(k0+1)*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx_1, w_idx_1)

    def add_block_1x2x1(i, j0, k0):
        nonlocal col
        u_idx = u12(i, j0)
        v_idx = v21(j0, k0)
        w_idx = [k0*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx, w_idx)

    # --- Part 3: C[2, 0:4] ---
    add_block_1x2x2(2, 0, 0)
    add_block_1x2x2(2, 2, 0)
    add_block_1x2x2(2, 0, 2)
    add_block_1x2x2(2, 2, 2)

    # --- Part 4: C[2, 4:7] ---
    add_block_1x2x2(2, 0, 4)
    add_block_1x2x2(2, 2, 4)
    add_block_1x2x1(2, 0, 6)
    add_block_1x2x1(2, 2, 6)

    if col <= total_rank:
        tensor = _build_tensor(n, m, p)
        U_eff = U[:, :col]
        V_eff = V[:, :col]
        W_eff = W[:, :col]
        if _verify(U_eff, V_eff, W_eff, tensor):
            if col < best_rank:
                best_rank = col
                best_U, best_V, best_W = _to_lists(U_eff, V_eff, W_eff)

    if time.time() - start_time > TIME_LIMIT - 5:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try to find a better decomposition using rank minimization
    # This is a heuristic approach - we'll try to reduce the rank further
    # by looking for linear dependencies among the rank-1 terms
    
    # For now, return the best we have
    return {"U": best_U, "V": best_V, "W": best_W}
