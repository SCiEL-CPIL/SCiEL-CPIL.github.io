"""Matrix multiplication 4x5x6 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 5, 6]
_START = time.time()
_DEADLINE = 145

def _time_check():
    return (time.time() - _START) < _DEADLINE

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0

    def add_term(Ue, Ve, We):
        nonlocal rank
        col = rank
        for (r, v) in Ue:
            if v != 0: U_dict[(r, col)] = U_dict.get((r, col), 0) + v
        for (r, v) in Ve:
            if v != 0: V_dict[(r, col)] = V_dict.get((r, col), 0) + v
        for (r, v) in We:
            if v != 0: W_dict[(r, col)] = W_dict.get((r, col), 0) + v
        rank += 1

    u_idx = lambda i, j: i * m + j
    v_idx = lambda j, k: j * p + k
    w_idx = lambda k, i: k * n + i

    def strassen222(ar, ac, br, bc, cr, cc):
        # ar, br: lists of 2 row indices; ac, bc: lists of 2 col indices
        # cr, cc: C row/col indices (lists of 2)
        a00, a01 = u_idx(ar[0],ac[0]), u_idx(ar[0],ac[1])
        a10, a11 = u_idx(ar[1],ac[0]), u_idx(ar[1],ac[1])
        b00, b01 = v_idx(br[0],bc[0]), v_idx(br[0],bc[1])
        b10, b11 = v_idx(br[1],bc[0]), v_idx(br[1],bc[1])
        c00, c01 = w_idx(cc[0],cr[0]), w_idx(cc[1],cr[0])
        c10, c11 = w_idx(cc[0],cr[1]), w_idx(cc[1],cr[1])
        
        # M1 = (A00+A11)(B00+B11) -> C00+=, C11+=
        add_term([(a00,1),(a11,1)], [(b00,1),(b11,1)], [(c00,1),(c11,1)])
        # M2 = (A10+A11)B00 -> C10+=, C00+=
        add_term([(a10,1),(a11,1)], [(b00,1)], [(c10,1),(c00,1)])
        # M3 = A00(B11-B01) -> C11+=, C01-=
        add_term([(a00,1)], [(b11,1),(b01,-1)], [(c11,1),(c01,-1)])
        # M4 = A11(B00-B10) -> C00+=, C10-=
        add_term([(a11,1)], [(b00,1),(b10,-1)], [(c00,1),(c10,-1)])
        # M5 = (A00+A01)B11 -> C01+=, C11+=
        add_term([(a00,1),(a01,1)], [(b11,1)], [(c01,1),(c11,1)])
        # M6 = (A10-A00)(B00+B01) -> C10+=, C00+=
        add_term([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(c10,1),(c00,1)])
        # M7 = (A01-A11)(B10+B11) -> C01+=, C11+=
        add_term([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(c01,1),(c11,1)])

    def direct(ar, ac, br, bc, cr, cc):
        for i in range(len(cr)):
            for k in range(len(cc)):
                for t in range(len(ac)):
                    add_term([(u_idx(ar[i],ac[t]),1)], [(v_idx(br[t],bc[k]),1)], [(w_idx(bc[k],ar[i]),1)])

    for Ar in [[0,1],[2,3]]:
        for Bc in [[0,1],[2,3],[4,5]]:
            cr, cc = Ar, Bc
            if _time_check(): strassen222(Ar,[0,1],[0,1],Bc,cr,cc)
            if _time_check(): direct(Ar,[2],[2],Bc,cr,cc)
            if _time_check(): strassen222(Ar,[3,4],[3,4],Bc,cr,cc)

    U_rows, V_rows, W_rows = n*m, m*p, p*n
    U = [[0]*rank for _ in range(U_rows)]
    V = [[0]*rank for _ in range(V_rows)]
    W = [[0]*rank for _ in range(W_rows)]
    for (r,c),v in U_dict.items(): U[r][c] = v
    for (r,c),v in V_dict.items(): V[r][c] = v
    for (r,c),v in W_dict.items(): W[r][c] = v
    return {"U": U, "V": V, "W": W}
