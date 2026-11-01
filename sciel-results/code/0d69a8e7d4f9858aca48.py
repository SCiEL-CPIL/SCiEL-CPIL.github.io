"""Program for 2x3x7 matrix multiplication tensor decomposition."""
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate the current constructive decomposition (Rank 39)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(u_vec, v_vec, w_vec):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for idx, coeff in u_vec:
            U_col[idx] = coeff
        for idx, coeff in v_vec:
            V_col[idx] = coeff
        for idx, coeff in w_vec:
            W_col[idx] = coeff
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Part 1: A2 * B2 (2x1 x 7) -> Rank 14
    for i in range(n):
        for k in range(p):
            add_decomp([(u_idx(i, 2), 1.0)], [(v_idx(2, k), 1.0)], [(w_idx(k, i), 1.0)])

    # Part 2: A1 * B1 (2x2 x 7)
    def strassen_decomp(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        # M1
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        # M2
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1), (w_idx(b_col_offset+1, a_row_offset+1), -1)]
        add_decomp(u, v, w)
        # M3
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+1), 1), (v_idx(b_row_offset+1, b_col_offset+1), -1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+0), 1), (w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        # M4
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+0), -1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1), (w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        # M5
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), -1), (w_idx(b_col_offset+1, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        # M6
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1), (u_idx(a_row_offset+0, a_col_offset+0), -1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1), (v_idx(b_row_offset+0, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+1, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        # M7
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1), (u_idx(a_row_offset+1, a_col_offset+1), -1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1), (v_idx(b_row_offset+1, b_col_offset+1), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)

    def schoolbook_221(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        u = [(u_idx(a_row_offset+0, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        u = [(u_idx(a_row_offset+0, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+0), 1)]
        add_decomp(u, v, w)
        u = [(u_idx(a_row_offset+1, a_col_offset+0), 1)]
        v = [(v_idx(b_row_offset+0, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)
        u = [(u_idx(a_row_offset+1, a_col_offset+1), 1)]
        v = [(v_idx(b_row_offset+1, b_col_offset+0), 1)]
        w = [(w_idx(b_col_offset+0, a_row_offset+1), 1)]
        add_decomp(u, v, w)

    strassen_decomp(0, 0, 0, 0)
    strassen_decomp(0, 0, 0, 2)
    strassen_decomp(0, 0, 0, 4)
    schoolbook_221(0, 0, 0, 6)

    U_init = np.array(U_cols).T
    V_init = np.array(V_cols).T
    W_init = np.array(W_cols).T
    R_init = U_init.shape[1]
    
    # 2. Attempt to reduce rank using numerical optimization
    # Target: Minimize residual of sum_r U_{im,r} V_{jk,r} W_{ki,r} - 1 for all i,j,k
    
    def residuals(params, R):
        U = params[:R * n * m].reshape(R, n * m).T
        V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
        W = params[R * (n * m + m * p):].reshape(R, p * n).T
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res.append(val - 1.0)
        return np.array(res)

    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = R_init
    best_res = np.inf

    # Try ranks from R_init-1 down to max(1, R_init-3)
    for R in range(R_init - 1, max(0, R_init - 4), -1):
        # Initialize with first R columns
        x0 = np.concatenate([
            U_init[:, :R].flatten(),
            V_init[:, :R].flatten(),
            W_init[:, :R].flatten()
        ])
        
        try:
            result = opt.least_squares(residuals, x0, args=(R,), method='trf', max_nfev=2000)
            res_val = result.cost
            if res_val < best_res:
                best_res = res_val
                best_rank = R
                # Extract best
                best_U = result.x[:R * n * m].reshape(n * m, R).T
                best_V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R).T
                best_W = result.x[R * (n * m + m * p):].reshape(p * n, R).T
                # Check if solution is effectively exact
                if res_val < 1e-8:
                    break
        except Exception:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
