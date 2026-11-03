"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 2, 4, 8
    dim_a, dim_b, dim_c = n * m, m * p, n * p  # 8, 32, 16

    # Build non-zero entries of the tensor
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz.append((i * m + j, j * p + k, k * n + i))
    nz_arr = np.array(nz)

    # Precompute flattened target matrices for ALS
    T_flat = np.zeros((dim_a, dim_b * dim_c))
    T_flat2 = np.zeros((dim_b, dim_a * dim_c))
    T_flat3 = np.zeros((dim_c, dim_a * dim_b))
    for a, b, c in nz:
        T_flat[a, b * dim_c + c] = 1.0
        T_flat2[b, a * dim_c + c] = 1.0
        T_flat3[c, a * dim_b + b] = 1.0

    def check_exact(U, V, W):
        rec = np.einsum('at,bt,ct->abc', U, V, W)
        T = np.zeros((dim_a, dim_b, dim_c))
        for a, b, c in nz:
            T[a, b, c] = 1.0
        return np.max(np.abs(rec - T)) < 1e-9

    # Build Strassen rank-56 solution as fallback
    def strassen_56():
        U = np.zeros((dim_a, 56))
        V = np.zeros((dim_b, 56))
        W = np.zeros((dim_c, 56))
        # 8 independent 2x2 x 2x2 Strassen blocks
        # A split: [A11 A12; A21 A22] where each is 2x2
        # B split: rows 0-1 and 2-3, cols 0-3 and 4-7
        # For each (row_block, col_block) pair, apply Strassen
        col = 0
        for rb in range(2):  # B row block: 0 or 1
            for cb in range(2):  # B col block: 0 or 1
                # A indices: i in 0..1, j = rb*2 .. rb*2+1
                # B indices: k = cb*2 .. cb*2+1, l = (2-cb)*4+cb*4 ... 
                # Actually: B is 4x8, split rows into [0,1] and [2,3], cols into [0..3] and [4..7]
                # A is 2x4, split cols into [0,1] and [2,3]
                # C is 2x8, split cols into [0..3] and [4..7]
                
                # For block (rb, cb):
                # A part: columns rb*2 to rb*2+1 (2x2)
                # B part: rows cb*2 to cb*2+1, columns cb*4 to cb*4+3 (2x4)
                # C part: rows 0..1, columns cb*4 to cb*4+3 (2x4)
                
                # Split B further into two 2x2 blocks: cols cb*4..cb*4+1 and cb*4+2..cb*4+3
                for sub_cb in range(2):
                    # B sub-block: rows cb*2..cb*2+1, cols cb*4+sub_cb*2..cb*4+sub_cb*2+1
                    # C sub-block: rows 0..1, cols cb*4+sub_cb*2..cb*4+sub_cb*2+1
                    a_base = rb * 2  # A col offset
                    b_row_base = cb * 2  # B row offset
                    b_col_base = cb * 4 + sub_cb * 2  # B col offset
                    c_col_base = cb * 4 + sub_cb * 2  # C col offset
                    
                    # Strassen for 2x2 x 2x2:
                    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
                    # M1 = (a00+a11)(b00+b11)
                    # M2 = (a10+a11)(b00)
                    # M3 = a00(b11-b10)
                    # M4 = a11(b00-b01)
                    # M5 = (a00+a01)(b10)
                    # M6 = (a10-a00)(b01+b11)
                    # M7 = (a01-a11)(b00+b10)
                    # C00 = M1+M4-M5+M7
                    # C01 = M3+M5
                    # C10 = M2+M6
                    # C11 = M1-M2+M3+M6
                    
                    # A indices: (i, a_base+j) for i,j in 0..1
                    # B indices: (b_row_base+i, b_col_base+j) for i,j in 0..1
                    # C indices: (i, c_col_base+j) for i,j in 0..1
                    
                    a_idx = [[i * m + (a_base + j) for j in range(2)] for i in range(2)]
                    b_idx = [[(b_row_base + i) * p + (b_col_base + j) for j in range(2)] for i in range(2)]
                    c_idx = [[i * p + (c_col_base + j) for j in range(2)] for i in range(2)]
                    
                    # 7 Strassen products
                    # M1 = (a00+a11)(b00+b11)
                    U[a_idx[0][0], col] = 1; U[a_idx[1][1], col] = 1
                    V[b_idx[0][0], col] = 1; V[b_idx[1][1], col] = 1
                    W[c_idx[0][0], col] = 1; W[c_idx[1][1], col] = 1
                    W[c_idx[0][1], col] += -1; W[c_idx[1][0], col] += -1
                    col += 1
                    # M2 = (a10+a11)(b00)
                    U[a_idx[1][0], col] = 1; U[a_idx[1][1], col] = 1
                    V[b_idx[0][0], col] = 1
                    W[c_idx[1][0], col] = 1; W[c_idx[1][1], col] += -1
                    col += 1
                    # M3 = a00(b11-b10)
                    U[a_idx[0][0], col] = 1
                    V[b_idx[1][1], col] = 1; V[b_idx[1][0], col] = -1
                    W[c_idx[0][1], col] = 1; W[c_idx[1][1], col] = 1
                    col += 1
                    # M4 = a11(b00-b01)
                    U[a_idx[1][1], col] = 1
                    V[b_idx[0][0], col] = 1; V[b_idx[0][1], col] = -1
                    W[c_idx[0][0], col] = 1; W[c_idx[1][1], col] = 1
                    col += 1
                    # M5 = (a00+a01)(b10)
                    U[a_idx[0][0], col] = 1; U[a_idx[0][1], col] = 1
                    V[b_idx[1][0], col] = 1
                    W[c_idx[0][0], col] = -1; W[c_idx[1][0], col] = 1
                    col += 1
                    # M6 = (a10-a00)(b01+b11)
                    U[a_idx[1][0], col] = 1; U[a_idx[0][0], col] = -1
                    V[b_idx[0][1], col] = 1; V[b_idx[1][1], col] = 1
                    W[c_idx[1][0], col] = 1; W[c_idx[1][1], col] = 1
                    col += 1
                    # M7 = (a01-a11)(b00+b10)
                    U[a_idx[0][1], col] = 1; U[a_idx[1][1], col] = -1
                    V[b_idx[0][0], col] = 1; V[b_idx[1][0], col] = 1
                    W[c_idx[0][0], col] = 1
                    col += 1
        
        return U, V, W

    U56, V56, W56 = strassen_56()
    if check_exact(U56, V56, W56):
        best_rank = 56
        U_best, V_best, W_best = U56, V56, W56
    else:
        # Fall back to schoolbook
        best_rank = 64
        U_best = np.zeros((dim_a, 64))
        V_best = np.zeros((dim_b, 64))
        W_best = np.zeros((dim_c, 64))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_best[i * m + j, col] = 1
                    V_best[j * p + k, col] = 1
                    W_best[k * n + i, col] = 1

    # Try to find lower rank via ALS
    for rank in range(55, 44, -1):
        if time.time() > deadline - 40:
            break
        for seed in range(4):
            if time.time() > deadline - 35:
                break
            rng = np.random.RandomState(seed * 1000 + rank * 7)
            U = rng.randn(dim_a, rank)
            V = rng.randn(dim_b, rank)
            W = rng.randn(dim_c, rank)

            for it in range(40):
                if time.time() > deadline - 30:
                    break
                # Solve for U
                M = np.einsum('bt,ct->btc', V, W).reshape(rank, dim_b * dim_c)
                U = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
                # Solve for V
                M2 = np.einsum('at,ct->atc', U, W).reshape(rank, dim_a * dim_c)
                V = np.linalg.lstsq(M2.T, T_flat2.T, rcond=None)[0].T
                # Solve for W
                M3 = np.einsum('at,bt->abt', U, V).reshape(rank, dim_a * dim_b)
                W = np.linalg.lstsq(M3.T, T_flat3.T, rcond=None)[0].T

                if it % 8 == 7:
                    if check_exact(U, V, W):
                        best_rank = rank
                        U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                        break
            if best_rank == rank:
                break
        if best_rank == rank:
            break

    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        lst = []
        for row in mat:
            lst.append([round(x * 2) / 2.0 for x in row])
        return lst

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
