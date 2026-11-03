import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 140
    n, m, p = 2, 3, 8

    def verify(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def build_2x2x4_rank13():
        """Rank-13 decomposition of M_{2,2,4}."""
        rank = 13
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]

        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[5][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[4][1] = 1

        U[0][2] = 1
        V[1][2] = 1; V[5][2] = 1
        W[1][2] = 1; W[5][2] = 1

        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[4][3] = 1
        W[0][3] = 1; W[4][3] = 1

        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[4][4] = -1
        W[0][4] = -1; W[1][4] = 1

        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[5][5] = -1
        W[5][5] = 1

        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[5][6] = 1
        W[0][6] = 1

        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[6][7] = 1
        W[4][7] = 1; W[7][7] = 1

        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[6][8] = 1

        U[0][9] = 1
        V[3][9] = 1; V[7][9] = 1
        W[5][9] = 1; W[7][9] = 1

        U[0][10] = 1; U[2][10] = -1
        V[3][10] = 1; V[6][10] = 1
        W[4][10] = 1; W[6][10] = 1

        U[1][11] = 1; U[2][11] = 1
        V[2][11] = 1; V[6][11] = -1
        W[4][11] = -1; W[5][11] = 1

        U[1][12] = -1; U[3][12] = 1
        V[3][12] = 1; V[7][12] = -1
        W[7][12] = 1

        return U, V, W

    def build_1plus2_split():
        """Split m=3 as 1+2, use rank-13 M_{2,2,4} for the 2x2 part."""
        n, m, p = 2, 3, 8
        rank1 = 16
        rank2 = 26
        total_rank = rank1 + rank2

        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        col = 0
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        U224, V224, W224 = build_2x2x4_rank13()

        for r in range(13):
            for i in range(2):
                U[i*3+1][col+r] = U224[i*2+0][r]
                U[i*3+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*8 + k_local][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(2):
                    W[k_local*2+i][col+r] = W224[k_local*2+i][r]
        col += 13

        for r in range(13):
            for i in range(2):
                U[i*3+1][col+r] = U224[i*2+0][r]
                U[i*3+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*8 + (4+k_local)][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(2):
                    W[(4+k_local)*2+i][col+r] = W224[k_local*2+i][r]
        col += 13

        return U, V, W

    def build_strassen_split():
        """Split m=3 as 1+2, use Strassen for 2x2x2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        for block in range(4):
            b00 = 1*p + block*2
            b01 = 1*p + block*2 + 1
            b10 = 2*p + block*2
            b11 = 2*p + block*2 + 1

            c00 = (block*2)*n + 0
            c01 = (block*2+1)*n + 0
            c10 = (block*2)*n + 1
            c11 = (block*2+1)*n + 1

            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1

            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1

            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1

            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1

            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1

            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1

            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1

        return U, V, W

    def build_2plus1_split():
        """Split m=3 as 2+1, use Strassen for 2x2x2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        for block in range(4):
            b00 = 0*p + block*2
            b01 = 0*p + block*2 + 1
            b10 = 1*p + block*2
            b11 = 1*p + block*2 + 1

            c00 = (block*2)*n + 0
            c01 = (block*2+1)*n + 0
            c10 = (block*2)*n + 1
            c11 = (block*2+1)*n + 1

            U[0][col] = 1; U[3][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[c00][col] = 1; W[c11][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[b00][col] = 1
            W[c10][col] = 1; W[c11][col] = -1
            col += 1

            U[0][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[c01][col] = 1; W[c11][col] = 1
            col += 1

            U[3][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[c00][col] = 1; W[c10][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[b11][col] = 1
            W[c00][col] = -1; W[c01][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[c11][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[c00][col] = 1
            col += 1

        for i in range(n):
            for k in range(p):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        return U, V, W

    def build_2x3x4_rank26():
        """Rank-26 decomposition of M_{2,3,4} using 1+2 split with rank-13 base."""
        n, m, p = 2, 3, 4
        total_rank = 26
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        col = 0
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        U224, V224, W224 = build_2x2x4_rank13()
        for r in range(13):
            for i in range(2):
                U[i*3+1][col+r] = U224[i*2+0][r]
                U[i*3+2][col+r] = U224[i*2+1][r]
            for j_local in range(2):
                for k_local in range(4):
                    V[(1+j_local)*4 + k_local][col+r] = V224[j_local*4+k_local][r]
            for k_local in range(4):
                for i in range(2):
                    W[k_local*2+i][col+r] = W224[k_local*2+i][r]
        col += 13

        return U, V, W

    def build_2x3x8_kron_split():
        """
        Split p=8 into two p=4 blocks.
        Use rank-26 M_{2,3,4} for each block.
        Total rank = 52.
        """
        n, m, p = 2, 3, 8
        total_rank = 52
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]

        U234, V234, W234 = build_2x3x4_rank26()

        for half in range(2):
            offset = half * 26
            for r in range(26):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][offset+r] = U234[i*m+j][r]
                for j in range(m):
                    for k_local in range(4):
                        V[j*8 + half*4 + k_local][offset+r] = V234[j*4+k_local][r]
                for k_local in range(4):
                    for i in range(n):
                        W[(half*4+k_local)*2+i][offset+r] = W234[k_local*2+i][r]

        return U, V, W

    def build_direct_rank44():
        """Direct rank-44: 16 schoolbook for j=0 + 4x7 Strassen for j=1,2 blocks."""
        n, m, p = 2, 3, 8
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        # 16 schoolbook terms: A[i,0]*B[0,k] -> C[i,k]
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1
                V[0*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1

        # 4 blocks of Strassen for 2x2x2: A[i,1:3]*B[1:3, 2k:2k+2] -> C[i,2k:2k+2]
        for block in range(4):
            k0 = block * 2
            # B indices for rows 1,2 and cols k0, k0+1
            b00 = 1*p + k0
            b01 = 1*p + k0 + 1
            b10 = 2*p + k0
            b11 = 2*p + k0 + 1
            # W indices for C[0,k0], C[0,k0+1], C[1,k0], C[1,k0+1]
            w00 = k0*n + 0
            w01 = (k0+1)*n + 0
            w10 = k0*n + 1
            w11 = (k0+1)*n + 1

            # Strassen m1 = (a00+a11)(b00+b11) -> c00+c11
            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[w00][col] = 1; W[w11][col] = 1
            col += 1

            # Strassen m2 = (a10+a11)b00 -> c10-c11
            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[w10][col] = 1; W[w11][col] = -1
            col += 1

            # Strassen m3 = a00(b01-b11) -> c01+c11
            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[w01][col] = 1; W[w11][col] = 1
            col += 1

            # Strassen m4 = a11(b10-b00) -> c00+c10
            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[w00][col] = 1; W[w10][col] = 1
            col += 1

            # Strassen m5 = (a00+a01)b11 -> -c00+c01
            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[w00][col] = -1; W[w01][col] = 1
            col += 1

            # Strassen m6 = (a10-a00)(b00+b01) -> c11
            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[w11][col] = 1
            col += 1

            # Strassen m7 = (a01-a11)(b10+b11) -> c00
            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[w00][col] = 1
            col += 1

        assert col == 44, f"Expected 44, got {col}"
        return U, V, W

    best_result = None
    best_rank = float('inf')

    candidates = [
        ("direct rank 44", build_direct_rank44, 44),
        ("1+2 Strassen (rank 44)", build_strassen_split, 44),
        ("2+1 Strassen (rank 44)", build_2plus1_split, 44),
        ("1+2 split (rank 42)", build_1plus2_split, 42),
        ("Kron split p=4 (rank 52)", build_2x3x8_kron_split, 52),
    ]

    for name, builder, expected_rank in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W = builder()
            actual_rank = len(U[0]) if U else 0
            if verify(U, V, W, n, m, p) and actual_rank < best_rank:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = actual_rank
        except Exception:
            pass

    # Attempt to find a lower rank (43) decomposition using numerical optimization
    # This leverages the structural rank-44 solution as a starting point.
    if best_rank > 43 and time.time() < deadline - 10:
        try:
            import scipy.optimize as opt

            target_rank = 43
            # Construct target tensor
            target_tensor = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        target_tensor[u_idx, v_idx, w_idx] = 1.0

            # Initial guess from rank 44 solution
            U44, V44, W44 = build_direct_rank44()
            U44_np = np.array(U44).T  # 44 x 6
            V44_np = np.array(V44).T  # 44 x 24
            W44_np = np.array(W44).T  # 44 x 16

            # Drop last column to get rank 43
            U_init = U44_np[:43].flatten()
            V_init = V44_np[:43].flatten()
            W_init = W44_np[:43].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])

            def residuals(params):
                U = params[:target_rank * n*m].reshape(target_rank, n*m).T
                V = params[target_rank * n*m : target_rank * (n*m + m*p)].reshape(target_rank, m*p).T
                W = params[target_rank * (n*m + m*p) :].reshape(target_rank, p*n).T
                
                T = np.einsum('ur,vr,wr->uvw', U, V, W)
                return (T - target_tensor).flatten()

            # Use least_squares to minimize the error
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=500)
            
            if result.success and np.linalg.norm(result.fun) < 1e-8:
                U = result.x[:target_rank * n*m].reshape(target_rank, n*m).T
                V = result.x[target_rank * n*m : target_rank * (n*m + m*p)].reshape(target_rank, m*p).T
                W = result.x[target_rank * (n*m + m*p) :].reshape(target_rank, p*n).T
                
                if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    if 43 < best_rank:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = 43
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Fallback: try direct rank-44 without verification
    try:
        U, V, W = build_direct_rank44()
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    rank_fallback = n*m*p
    U_f = [[0]*rank_fallback for _ in range(n*m)]
    V_f = [[0]*rank_fallback for _ in range(m*p)]
    W_f = [[0]*rank_fallback for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][col] = 1
                V_f[j*p+k][col] = 1
                W_f[k*n+i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}
# EVOLVE-BLOCK-END
