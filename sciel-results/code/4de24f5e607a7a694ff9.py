import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build target tensor T[x,y,z] where x=i*m+k, y=k*p+j, z=k*n+i (W[k*n+i,r] convention)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = k * n + i
                T[x, y, z] = 1.0

    # Strassen-based rank-28 solution
    rank28 = 28
    U28 = np.zeros((n*m, rank28), dtype=np.float64)
    V28 = np.zeros((m*p, rank28), dtype=np.float64)
    W28 = np.zeros((p*n, rank28), dtype=np.float64)

    def add_strassen_block(col_off, X, Y, Z):
        # X = [x00, x01, x10, x11] mapped to U rows
        # Y = [y00, y01, y10, y11] mapped to V rows
        # Z = [c00, c01, c10, c11] mapped to W rows
        # P1 = (x00+x11)(y00+y11)
        U28[X[0]][col_off+0] += 1; U28[X[3]][col_off+0] += 1
        V28[Y[0]][col_off+0] += 1; V28[Y[3]][col_off+0] += 1
        W28[Z[0]][col_off+0] += 1; W28[Z[3]][col_off+0] += 1
        # P2 = (x10+x11)(y00)
        U28[X[2]][col_off+1] += 1; U28[X[3]][col_off+1] += 1
        V28[Y[0]][col_off+1] += 1
        W28[Z[2]][col_off+1] += 1; W28[Z[3]][col_off+1] -= 1
        # P3 = x00(y01-y11)
        U28[X[0]][col_off+2] += 1
        V28[Y[1]][col_off+2] += 1; V28[Y[3]][col_off+2] -= 1
        W28[Z[1]][col_off+2] += 1; W28[Z[3]][col_off+2] += 1
        # P4 = x11(y10-y00)
        U28[X[3]][col_off+3] += 1
        V28[Y[2]][col_off+3] += 1; V28[Y[0]][col_off+3] -= 1
        W28[Z[0]][col_off+3] += 1; W28[Z[2]][col_off+3] += 1
        # P5 = (x00+x01)(y11)
        U28[X[0]][col_off+4] += 1; U28[X[1]][col_off+4] += 1
        V28[Y[3]][col_off+4] += 1
        W28[Z[0]][col_off+4] -= 1; W28[Z[1]][col_off+4] += 1
        # P6 = (x10-x00)(y00+y01)
        U28[X[2]][col_off+5] += 1; U28[X[0]][col_off+5] -= 1
        V28[Y[0]][col_off+5] += 1; V28[Y[1]][col_off+5] += 1
        W28[Z[3]][col_off+5] += 1
        # P7 = (x01-x11)(y10+y11)
        U28[X[1]][col_off+6] += 1; U28[X[3]][col_off+6] -= 1
        V28[Y[2]][col_off+6] += 1; V28[Y[3]][col_off+6] += 1
        W28[Z[0]][col_off+6] += 1

    # A = [A_L | A_R], A_L cols 0,1; A_R cols 2,3
    # B = [[B11, B12],[B21, B22]]
    # C = A_L*B11 + A_R*B21 (left 2 cols) and A_L*B12 + A_R*B22 (right 2 cols)
    # X for A_L: rows 0,1,4,5 (i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1)
    # X for A_R: rows 2,3,6,7 (i=0,k=2; i=0,k=3; i=1,k=2; i=1,k=3)
    # Y for B11: rows 0,1,4,5 (k=0,j=0; k=0,j=1; k=1,j=0; k=1,j=1)
    # Y for B12: rows 2,3,6,7 (k=0,j=2; k=0,j=3; k=1,j=2; k=1,j=3)
    # Y for B21: rows 8,9,12,13 (k=2,j=0; k=2,j=1; k=3,j=0; k=3,j=1)
    # Y for B22: rows 10,11,14,15 (k=2,j=2; k=2,j=3; k=3,j=2; k=3,j=3)
    # Z for left C (cols 0,1): z = k*2+i. k=0: 0,1; k=1: 2,3 -> [0,1,2,3]
    # Z for right C (cols 2,3): k=2: 4,5; k=3: 6,7 -> [4,5,6,7]
    # But we need Z in order [c00, c01, c10, c11] for the 2x2 block
    # For left: c00=z(0,0)=0, c01=z(0,1)=2, c10=z(1,0)=1, c11=z(1,1)=3 -> [0,2,1,3]
    # For right: c00=z(2,0)=4, c01=z(2,1)=6, c10=z(3,0)=5, c11=z(3,1)=7 -> [4,6,5,7]

    X_L = [0, 1, 4, 5]
    X_R = [2, 3, 6, 7]
    Y_B11 = [0, 1, 4, 5]
    Y_B12 = [2, 3, 6, 7]
    Y_B21 = [8, 9, 12, 13]
    Y_B22 = [10, 11, 14, 15]
    Z_L = [0, 2, 1, 3]
    Z_R = [4, 6, 5, 7]

    add_strassen_block(0, X_L, Y_B11, Z_L)
    add_strassen_block(7, X_R, Y_B21, Z_L)
    add_strassen_block(14, X_L, Y_B12, Z_R)
    add_strassen_block(21, X_R, Y_B22, Z_R)

    # Verify rank-28 solution
    def tensor_from_factors(U, V, W, rank):
        return np.einsum('xr,yr,zr->xyz', U[:, :rank], V[:, :rank], W[:, :rank])

    T28 = tensor_from_factors(U28, V28, W28, 28)
    assert np.allclose(T28, T, atol=1e-9), "Rank-28 solution failed verification"

    best_rank = 28
    best_U = U28.copy()
    best_V = V28.copy()
    best_W = W28.copy()

    def residual(U, V, W, rank):
        T_est = np.einsum('xr,yr,zr->xyz', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.sum((T_est - T) ** 2)

    def is_exact(U, V, W, rank):
        T_est = np.einsum('xr,yr,zr->xyz', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.allclose(T_est, T, atol=1e-9)

    def local_search(U_init, V_init, W_init, target_rank, max_iter=2000, perturb_range=2):
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        # Zero out columns beyond what we want to modify (last 1-2 columns)
        if target_rank < 28:
            U[:, target_rank-1:] = 0
            V[:, target_rank-1:] = 0
            W[:, target_rank-1:] = 0

        cur_res = residual(U, V, W, target_rank)
        best_res = cur_res
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if best_res < 1e-9:
                if is_exact(best_U, best_V, best_W, target_rank):
                    return best_U, best_V, best_W, best_res

            # Try a random perturbation
            r = random.randint(0, target_rank - 1)
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n*m-1)
                old = U[row, r]
                delta = random.randint(-perturb_range, perturb_range)
                if delta == 0:
                    delta = 1
                U[row, r] += delta
            elif mat == 'V':
                row = random.randint(0, m*p-1)
                old = V[row, r]
                delta = random.randint(-perturb_range, perturb_range)
                if delta == 0:
                    delta = 1
                V[row, r] += delta
            else:
                row = random.randint(0, p*n-1)
                old = W[row, r]
                delta = random.randint(-perturb_range, perturb_range)
                if delta == 0:
                    delta = 1
                W[row, r] += delta

            new_res = residual(U, V, W, target_rank)
            if new_res < cur_res:
                cur_res = new_res
                if new_res < best_res:
                    best_res = new_res
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                # Revert
                if mat == 'U':
                    U[row, r] = old
                elif mat == 'V':
                    V[row, r] = old
                else:
                    W[row, r] = old

        return None

    # Try rank 27
    if time.time() < deadline - 20:
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            result = local_search(U28, V28, W28, 27, max_iter=1500, perturb_range=2)
            if result is not None:
                U27, V27, W27, res = result
                if is_exact(U27, V27, W27, 27):
                    best_rank = 27
                    best_U = U27
                    best_V = V27
                    best_W = W27
                    break

    # Try rank 26 if still have time
    if best_rank == 28 and time.time() < deadline - 15:
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            result = local_search(U28, V28, W28, 26, max_iter=1000, perturb_range=3)
            if result is not None:
                U26, V26, W26, res = result
                if is_exact(U26, V26, W26, 26):
                    best_rank = 26
                    best_U = U26
                    best_V = V26
                    best_W = W26
                    break

    # Convert to list-of-lists with integer entries
    U_out = [[int(round(x)) for x in row] for row in best_U[:, :best_rank]]
    V_out = [[int(round(x)) for x in row] for row in best_V[:, :best_rank]]
    W_out = [[int(round(x)) for x in row] for row in best_W[:, :best_rank]]

    return {"U": U_out, "V": V_out, "W": W_out}
