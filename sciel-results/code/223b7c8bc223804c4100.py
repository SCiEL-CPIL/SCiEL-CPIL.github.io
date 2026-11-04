import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Convention: for 2x2, U has rows indexed by (i,j) -> i*2+j
    # V has rows indexed by (j,k) -> j*2+k
    # W has rows indexed by (k,i) -> k*2+i
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=float)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=float)

    # Kronecker product gives 4x4 from 2x2: rank 49
    U_kron = np.kron(U2, U2)  # shape (16, 49)
    V_kron = np.kron(V2, V2)  # shape (16, 49)
    W_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permute rows to standard row-major ordering
    def permute_rows(kron_mat, a_idx, b_idx):
        perm = []
        for S in range(16):
            a_out, a_in, b_out, b_in = a_idx(S), b_idx(S)
            block = a_out * 2 + b_out
            sub = a_in * 2 + b_in
            perm.append(block * 4 + sub)
        return kron_mat[perm, :]

    def u_split(S):
        i = S // 4
        j = S % 4
        return i // 2, i % 2, j // 2, j % 2

    def v_split(S):
        j = S // 4
        k = S % 4
        return j // 2, j % 2, k // 2, k % 2

    def w_split(S):
        k = S // 4
        i = S % 4
        return k // 2, k % 2, i // 2, i % 2

    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    for S in range(16):
        i_out, i_in, j_out, j_in = u_split(S)
        U[S, :] = U_kron[(i_out * 2 + j_out) * 4 + (i_in * 2 + j_in), :]

        j_out, j_in, k_out, k_in = v_split(S)
        V[S, :] = V_kron[(j_out * 2 + k_out) * 4 + (j_in * 2 + k_in), :]

        k_out, k_in, i_out, i_in = w_split(S)
        W[S, :] = W_kron[(k_out * 2 + i_out) * 4 + (k_in * 2 + i_in), :]

    # Verification
    if time.time() - t0 > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ui = U[i * 4 + j, :]
                vj = V[j * 4 + k, :]
                wk = W[k * 4 + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 4 + k, col] = 1.0
                    W_fb[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Build the constraint matrix for rank reduction via greedy column elimination
    # For each (i,j,k), the equation is: sum_r c_r * U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    A_full = np.zeros((64, 49))
    b_vec = np.ones(64)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    A_full[idx, r] = U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                idx += 1

    # Greedy column elimination: try to remove one column at a time
    active = set(range(49))
    
    for iteration in range(49):
        if time.time() - t0 > deadline - 20.0:
            break
        
        progress = False
        for r in sorted(active):
            if time.time() - t0 > deadline - 20.0:
                break
            
            test_active = list(active - {r})
            A_sub = A_full[:, test_active]
            
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A_sub, b_vec, rcond=None)
                residual = np.linalg.norm(A_sub @ c - b_vec)
                if residual < 1e-9:
                    active.remove(r)
                    progress = True
                    break
            except:
                pass
        
        if not progress:
            break

    if time.time() - t0 > deadline - 15.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if len(active) < 49:
        active_list = sorted(active)
        U_red = U[:, active_list]
        V_red = V[:, active_list]
        W_red = W[:, active_list]

        # Final verification
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    ui = U_red[i * 4 + j, :]
                    vj = V_red[j * 4 + k, :]
                    wk = W_red[k * 4 + i, :]
                    val = np.dot(ui, vj * wk)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err

        if max_err > 1e-9:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
