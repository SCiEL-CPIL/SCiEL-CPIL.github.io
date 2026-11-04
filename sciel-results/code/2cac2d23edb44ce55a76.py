import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication.
    # U2 indexed by (i,j) -> i*2+j ; V2 by (j,k) -> j*2+k ; W2 by (k,i) -> k*2+i
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
        [1, -1,1,  0,  0,  1,  0]
    ], dtype=float)

    # Kronecker product gives 4x4 from 2x2: rank 49.
    U_kron = np.kron(U2, U2)  # shape (16, 49), row index (a_out*2+a_in)*4 + (b_out*2+b_in)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)

    # Map standard row-major index S -> Kronecker row index K.
    # For U: S = i*4+j -> i=i1*2+i2, j=j1*2+j2 -> K=(i1*2+j1)*4+(i2*2+j2)
    # For V: S = j*4+k -> j=j1*2+j2, k=k1*2+k2 -> K=(j1*2+k1)*4+(j2*2+k2)
    # For W: S = k*4+i -> k=k1*2+k2, i=i1*2+i2 -> K=(k1*2+i1)*4+(k2*2+i2)
    def std_to_kron(S):
        a = S // 4
        b = S % 4
        a1, a2 = a // 2, a % 2
        b1, b2 = b // 2, b % 2
        return (a1 * 2 + b1) * 4 + (a2 * 2 + b2)

    perm = np.array([std_to_kron(S) for S in range(16)], dtype=int)

    U = U_kron[perm, :].copy()  # shape (16, 49)
    V = V_kron[perm, :].copy()
    W = W_kron[perm, :].copy()

    # Verification helper
    def verify(Um, Vm, Wm, tol=1e-9):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    ui = Um[i * 4 + j, :]
                    vj = Vm[j * 4 + k, :]
                    wk = Wm[k * 4 + i, :]
                    if abs(np.dot(ui, vj * wk) - 1.0) > tol:
                        return False
        return True

    # Check if the base decomposition is valid
    if not verify(U, V, W):
        # Fallback to schoolbook (rank 64)
        rank_fb = 64
        Uf = np.zeros((16, rank_fb), dtype=float)
        Vf = np.zeros((16, rank_fb), dtype=float)
        Wf = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    Uf[i * 4 + j, col] = 1.0
                    Vf[j * 4 + k, col] = 1.0
                    Wf[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # Try to remove columns that are entirely zero
    active = set(range(49))
    for r in range(49):
        if time.time() - t0 > deadline - 10.0:
            break
        if not np.any(U[:, r]) or not np.any(V[:, r]) or not np.any(W[:, r]):
            active.remove(r)

    if len(active) < 49:
        active_list = sorted(active)
        U_red = U[:, active_list]
        V_red = V[:, active_list]
        W_red = W[:, active_list]
        if verify(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
