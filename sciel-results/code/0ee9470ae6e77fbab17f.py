import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118

    n, m, p = 3, 4, 6

    def build_base_66():
        rank = 66
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            if time.time() > deadline:
                return
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1
            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1
            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1
            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1
            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1
            U[A10, col] += 1; U[A00, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1
            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            if time.time() > deadline:
                return
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)

        return U[:, :col], V[:, :col], W[:, :col]

    def verify_fast(U, V, W):
        """Fast verification using vectorized numpy."""
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        # T[r,s,c] = sum_rank U[r,s,rank] * V[s,c,rank] * W[c,r,rank]
        # Use einsum: T[r,s,c] = einsum('sr,sc,cr->', ...) no, need 3D output
        # T = einsum('sr,sc,cr->rsc', U3, V3, W3)
        T = np.einsum('sr,sc,cr->rsc', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-10)

    U, V, W = build_base_66()
    base_rank = U.shape[1]

    if not verify_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding removable columns
    # For each column r, check if removing it still gives valid decomposition
    # Use fast verification
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    cur_rank = current_U.shape[1]

    for _ in range(10):  # Try up to 10 rounds of removal
        if time.time() > deadline - 3:
            break
        if cur_rank <= 60:
            break
        removed = False
        for r in range(cur_rank):
            if time.time() > deadline - 3:
                break
            U_test = np.delete(current_U, r, axis=1)
            V_test = np.delete(current_V, r, axis=1)
            W_test = np.delete(current_W, r, axis=1)
            if verify_fast(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                cur_rank -= 1
                removed = True
                break
        if not removed:
            break

    # Final check
    if not verify_fast(current_U, current_V, current_W):
        current_U, current_V, current_W = U, V, W

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
