import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 6

    # Strassen factors for 2x2 @ 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_U = [
        {0: 1, 3: 1},  # M1: (A00+A11)
        {2: 1, 3: 1},  # M2: (A10+A11)
        {0: 1},         # M3: A00
        {3: 1},         # M4: A11
        {0: 1, 1: 1},  # M5: (A00+A01)
        {2: 1, 0: -1}, # M6: (A10-A00)
        {1: 1, 3: -1}, # M7: (A01-A11)
    ]
    strassen_V = [
        {0: 1, 3: 1},  # M1: (B00+B11)
        {0: 1},         # M2: B00
        {2: 1, 3: -1}, # M3: (B10-B11)
        {1: 1, 0: -1}, # M4: (B01-B00)
        {3: 1},         # M5: B11
        {0: 1, 1: 1},  # M6: (B00+B01)
        {2: 1, 3: 1},  # M7: (B10+B11)
    ]
    strassen_W = [
        {0: 1, 3: 1},  # M1: C00+=1, C11+=1
        {2: 1},         # M2: C10+=1
        {1: 1, 3: -1}, # M3: C01+=1, C11+=-1
        {0: -1, 2: 1}, # M4: C00+=-1, C10+=1
        {0: 1, 1: 1},  # M5: C00+=1, C01+=1
        {3: 1},         # M6: C11+=1
        {0: 1, 3: 1},  # M7: C00+=1, C11+=1
    ]

    def build_rank_42():
        rank_total = 42
        U = np.zeros((8, rank_total))   # 2x4
        V = np.zeros((24, rank_total))  # 4x6
        W = np.zeros((12, rank_total))  # 2x6

        # Two blocks: block 0 uses A_L (cols 0,1 of A) and B_T (rows 0,1 of B)
        # block 1 uses A_R (cols 2,3 of A) and B_B (rows 2,3 of B)
        for block in range(2):
            a_col_offset = block * 2  # 0 or 2
            b_row_offset = block * 2  # 0 or 2

            for col_pair in range(3):  # 3 column pairs in the 6 columns
                for s in range(7):  # 7 Strassen terms
                    col = block * 21 + col_pair * 7 + s

                    # U: A indices
                    a_local_to_full = {0: a_col_offset, 1: a_col_offset + 1,
                                       2: 4 + a_col_offset, 3: 4 + a_col_offset + 1}
                    for local_idx, sign in strassen_U[s].items():
                        u_idx = a_local_to_full[local_idx]
                        U[u_idx, col] += sign

                    # V: B indices
                    b_local_to_full = {0: b_row_offset * 6 + 2 * col_pair,
                                       1: b_row_offset * 6 + 2 * col_pair + 1,
                                       2: (b_row_offset + 1) * 6 + 2 * col_pair,
                                       3: (b_row_offset + 1) * 6 + 2 * col_pair + 1}
                    for local_idx, sign in strassen_V[s].items():
                        v_idx = b_local_to_full[local_idx]
                        V[v_idx, col] += sign

                    # W: C indices
                    c_local_to_full = {0: 0 * 6 + 2 * col_pair,
                                       1: 0 * 6 + 2 * col_pair + 1,
                                       2: 1 * 6 + 2 * col_pair,
                                       3: 1 * 6 + 2 * col_pair + 1}
                    for local_idx, sign in strassen_W[s].items():
                        w_idx = c_local_to_full[local_idx]
                        W[w_idx, col] += sign

        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_true = np.zeros((8, 24, 12))
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    T_true[i*4+j, j*6+k, k*2+i] = 1.0

        if not np.allclose(T_recon, T_true, atol=1e-9):
            return None

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find better decomposition via optimization
    def try_optimize(rank_target, max_seeds=20):
        T_true = np.zeros((8, 24, 12))
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    T_true[i*4+j, j*6+k, k*2+i] = 1.0

        def loss(params):
            U = params[:8*rank_target].reshape(8, rank_target)
            V = params[8*rank_target:8*rank_target+24*rank_target].reshape(24, rank_target)
            W = params[8*rank_target+24*rank_target:].reshape(12, rank_target)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((T_recon - T_true)**2)

        best = None
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break
            np.random.seed(seed)
            U0 = np.random.randn(8, rank_target) * 0.1
            V0 = np.random.randn(24, rank_target) * 0.1
            W0 = np.random.randn(12, rank_target) * 0.1
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            from scipy.optimize import minimize
            res = minimize(loss, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-14})

            if res.fun < 1e-8:
                U = res.x[:8*rank_target].reshape(8, rank_target)
                V = res.x[8*rank_target:8*rank_target+24*rank_target].reshape(24, rank_target)
                W = res.x[8*rank_target+24*rank_target:].reshape(12, rank_target)

                # Snap to half-integers
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                T_r = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                if np.allclose(T_r, T_true, atol=1e-9):
                    best = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    break

                # Try integer snap
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                T_ri = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
                if np.allclose(T_ri, T_true, atol=1e-9):
                    best = {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                    break

        return best

    # Try rank 40
    if time.time() < deadline - 30:
        result = try_optimize(40, max_seeds=10)
        if result is not None:
            return result

    # Try rank 38
    if time.time() < deadline - 20:
        result = try_optimize(38, max_seeds=10)
        if result is not None:
            return result

    # Fallback to rank 42
    result = build_rank_42()
    if result is not None:
        return result

    # Ultimate fallback: schoolbook rank 48
    rank_total = 48
    U = np.zeros((8, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((12, rank_total))
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(6):
                U[i*4+j, col] = 1.0
                V[j*6+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
