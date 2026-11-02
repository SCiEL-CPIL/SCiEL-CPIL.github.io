import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 130

    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def get_strassen_28():
        # Fallback solution
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        # Note: The Strassen decomposition in the prompt was specific to the block structure.
        # I will use the exact block construction from the prompt to ensure correctness.
        
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))

        # Re-implementing the block logic from the prompt to be safe
        # ... (copying logic from prompt) ...
        
        # For brevity in this thought block, I'll assume the prompt's logic is correct and copy it.
        # Actually, I'll just paste the prompt's Strassen construction code into this function.
        
        U_s, V_s, W_s = strassen_2x2_decomp()
        # ... (rest of block construction) ...
        
        # To save space in the final code, I will include the full function.
        pass

    def strassen_2x2_decomp():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0
        return U, V, W

    def build_strassen_28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))

        # Block 1
        for t in range(7):
            col = 0 * 7 + t
            for li in range(2):
                for lj in range(2):
                    local_a = li * 2 + lj
                    global_a = li * 4 + lj
                    U_full[global_a, col] = U_s[local_a, t]
            for lj in range(2):
                for lk in range(2):
                    local_b = lj * 2 + lk
                    global_b = lj * 4 + lk
                    V_full[global_b, col] = V_s[local_b, t]
            for lk in range(2):
                for li in range(2):
                    local_c = lk * 2 + li
                    global_c = lk * 2 + li
                    W_full[global_c, col] = W_s[local_c, t]

        # Block 2
        for t in range(7):
            col = 1 * 7 + t
            for li in range(2):
                for lj in range(2):
                    local_a = li * 2 + lj
                    global_a = li * 4 + (lj + 2)
                    U_full[global_a, col] = U_s[local_a, t]
            for lj in range(2):
                for lk in range(2):
                    local_b = lj * 2 + lk
                    global_b = (lj + 2) * 4 + lk
                    V_full[global_b, col] = V_s[local_b, t]
            for lk in range(2):
                for li in range(2):
                    local_c = lk * 2 + li
                    global_c = lk * 2 + li
                    W_full[global_c, col] = W_s[local_c, t]

        # Block 3
        for t in range(7):
            col = 2 * 7 + t
            for li in range(2):
                for lj in range(2):
                    local_a = li * 2 + lj
                    global_a = li * 4 + lj
                    U_full[global_a, col] = U_s[local_a, t]
            for lj in range(2):
                for lk in range(2):
                    local_b = lj * 2 + lk
                    global_b = lj * 4 + (lk + 2)
                    V_full[global_b, col] = V_s[local_b, t]
            for lk in range(2):
                for li in range(2):
                    local_c = lk * 2 + li
                    global_c = (lk + 2) * 2 + li
                    W_full[global_c, col] = W_s[local_c, t]

        # Block 4
        for t in range(7):
            col = 3 * 7 + t
            for li in range(2):
                for lj in range(2):
                    local_a = li * 2 + lj
                    global_a = li * 4 + (lj + 2)
                    U_full[global_a, col] = U_s[local_a, t]
            for lj in range(2):
                for lk in range(2):
                    local_b = lj * 2 + lk
                    global_b = (lj + 2) * 4 + (lk + 2)
                    V_full[global_b, col] = V_s[local_b, t]
            for lk in range(2):
                for li in range(2):
                    local_c = lk * 2 + li
                    global_c = (lk + 2) * 2 + li
                    W_full[global_c, col] = W_s[local_c, t]

        return U_full, V_full, W_full

    def try_rank(r, num_restarts=20):
        if time.time() > deadline - 15:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                return None
                
            np.random.seed(42 + restart)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS Iterations
            for _ in range(50):
                if time.time() > deadline - 15:
                    return None
                    
                # Update U
                M = V[:, :, None] * W[:, None, :]
                M_flat = M.transpose(2, 0, 1).reshape(r, m * p * p * n)
                T_flat = T.reshape(n * m, m * p * p * n)
                try:
                    U, _, _, _ = np.linalg.lstsq(M_flat.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    return None

                # Update V
                N = U[:, :, None] * W[:, None, :]
                N_flat = N.transpose(2, 0, 1).reshape(r, n * m * p * n)
                T_flat_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
                try:
                    V, _, _, _ = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)
                    V = V.T
                except:
                    return None

                # Update W
                P = U[:, :, None] * V[:, None, :]
                P_flat = P.transpose(2, 0, 1).reshape(r, n * m * m * p)
                T_flat_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
                try:
                    W, _, _, _ = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)
                    W = W.T
                except:
                    return None

            recon = np.einsum('at,bt,ct->abc', U, V, W)
            residual = np.max(np.abs(recon - T))
            
            if residual < 1e-6:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, tol=1e-8):
                    return to_lists(U_rounded, V_rounded, W_rounded, r)
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, tol=1e-8):
                    return to_lists(U_int, V_int, W_int, r)
        
        return None

    # Try ranks 24 to 27
    for r in [24, 25, 26, 27]:
        res = try_rank(r, num_restarts=30)
        if res:
            return res

    # Fallback
    U, V, W = build_strassen_28()
    return to_lists(U, V, W, 28)
