import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (16, R)
    # W: (p*n, R) = (8, R)
    # Output convention: W[k*n + i, r] where k is output col, i is output row?
    # Actually, standard matrix mult C = A @ B.
    # C is n x p.
    # Index i for U: A's row * m + A's col. (0..7)
    # Index j for V: B's row * p + B's col. (0..15)
    # Index k for W: C's row * p + C's col? Or C's col * n + C's row?
    # Prompt says: "W[k*n+i,r]".
    # If C is n x p, then C[row, col].
    # Usually tensor T[i,j,k] corresponds to C[i, k] = sum_j A[i,j] B[j,k].
    # Here A is n x m, B is m x p, C is n x p.
    # So i indexes A's rows (and C's rows), j indexes A's cols (and B's rows), k indexes B's cols (and C's cols).
    # U[i, r] where i in 0..n*m-1. i = row_a * m + col_a.
    # V[j, r] where j in 0..m*p-1. j = row_b * p + col_b.
    # W[k, r] where k in 0..p*n-1.
    # The prompt says "W[k*n+i,r]". This looks like an indexing into a flattened vector of size p*n?
    # If W is shape (p*n, R), then the first index is in 0..p*n-1.
    # If the convention is C[col_c, row_c] or C[row_c, col_c]?
    # Let's look at the standard: C = A @ B. C is (2, 4).
    # T[i, j, k] = 1 if A[i] and B[j] contribute to C[k].
    # i = r_a * m + c_a.
    # j = r_b * p + c_b.
    # k = ?
    # C is n x p. Let k index C.
    # If k = c_c * n + r_c (col major) or r_c * p + c_c (row major)?
    # The prompt says "W[k*n+i, r]". This implies the first dimension of W is size p*n.
    # And the index is k*n + i? No, that would mix k and i.
    # Wait, the prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # This likely refers to the layout of the W matrix rows.
    # If W has shape (p*n, R), the rows are indexed by some quantity.
    # Let's assume the standard: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    # The index k must run from 0 to p*n-1.
    # For C (2x4), there are 8 elements.
    # Let's check the seed code's T construction:
    # k = col_c * n + row_c.
    # So k is col-major indexing of C? C[row_c, col_c].
    # Let's stick to the seed's definition of T to be safe, as it likely passed tests or is close.
    # Seed:
    # if col_a == row_b and row_a == row_c and col_b == col_c:
    #     T[i, j, k] = 1.0
    # with i = row_a * m + col_a
    # j = row_b * p + col_b
    # k = col_c * n + row_c
    
    # Let's verify this matches C = A @ B.
    # C[row_a, col_c] = sum_{c_a} A[row_a, c_a] B[c_a, col_c].
    # Term exists if c_a == row_b (B's row index) and row_a == row_c (C's row index) and col_b == col_c (B's col index).
    # Yes, this is correct.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    T_flat = T.flatten()

    # --- Construction of Rank 28 Baseline (Strassen 2x2x2 blocks) ---
    
    # Strassen 2x2x2
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)
    
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0
    
    M_sys = np.zeros((64, 28), dtype=float)
    T_2_vec = T_2.flatten()
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]
    W_s, _, _, _ = np.linalg.lstsq(M_sys, T_2_vec, rcond=None)
    W_s = W_s.reshape(4, 7)

    U_28 = np.zeros((8, 28), dtype=float)
    V_28 = np.zeros((16, 28), dtype=float)
    W_28 = np.zeros((8, 28), dtype=float)
    
    # Block 1: C_00 += A_00 B_00
    U_28[0:4, 0:7] = U_s
    V_28[0:4, 0:7] = V_s
    W_28[0:4, 0:7] = W_s
    
    # Block 2: C_00 += A_01 B_10
    U_28[4:8, 7:14] = U_s
    V_28[8:12, 7:14] = V_s
    W_28[0:4, 7:14] = W_s
    
    # Block 3: C_01 += A_00 B_01
    U_28[0:4, 14:21] = U_s
    V_28[4:8, 14:21] = V_s
    W_28[4:8, 14:21] = W_s
    
    # Block 4: C_01 += A_01 B_11
    U_28[4:8, 21:28] = U_s
    V_28[12:16, 21:28] = V_s
    W_28[4:8, 21:28] = W_s

    def verify(U, V, W):
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(computed - T))

    err_28 = verify(U_28, V_28, W_28)
    if err_28 > 1e-9:
        # Fallback
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # --- Search for Rank 27 using Structural Elimination ---
    # The 2x4x4 multiplication can be viewed as two 2x4x2 multiplications?
    # Or better: We have 4 blocks of 2x2x2.
    # Can we merge them?
    # A is 2x4. B is 4x4.
    # C = A B.
    # Let's try to construct a rank 27 solution by perturbing the rank 28 solution
    # and using ALS with rounding to half-integers.
    
    def build_A_U(V, W, R):
        # A_U shape 1024 x (8*R)
        # Row (i,j,k), Col (i', r) -> V[j,r] W[k,r] if i==i'
        # Using sparse kron
        import scipy.sparse as sp
        A_U = sp.lil_matrix((1024, 8 * R), dtype=float)
        for r in range(R):
            M_r = (V[:, r] * W[:, r]).reshape(16, 8)
            M_r_flat = M_r.flatten().reshape(128, 1)
            Block = sp.kron(sp.eye(8), sp.csr_matrix(M_r_flat))
            A_U[:, r*8:(r+1)*8] = Block
        return A_U.tocsr()

    def build_A_V(U, W, R):
        # A_V shape 1024 x (16*R)
        # Row (i,j,k), Col (j', r) -> U[i,r] W[k,r] if j==j'
        import scipy.sparse as sp
        A_V = sp.lil_matrix((1024, 16 * R), dtype=float)
        for r in range(R):
            u_r = U[:, r]
            w_r = W[:, r]
            for j in range(16):
                col_idx = j * R + r
                for i in range(8):
                    for k in range(8):
                        row_idx = (i * 16 + j) * 8 + k
                        A_V[row_idx, col_idx] = u_r[i] * w_r[k]
        return A_V.tocsr()

    def build_A_W(U, V, R):
        # A_W shape 1024 x (8*R)
        # Row (i,j,k), Col (k', r) -> U[i,r] V[j,r] if k==k'
        import scipy.sparse as sp
        A_W = sp.lil_matrix((1024, 8 * R), dtype=float)
        for r in range(R):
            u_r = U[:, r]
            v_r = V[:, r]
            for k in range(8):
                col_idx = k * R + r
                for i in range(8):
                    for j in range(16):
                        row_idx = (i * 16 + j) * 8 + k
                        A_W[row_idx, col_idx] = u_r[i] * v_r[j]
        return A_W.tocsr()

    import scipy.sparse.linalg as spla

    def solve_als(U, V, W, R, max_iters=50):
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                return U, V, W, False
            
            # Solve U
            A_U = build_A_U(V, W, R)
            U_flat, _, _, _ = spla.lstsq(A_U, T_flat, cond=None)
            U = U_flat.reshape(8, R)
            U = np.round(U * 2) / 2
            
            # Solve V
            A_V = build_A_V(U, W, R)
            V_flat, _, _, _ = spla.lstsq(A_V, T_flat, cond=None)
            V = V_flat.reshape(16, R)
            V = np.round(V * 2) / 2
            
            # Solve W
            A_W = build_A_W(U, V, R)
            W_flat, _, _, _ = spla.lstsq(A_W, T_flat, cond=None)
            W = W_flat.reshape(8, R)
            W = np.round(W * 2) / 2
            
            err = verify(U, V, W)
            if err < 1e-9:
                return U, V, W, True
        return U, V, W, False

    best_solution = (U_28, V_28, W_28)
    best_rank = 28

    # Try Rank 27
    R = 27
    found = False
    for restart in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        if restart == 0:
            # Truncate
            U_init = U_28[:, :R].copy()
            V_init = V_28[:, :R].copy()
            W_init = W_28[:, :R].copy()
        else:
            # Random noise on truncated
            U_init = U_28[:, :R].copy() + np.random.normal(0, 0.1, (8, R))
            V_init = V_28[:, :R].copy() + np.random.normal(0, 0.1, (16, R))
            W_init = U_28[:, :R].copy() + np.random.normal(0, 0.1, (8, R))
            U_init = np.round(U_init * 2) / 2
            V_init = np.round(V_init * 2) / 2
            W_init = np.round(W_init * 2) / 2
        
        U_sol, V_sol, W_sol, success = solve_als(U_init, V_init, W_init, R, max_iters=30)
        if success:
            best_rank = R
            best_solution = (U_sol, V_sol, W_sol)
            found = True
            break

    if not found and time.time() - start_time < deadline - 10:
        # Try Rank 26
        R = 26
        for restart in range(5):
            if time.time() - start_time > deadline - 5:
                break
            U_init = U_28[:, :R].copy()
            V_init = V_28[:, :R].copy()
            W_init = W_28[:, :R].copy()
            U_sol, V_sol, W_sol, success = solve_als(U_init, V_init, W_init, R, max_iters=30)
            if success:
                best_rank = R
                best_solution = (U_sol, V_sol, W_sol)
                break

    U_out, V_out, W_out = best_solution
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
