import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Target Tensor
    targets_alpha = []
    targets_beta = []
    targets_gamma = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_alpha.append(a)
                targets_beta.append(b)
                targets_gamma.append(c)
                
    targets_alpha = np.array(targets_alpha)
    targets_beta = np.array(targets_beta)
    targets_gamma = np.array(targets_gamma)
    
    T_target_flat = np.zeros(size_U * size_V * size_W)
    for a, b, c in zip(targets_alpha, targets_beta, targets_gamma):
        idx = a * size_V * size_W + b * size_W + c
        T_target_flat[idx] = 1.0

    def check_solution(U, V, W):
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_full.flatten(), T_target_flat, atol=1e-9)

    def solve_for_W(U, V, R):
        A = np.zeros((size_U * size_V * size_W, size_W * R))
        for a in range(size_U):
            for b in range(size_V):
                for c in range(size_W):
                    row_idx = a * size_V * size_W + b * size_W + c
                    for r in range(R):
                        col_idx = c * R + r
                        A[row_idx, col_idx] = U[a, r] * V[b, r]
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, T_target_flat, rcond=None)
            W = x.reshape(size_W, R)
            return W
        except:
            return None

    def solve_for_U(V, W, R):
        A = np.zeros((size_U * size_V * size_W, size_U * R))
        for a in range(size_U):
            for b in range(size_V):
                for c in range(size_W):
                    row_idx = a * size_V * size_W + b * size_W + c
                    for r in range(R):
                        col_idx = a * R + r
                        A[row_idx, col_idx] = V[b, r] * W[c, r]
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, T_target_flat, rcond=None)
            U = x.reshape(size_U, R)
            return U
        except:
            return None

    def solve_for_V(U, W, R):
        A = np.zeros((size_U * size_V * size_W, size_V * R))
        for a in range(size_U):
            for b in range(size_V):
                for c in range(size_W):
                    row_idx = a * size_V * size_W + b * size_W + c
                    for r in range(R):
                        col_idx = b * R + r
                        A[row_idx, col_idx] = U[a, r] * W[c, r]
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, T_target_flat, rcond=None)
            V = x.reshape(size_V, R)
            return V
        except:
            return None

    def snap_to_half_integers(M, tol=1e-6):
        M_snapped = np.round(M * 2) / 2.0
        if np.max(np.abs(M - M_snapped)) < tol:
            return M_snapped
        return None

    start_time = time.time()
    deadline = start_time + 110.0

    for R in range(14, 21):
        if time.time() > deadline:
            break
            
        num_restarts = 200
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 100 + R)
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            
            # Case 1: Fix U, V -> Solve W
            if restart % 3 == 0:
                U = np.random.choice(vals, size=(size_U, R))
                V = np.random.choice(vals, size=(size_V, R))
                W = solve_for_W(U, V, R)
                if W is not None:
                    W_snapped = snap_to_half_integers(W)
                    if W_snapped is not None and check_solution(U, V, W_snapped):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W_snapped.tolist()}

            # Case 2: Fix V, W -> Solve U
            elif restart % 3 == 1:
                V = np.random.choice(vals, size=(size_V, R))
                W = np.random.choice(vals, size=(size_W, R))
                U = solve_for_U(V, W, R)
                if U is not None:
                    U_snapped = snap_to_half_integers(U)
                    if U_snapped is not None and check_solution(U_snapped, V, W):
                        return {"U": U_snapped.tolist(), "V": V.tolist(), "W": W.tolist()}

            # Case 3: Fix U, W -> Solve V
            else:
                U = np.random.choice(vals, size=(size_U, R))
                W = np.random.choice(vals, size=(size_W, R))
                V = solve_for_V(U, W, R)
                if V is not None:
                    V_snapped = snap_to_half_integers(V)
                    if V_snapped is not None and check_solution(U, V_snapped, W):
                        return {"U": U.tolist(), "V": V_snapped.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
