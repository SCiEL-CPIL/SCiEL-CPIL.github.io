import time
import random
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if (i_a, j_a, k_a) matches (i_b, j_b, k_b) appropriately
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Condition: i_a == i_c, j_a == j_b, k_b == k_c
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    def tensor_error(U, V, W):
        err = np.einsum('ar,br,cr->abc', U, V, W) - T
        return np.linalg.norm(err)

    def als_solve(U, V, W, R, max_iter=50, tol=1e-12):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        for _ in range(max_iter):
            # Solve U
            new_U = np.zeros_like(U)
            # Precompute M_U? No, V and W change.
            # M_U is (m*p * p*n, R) where M_U[b_idx, r] = V[b,r] * W[c,r] ?
            # Equation: sum_r U[a,r] (V[b,r] W[c,r]) = T[a,b,c]
            # For fixed a, variables U[a, :].
            # Matrix A_a has rows (b,c), cols r. A_a[(b,c), r] = V[b,r] W[c,r].
            # This matrix is the same for all a!
            # Let's compute it once per iteration.
            # Shape of V: (m*p, R). Shape of W: (p*n, R).
            # We need outer product of V and W along the R dimension?
            # V[:, r] is a vector of size m*p.
            # W[:, r] is a vector of size p*n.
            # Outer product is (m*p, p*n).
            # Stack for all r: (R, m*p, p*n).
            # Transpose to (m*p, p*n, R).
            # Flatten first two dims: (m*p*p*n, R).
            
            # Efficient way:
            # V is (21, R), W is (14, R).
            # We want M such that M.flatten() corresponds to (b,c,r).
            # M[b, c, r] = V[b, r] * W[c, r].
            # This is element-wise multiplication of V[:, None, :] and W[None, :, :].
            M_U = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
            
            for a in range(n * m):
                t = T[a].flatten()
                # Solve M_U x = t
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_U, t, rcond=None)
                new_U[a] = sol
            U = new_U

            # Solve V
            # Equation: sum_r V[b,r] (U[a,r] W[c,r]) = T[a,b,c]
            # For fixed b, variables V[b, :].
            # Matrix A_b has rows (a,c), cols r. A_b[(a,c), r] = U[a,r] W[c,r].
            M_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
            new_V = np.zeros_like(V)
            for b in range(m * p):
                t = T[:, b, :].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_V, t, rcond=None)
                new_V[b] = sol
            V = new_V

            # Solve W
            # Equation: sum_r W[c,r] (U[a,r] V[b,r]) = T[a,b,c]
            # For fixed c, variables W[c, :].
            # Matrix A_c has rows (a,b), cols r. A_c[(a,b), r] = U[a,r] V[b,r].
            M_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
            new_W = np.zeros_like(W)
            for c in range(p * n):
                t = T[:, :, c].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_W, t, rcond=None)
                new_W[c] = sol
            W = new_W

            err = tensor_error(U, V, W)
            if err < tol:
                break
                
        return U, V, W, err

    def svd_init(R):
        # Matricize T to (n*m, m*p*p*n)
        T_mat = T.reshape(n * m, -1)
        U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=False)
        # Take top R
        U_init = U_svd[:, :R]
        # The rest is in Vh. We need to split Vh into V and W.
        # This is not straightforward.
        # Alternative: Random init for V and W, then solve U.
        # Or use the SVD of the "flattened" tensor in a different way.
        
        # Let's just use random init for V and W, but scaled.
        V_init = np.random.randn(m * p, R) * 0.1
        W_init = np.random.randn(p * n, R) * 0.1
        # Solve for U exactly
        M_U = (V_init[:, None, :] * W_init[None, :, :]).reshape(m * p * p * n, R)
        U_init = np.zeros((n * m, R))
        for a in range(n * m):
            t = T[a].flatten()
            sol, residuals, rank_s, sv = np.linalg.lstsq(M_U, t, rcond=None)
            U_init[a] = sol
        return U_init, V_init, W_init

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize
            U, V, W = svd_init(target_rank)
            
            # Refine
            U, V, W, err = als_solve(U, V, W, target_rank, max_iter=30)
            
            if err < 1e-9:
                # Try to quantize to integers or half-integers
                U_q = np.round(U * 2) / 2
                V_q = np.round(V * 2) / 2
                W_q = np.round(W * 2) / 2
                
                # Verify quantized solution
                U_list = U_q.tolist()
                V_list = V_q.tolist()
                W_list = W_q.tolist()
                
                if verify_exact(U_list, V_list, W_list, n, m, p):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best
