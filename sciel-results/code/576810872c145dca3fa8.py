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
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # Using einsum: 'ar,br,cr->abc'
        err = np.einsum('ar,br,cr->abc', U, V, W) - T
        return np.linalg.norm(err)

    def als_solve(U, V, W, R, max_iter=20, tol=1e-12):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Solve for U:
        # For each a, solve A_a u_a = t_a
        # A_a has shape (m*p * p*n, R)? No.
        # Equation: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # For fixed a, unknowns are U[a, 0..R-1].
        # Variables r.
        # For each pair (b,c), we have an equation:
        # sum_r U[a,r] (V[b,r] * W[c,r]) = T[a,b,c]
        # So matrix A_a has rows indexed by (b,c) and cols by r.
        # A_a[b, c, r] = V[b, r] * W[c, r]
        
        # We can form the full matrix for all a at once?
        # A is shape (n*m, m*p, p*n, R)
        # We need to solve for U (n*m, R)
        
        # Let's iterate over a
        for _ in range(max_iter):
            # Solve U
            new_U = np.zeros_like(U)
            for a in range(n * m):
                # Construct A_a: (m*p * p*n, R)
                # V is (m*p, R), W is (p*n, R)
                # We need outer product of V[:, r] and W[:, r] for each r?
                # Actually, A_a[b_idx, r] = V[b_idx, r] * W[c_idx, r] is not right.
                # The equation is: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This is not a standard matrix multiplication A x = b where A is constant.
                # The coefficient of U[a,r] is V[b,r] * W[c,r].
                # So for a fixed a, we have a linear system in variables U[a, 0..R-1].
                # The matrix M_a has entries M_a[(b,c), r] = V[b,r] * W[c,r].
                # The vector t_a has entries t_a[(b,c)] = T[a,b,c].
                
                # Reshape V and W to facilitate this?
                # V is (m*p, R). W is (p*n, R).
                # We can compute M_a by broadcasting?
                # M_a is (m*p, p*n, R) -> flatten first two dims.
                # M_a = (V[:, None, :] * W[None, :, :]).reshape(m*p*p*n, R)
                # This is expensive to do for every a, but V and W are fixed.
                # M is the same for all a!
                # M = (V[:, None, :] * W[None, :, :]).reshape(m*p*p*n, R)
                
                M = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
                t = T[a].flatten()
                # Solve M x = t
                sol, residuals, rank_s, sv = np.linalg.lstsq(M, t, rcond=None)
                new_U[a] = sol
                
            U = new_U
            
            # Solve V
            # Equation: sum_r V[b,r] * (U[a,r] * W[c,r]) = T[a,b,c]
            # For fixed b, unknowns V[b, 0..R-1].
            # Coeff of V[b,r] is U[a,r] * W[c,r].
            # M_b[(a,c), r] = U[a,r] * W[c,r]
            new_V = np.zeros_like(V)
            for b in range(m * p):
                M = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
                t = T[:, b, :].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M, t, rcond=None)
                new_V[b] = sol
            V = new_V
            
            # Solve W
            # Equation: sum_r W[c,r] * (U[a,r] * V[b,r]) = T[a,b,c]
            # For fixed c, unknowns W[c, 0..R-1].
            # Coeff of W[c,r] is U[a,r] * V[b,r].
            # M_c[(a,b), r] = U[a,r] * V[b,r]
            new_W = np.zeros_like(W)
            for c in range(p * n):
                M = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
                t = T[:, :, c].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M, t, rcond=None)
                new_W[c] = sol
            W = new_W
            
            err = tensor_error(U, V, W)
            if err < tol:
                break
                
        return U, V, W, err

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        # Try a few random initializations
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            U, V, W, err = als_solve(U, V, W, target_rank, max_iter=30)
            
            if err < 1e-9:
                # Try to quantize to integers or half-integers
                # Round to nearest 0.5
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
