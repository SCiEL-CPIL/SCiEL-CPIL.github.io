import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def solve_sub(n_sub, m_sub, p_sub, time_limit):
        """
        Solve for the tensor decomposition of n_sub x m_sub x p_sub.
        Returns U, V, W, rank.
        """
        trivial_rank = n_sub * m_sub * p_sub
        U_trivial = np.zeros((n_sub * m_sub, trivial_rank))
        V_trivial = np.zeros((m_sub * p_sub, trivial_rank))
        W_trivial = np.zeros((p_sub * n_sub, trivial_rank))
        
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_trivial[i * m_sub + j, col] = 1
                    V_trivial[j * p_sub + k, col] = 1
                    W_trivial[k * n_sub + i, col] = 1
                    col += 1
        
        best_U = U_trivial
        best_V = V_trivial
        best_W = W_trivial
        best_rank = trivial_rank

        def _is_exact(U, V, W, n, m, p, tol=1e-8):
            nm = n * m
            mp = m * p
            pn = p * n
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1.0) > tol:
                            return False
            
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if not (i == i2 and j == j2 and k == k2):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            if abs(val) > tol:
                                return False
            return True

        def _compute_error_fast(U, V, W, n, m, p):
            err = 0.0
            nm = n * m
            mp = m * p
            pn = p * n
            R = U.shape[1]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            
            np.random.seed(42)
            for _ in range(2000):
                a = np.random.randint(nm)
                b = np.random.randint(mp)
                c = np.random.randint(pn)
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (i == i2 and j == j2 and k == k2):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += val ** 2
            return err

        # Try to find lower rank
        # For 3x3x3, we hope to find 23.
        # We try ranks from trivial_rank - 1 down to some limit.
        
        # Optimization: If p_sub == 1, just return trivial (rank 9)
        if p_sub == 1:
            return U_trivial, V_trivial, W_trivial, trivial_rank

        min_rank_to_try = 20 if (n_sub == 3 and m_sub == 3 and p_sub == 3) else 20
        
        for R in range(trivial_rank - 1, min_rank_to_try, -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            # Try multiple random initializations
            for trial in range(10):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(1000 + R * 10 + trial)
                U = np.random.randn(n_sub * m_sub, R) * 0.3
                V = np.random.randn(m_sub * p_sub, R) * 0.3
                W = np.random.randn(p_sub * n_sub, R) * 0.3
                
                prev_err = float('inf')
                converged = False
                
                for iteration in range(100):
                    if time.time() > deadline - 2:
                        break
                    
                    # Update U
                    for i in range(n_sub):
                        for j in range(m_sub):
                            A = np.zeros((p_sub, R))
                            for k in range(p_sub):
                                A[k, :] = V[j * p_sub + k, :] * W[k * n_sub + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, np.ones(p_sub), rcond=None)
                                U[i * m_sub + j, :] = x
                            except:
                                pass
                    
                    # Update V
                    for j in range(m_sub):
                        for k in range(p_sub):
                            B = np.zeros((n_sub, R))
                            for i in range(n_sub):
                                B[i, :] = U[i * m_sub + j, :] * W[k * n_sub + i, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(B, np.ones(n_sub), rcond=None)
                                V[j * p_sub + k, :] = x
                            except:
                                pass
                    
                    # Update W
                    for k in range(p_sub):
                        for i in range(n_sub):
                            C = np.zeros((m_sub, R))
                            for j in range(m_sub):
                                C[j, :] = U[i * m_sub + j, :] * V[j * p_sub + k, :]
                            try:
                                x, _, _, _ = np.linalg.lstsq(C, np.ones(m_sub), rcond=None)
                                W[k * n_sub + i, :] = x
                            except:
                                pass
                    
                    err = _compute_error_fast(U, V, W, n_sub, m_sub, p_sub)
                    if err < 1e-10:
                        converged = True
                        break
                    if abs(prev_err - err) < 1e-12:
                        break
                    prev_err = err
                
                if converged:
                    if _is_exact(U, V, W, n_sub, m_sub, p_sub):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = R
                        found = True
                        break
            
            if found:
                continue
            else:
                if time.time() > deadline - 5:
                    break

        return best_U, best_V, best_W, best_rank

    # Strategy: Split 7 into 1, 3, 3
    # Solve 3x3x1 (Trivial, Rank 9)
    U1, V1, W1, r1 = solve_sub(3, 3, 1, 1)
    
    # Solve 3x3x3 (Target 23)
    U2, V2, W2, r2 = solve_sub(3, 3, 3, deadline - time.time() - 5)
    
    # Use U2, V2, W2 for the second 3x3x3 block as well
    # Since they are identical tensors, we can reuse the same factors.
    # However, we need to ensure they are independent in the sum?
    # No, the rank of sum of identical tensors T + T is not 2*R(T) necessarily?
    # Wait. T + T = 2T. The rank of 2T is the same as rank of T.
    # BUT, the indices are different!
    # The second 3x3x3 block corresponds to different k indices in the full 3x3x7 tensor.
    # So they are NOT the same tensor in the full space.
    # They are "shifted" versions.
    # So we can reuse the U, V, W matrices, but we map them to different rows in V and W.
    # The U matrix is the same.
    
    R_total = r1 + 2 * r2
    
    U_total = np.zeros((9, R_total))
    V_total = np.zeros((21, R_total))
    W_total = np.zeros((21, R_total))
    
    # Part 1: 3x3x1 (k=0)
    U_total[:, :r1] = U1
    for j in range(3):
        idx_sub = j
        idx_full = j * 7 + 0
        V_total[idx_full, :r1] = V1[idx_sub, :]
    for i in range(3):
        idx_sub = i
        idx_full = 0 * 3 + i
        W_total[idx_full, :r1] = W1[idx_sub, :]
        
    # Part 2: 3x3x3 (k=1,2,3)
    offset2 = r1
    U_total[:, offset2:offset2+r2] = U2
    for j in range(3):
        for k_local in range(3):
            k_full = 1 + k_local
            idx_sub = j * 3 + k_local
            idx_full = j * 7 + k_full
            V_total[idx_full, offset2:offset2+r2] = V2[idx_sub, :]
    for k_local in range(3):
        k_full = 1 + k_local
        for i in range(3):
            idx_sub = k_local * 3 + i
            idx_full = k_full * 3 + i
            W_total[idx_full, offset2:offset2+r2] = W2[idx_sub, :]
            
    # Part 3: 3x3x3 (k=4,5,6)
    offset3 = r1 + r2
    U_total[:, offset3:offset3+r2] = U2
    for j in range(3):
        for k_local in range(3):
            k_full = 4 + k_local
            idx_sub = j * 3 + k_local
            idx_full = j * 7 + k_full
            V_total[idx_full, offset3:offset3+r2] = V2[idx_sub, :]
    for k_local in range(3):
        k_full = 4 + k_local
        for i in range(3):
            idx_sub = k_local * 3 + i
            idx_full = k_full * 3 + i
            W_total[idx_full, offset3:offset3+r2] = W2[idx_sub, :]
            
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
