import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_2x2x3():
        """Return U (4,9), V (6,9), W (6,9) for 2x2 * 2x3."""
        n2, m2, p2 = 2, 2, 3
        rank = 9
        U = np.zeros((n2*m2, rank))
        V = np.zeros((m2*p2, rank))
        W = np.zeros((p2*n2, rank))
        
        # Strassen terms for 2x2 * 2x2 (first two columns of B)
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[4, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[4, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[4, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Additional terms for the 3rd column
        # M8 = (A11+A12)B13
        U[0, 7] = 1; U[1, 7] = 1
        V[2, 7] = 1
        W[4, 7] = 1
        
        # M9 = (A21+A22)B23
        U[2, 8] = 1; U[3, 8] = 1
        V[5, 8] = 1
        W[5, 8] = 1
        
        return U, V, W

    def build_decomp(n, m, p):
        """Build tensor decomposition for n x m * m x p matrix multiplication."""
        if n == 1 and m == 1 and p == 1:
            return np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])
        
        if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
            return U, V, W
        
        # For 2x2 * 2x3, use the special base case
        if n == 2 and m == 2 and p == 3:
            return build_2x2x3()
        
        n2, m2, p2 = n // 2, m // 2, p // 2
        U_sub, V_sub, W_sub = build_decomp(n2, m2, p2)
        rank_sub = U_sub.shape[1]
        total_rank = 7 * rank_sub
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        terms = [
            {'A': [(11, 1), (22, 1)], 'B': [(11, 1), (22, 1)], 'C': [(11, 1), (22, 1)]},
            {'A': [(21, 1), (22, 1)], 'B': [(11, 1)], 'C': [(21, 1), (22, -1)]},
            {'A': [(11, 1)], 'B': [(12, 1), (22, -1)], 'C': [(12, 1), (22, 1)]},
            {'A': [(22, 1)], 'B': [(21, 1), (11, -1)], 'C': [(11, 1), (21, 1)]},
            {'A': [(11, 1), (12, 1)], 'B': [(22, 1)], 'C': [(11, -1), (12, 1)]},
            {'A': [(21, 1), (11, -1)], 'B': [(11, 1), (12, 1)], 'C': [(22, 1)]},
            {'A': [(12, 1), (22, -1)], 'B': [(21, 1), (22, 1)], 'C': [(11, 1)]}
        ]
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            
            for u_idx in range(n2 * m2):
                u_val = U_sub[u_idx]
                if np.all(u_val == 0):
                    continue
                i = u_idx // m2
                j = u_idx % m2
                for block, coeff in term['A']:
                    if block == 11:
                        g_u = i * m + j
                    elif block == 12:
                        g_u = i * m + (j + m2)
                    elif block == 21:
                        g_u = (i + n2) * m + j
                    elif block == 22:
                        g_u = (i + n2) * m + (j + m2)
                    else:
                        continue
                    U[g_u, base_col:base_col+rank_sub] += u_val * coeff
            
            for v_idx in range(m2 * p2):
                v_val = V_sub[v_idx]
                if np.all(v_val == 0):
                    continue
                j = v_idx // p2
                k = v_idx % p2
                for block, coeff in term['B']:
                    if block == 11:
                        g_v = j * p + k
                    elif block == 12:
                        g_v = j * p + (k + p2)
                    elif block == 21:
                        g_v = (j + m2) * p + k
                    elif block == 22:
                        g_v = (j + m2) * p + (k + p2)
                    else:
                        continue
                    V[g_v, base_col:base_col+rank_sub] += v_val * coeff
            
            for w_idx in range(p2 * n2):
                w_val = W_sub[w_idx]
                if np.all(w_val == 0):
                    continue
                k = w_idx // n2
                i = w_idx % n2
                for block, coeff in term['C']:
                    if block == 11:
                        g_w = k * n + i
                    elif block == 12:
                        g_w = (k + p2) * n + i
                    elif block == 21:
                        g_w = k * n + (i + n2)
                    elif block == 22:
                        g_w = (k + p2) * n + (i + n2)
                    else:
                        continue
                    W[g_w, base_col:base_col+rank_sub] += w_val * coeff
        
        return U, V, W
    
    U, V, W = build_decomp(n, m, p)
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    elapsed = time.time() - t0
    if elapsed > 140:
        # Fallback to schoolbook if we're running out of time
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
