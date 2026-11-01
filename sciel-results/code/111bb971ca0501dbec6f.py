import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute valid tensor entries
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    n_am = n * m  # 8
    n_bm = m * p  # 20
    n_cm = p * n  # 10
    
    # Build list of (a, b, c, value) for nonzero entries
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c, 1.0))
    
    # Total number of (a,b) pairs
    total_ab = n_am * n_bm  # 160
    
    def try_rank(r, max_iters=500):
        """Try to find rank-r decomposition via alternating least squares."""
        np.random.seed(42)
        U = np.random.randn(n_am, r) * 0.5
        V = np.random.randn(n_bm, r) * 0.5
        W = np.random.randn(n_cm, r) * 0.5
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update W: for each c, solve least squares
            for c in range(n_cm):
                # Build 160 x r matrix M where M[ab_idx, r] = U[a,r] * V[b,r]
                # and rhs[ab_idx] = T[a,b,c]
                M = np.zeros((total_ab, r))
                rhs = np.zeros(total_ab)
                ab_idx = 0
                for a in range(n_am):
                    for b in range(n_bm):
                        M[ab_idx, :] = U[a, :] * V[b, :]
                        # Check if (a,b,c) is a valid nonzero entry
                        # a = i*m+j, b = j'*p+k, c = k'*n+i'
                        # Valid if j == j' and k == k' and i == i'
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        i_c = c % n
                        k_c = c // n
                        if j_a == j_b and k_b == k_c and i_a == i_c:
                            rhs[ab_idx] = 1.0
                        ab_idx += 1
                
                # Solve least squares
                W[c, :], residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
            
            # Update V: for each b, solve least squares
            for b in range(n_bm):
                M = np.zeros((total_ab, r))
                rhs = np.zeros(total_ab)
                ab_idx = 0
                for a in range(n_am):
                    for bb in range(n_bm):
                        M[ab_idx, :] = U[a, :] * W[(bb % p) * n + (a // m), :]  # c = (bb%p)*n + (a//m)
                        # Wait, this is wrong. Let me reconsider.
                        ab_idx += 1
                
                # Actually, for fixed b and V, the equation is:
                # For each (a, c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed b, V[b,:] is the unknown.
                # sum_r (U[a,r] * W[c,r]) * V[b,r] = T[a,b,c]
                # This is linear in V[b,:].
                # The (a,c) pairs: a ranges over 0..7, c ranges over 0..9
                # But T[a,b,c] is only nonzero for specific (a,c) given b.
                
                # Let me redo: for fixed b, unknown is V[b,:] (r values)
                # Equations: for each (a,c): sum_r U[a,r]*W[c,r]*V[b,r] = T[a,b,c]
                # Number of (a,c) pairs: 8*10 = 80
                # But many have T=0.
                
                M2 = np.zeros((n_am * n_cm, r))
                rhs2 = np.zeros(n_am * n_cm)
                ac_idx = 0
                for a in range(n_am):
                    for c in range(n_cm):
                        M2[ac_idx, :] = U[a, :] * W[c, :]
                        # Check if T[a,b,c] = 1
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        i_c = c % n
                        k_c = c // n
                        if j_a == j_b and k_b == k_c and i_a == i_c:
                            rhs2[ac_idx] = 1.0
                        ac_idx += 1
                
                V[b, :], _, _, _ = np.linalg.lstsq(M2, rhs2, rcond=None)
            
            # Update U: for each a, solve least squares
            for a in range(n_am):
                M3 = np.zeros((n_bm * n_cm, r))
                rhs3 = np.zeros(n_bm * n_cm)
                bc_idx = 0
                for b in range(n_bm):
                    for c in range(n_cm):
                        M3[bc_idx, :] = V[b, :] * W[c, :]
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        i_c = c % n
                        k_c = c // n
                        if j_a == j_b and k_b == k_c and i_a == i_c:
                            rhs3[bc_idx] = 1.0
                        bc_idx += 1
                
                U[a, :], _, _, _ = np.linalg.lstsq(M3, rhs3, rcond=None)
            
            # Check convergence: compute residual
            if iteration % 50 == 0:
                max_res = 0.0
                for (a, b, c, val) in nonzero_entries:
                    res = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                    max_res = max(max_res, abs(res - val))
                # Also check zero entries (sample)
                for _ in range(100):
                    a = np.random.randint(n_am)
                    b = np.random.randint(n_bm)
                    c = np.random.randint(n_cm)
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_c = c % n
                    k_c = c // n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        res = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                        max_res = max(max_res, abs(res))
                if max_res < 1e-8:
                    return (U, V, W)
        
        return None
    
    # Try rank 35
    result = try_rank(35, max_iters=300)
    
    if result is not None:
        U, V, W = result
        # Verify
        max_res = 0.0
        for (a, b, c, val) in nonzero_entries:
            res = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
            max_res = max(max_res, abs(res - val))
        for _ in range(500):
            a = np.random.randint(n_am)
            b = np.random.randint(n_bm)
            c = np.random.randint(n_cm)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            i_c = c % n
            k_c = c // n
            if not (j_a == j_b and k_b == k_c and i_a == i_c):
                res = np.dot(U[a, :], np.dot(V[b, :], W[c, :]))
                max_res = max(max_res, abs(res))
        
        if max_res < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 36
    rank = 36
    U = np.zeros((n_am, rank), dtype=np.float64)
    V = np.zeros((n_bm, rank), dtype=np.float64)
    W = np.zeros((n_cm, rank), dtype=np.float64)
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i

    def strassen_block(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices
        
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1
        W[w_idx(*w11)][col] -= 1
        col += 1
        
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        W[w_idx(*w01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        W[w_idx(*w00)][col] += 1
        W[w_idx(*w10)][col] += 1
        col += 1
        
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] -= 1
        W[w_idx(*w01)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    def schoolbook_2x2x1(y_indices, z_indices, w_indices):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z0, z1 = z_indices
        w0, w1 = w_indices
        
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w_idx(*w0)][col] += 1
        col += 1
        
        U[u_idx(*y10)][col] += 1
        V[v_idx(*z0)][col] += 1
        W[w_idx(*w1)][col] += 1
        col += 1
        
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z1)][col] += 1
        W[w_idx(*w1)][col] += 1
        col += 1

    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    y5 = [(0,0), (0,1), (1,0), (1,1)]
    z5 = [(0,4), (1,4)]
    w5 = [(4,0), (4,1)]
    schoolbook_2x2x1(y5, z5, w5)
    
    y6 = [(0,2), (0,3), (1,2), (1,3)]
    z6 = [(2,4), (3,4)]
    w6 = [(4,0), (4,1)]
    schoolbook_2x2x1(y6, z6, w6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
