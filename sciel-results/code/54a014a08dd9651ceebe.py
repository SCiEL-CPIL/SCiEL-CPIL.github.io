"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook rank is n*m*p = 4*4*9 = 144.
    rank = n * m * p
    
    # Build schoolbook solution
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank
    
    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Condition: sum_r U[i*m+j,r] V[j'*p+k,r] W[k'*n+i',r] = delta(i,i') delta(j,j') delta(k,k')
        """
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        # T[i, j, s, k, t, i'] = sum_r U[i,j,r] V[s,k,r] W[t,i',r]
        T = np.einsum('ijr,skr,tir->ijstkr', U_3d, V_3d, W_3d, optimize=True)
        
        # Check diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i, j, j, k, k, i] - 1.0) > 1e-9:
                        return False
        
        # Check off-diagonal entries
        for i in range(n):
            for j in range(m):
                for s in range(m):
                    for k in range(p):
                        for t in range(p):
                            for ip in range(n):
                                if not (i == ip and j == s and k == t):
                                    if abs(T[i, j, s, k, t, ip]) > 1e-9:
                                        return False
        return True
    
    def try_merge(U, V, W, r1, r2):
        """
        Try to merge two rank-1 terms (columns r1 and r2) into one.
        Returns (U_new, V_new, W_new) if successful, else None.
        """
        R = U.shape[1]
        if r1 == r2:
            return None
        
        a = U[:, r1].copy()
        b = V[:, r1].copy()
        c = W[:, r1].copy()
        d = U[:, r2].copy()
        e = V[:, r2].copy()
        f = W[:, r2].copy()
        
        # Check if a and d are proportional
        nz_a = np.nonzero(a)[0]
        nz_d = np.nonzero(d)[0]
        
        if len(nz_a) == 0 or len(nz_d) == 0:
            return None
        
        if abs(a[nz_a[0]]) < 1e-12:
            return None
        alpha = d[nz_a[0]] / a[nz_a[0]]
        
        if not np.allclose(d, alpha * a, atol=1e-9):
            return None
        
        # Similarly for b and e
        nz_b = np.nonzero(b)[0]
        nz_e = np.nonzero(e)[0]
        if len(nz_b) == 0 or len(nz_e) == 0:
            return None
        if abs(b[nz_b[0]]) < 1e-12:
            return None
        beta = e[nz_b[0]] / b[nz_b[0]]
        if not np.allclose(e, beta * b, atol=1e-9):
            return None
        
        # Similarly for c and f
        nz_c = np.nonzero(c)[0]
        nz_f = np.nonzero(f)[0]
        if len(nz_c) == 0 or len(nz_f) == 0:
            return None
        if abs(c[nz_c[0]]) < 1e-12:
            return None
        gamma = f[nz_c[0]] / c[nz_c[0]]
        if not np.allclose(f, gamma * c, atol=1e-9):
            return None
        
        scale = 1.0 + alpha * beta * gamma
        if abs(scale) < 1e-12:
            # The terms cancel out! Remove both.
            mask = np.ones(R, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            return (U_new, V_new, W_new)
        
        # Merge: new term = scale * (a, b, c)
        keep = min(r1, r2)
        mask = np.ones(R, dtype=bool)
        mask[r1] = False
        mask[r2] = False
        
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        U_new = np.insert(U_new, keep, a, axis=1)
        V_new = np.insert(V_new, keep, b, axis=1)
        W_new = np.insert(W_new, keep, scale * c, axis=1)
        
        return (U_new, V_new, W_new)
    
    # Try to reduce rank by merging pairs
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    cur_rank = rank
    
    for round_num in range(5):
        if time.time() > deadline:
            break
        
        merged = False
        for r1 in range(cur_rank - 1):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, cur_rank):
                result = try_merge(U_cur, V_cur, W_cur, r1, r2)
                if result is not None:
                    U_cur, V_cur, W_cur = result
                    cur_rank -= 1
                    merged = True
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": U_cur.tolist(),
                            "V": V_cur.tolist(),
                            "W": W_cur.tolist()
                        }
                    break
            if merged and time.time() > deadline:
                break
    
    # Also try random search from current best
    values = [1, -1, 0.5, -0.5]
    
    for target_rank in range(min(best_rank - 1, rank - 1), max(120, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                num_nonzero_U = random.randint(1, 3)
                rows_U = random.sample(range(n * m), num_nonzero_U)
                for row in rows_U:
                    U_trial[row, r] = random.choice(values)
                    
                num_nonzero_V = random.randint(1, 3)
                rows_V = random.sample(range(m * p), num_nonzero_V)
                for row in rows_V:
                    V_trial[row, r] = random.choice(values)
                    
                num_nonzero_W = random.randint(1, 3)
                rows_W = random.sample(range(p * n), num_nonzero_W)
                for row in rows_W:
                    W_trial[row, r] = random.choice(values)
            
            if verify(U_trial, V_trial, W_trial):
                best_rank = target_rank
                best_result = {
                    "U": U_trial.tolist(),
                    "V": V_trial.tolist(),
                    "W": W_trial.tolist()
                }
                break
    
    return best_result
# EVOLVE-BLOCK-END
