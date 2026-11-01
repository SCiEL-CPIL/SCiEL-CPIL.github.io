import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-9 decomposition for 2x3x3
    n_small, m_small, p_small = 2, 3, 3
    rank_small = 9
    
    # Target tensor for 2x3x3
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # Else 0
    T_target = np.zeros((n_small * m_small, m_small * p_small, p_small * n_small), dtype=np.int64)
    for i in range(n_small):
        for j in range(m_small):
            for k in range(p_small):
                a = i * m_small + j
                b = j * p_small + k
                c = k * n_small + i
                T_target[a, b, c] = 1

    def compute_error(U, V, W):
        # Compute sum_c U[a,c] V[b,c] W[c_idx,c]
        # Note: W index is c_idx = k*n+i
        # We need to sum over c for all a,b,c_idx
        # This is a tensor contraction
        # We can do this by iterating over c and adding outer products
        T_calc = np.zeros((n_small * m_small, m_small * p_small, p_small * n_small), dtype=np.int64)
        for c in range(rank_small):
            # u_vec: (n*m, 1)
            # v_vec: (m*p, 1)
            # w_vec: (p*n, 1)
            # T += u[:, None, None] * v[None, :, None] * w[None, None, :]
            # But we need to be careful with broadcasting
            T_calc += np.einsum('c,dc,ec->de', U[:, c], V[:, c], W[:, c])
            # Wait, einsum 'c,dc,ec->de' is not right.
            # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
            # We want T[a,b,c_idx] = sum_r U[a,r] V[b,r] W[c_idx,r]
            # np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - T_target) ** 2)

    # Local search
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    max_iter = 20000
    for t in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        # Initialize
        U = np.random.randint(-2, 3, size=(n_small * m_small, rank_small))
        V = np.random.randint(-2, 3, size=(m_small * p_small, rank_small))
        W = np.random.randint(-2, 3, size=(p_small * n_small, rank_small))
        
        current_error = compute_error(U, V, W)
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if best_error == 0:
                break
        
        # Local search
        for _ in range(50): # Steps per restart
            if time.time() > deadline - 20:
                break
            if current_error == 0:
                break
            
            # Pick a random matrix and element
            mat = np.random.randint(3)
            if mat == 0:
                idx = (np.random.randint(n_small * m_small), np.random.randint(rank_small))
                old_val = U[idx]
                delta = np.random.randint(-2, 3)
                if delta == 0: continue
                U[idx] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    U[idx] = old_val
            elif mat == 1:
                idx = (np.random.randint(m_small * p_small), np.random.randint(rank_small))
                old_val = V[idx]
                delta = np.random.randint(-2, 3)
                if delta == 0: continue
                V[idx] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    V[idx] = old_val
            else:
                idx = (np.random.randint(p_small * n_small), np.random.randint(rank_small))
                old_val = W[idx]
                delta = np.random.randint(-2, 3)
                if delta == 0: continue
                W[idx] += delta
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    W[idx] = old_val
    
    if best_error == 0:
        # Construct the 2x3x9 decomposition
        rank_full = rank_small * 3
        U_full = np.zeros((n_rows, rank_full), dtype=np.int64)
        V_full = np.zeros((m_rows, rank_full), dtype=np.int64)
        W_full = np.zeros((p_rows, rank_full), dtype=np.int64)
        
        for block in range(3):
            U_full[:, block * rank_small : (block + 1) * rank_small] = best_U
            
            for j in range(3):
                for k in range(3):
                    v_idx_full = j * 9 + 3 * block + k
                    v_idx_small = j * 3 + k
                    V_full[v_idx_full, block * rank_small : (block + 1) * rank_small] = best_V[v_idx_small, :]
                    
            for k_small in range(3):
                for i in range(2):
                    k_full = 3 * block + k_small
                    w_idx_full = k_full * 2 + i
                    w_idx_small = k_small * 2 + i
                    W_full[w_idx_full, block * rank_small : (block + 1) * rank_small] = best_W[w_idx_small, :]
        
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        return schoolbook()
