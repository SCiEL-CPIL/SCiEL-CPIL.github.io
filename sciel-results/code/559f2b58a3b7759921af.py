import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # First, build the rank-28 Strassen solution
    rank_init = 28
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    strassen_products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            for (i, j), coeff in prod['A'].items():
                U[i * m + j, t] = coeff
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V[j * p + k_global, t] = coeff
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W[k_global * n + i, t] = coeff
    
    # Fast and correct verification
    def verify(U_mat, V_mat, W_mat):
        """Verify that sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] equals the multiplication tensor."""
        rank = U_mat.shape[1]
        nm, mp, pn = n * m, m * p, p * n
        
        # Build expected tensor
        T_expected = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute decomposition tensor efficiently
        T_decomp = np.zeros((nm, mp, pn), dtype=float)
        for t in range(rank):
            T_decomp += np.einsum('i,j,k->ijk', U_mat[:,t], V_mat[:,t], W_mat[:,t])
        
        return np.allclose(T_decomp, T_expected, atol=1e-10)
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find linear dependencies using SVD
    nm, mp, pn = n * m, m * p, p * n
    num_constraints = nm * mp * pn
    
    # Build system matrix: A_sys[constraint, t] = U[a,t]*V[b,t]*W[c,t]
    A_sys = np.zeros((num_constraints, rank_init), dtype=float)
    
    idx = 0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                for t in range(rank_init):
                    A_sys[idx, t] = U[a, t] * V[b, t] * W[c, t]
                idx += 1
    
    # Find null space of A_sys
    try:
        _, s, vh = np.linalg.svd(A_sys, full_matrices=False)
        tol = 1e-10
        null_mask = s < tol
        null_dim = np.sum(null_mask)
        
        if null_dim > 0:
            # We have null space vectors. Use them to reduce rank.
            # Each null space vector c gives: sum_t c[t] * U_t ⊗ V_t ⊗ W_t = 0
            # We can use this to eliminate terms.
            
            for ni in range(null_dim):
                if time.time() > deadline - 10:
                    break
                c = vh[null_mask[ni], :]
                max_t = np.argmax(np.abs(c))
                
                if abs(c[max_t]) > 1e-10:
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    
                    for t in range(rank_init):
                        if t != max_t:
                            U_new[:, t] -= c[t] / c[max_t] * U[:, max_t]
                            V_new[:, t] -= c[t] / c[max_t] * V[:, max_t]
                            W_new[:, t] -= c[t] / c[max_t] * W[:, max_t]
                    
                    keep = [t for t in range(rank_init) if t != max_t]
                    U_new = U_new[:, keep]
                    V_new = V_new[:, keep]
                    W_new = W_new[:, keep]
                    
                    if verify(U_new, V_new, W_new):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    except Exception:
        pass
    
    # Try removing single terms
    for t_remove in range(rank_init):
        if time.time() > deadline - 10:
            break
        
        U_test = np.delete(U, t_remove, axis=1)
        V_test = np.delete(V, t_remove, axis=1)
        W_test = np.delete(W, t_remove, axis=1)
        
        if verify(U_test, V_test, W_test):
            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
    
    # Try removing pairs of terms
    for t1 in range(rank_init):
        if time.time() > deadline - 5:
            break
        for t2 in range(t1 + 1, rank_init):
            if time.time() > deadline - 5:
                break
            
            indices_to_remove = [t1, t2]
            U_test = np.delete(U, indices_to_remove, axis=1)
            V_test = np.delete(V, indices_to_remove, axis=1)
            W_test = np.delete(W, indices_to_remove, axis=1)
            
            if verify(U_test, W_test, W_test):
                return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
    
    # Return the rank-28 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
