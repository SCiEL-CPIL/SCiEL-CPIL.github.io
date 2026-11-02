import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    def calc_error(U, V, W, r):
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        return np.sum((T - 1.0)**2)

    def verify(U, V, W, r):
        return calc_error(U, V, W, r) < 1e-9

    def local_search(U, V, W, r, time_limit):
        U_r = U.reshape(n, m, r).copy()
        V_r = V.reshape(m, p, r).copy()
        W_r = W.reshape(p, n, r).copy()
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        E = np.sum((T - 1.0)**2)
        
        if E < 1e-9:
            return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # U
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        old_val = U_r[i, j, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-0.5, 0.5]:
                            updates = delta * (V_r[j, :, c] * W_r[:, i, c])
                            T_slice = T[i, j, :]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            U_r[i, j, c] += best_delta
                            T[i, j, :] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E
            
            # V
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        old_val = V_r[j, k, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-0.5, 0.5]:
                            updates = delta * (U_r[:, j, c] * W_r[k, :, c])
                            T_slice = T[:, j, k]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            V_r[j, k, c] += best_delta
                            T[:, j, k] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

            # W
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        old_val = W_r[k, i, c]
                        best_delta = 0
                        best_E = E
                        best_T_slice = None
                        for delta in [-0.5, 0.5]:
                            updates = delta * (U_r[i, :, c] * V_r[:, k, c])
                            T_slice = T[i, :, k]
                            T_new_slice = T_slice + updates
                            E_slice = np.sum((T_slice - 1.0)**2)
                            E_new_slice = np.sum((T_new_slice - 1.0)**2)
                            E_new = E - E_slice + E_new_slice
                            if E_new < best_E:
                                best_E = E_new
                                best_delta = delta
                                best_T_slice = T_new_slice
                        if best_delta != 0:
                            W_r[k, i, c] += best_delta
                            T[i, :, k] = best_T_slice
                            E = best_E
                            improved = True
                            if E < 1e-9:
                                return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

        return U_r.reshape(-1), V_r.reshape(-1), W_r.reshape(-1), E

    def to_half_int(x):
        return round(x * 2) / 2

    best = None
    best_rank = None
    
    # Try decreasing ranks from 59 down to 50
    # We know 60 is always valid (schoolbook).
    # We try to find a lower one.
    for r in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
            
        # Multiple random restarts
        found = False
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(restart * 1000 + r)
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Alternating minimization with vectorized updates
            # We will update U, then V, then W.
            
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                # For each i, j: A[i,j] @ U[i*m+j, :] = 1_p
                # A[i,j] is p x r, A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
                # We can compute A for all i,j at once?
                # V is (m, p, r), W is (p, n, r)
                # A[i,j] = sum_c? No, A is a matrix of coefficients.
                # A_{i,j,k,c} = V_{j,k,c} W_{k,i,c}
                # We want to solve for U_{i,j,:}
                
                # Let's construct the linear system for all i,j simultaneously?
                # The system is: sum_c U_{i,j,c} * (V_{j,k,c} * W_{k,i,c}) = 1 for all k.
                # This is a linear system: M_{i,j} @ u_{i,j} = 1
                # where M_{i,j} is p x r.
                # M_{i,j,k,c} = V_{j,k,c} * W_{k,i,c}
                
                # We can compute M for all i,j using einsum
                # M shape: (n, m, p, r)
                # M[i,j,k,c] = V[j,k,c] * W[k,i,c]
                # V_reshaped: (m, p, r)
                # W_reshaped: (p, n, r)
                # We need to align indices.
                # V depends on j, k, c. W depends on k, i, c.
                # M[i,j,k,c] = V_reshaped[j,k,c] * W_reshaped[k,i,c]
                
                try:
                    # Compute M: (n, m, p, r)
                    # V_reshaped: (m, p, r) -> expand to (1, m, p, r)
                    # W_reshaped: (p, n, r) -> expand to (n, 1, p, r) ... wait, indices are (k, i, c)
                    # So W_reshaped[k, i, c]. We need to transpose to (i, k, c) or use einsum.
                    # M[i,j,k,c] = V_reshaped[j,k,c] * W_reshaped[k,i,c]
                    # einsum: 'jkc,kic->ijkc'
                    M = np.einsum('jkc,kic->ijkc', V_reshaped, W_reshaped, optimize=True)
                    
                    # Now solve M[i,j] @ U[i*m+j, :] = 1_p
                    # M is (n, m, p, r). U is (n, m, r).
                    # We can reshape M to (n*m, p, r) and U to (n*m, r)
                    M_flat = M.reshape(n*m, p, r)
                    U_flat = U.reshape(n*m, r)
                    b = np.ones((n*m, p))
                    
                    # Solve linear least squares for each of the n*m systems
                    # np.linalg.lstsq doesn't support batched input directly in older versions, 
                    # but we can loop or use a trick.
                    # Since n*m is small (12), a loop is fine.
                    for idx in range(n*m):
                        try:
                            U_flat[idx, :] = np.linalg.lstsq(M_flat[idx], b[idx], rcond=None)[0]
                        except:
                            pass
                    U = U_flat.reshape(n*m, r)
                    
                except Exception as e:
                    pass
                
                # Update V
                # For each j, k: A[j,k] @ V[j*p+k, :] = 1_n
                # A[j,k] is n x r. A[j,k,i,c] = U[i*m+j, c] * W[k*n+i, c]
                # U_reshaped: (n, m, r). W_reshaped: (p, n, r).
                # A[j,k,i,c] = U_reshaped[i,j,c] * W_reshaped[k,i,c]
                try:
                    # M_V shape: (m, p, n, r)
                    # einsum: 'ijc,kic->jkic'
                    M_V = np.einsum('ijc,kic->jkic', U_reshaped, W_reshaped, optimize=True)
                    M_V_flat = M_V.reshape(m*p, n, r)
                    V_flat = V.reshape(m*p, r)
                    b_V = np.ones((m*p, n))
                    
                    for idx in range(m*p):
                        try:
                            V_flat[idx, :] = np.linalg.lstsq(M_V_flat[idx], b_V[idx], rcond=None)[0]
                        except:
                            pass
                    V = V_flat.reshape(m*p, r)
                    
                except Exception as e:
                    pass
                
                # Update W
                # For each k, i: A[k,i] @ W[k*n+i, :] = 1_m
                # A[k,i] is m x r. A[k,i,j,c] = U[i*m+j, c] * V[j*p+k, c]
                # U_reshaped: (n, m, r). V_reshaped: (m, p, r).
                # A[k,i,j,c] = U_reshaped[i,j,c] * V_reshaped[j,k,c]
                try:
                    # M_W shape: (p, n, m, r)
                    # einsum: 'ijc,jkc->kimc' ? No, indices: i, j, c and j, k, c -> i, j, k, c
                    # We need A[k,i,j,c] = U[i,j,c] * V[j,k,c]
                    # einsum: 'ijc,jkc->kijc'
                    M_W = np.einsum('ijc,jkc->kijc', U_reshaped, V_reshaped, optimize=True)
                    M_W_flat = M_W.reshape(p*n, m, r)
                    W_flat = W.reshape(p*n, r)
                    b_W = np.ones((p*n, m))
                    
                    for idx in range(p*n):
                        try:
                            W_flat[idx, :] = np.linalg.lstsq(M_W_flat[idx], b_W[idx], rcond=None)[0]
                        except:
                            pass
                    W = W_flat.reshape(p*n, r)
                    
                except Exception as e:
                    pass
                
                # Check convergence
                if it % 10 == 0:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    err = calc_error(U_r, V_r, W_r, r)
                    
                    if err < 1e-9:
                        if best_rank is None or r < best_rank:
                            best_rank = r
                            best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                        found = True
                        break
                    elif err < 1.0: # Potential candidate for local search
                        # Perform local search
                        time_limit_local = time.time() + 2.0
                        U_r, V_r, W_r, err = local_search(U_r, V_r, W_r, r, time_limit_local)
                        if err < 1e-9:
                            if best_rank is None or r < best_rank:
                                best_rank = r
                                best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                            found = True
                            break
                            
            if found:
                break
                
    if best is not None:
        U_list, V_list, W_list = best
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
