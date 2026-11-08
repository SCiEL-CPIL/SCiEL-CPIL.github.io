import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 100.0 # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor entries
    # T[i*m + j, j*p + k, k*n + i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school
    
    # Precompute vectorized target slices
    # M_a is a matrix of shape (mp, np_)
    # vec(M_a) is a vector of length mp * np_
    M_slices_a = [np.zeros(mp * np_) for _ in range(nm)]
    M_slices_b = [np.zeros(nm * np_) for _ in range(mp)]
    M_slices_c = [np.zeros(nm * mp) for _ in range(np_)]
    
    for a, b, c in target_indices:
        # For slice a: index in M_a is b * np_ + c
        M_slices_a[a][b * np_ + c] = 1.0
        # For slice b: index in M_b is a * np_ + c
        M_slices_b[b][a * np_ + c] = 1.0
        # For slice c: index in M_c is a * mp + b
        M_slices_c[c][a * mp + b] = 1.0

    def snap_to_half_int(matrix):
        snapped = []
        for row in matrix:
            new_row = []
            for val in row:
                v2 = val * 2
                r2 = round(v2)
                new_row.append(r2 / 2.0)
            snapped.append(new_row)
        return snapped

    def check_validity(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (np_, R)
        # T[abc] = sum_r U[ar] V[br] W[cr]
        # Check target indices
        for a, b, c in target_indices:
            s = 0.0
            for r in range(R):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s - 1.0) > 1e-6:
                return False
        
        # Check zero entries
        # To speed up, we can use einsum to compute full tensor and check
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates an array of size 6x15x10 = 900. Very fast.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if (a, b, c) in set(target_indices):
                        continue
                    if abs(T_rec[a, b, c]) > 1e-6:
                        return False
        return True

    def check_snapped_validity(U, V, W, R):
        Uf = np.array(U)
        Vf = np.array(V)
        Wf = np.array(W)
        T_rec = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        
        for a, b, c in target_indices:
            if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                return False
        
        target_set = set(target_indices)
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if (a, b, c) in target_set:
                        continue
                    if abs(T_rec[a, b, c]) > 1e-9:
                        return False
        return True

    # ALS solver
    def solve_rank(R, num_restarts=50):
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                return None
            
            # Random initialization
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(np_, R)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(20):
                if time.time() - start_time > timeout:
                    return None
                
                # Update U
                for a in range(nm):
                    # Build K_a: (mp*np_, R)
                    # Column r is W[:, r] tensor V[:, r]
                    # vec(W[:, r] V[:, r]^T) = V[:, r] tensor W[:, r] ?
                    # Let's check:
                    # (V[:, r] W[:, r]^T)[b, c] = V[b, r] W[c, r]
                    # vec(M) index b*np_ + c.
                    # So K_a[:, r] = V[:, r] tensor W[:, r] is wrong order?
                    # np.kron(V[:, r], W[:, r]) has index b*np_ + c?
                    # np.kron(A, B) with A size m, B size n.
                    # Index i*n + j corresponds to A[i] B[j].
                    # So np.kron(V[:, r], W[:, r]) gives V[b] W[c] at index b*np_ + c.
                    # This matches M_a index b*np_ + c.
                    K_a = np.zeros((mp * np_, R))
                    for r in range(R):
                        K_a[:, r] = np.kron(V[:, r], W[:, r])
                    
                    # Solve K_a u_a = M_slices_a[a]
                    u_a, residuals, rank, sv = np.linalg.lstsq(K_a, M_slices_a[a], rcond=None)
                    U[a, :] = u_a
                
                # Update V
                for b in range(mp):
                    # M_b index a * np_ + c
                    # Equation: sum_r v_br U[ar] W[cr] = M_b[ac]
                    # vec(M_b) index a * np_ + c
                    # K_b[:, r] should be U[:, r] tensor W[:, r]
                    # np.kron(U[:, r], W[:, r]) index a*np_ + c -> U[a] W[c]. Correct.
                    K_b = np.zeros((nm * np_, R))
                    for r in range(R):
                        K_b[:, r] = np.kron(U[:, r], W[:, r])
                    
                    v_b, residuals, rank, sv = np.linalg.lstsq(K_b, M_slices_b[b], rcond=None)
                    V[b, :] = v_b
                
                # Update W
                for c in range(np_):
                    # M_c index a * mp + b
                    # Equation: sum_r w_cr U[ar] V[br] = M_c[ab]
                    # vec(M_c) index a * mp + b
                    # K_c[:, r] should be U[:, r] tensor V[:, r]
                    # np.kron(U[:, r], V[:, r]) index a*mp + b -> U[a] V[b]. Correct.
                    K_c = np.zeros((nm * mp, R))
                    for r in range(R):
                        K_c[:, r] = np.kron(U[:, r], V[:, r])
                    
                    w_c, residuals, rank, sv = np.linalg.lstsq(K_c, M_slices_c[c], rcond=None)
                    W[c, :] = w_c
            
            # Check validity
            if check_validity(U, V, W, R):
                # Snap to half integers
                U_snap = snap_to_half_int(U.tolist())
                V_snap = snap_to_half_int(V.tolist())
                W_snap = snap_to_half_int(W.tolist())
                
                if check_snapped_validity(U_snap, V_snap, W_snap, R):
                    return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # Try ranks from 1 to 30
    for R in range(1, R_school + 1):
        if time.time() - start_time > timeout:
            break
            
        result = solve_rank(R, num_restarts=20)
        if result is not None:
            return result
            
    return best_result
