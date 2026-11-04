import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # 1. Construct the target tensor T
    # T shape: (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    # Schoolbook factors as fallback
    rank_school = nm * mp # Wait, schoolbook rank is n*m*p = 36
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_factors = {
        'U': U_school.tolist(),
        'V': V_school.tolist(),
        'W': W_school.tolist()
    }

    # Try to find lower rank using SVD truncation + rounding
    # Theoretical lower bound for 3x3x4 is likely > 20.
    # Try ranks from 25 to 35.
    
    for target_rank in range(25, 36):
        if time.time() > deadline:
            break
            
        # Matricize T
        T1 = T.reshape(nm, mp * pn)
        T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        # SVD
        try:
            U1, S1, Vh1 = np.linalg.svd(T1, full_matrices=False)
            U2, S2, Vh2 = np.linalg.svd(T2, full_matrices=False)
            U3, S3, Vh3 = np.linalg.svd(T3, full_matrices=False)
        except:
            continue
            
        # Take top target_rank components
        U = U1[:, :target_rank]
        V = U2[:, :target_rank]
        W = U3[:, :target_rank]
        
        # The factors from SVD are not directly the tensor factors.
        # T1 = U1 S1 Vh1. T1 = U @ (V \otimes W)^T ? No.
        # This approach is tricky because the SVD of matricizations doesn't directly give the CP factors.
        # We need to solve for the CP factors.
        
        # Let's stick to the ALS approach but with a better initialization or different restarts.
        # Or, simply return schoolbook if ALS fails quickly.
        
        # Given the time constraint and the difficulty of finding < 36,
        # and the fact that schoolbook is 36,
        # let's try a few random restarts of ALS for rank 32-35.
        
        found = False
        for r in range(10): # Fewer restarts to save time
            if time.time() > deadline:
                break
                
            np.random.seed(r) # Reproducible
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for it in range(20): # Fewer iterations
                # Update U
                M = np.zeros((mp * pn, target_rank))
                for r_idx in range(target_rank):
                    M[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
                T_mat = T.reshape(nm, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                M_v = np.zeros((nm * pn, target_rank))
                for r_idx in range(target_rank):
                    M_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
                T_mat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_v.T, T_mat_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                M_w = np.zeros((nm * mp, target_rank))
                for r_idx in range(target_rank):
                    M_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
                T_mat_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_w.T, T_mat_w.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Check error
                if it % 5 == 0:
                    T_rec = np.zeros_like(T)
                    for r_idx in range(target_rank):
                        T_rec += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    err = np.linalg.norm(T - T_rec)
                    if err < 1e-8:
                        # Try to snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        T_rec_snap = np.zeros_like(T)
                        for r_idx in range(target_rank):
                            T_rec_snap += np.einsum('i,j,k->ijk', U_snap[:, r_idx], V_snap[:, r_idx], W_snap[:, r_idx])
                        
                        if np.allclose(T, T_rec_snap, atol=1e-9):
                            U_out = [[int(x) if x == int(x) else float(x) for x in row] for row in U_snap]
                            V_out = [[int(x) if x == int(x) else float(x) for x in row] for row in V_snap]
                            W_out = [[int(x) if x == int(x) else float(x) for x in row] for row in W_snap]
                            
                            best_rank = target_rank
                            best_factors = {'U': U_out, 'V': V_out, 'W': W_out}
                            found = True
                            break
            if found:
                break

    return best_factors
