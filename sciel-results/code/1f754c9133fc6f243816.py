import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # In our factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # So T is a 3D array of shape (nm, mp, pn) with T[idx_u, idx_v, idx_w] = 1
    # where idx_u = i*m+j, idx_v = j*p+k, idx_w = k*n+i
    # But note: not all (idx_u, idx_v, idx_w) are valid; only those where j matches and i,k match.
    # Actually, for the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    # The flattening is: U has index (i,j), V has index (j,k), W has index (k,i).
    # So the tensor is defined on the "compatible" triples. But in the standard tensor
    # formulation, we define a 3D tensor of shape (nm, mp, pn) where most entries are 0,
    # and T[(i,j), (j,k), (k,i)] = 1 for valid i,j,k.
    
    # Let's build the target tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[:,c] V[:,c] W[:,c] (outer product sum) equals T.
    # This is a rank-r tensor decomposition.
    
    # Strategy: try ranks from low to high, use ALS to find a decomposition.
    # Start with a reasonable lower bound. Schoolbook is 100. Strassen blocks give 96.
    # Let's try from 90 up to 100, and also try lower.
    
    best_result = None
    best_rank = 100
    
    # First, let's get the schoolbook solution as a fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    
    # Verify schoolbook
    T_recon = np.einsum('ic,jc,kc->ijk', U_school, V_school, W_school)
    err = np.max(np.abs(T_recon - T))
    if err > 1e-10:
        # Should not happen
        pass
    
    # Now try to find lower rank decompositions using ALS
    # We'll try ranks from 90 to 99, and for each, run ALS for a few iterations.
    
    for r in range(90, 100):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize with random values
        np.random.seed(42)  # Fixed seed for reproducibility
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # ALS: alternate minimizing over U, V, W
        for iteration in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Fix V, W; solve for U
            # T[i, :, :] = sum_c U[i,c] * (V[:,c] outer W[:,c])
            # Let M_VW[:, i, j] = sum_c V[j,c] * W[i,c] ... this is getting messy.
            # Better: reshape T to 2D for each mode.
            
            # Mode-1 unfolding: T_1[i, (j,k)] = T[i, j, k]
            # T_1 = U * (W ⊙ V) where ⊙ is Khatri-Rao product
            # So U = T_1 @ (W ⊙ V)^+ 
            VW = np.kron(W, V)  # shape (pn*mp, r)
            # Actually, Khatri-Rao product of W (pn x r) and V (mp x r) is (pn*mp x r)
            # where the row index is (i_w, i_v) and the entry is W[i_w, c] * V[i_v, c]
            # But we need it in the right order. Let's use a different approach.
            
            # For mode-1: T_1[i, (j,k)] = T[i, j, k]
            # T_1 = U @ (W ⊙ V)^T
            # So U = T_1 @ (W ⊙ V)^+
            
            # Build Khatri-Rao product properly
            # (W ⊙ V)[(i_w, i_v), c] = W[i_w, c] * V[i_v, c]
            # We need to map (i, j, k) to (i, (j, k)) for mode-1
            # T_1[i, (j,k)] = T[i, j, k]
            
            T_1 = T.reshape(nm, mp * pn)  # This is not quite right because the ordering matters
            
            # Actually, let's just use the direct formula:
            # For each i, T[i, :, :] = sum_c U[i, c] * (V[:, c] outer W[:, c])
            # Let B_c = V[:, c] outer W[:, c], shape (mp, pn)
            # T[i, j, k] = sum_c U[i, c] * B_c[j, k]
            
            # So for fixed i, T[i, :, :] = sum_c U[i, c] * B_c
            # This is a linear system: T[i] = U[i, :] @ (B_1, B_2, ..., B_r)
            # where (B_1, ..., B_r) is a matrix of shape (mp*pn, r)
            
            # Build the matrix B of shape (mp*pn, r)
            # B[(j, k), c] = V[j, c] * W[k, c]
            B = np.zeros((mp * pn, r))
            for c in range(r):
                B[:, c] = np.outer(V[:, c], W[:, c]).flatten()
            
            T_1 = T.reshape(nm, mp * pn)
            # U = T_1 @ B^+
            try:
                U = T_1 @ np.linalg.pinv(B)
            except:
                break
            
            # Fix U, W; solve for V
            # T[:, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
            # For fixed j, T[:, j, k] = sum_c V[j, c] * (U[:, c] outer W[k, c])... 
            # Better: T_2[j, (i, k)] = T[i, j, k]
            # T_2 = V @ (W ⊙ U)^T
            C = np.zeros((pn * nm, r))
            for c in range(r):
                C[:, c] = np.outer(W[:, c], U[:, c]).flatten()
            
            T_2 = np.zeros((mp, pn * nm))
            for j in range(mp):
                for i in range(nm):
                    for k in range(pn):
                        T_2[j, k * nm + i] = T[i, j, k]
            
            try:
                V = T_2 @ np.linalg.pinv(C)
            except:
                break
            
            # Fix U, V; solve for W
            # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
            # For fixed k, T[i, j, k] = sum_c W[k, c] * (U[i, c] * V[j, c])
            # T_3[k, (i, j)] = T[i, j, k]
            D = np.zeros((nm * mp, r))
            for c in range(r):
                D[:, c] = np.outer(U[:, c], V[:, c]).flatten()
            
            T_3 = np.zeros((pn, nm * mp))
            for k in range(pn):
                for i in range(nm):
                    for j in range(mp):
                        T_3[k, j * nm + i] = T[i, j, k]
            
            try:
                W = T_3 @ np.linalg.pinv(D)
            except:
                break
            
            # Check convergence
            T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-8:
                # Found a solution! Check if entries are close to half-integers
                # If so, round them
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_recon_round = np.einsum('ic,jc,kc->ijk', U_round, V_round, W_round)
                err_round = np.max(np.abs(T_recon_round - T))
                
                if err_round < 1e-10:
                    best_result = (U_round, V_round, W_round, r)
                    best_rank = r
                    break
                else:
                    # Use the unrounded version if it's exact enough
                    best_result = (U, V, W, r)
                    best_rank = r
                    break
            if time.time() > deadline - 5.0:
                break
        
        if best_rank < 100:
            break
    
    if best_result is not None:
        U_best, V_best, W_best, r_best = best_result
        # Convert to lists
        U_list = U_best.tolist()
        V_list = V_best.tolist()
        W_list = W_best.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
