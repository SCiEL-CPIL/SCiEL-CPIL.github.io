import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    rank_school = n * m * p  # 56

    # Build the multiplication tensor T as a flat vector
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0

    # Schoolbook solution
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Helper functions
    def is_half_integer(x, atol=1e-10):
        x2 = x * 2
        return np.allclose(x2, np.round(x2), atol=atol)

    def clean_half_int(x):
        x2 = x * 2
        return np.round(x2) / 2

    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)

    # Strategy: Exploit the Kronecker structure of m=4 = 2x2
    # j = j1*2 + j2 where j1, j2 in {0,1}
    # We decompose into blocks based on j1 (the "row" of the 2x2 block structure)
    
    # For each j1 in {0,1}, we have a sub-problem:
    # Multiplying 2x2 by 2x7 (for fixed j1, the j2 index)
    # But we need to be careful about the W index which couples i and k
    
    # Let's use a different approach: direct structured search
    # The tensor has a specific structure that we can exploit
    
    # Approach: Use the fact that for fixed (i,k), the tensor is constant in j
    # T[i, j, k] = 1 for all j when (i,k) is valid
    
    # This means we can factor out the j-dependence in some cases
    
    # Let's try a greedy approach with structured random vectors
    residual = T_flat.copy()
    U_list, V_list, W_list = [], [], []
    
    # Try different sparsity patterns
    max_steps = 60
    for step in range(max_steps):
        if time.time() - start_time > deadline:
            break
        
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        best_pattern = None
        
        # Try multiple random patterns
        n_trials = 30
        for trial in range(n_trials):
            if time.time() - start_time > deadline:
                break
            
            # Choose sparsity based on the structure
            # U has 8 elements (2x4), V has 28 (4x7), W has 14 (7x2)
            
            # Pattern 1: Dense-ish
            if trial < 10:
                nnz_u = random.randint(4, 8)
                nnz_v = random.randint(14, 28)
                nnz_w = random.randint(7, 14)
            # Pattern 2: Sparse
            elif trial < 20:
                nnz_u = random.randint(2, 4)
                nnz_v = random.randint(4, 10)
                nnz_w = random.randint(2, 5)
            # Pattern 3: Structured (block diagonal-like)
            else:
                # Exploit the 2x2 block structure in m
                # j = j1*2 + j2
                nnz_u = 2 * random.randint(1, 2)  # Pick j1 blocks
                nnz_v = 2 * random.randint(2, 4)  # Corresponding j blocks
                nnz_w = random.randint(2, 6)
            
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            idx_u = random.sample(range(dim_u), nnz_u)
            idx_v = random.sample(range(dim_v), nnz_v)
            idx_w = random.sample(range(dim_w), nnz_w)
            
            # Use ±1 values with some probability of 0
            u[idx_u] = np.random.choice([-1, 0, 1], nnz_u, p=[0.4, 0.2, 0.4])
            v[idx_v] = np.random.choice([-1, 0, 1], nnz_v, p=[0.4, 0.2, 0.4])
            w[idx_w] = np.random.choice([-1, 0, 1], nnz_w, p=[0.4, 0.2, 0.4])
            
            tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
            score = np.dot(tensor, residual)
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                best_pattern = (nnz_u, nnz_v, nnz_w)
        
        if best_score <= 1e-10 or best_u is None:
            break
        
        tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
        residual -= tensor
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(residual, 0, atol=1e-10):
            break
    
    # Also try a second phase with different random seeds
    if time.time() - start_time < deadline - 5 and len(U_list) < 40:
        residual2 = T_flat.copy()
        U_list2, V_list2, W_list2 = [], [], []
        
        for step in range(min(40, max_steps - len(U_list))):
            if time.time() - start_time > deadline:
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for trial in range(20):
                if time.time() - start_time > deadline:
                    break
                
                # Different sparsity emphasis
                nnz_u = random.randint(3, 6)
                nnz_v = random.randint(8, 16)
                nnz_w = random.randint(4, 10)
                
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                idx_u = random.sample(range(dim_u), nnz_u)
                idx_v = random.sample(range(dim_v), nnz_v)
                idx_w = random.sample(range(dim_w), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual2)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10 or best_u is None:
                break
            
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual2 -= tensor
            
            U_list2.append(best_u)
            V_list2.append(best_v)
            W_list2.append(best_w)
            
            if np.allclose(residual2, 0, atol=1e-10):
                break
        
        # Combine both lists
        if len(U_list2) > 0:
            U_list = U_list + U_list2
            V_list = V_list + V_list2
            W_list = W_list + W_list2
    
    # Verify the combined decomposition
    if len(U_list) > 0 and len(U_list) < best_rank:
        U_g = np.column_stack(U_list)
        V_g = np.column_stack(V_list)
        W_g = np.column_stack(W_list)
        rank_g = len(U_list)
        
        # Try least squares refinement
        M = np.einsum('it,jt,kt->ijt', U_g, V_g, W_g).reshape(-1, rank_g)
        try:
            c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            T_recon = (M @ c).reshape(dim_u, dim_v, dim_w)
            if np.allclose(T_recon, T_flat, atol=1e-10):
                # Check if c gives half-integer results
                if is_half_integer(c, atol=1e-8):
                    U_final = U_g * c[None, :]
                    V_final = V_g
                    W_final = W_g
                    U_final = clean_half_int(U_final)
                    V_final = clean_half_int(V_final)
                    W_final = clean_half_int(W_final)
                    
                    if verify_decomposition(U_final, V_final, W_final, T_flat):
                        if rank_g < best_rank:
                            best_rank = rank_g
                            best_U = U_final
                            best_V = V_final
                            best_W = W_final
        except:
            pass
    
    # Try a Kronecker product approach as well
    if time.time() - start_time < deadline - 10 and best_rank == rank_school:
        # Split m=4 into two m=2 blocks
        # For j1 in {0,1}, j2 in {0,1}: j = j1*2 + j2
        # We can try to decompose each 2x2x7 sub-tensor and combine
        
        n_sub, m_sub, p_sub = 2, 2, 7
        dim_u_sub = n_sub * m_sub  # 4
        dim_v_sub = m_sub * p_sub  # 14
        dim_w_sub = p_sub * n_sub  # 14
        
        T_sub_flat = np.zeros(dim_u_sub * dim_v_sub * dim_w_sub)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    idx_u = i * m_sub + j
                    idx_v = j * p_sub + k
                    idx_w = k * n_sub + i
                    T_sub_flat[idx_u * dim_v_sub * dim_w_sub + idx_v * dim_w_sub + idx_w] = 1.0
        
        # Schoolbook for sub: rank 28
        U_sub = np.zeros((dim_u_sub, 28))
        V_sub = np.zeros((dim_v_sub, 28))
        W_sub = np.zeros((dim_w_sub, 28))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_sub[i * m_sub + j, col] = 1
                    V_sub[j * p_sub + k, col] = 1
                    W_sub[k * n_sub + i, col] = 1
                    col += 1
        
        # Try to find a lower-rank decomposition of the sub-tensor
        residual_sub = T_sub_flat.copy()
        U_sub_list, V_sub_list, W_sub_list = [], [], []
        
        for step in range(25):
            if time.time() - start_time > deadline:
                break
            
            best_score = -np.inf
            best_u, best_v, best_w = None, None, None
            
            for trial in range(20):
                if time.time() - start_time > deadline:
                    break
                
                nnz_u = random.randint(2, 4)
                nnz_v = random.randint(4, 8)
                nnz_w = random.randint(2, 4)
                
                u = np.zeros(dim_u_sub)
                v = np.zeros(dim_v_sub)
                w = np.zeros(dim_w_sub)
                
                idx_u = random.sample(range(dim_u_sub), nnz_u)
                idx_v = random.sample(range(dim_v_sub), nnz_v)
                idx_w = random.sample(range(dim_w_sub), nnz_w)
                
                u[idx_u] = np.random.choice([-1, 1], nnz_u)
                v[idx_v] = np.random.choice([-1, 1], nnz_v)
                w[idx_w] = np.random.choice([-1, 1], nnz_w)
                
                tensor = np.einsum('i,j,k->ijk', u, v, w).flatten()
                score = np.dot(tensor, residual_sub)
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_score <= 1e-10 or best_u is None:
                break
            
            tensor = np.einsum('i,j,k->ijk', best_u, best_v, best_w).flatten()
            residual_sub -= tensor
            
            U_sub_list.append(best_u)
            V_sub_list.append(best_v)
            W_sub_list.append(best_w)
            
            if np.allclose(residual_sub, 0, atol=1e-10):
                break
        
        if len(U_sub_list) > 0 and len(U_sub_list) < 28:
            U_sub_g = np.column_stack(U_sub_list)
            V_sub_g = np.column_stack(V_sub_list)
            W_sub_g = np.column_stack(W_sub_list)
            rank_sub = len(U_sub_list)
            
            M_sub = np.einsum('it,jt,kt->ijt', U_sub_g, V_sub_g, W_sub_g).reshape(-1, rank_sub)
            try:
                c_sub, _, _, _ = np.linalg.lstsq(M_sub, T_sub_flat, rcond=None)
                T_recon_sub = (M_sub @ c_sub).reshape(dim_u_sub, dim_v_sub, dim_w_sub)
                if np.allclose(T_recon_sub, T_sub_flat, atol=1e-10):
                    if is_half_integer(c_sub, atol=1e-8):
                        U_sub_final = U_sub_g * c_sub[None, :]
                        V_sub_final = V_sub_g
                        W_sub_final = W_sub_g
                        U_sub_final = clean_half_int(U_sub_final)
                        V_sub_final = clean_half_int(V_sub_final)
                        W_sub_final = clean_half_int(W_sub_final)
                        
                        if verify_decomposition(U_sub_final, V_sub_final, W_sub_final, T_sub_flat):
                            # Build full decomposition using Kronecker structure
                            rank_full = rank_sub * 2
                            
                            U_full = np.zeros((dim_u, rank_full))
                            V_full = np.zeros((dim_v, rank_full))
                            W_full = np.zeros((dim_w, rank_full))
                            
                            for t1 in range(rank_sub):
                                for j2 in range(2):
                                    t = t1 * 2 + j2
                                    
                                    for i in range(n_sub):
                                        for j1 in range(m_sub):
                                            j = j1 * 2 + j2
                                            U_full[i * m + j, t] = U_sub_final[i * m_sub + j1, t1]
                                    
                                    for j1 in range(m_sub):
                                        for k in range(p_sub):
                                            j = j1 * 2 + j2
                                            V_full[j * p + k, t] = V_sub_final[j * p_sub + k, t1]
                                    
                                    for k in range(p_sub):
                                        for i in range(n_sub):
                                            W_full[k * n + i, t] = W_sub_final[k * n_sub + i, t1]
                            
                            if verify_decomposition(U_full, V_full, W_full, T_flat):
                                if rank_full < best_rank:
                                    best_rank = rank_full
                                    best_U = U_full
                                    best_V = V_full
                                    best_W = W_full
            except:
                pass
    
    if best_rank == rank_school:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
