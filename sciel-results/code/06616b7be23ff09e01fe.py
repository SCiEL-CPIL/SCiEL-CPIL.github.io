"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Strategy: Decompose into 3 blocks of 2x3 by 3x3
    # Each block has rank 15 (known result). Total rank 45.
    # We construct the rank-15 decomposition explicitly.
    
    n_block, m_block, p_block = 2, 3, 3
    target_rank_b = 15
    
    def construct_block_decomposition():
        """Construct a rank-15 decomposition for 2x3 by 3x3 matrix multiplication."""
        # Indices: U[i*m+j, d], V[j*p+k, d], W[k*n+i, d]
        # We need: sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] = 1 if valid, 0 otherwise
        
        # Known construction for 2x3 by 3x3 with rank 15:
        # We use the fact that this can be built from the 2x2 by 2x2 rank-7 decomposition
        # plus additional terms for the extra row/column.
        
        # Let's use a direct construction based on the structure.
        # The 18 schoolbook terms are (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
        # We need to reduce from 18 to 15 by merging 3 pairs.
        
        # A known approach: use the rank-7 decomposition of 2x2 by 2x2 for the 
        # (i in {0,1}, j in {0,1}, k in {0,1}) sub-block, then handle the rest.
        
        # Actually, let me try a direct algebraic construction.
        # For 2x3 by 3x3, we can write:
        # C_{k,i} = sum_j A_{i,j} B_{j,k}
        
        # Let me try to find a rank-15 decomposition by solving a system.
        dim_u_b = 6
        dim_v_b = 9
        dim_w_b = 6
        rank_b = 15
        
        # Build the full constraint system
        # For each (a,b,c) in {0..5}x{0..8}x{0..5}:
        #   sum_d U[a,d] V[b,d] W[c,d] = T[a,b,c]
        # where T[a,b,c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j', k=k', i=i'
        
        # This is a nonlinear system. Let's use alternating minimization with 
        # a good initialization.
        
        # Initialize with schoolbook and then try to merge terms.
        # Start with 18 terms, then try to eliminate 3 by finding linear dependencies.
        
        # Alternative: use a structured initialization based on the 
        # known rank-7 decomposition for 2x2 by 2x2.
        
        # Rank-7 decomposition for 2x2 by 2x2 (Strassen-like):
        # U, V, W are 4x7 matrices.
        # Let me use the standard rank-7 decomposition.
        
        # For 2x2 by 2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # Rank-7 decomposition (one known form):
        # This is getting complex. Let me try a different approach.
        
        # Use a random initialization with multiple restarts and 
        # alternating minimization, but with a more efficient solver.
        
        for trial in range(50):
            if time.time() > deadline - 10:
                return None
            
            rng = np.random.RandomState(trial * 137 + 42)
            
            # Initialize with a mix of structured and random
            U = np.zeros((dim_u_b, rank_b))
            V = np.zeros((dim_v_b, rank_b))
            W = np.zeros((dim_w_b, rank_b))
            
            # First 18 columns: schoolbook-like but permuted
            # Actually, let's just use random init with small values
            U = rng.randn(dim_u_b, rank_b) * 0.3
            V = rng.randn(dim_v_b, rank_b) * 0.3
            W = rng.randn(dim_w_b, rank_b) * 0.3
            
            # Alternating minimization with efficient solving
            for it in range(200):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U: for each a, sum_d U[a,d] * (V[b,d]*W[c,d]) = T[a,b,c]
                # This is a linear system in U for fixed V, W.
                # We can vectorize: for each a, the equation involves all (b,c) pairs.
                
                # Build the system efficiently
                # For U: unknowns are U[a,d] for a in 0..5, d in 0..14 (90 unknowns)
                # Equations: for each (a,b,c), sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                
                # Vectorize: create matrix A where A[(a,b,c), (a',d)] = V[b,d]*W[c,d] if a==a' else 0
                # This is a 324 x 90 matrix (sparse-ish but we'll use dense)
                
                # More efficient: for each a, we have a 9x6 = 54 equation subsystem
                # with 15 unknowns (U[a,0..14])
                
                U_new = np.zeros_like(U)
                for a in range(dim_u_b):
                    # For this a, equations: for each (b,c), sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                    # Build 54x15 system
                    A_sub = np.zeros((dim_v_b * dim_w_b, rank_b))
                    b_sub = np.zeros(dim_v_b * dim_w_b)
                    idx = 0
                    for b in range(dim_v_b):
                        for c in range(dim_w_b):
                            # T[a,b,c]
                            i = a // m_block
                            j = a % m_block
                            j2 = b // p_block
                            k = b % p_block
                            k2 = c // n_block
                            i2 = c % n_block
                            if j == j2 and k == k2 and i == i2:
                                b_sub[idx] = 1.0
                            else:
                                b_sub[idx] = 0.0
                            A_sub[idx, :] = V[b, :] * W[c, :]
                            idx += 1
                    try:
                        U_new[a, :], _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    except:
                        continue
                
                U = U_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for b in range(dim_v_b):
                    A_sub = np.zeros((dim_u_b * dim_w_b, rank_b))
                    b_sub = np.zeros(dim_u_b * dim_w_b)
                    idx = 0
                    for a in range(dim_u_b):
                        for c in range(dim_w_b):
                            i = a // m_block
                            j = a % m_block
                            j2 = b // p_block
                            k = b % p_block
                            k2 = c // n_block
                            i2 = c % n_block
                            if j == j2 and k == k2 and i == i2:
                                b_sub[idx] = 1.0
                            else:
                                b_sub[idx] = 0.0
                            A_sub[idx, :] = U[a, :] * W[c, :]
                            idx += 1
                    try:
                        V_new[b, :], _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    except:
                        continue
                V = V_new
                
                # Solve for W
                W_new = np.zeros_like(W)
                for c in range(dim_w_b):
                    A_sub = np.zeros((dim_u_b * dim_v_b, rank_b))
                    b_sub = np.zeros(dim_u_b * dim_v_b)
                    idx = 0
                    for a in range(dim_u_b):
                        for b in range(dim_v_b):
                            i = a // m_block
                            j = a % m_block
                            j2 = b // p_block
                            k = b % p_block
                            k2 = c // n_block
                            i2 = c % n_block
                            if j == j2 and k == k2 and i == i2:
                                b_sub[idx] = 1.0
                            else:
                                b_sub[idx] = 0.0
                            A_sub[idx, :] = U[a, :] * V[b, :]
                            idx += 1
                    try:
                        W_new[c, :], _, _, _ = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    except:
                        continue
                W = W_new
                
                # Check error
                err = 0.0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            a = i*m_block+j
                            b_idx = j*p_block+k
                            c_idx = k*n_block+i
                            val = np.dot(U[a,:], V[b_idx,:] * W[c_idx,:])
                            err += (val - 1)**2
                # Sample zero constraints
                rng_check = np.random.RandomState(it)
                for _ in range(50):
                    a = rng_check.randint(dim_u_b)
                    b = rng_check.randint(dim_v_b)
                    c = rng_check.randint(dim_w_b)
                    i = a // m_block
                    j = a % m_block
                    j2 = b // p_block
                    k = b % p_block
                    k2 = c // n_block
                    i2 = c % n_block
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.dot(U[a,:], V[b,:] * W[c,:])
                        err += val**2
                
                if err < 1e-12:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_r = 0.0
                    for i in range(n_block):
                        for j in range(m_block):
                            for k in range(p_block):
                                a = i*m_block+j
                                b_idx = j*p_block+k
                                c_idx = k*n_block+i
                                val = np.dot(U_r[a,:], V_r[b_idx,:] * W_r[c_idx,:])
                                err_r += (val - 1)**2
                    for a in range(dim_u_b):
                        for b in range(dim_v_b):
                            for c in range(dim_w_b):
                                i = a // m_block
                                j = a % m_block
                                j2 = b // p_block
                                k = b % p_block
                                k2 = c // n_block
                                i2 = c % n_block
                                if not (j == j2 and k == k2 and i == i2):
                                    val = np.dot(U_r[a,:], V_r[b,:] * W_r[c,:])
                                    err_r += val**2
                            
                    if err_r < 1e-10:
                        return U_r, V_r, W_r
        
        return None
    
    # Try to find the block decomposition
    block_decomp = construct_block_decomposition()
    
    if block_decomp is not None:
        U_b, V_b, W_b = block_decomp
        # Construct full decomposition
        full_rank = target_rank_b * 3
        U_full = np.zeros((dim_u, full_rank))
        V_full = np.zeros((dim_v, full_rank))
        W_full = np.zeros((dim_w, full_rank))
        
        for block_idx in range(3):
            k_offset = block_idx * p_block
            for c in range(target_rank_b):
                full_c = block_idx * target_rank_b + c
                U_full[:, full_c] = U_b[:, c]
                
                for j in range(m_block):
                    for k in range(p_block):
                        v_block_idx = j * p_block + k
                        v_full_idx = j * p + (k_offset + k)
                        V_full[v_full_idx, full_c] = V_b[v_block_idx, c]
                        
                for k in range(p_block):
                    for i in range(n_block):
                        w_block_idx = k * n_block + i
                        w_full_idx = (k_offset + k) * n + i
                        W_full[w_full_idx, full_c] = W_b[w_block_idx, c]
        
        # Verify full decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    val = np.dot(U_full[a,:], V_full[b,:] * W_full[c,:])
                    if abs(val - 1) > 1e-10:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            # Check zero entries
            for a in range(dim_u):
                for b in range(dim_v):
                    for c in range(dim_w):
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if not (j == j2 and k == k2 and i == i2):
                            val = np.dot(U_full[a,:], V_full[b,:] * W_full[c,:])
                            if abs(val) > 1e-10:
                                valid = False
                                break
                    if not valid: break
                if not valid: break
            
            if valid:
                best_result = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
                best_rank = full_rank
    
    return best_result
# EVOLVE-BLOCK-END
