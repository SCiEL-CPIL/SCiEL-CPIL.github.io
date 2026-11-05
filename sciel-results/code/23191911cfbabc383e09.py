"""Matrix multiplication 4,4,9 tensor decomposition - Random sparse support search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the target tensor T[i,j,k] where i in [0,16), j in [0,36), k in [0,36)
    # T[i,j,k] = 1 if k == i*m+j (i.e., output position k corresponds to A row i//m, col i%m
    # times B row i%m, col j//p... wait let me think about the convention)
    # 
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition: T[A_idx, B_idx, C_idx] = sum_r U[A_idx, r] * V[B_idx, r] * W[C_idx, r]
    # where A_idx = i*m + j (row i, col j of A), B_idx = j*p + k (row j, col k of B), C_idx = ?
    #
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So C_idx = k*n + i where i is the row of C (0..n-1) and k is the col of C (0..p-1)
    #
    # T[A_idx, B_idx, C_idx] = 1 iff A_idx encodes (i,j), B_idx encodes (j,k), C_idx encodes (i,k)
    # i.e., A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i
    #
    # So the tensor has 1s at positions where the indices are consistent.
    
    # Build tensor as flat vector of length 16*36*36 = 20736
    # T_flat[idx] where idx = A_idx * (m*p) * (p*n) + B_idx * (p*n) + C_idx
    
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    # Target tensor: T[i,j,k] = 1 if i=m+j (A row i//m, col i%m), j=j*p+k (B row j//p, col j%p), k=k*n+i (C row i, col k)
    # More precisely: for each (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
    #   A_idx = i*m + j
    #   B_idx = j*p + k  
    #   C_idx = k*n + i
    #   T[A_idx, B_idx, C_idx] = 1
    
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    target_flat = target  # already flat
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == target tensor."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Build as: for each r, outer product U[:,r] * V[:,r] * W[:,r]
        # Reshape target to (n_dim, m_dim, p_dim)
        T = np.zeros((n_dim, m_dim, p_dim))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T.flatten(), target_flat, atol=1e-10)
    
    def try_random_search(rank, max_iters, seed_offset=0):
        """Try to find a rank-rank decomposition with structured sparse supports."""
        rng = random.Random(42 + seed_offset)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n_dim, rank))
            V = np.zeros((m_dim, rank))
            W = np.zeros((p_dim, rank))
            
            for r in range(rank):
                # Structured support: each term corresponds to a (i,j,k) pattern
                # U support: rows i*m+j for some set of (i,j) pairs
                # V support: rows j*p+k for some set of (j,k) pairs  
                # W support: rows k*n+i for some set of (k,i) pairs
                
                # Pick 1-3 i indices
                n_i = rng.randint(1, 3)
                i_set = rng.sample(range(n), n_i)
                
                # For each i, pick 1-3 j indices
                u_positions = []
                v_positions = []
                w_positions = []
                
                for i in i_set:
                    n_j = rng.randint(1, 3)
                    j_set = rng.sample(range(m), n_j)
                    for j in j_set:
                        u_positions.append(i * m + j)
                        
                        # For each j, pick 1-3 k indices
                        n_k = rng.randint(1, 3)
                        k_set = rng.sample(range(p), n_k)
                        for k in k_set:
                            v_positions.append(j * p + k)
                            w_positions.append(k * n + i)
                
                # Remove duplicates
                u_positions = list(set(u_positions))
                v_positions = list(set(v_positions))
                w_positions = list(set(w_positions))
                
                # Assign random signs
                for pos in u_positions:
                    U[pos, r] = rng.choice([-1, 1])
                for pos in v_positions:
                    V[pos, r] = rng.choice([-1, 1])
                for pos in w_positions:
                    W[pos, r] = rng.choice([-1, 1])
            
            # Build D matrix: shape (n_dim * m_dim, rank)
            D = np.zeros((n_dim * m_dim, rank))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b, :] = U[a, :] * V[b, :]
            
            # Target reshaped: shape (n_dim * m_dim, p_dim)
            T_reshaped = target_flat.reshape(n_dim * m_dim, p_dim)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except Exception:
                continue
            
            T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
            if np.allclose(T_check.flatten(), target_flat, atol=1e-9):
                all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-6):
                    U_exact = np.round(U * 2) / 2.0
                    V_exact = np.round(V * 2) / 2.0
                    W_exact = np.round(W_sol * 2) / 2.0
                    T_final = np.einsum('ar,br,cr->abc', U_exact, V_exact, W_exact)
                    if np.allclose(T_final.flatten(), target_flat, atol=1e-10):
                        return U_exact, V_exact, W_exact, rank
        
        return None
    
    # Try various ranks starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But realistically, for 4x4x9, we expect rank around 80-120
    # The current best is 114 (score 1.2632)
    # Let's try ranks from 100 down to see if we can find something better
    
    best_result = None
    
    # First, let's try the known Strassen-based construction as fallback
    def construct_strassen():
        U_cols = []
        V_cols = []
        W_cols = []
        
        def add_term(u_vec, v_vec, w_vec):
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
        
        strassen_2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        def add_strassen_4x4(B_col_offset):
            def add_inner(A_blocks, B_blocks, C_blocks):
                for u_c, v_c, w_c in strassen_2x2:
                    u_vec = [0] * 16
                    v_vec = [0] * 36
                    w_vec = [0] * 36
                    
                    for loc, coeff in u_c.items():
                        li = loc // 2
                        lj = loc % 2
                        for br, bc, bc_coeff in A_blocks:
                            gi = br * 2 + li
                            gj = bc * 2 + lj
                            u_vec[gi * m + gj] += coeff * bc_coeff
                    
                    for loc, coeff in v_c.items():
                        lj = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in B_blocks:
                            gj = br * 2 + lj
                            gk = B_col_offset + bc * 2 + lk
                            v_vec[gj * p + gk] += coeff * bc_coeff
                    
                    for loc, coeff in w_c.items():
                        li = loc // 2
                        lk = loc % 2
                        for br, bc, bc_coeff in C_blocks:
                            gi = br * 2 + li
                            gk = B_col_offset + bc * 2 + lk
                            w_vec[gk * n + gi] += coeff * bc_coeff
                    
                    add_term(u_vec, v_vec, w_vec)
            
            add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
            add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
            add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
            add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
            add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
            add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
            add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        add_strassen_4x4(0)
        add_strassen_4x4(4)
        
        for i in range(4):
            for j in range(4):
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                u_vec[i * m + j] = 1
                v_vec[j * p + 8] = 1
                w_vec[8 * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
        
        rank = len(U_cols)
        U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
        V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
        W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
        return U, V, W
    
    # Try to find a better decomposition for 4x4x4 blocks
    def search_4x4x4(rank, max_iters=50, seed=42):
        n_s, m_s, p_s = 4, 4, 4
        n_d, m_d, p_d = 16, 16, 16
        target = np.zeros(n_d * m_d * p_d, dtype=float)
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    a = i * m_s + j
                    b = j * p_s + k
                    c = k * n_s + i
                    target[a * (m_d * p_d) + b * p_d + c] = 1.0
        
        rng = random.Random(seed)
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            U = np.zeros((n_d, rank))
            V = np.zeros((m_d, rank))
            W = np.zeros((p_d, rank))
            
            # Structured supports aligned with tensor structure
            for r in range(rank):
                n_i = rng.randint(1, 2)
                i_set = rng.sample(range(n_s), n_i)
                u_positions = []
                v_positions = []
                w_positions = []
                for i in i_set:
                    n_j = rng.randint(1, 2)
                    j_set = rng.sample(range(m_s), n_j)
                    for j in j_set:
                        u_positions.append(i * m_s + j)
                        n_k = rng.randint(1, 2)
                        k_set = rng.sample(range(p_s), n_k)
                        for k in k_set:
                            v_positions.append(j * p_s + k)
                            w_positions.append(k * n_s + i)
                
                u_positions = list(set(u_positions))
                v_positions = list(set(v_positions))
                w_positions = list(set(w_positions))
                
                for pos in u_positions: U[pos, r] = rng.choice([-1, 1])
                for pos in v_positions: V[pos, r] = rng.choice([-1, 1])
                for pos in w_positions: W[pos, r] = rng.choice([-1, 1])
            
            D = np.zeros((n_d * m_d, rank))
            for a in range(n_d):
                for b in range(m_d):
                    D[a * m_d + b, :] = U[a, :] * V[b, :]
            
            T_reshaped = target.reshape(n_d * m_d, p_d)
            
            try:
                X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except:
                continue
                
            T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-9):
                all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
                scaled = all_vals * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-6):
                    return np.round(U*2)/2, np.round(V*2)/2, np.round(W_sol*2)/2
        return None

    better_4x4x4 = None
    for r in [48, 47, 46, 45, 44]:
        if time.time() > deadline - 30: break
        res = search_4x4x4(r, max_iters=30, seed=1000+r)
        if res:
            better_4x4x4 = res
            break
            
    if better_4x4x4:
        U4, V4, W4 = better_4x4x4
        rank4 = U4.shape[1]
        
        U_full = np.zeros((16, rank4 * 2 + 16))
        V_full = np.zeros((36, rank4 * 2 + 16))
        W_full = np.zeros((36, rank4 * 2 + 16))
        
        # Block 1: cols 0-3
        U_full[:, 0:rank4] = U4
        V_full[:, 0:rank4] = V4
        W_full[0:16, 0:rank4] = W4
        
        # Block 2: cols 4-7
        U_full[:, rank4:2*rank4] = U4
        V4_full = np.zeros((36, rank4))
        for j in range(4):
            for k_local in range(4):
                V4_full[j*9 + 4 + k_local, :] = V4[j*4 + k_local, :]
        V_full[:, rank4:2*rank4] = V4_full
        W_full[16:32, rank4:2*rank4] = W4
        
        # Block 3: col 8
        U3 = np.zeros((16, 16))
        V3 = np.zeros((36, 16))
        W3 = np.zeros((36, 16))
        for i in range(4):
            for j in range(4):
                U3[i*4+j, j] = 1
                V3[j*9+8, j] = 1
                if i == j:
                    W3[32+i, j] = 1
                    
        U_full[:, 2*rank4:2*rank4+16] = U3
        V_full[:, 2*rank4:2*rank4+16] = V3
        W_full[:, 2*rank4:2*rank4+16] = W3
        
        # Verify
        T_final = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        # Build full target
        target_full = np.zeros(16 * 36 * 36, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    a = i * 4 + j
                    b = j * 9 + k
                    c = k * 4 + i
                    target_full[a * (36*36) + b * 36 + c] = 1.0
                    
        if np.allclose(T_final.flatten(), target_full, atol=1e-9):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to list format
        U_out = [[U[row, col] for col in range(rank)] for row in range(n * m)]
        V_out = [[V[row, col] for col in range(rank)] for row in range(m * p)]
        W_out = [[W[row, col] for col in range(rank)] for row in range(p * n)]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to Strassen construction
        U, V, W = construct_strassen()
        return {"U": U, "V": V, "W": W}
