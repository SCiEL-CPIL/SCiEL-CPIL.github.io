"""Matrix multiplication 4,4,9 tensor decomposition - Random sparse support search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build the target tensor
    n_dim = n * m  # 16
    m_dim = m * p  # 36
    p_dim = p * n  # 36
    
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                flat_idx = a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx
                target[flat_idx] = 1.0
    
    target_flat = target
    
    def verify_decomposition(U, V, W, rank):
        T = np.zeros((n_dim, m_dim, p_dim))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T.flatten(), target_flat, atol=1e-10)

    # Construct the known rank 114 solution (2x Strassen 4x4x4 + 1x Schoolbook 4x4x1)
    def construct_rank_114():
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
        U = np.array([[U_cols[col][row] for col in range(rank)] for row in range(n * m)])
        V = np.array([[V_cols[col][row] for col in range(rank)] for row in range(m * p)])
        W = np.array([[W_cols[col][row] for col in range(rank)] for row in range(p * n)])
        return U, V, W

    U_114, V_114, W_114 = construct_rank_114()
    
    # Check if 114 is valid
    if not verify_decomposition(U_114, V_114, W_114, 114):
        # Fallback to simple schoolbook if Strassen construction fails (should not happen)
        U = np.zeros((16, 144))
        V = np.zeros((36, 144))
        W = np.zeros((36, 144))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i*4+j, idx] = 1
                    V[j*9+k, idx] = 1
                    W[k*4+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to find rank 113
    # Strategy: Random sparse search biased by the 114 solution's supports
    target_rank = 113
    max_iters = 50
    rng = random.Random(42)
    
    # Extract active indices from 114 solution
    u_active = set()
    v_active = set()
    w_active = set()
    for r in range(114):
        for i in range(16):
            if U_114[i, r] != 0: u_active.add(i)
        for j in range(36):
            if V_114[j, r] != 0: v_active.add(j)
        for k in range(36):
            if W_114[k, r] != 0: w_active.add(k)
    
    u_pool = list(u_active)
    v_pool = list(v_active)
    w_pool = list(w_active)
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        U = np.zeros((16, target_rank))
        V = np.zeros((36, target_rank))
        W = np.zeros((36, target_rank))
        
        # Generate random supports from the pool
        # We want sparse columns, similar to the 114 solution
        # Average sparsity in 114 solution:
        # U: 16*114 nonzeros? No, each col has ~4 nonzeros.
        # Let's use 3-5 nonzeros.
        
        for r in range(target_rank):
            u_supp = rng.sample(u_pool, rng.randint(3, 5))
            v_supp = rng.sample(v_pool, rng.randint(3, 5))
            w_supp = rng.sample(w_pool, rng.randint(3, 5))
            
            for pos in u_supp: U[pos, r] = rng.choice([-1, 1])
            for pos in v_supp: V[pos, r] = rng.choice([-1, 1])
            for pos in w_supp: W[pos, r] = rng.choice([-1, 1])
        
        # Solve for W
        D = np.zeros((16 * 36, target_rank))
        for a in range(16):
            for b in range(36):
                D[a * 36 + b, :] = U[a, :] * V[b, :]
        
        T_reshaped = target_flat.reshape(16 * 36, 36)
        
        try:
            X, residuals, rank_sys, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
            W_sol = X.T
        except Exception:
            continue
        
        # Check exactness
        T_check = np.einsum('ar,br,cr->abc', U, V, W_sol)
        if np.allclose(T_check.flatten(), target_flat, atol=1e-9):
            # Check if values are close to integers/half-integers
            all_vals = np.concatenate([U.flatten(), V.flatten(), W_sol.flatten()])
            scaled = all_vals * 2
            if np.allclose(scaled, np.round(scaled), atol=1e-6):
                U_exact = np.round(U * 2) / 2.0
                V_exact = np.round(V * 2) / 2.0
                W_exact = np.round(W_sol * 2) / 2.0
                
                if verify_decomposition(U_exact, V_exact, W_exact, target_rank):
                    return {"U": U_exact.tolist(), "V": V_exact.tolist(), "W": W_exact.tolist()}

    # Return rank 114 solution
    return {"U": U_114.tolist(), "V": V_114.tolist(), "W": W_114.tolist()}
