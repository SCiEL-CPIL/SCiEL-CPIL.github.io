import numpy as np
import scipy.optimize as so
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    # Helper to build target tensor
    def build_tensor(n, m, p):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    T[a, b, w] = 1.0
        indices = np.argwhere(T == 1)
        return indices

    # Helper to verify solution
    def verify_solution(U, V, W, indices, atol=1e-9):
        idx_a = indices[:, 0]
        idx_b = indices[:, 1]
        idx_w = indices[:, 2]
        vals = np.einsum('ar,br,wr->ab', U[idx_a], V[idx_b], W[idx_w])
        return np.allclose(vals, 1.0, atol=atol)

    # Helper to try rank r
    def try_rank(n, m, p, r, time_budget, init_U=None, init_V=None, init_W=None):
        dim_A = n * m
        dim_B = m * p
        dim_C = p * n
        indices = build_tensor(n, m, p)
        idx_a = indices[:, 0]
        idx_b = indices[:, 1]
        idx_w = indices[:, 2]

        def res_func(t):
            U = t[:dim_A * r].reshape(dim_A, r)
            V = t[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
            W = t[dim_A * r + dim_B * r :].reshape(dim_C, r)
            vals = np.einsum('ar,br,wr->ab', U[idx_a], V[idx_b], W[idx_w])
            return vals - 1.0

        start_time_local = time.time()
        deadline_local = start_time_local + time_budget
        
        # Initializations
        inits = []
        if init_U is not None:
            inits.append((init_U, init_V, init_W))
            
        # Random initializations
        for i in range(10):
            if time.time() > deadline_local:
                break
            np.random.seed(42 + i)
            U0 = np.random.randn(dim_A, r) * 0.5
            V0 = np.random.randn(dim_B, r) * 0.5
            W0 = np.random.randn(dim_C, r) * 0.5
            inits.append((U0, V0, W0))

        best_sol = None
        best_cost = float('inf')

        for U0, V0, W0 in inits:
            if time.time() > deadline_local:
                break
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                res = so.least_squares(res_func, theta0, method='trf', bounds=(-10, 10), 
                                       ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
                if res.cost < best_cost:
                    best_cost = res.cost
                    if res.cost < 1e-10:
                        best_sol = res.x
                        break
            except Exception:
                pass

        if best_sol is None:
            return None

        U_sol = best_sol[:dim_A * r].reshape(dim_A, r)
        V_sol = best_sol[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W_sol = best_sol[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        # Round to half-integers
        U_int = np.round(U_sol * 2) / 2
        V_int = np.round(V_sol * 2) / 2
        W_int = np.round(W_sol * 2) / 2
        
        if verify_solution(U_int, V_int, W_int, indices):
            return U_int, V_int, W_int
            
        return None

    # Stage 1: Find rank-9 decomposition for M_{2,4,2}
    # n=2, m=4, p=2
    n2, m2, p2 = 2, 4, 2
    rank2 = 9
    time_budget_2 = 10.0
    sol_242 = try_rank(n2, m2, p2, rank2, time_budget_2)
    
    if sol_242 is not None:
        U_242, V_242, W_242 = sol_242
        # Construct initial guess for M_{2,4,6}
        # M_{2,4,6} is 3 blocks of M_{2,4,2}
        # U is same for all blocks?
        # In the tensor decomposition, U vectors are associated with A indices.
        # If we use the same U vectors, we need to be careful.
        # The terms are u_r \otimes v_r \otimes w_r.
        # For block 1: u_r \otimes v_{1,r} \otimes w_{1,r}
        # For block 2: u_r \otimes v_{2,r} \otimes w_{2,r}
        # For block 3: u_r \otimes v_{3,r} \otimes w_{3,r}
        # Total 27 terms.
        # U matrix for M_{2,4,6} will have 27 columns.
        # Columns 0-8: U_242
        # Columns 9-17: U_242
        # Columns 18-26: U_242
        
        U_246 = np.hstack([U_242, U_242, U_242])
        
        # V matrix for M_{2,4,6}
        # V_242 has shape (24, 9).
        # Block 1 V: support on B indices 0-7 (rows 0,1 of B? No, B is 4x2, indices 0-7).
        # Wait, B for M_{2,4,2} is 4x2. Indices 0..7.
        # For M_{2,4,6}, B is 4x6. Indices 0..23.
        # Block 1 uses B cols 0,1. Indices 0..7.
        # Block 2 uses B cols 2,3. Indices 12..19? No.
        # B index is j*p + k.
        # For p=6:
        # Row 0: 0,1,2,3,4,5
        # Row 1: 6,7,8,9,10,11
        # Row 2: 12,13,14,15,16,17
        # Row 3: 18,19,20,21,22,23
        # Block 1 (cols 0,1): indices 0,1,6,7,12,13,18,19.
        # Block 2 (cols 2,3): indices 2,3,8,9,14,15,20,21.
        # Block 3 (cols 4,5): indices 4,5,10,11,16,17,22,23.
        
        # V_242 indices are 0..7 corresponding to B indices 0..7 in the 4x2 case.
        # In 4x2 case:
        # Row 0: 0,1
        # Row 1: 2,3
        # Row 2: 4,5
        # Row 3: 6,7
        
        # Mapping:
        # Block 1: 0->0, 1->1, 2->6, 3->7, 4->12, 5->13, 6->18, 7->19
        # Block 2: 0->2, 1->3, 2->8, 3->9, 4->14, 5->15, 6->20, 7->21
        # Block 3: 0->4, 1->5, 2->10, 3->11, 4->16, 5->17, 6->22, 7->23
        
        V_246 = np.zeros((24, 27))
        W_246 = np.zeros((12, 27))
        
        # W for M_{2,4,2} is 2x2, indices 0..3.
        # W index is k*n + i.
        # Block 1 (cols 0,1): W indices 0,2,1,3?
        # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3.
        # Block 2 (cols 2,3): W indices 4,6,5,7.
        # Block 3 (cols 4,5): W indices 8,10,9,11.
        
        # V mapping
        map_1 = [0, 1, 6, 7, 12, 13, 18, 19]
        map_2 = [2, 3, 8, 9, 14, 15, 20, 21]
        map_3 = [4, 5, 10, 11, 16, 17, 22, 23]
        
        for r in range(9):
            for i_local, i_global in enumerate(map_1):
                V_246[i_global, r] = V_242[i_local, r]
            for i_local, i_global in enumerate(map_2):
                V_246[i_global, r + 9] = V_242[i_local, r]
            for i_local, i_global in enumerate(map_3):
                V_246[i_global, r + 18] = V_242[i_local, r]
                
        # W mapping
        w_map_1 = [0, 2, 1, 3] # C[0,0], C[0,1], C[1,0], C[1,1] -> W[0], W[2], W[1], W[3]
        w_map_2 = [4, 6, 5, 7]
        w_map_3 = [8, 10, 9, 11]
        
        # W_242 indices 0..3 correspond to C[0,0], C[0,1], C[1,0], C[1,1]?
        # W index is k*n + i.
        # C[0,0] -> k=0, i=0 -> 0
        # C[0,1] -> k=1, i=0 -> 2
        # C[1,0] -> k=0, i=1 -> 1
        # C[1,1] -> k=1, i=1 -> 3
        # So W_242[0] is for C[0,0], W_242[2] for C[0,1], etc.
        # In the 4x2 case, C is 2x2.
        # The mapping from local C index to global W index:
        # Block 1: Local 0->0, 1->2, 2->1, 3->3?
        # Wait, W_242 has 4 rows.
        # Row 0: C[0,0]
        # Row 1: C[1,0]
        # Row 2: C[0,1]
        # Row 3: C[1,1]
        # So W_242[0] -> W_246[0]
        # W_242[1] -> W_246[1]
        # W_242[2] -> W_246[2]
        # W_242[3] -> W_246[3]
        
        for r in range(9):
            for i_local in range(4):
                W_246[w_map_1[i_local], r] = W_242[i_local, r]
                W_246[w_map_2[i_local], r + 9] = W_242[i_local, r]
                W_246[w_map_3[i_local], r + 18] = W_242[i_local, r]

        # Now refine the rank-27 solution
        time_budget_246 = min(60.0, deadline - time.time())
        sol_246 = try_rank(2, 4, 6, 27, time_budget_246, U_246, V_246, W_246)
        
        if sol_246 is not None:
            U, V, W = sol_246
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 42
    def get_fallback_solution():
        # ... (same as before) ...
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W

    U, V, W = get_fallback_solution()
    return {"U": U, "V": V, "W": W}
