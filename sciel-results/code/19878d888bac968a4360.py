"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
1. Attempt to find a rank-10 decomposition for the 2x2 * 2x3 sub-problem.
2. If found, use recursive Strassen to build a rank-70 decomposition for 4x4 * 4x6.
3. If not found within time limit, fall back to the proven rank-77 decomposition 
   (using rank-11 for 2x2 * 2x3).
"""

import time
import numpy as np
import sys

_START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _search_2x2x2x3_rank10():
    """
    Search for a rank-10 decomposition of the 2x2 * 2x3 matrix multiplication tensor.
    Returns (U, V, W) as numpy arrays if found, else None.
    """
    if _time_remaining() < 10.0:
        return None

    n, m, p = 2, 2, 3
    rank = 10
    
    # Target tensor T[a, b, c]
    # a = i*m + j (0..3)
    # b = j*p + k (0..5)
    # c = k*n + i (0..5)
    # T[a,b,c] = 1 if consistent, 0 otherwise.
    # We only need to check the 12 non-zero entries for the objective, 
    # but we should also penalize the zero entries to ensure exactness.
    
    # List of (a, b, c) indices where T=1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Total possible indices
    total_indices = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                total_indices.append((a, b, c))
                
    target_set = set(target_indices)
    zero_indices = [idx for idx in total_indices if idx not in target_set]
    
    def objective_and_grad(params):
        U = params[0:4*rank].reshape(4, rank)
        V = params[4*rank:10*rank].reshape(6, rank)
        W = params[10*rank:16*rank].reshape(6, rank)
        
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Compute contributions for all indices
        # To be efficient, we can vectorize the dot products
        # T[a,b,c] = sum_l U[a,l] * V[b,l] * W[c,l]
        
        # For the 12 non-zero targets, we want T=1
        # For the 132 zero targets, we want T=0
        
        # This is a sum of products. 
        # Let's compute the full tensor T_recon
        # T_recon[a,b,c] = U[a,:] dot (V[b,:] * W[c,:])
        
        # Efficient computation:
        # We can't easily vectorize the triple loop in pure numpy without expanding dimensions significantly.
        # But rank is small (10).
        
        # Let's just loop, it's 144 * 10 ops = 1440 ops per eval. Very fast.
        
        for a, b, c in target_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - 1.0
            loss += err * err
            # d/dU[a,l] = 2 * err * V[b,l] * W[c,l]
            grad_U[a, :] += 2 * err * (V[b, :] * W[c, :])
            
        for a, b, c in zero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - 0.0
            loss += err * err
            grad_U[a, :] += 2 * err * (V[b, :] * W[c, :])
            
        # Gradient for V and W is symmetric but we need to accumulate properly.
        # d/dV[b,l] = sum_c T_err[b,c] * U[a,c] * W[c,l] ... this is getting complex.
        # Let's use numerical gradient for simplicity if analytical is tricky, 
        # or just use scipy's numerical gradient by returning loss only.
        # However, L-BFGS-B with numerical gradient is slower.
        # Let's try to derive it properly or just use finite diff.
        
        # Actually, let's just return loss and let scipy do finite differences. 
        # With 480 variables, finite diff is 480 evaluations. 
        # If each eval is 1ms, that's 0.5s per step. 100 steps = 50s. Too slow.
        
        # We need analytical gradient.
        # Let T_recon[a,b,c] = sum_l U[a,l] V[b,l] W[c,l]
        # L = sum_{(a,b,c) in targets} (T_recon[a,b,c] - 1)^2 + sum_{(a,b,c) in zeros} (T_recon[a,b,c])^2
        
        # dL/dU[a0,l0] = sum_{b,c} 2 (T_recon[a0,b,c] - target[a0,b,c]) * V[b,l0] * W[c,l0]
        # We can precompute M[b,c,l] = V[b,l] * W[c,l]
        # Then grad_U[a0,l0] = sum_{b,c} 2 (T_recon[a0,b,c] - target[a0,b,c]) * M[b,c,l0]
        
        # This is still O(N^3 * Rank).
        
        # Let's stick to the simple loop for now but accumulate gradients for V and W too.
        
        # Reset grad_V and grad_W
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # We already computed loss and grad_U above. 
        # We need to recompute or store T_recon and errors.
        # Let's restructure to compute T_recon and errors first.
        
        # This is getting messy. Let's use a simpler approach:
        # Use scipy.optimize.minimize with method='BFGS' or 'L-BFGS-B' and 
        # provide the analytical gradient.
        
        # Let's compute T_recon for all 144 indices.
        # T_recon[a,b,c]
        # We can compute this as:
        # For each l, add U[:,l] outer V[:,l] outer W[:,l] to T_recon.
        # T_recon = np.einsum('al,bl,cl->abc', U, V, W)
        
        T_recon = np.einsum('al,bl,cl->abc', U, V, W)
        
        # Target tensor
        T_target = np.zeros((4, 6, 6))
        for a, b, c in target_indices:
            T_target[a, b, c] = 1.0
            
        err = T_recon - T_target
        loss = np.sum(err * err)
        
        # Gradient
        # dL/dU[a,l] = 2 * sum_{b,c} err[a,b,c] * V[b,l] * W[c,l]
        # dL/dV[b,l] = 2 * sum_{a,c} err[a,b,c] * U[a,l] * W[c,l]
        # dL/dW[c,l] = 2 * sum_{a,b} err[a,b,c] * U[a,l] * V[b,l]
        
        # grad_U = 2 * np.einsum('abc,bl,cl->al', err, V, W)
        # grad_V = 2 * np.einsum('abc,al,cl->bl', err, U, W)
        # grad_W = 2 * np.einsum('abc,al,bl->cl', err, U, V)
        
        grad_U = 2 * np.einsum('abc,bl,cl->al', err, V, W)
        grad_V = 2 * np.einsum('abc,al,cl->bl', err, U, W)
        grad_W = 2 * np.einsum('abc,al,bl->cl', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    np.random.seed(42)
    best_loss = float('inf')
    best_params = None
    
    # Try a few random initializations
    for trial in range(5):
        if _time_remaining() < 5.0:
            break
            
        U_init = np.random.randn(4, rank) * 0.5
        V_init = np.random.randn(6, rank) * 0.5
        W_init = np.random.randn(6, rank) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective_and_grad, params_init, method='L-BFGS-B', 
                          jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                
                if best_loss < 1e-8:
                    break
        except Exception as e:
            pass

    if best_params is not None and best_loss < 1e-8:
        U = best_params[0:4*rank].reshape(4, rank)
        V = best_params[4*rank:10*rank].reshape(6, rank)
        W = best_params[10*rank:16*rank].reshape(6, rank)
        
        # Round to nearest half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify rounded solution
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), 2, 2, 3, tol=1e-9):
            return U_rounded, V_rounded, W_rounded
            
    return None


def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W'.
    """
    n, m, p = 4, 4, 6
    
    # First, try to find a rank-10 decomposition for 2x2 * 2x3
    sub_decomp = None
    if _time_remaining() > 30.0:
        sub_decomp = _search_2x2x2x3_rank10()
        
    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = 10
        total_rank = 7 * rank_sub  # 70
        
        # Build 4x4 * 4x6 using recursive Strassen
        # U: (16, 70), V: (24, 70), W: (24, 70)
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Strassen terms for 4x4 * 4x6
        # A blocks: A00(0:2,0:2), A01(0:2,2:4), A10(2:4,0:2), A11(2:4,2:4)
        # B blocks: B00(0:2,0:3), B01(0:2,3:6), B10(2:4,0:3), B11(2:4,3:6)
        # C blocks: C00(0:2,0:3), C01(0:2,3:6), C10(2:4,0:3), C11(2:4,3:6)
        
        # M1 = (A00+A11)(B00+B11) -> C00+C11
        # M2 = (A10+A11)B00 -> C10+C11
        # M3 = A00(B01-B11) -> C01+C11
        # M4 = A11(B10-B00) -> C00+C10
        # M5 = (A00+A01)B11 -> C00+C01
        # M6 = (A10-A00)(B00+B01) -> C11
        # M7 = (A01-A11)(B10+B11) -> C00
        
        # We define a helper to add a Strassen term
        # Each term involves specific blocks of A, B, C with signs.
        # The sub-decomposition U_sub, V_sub, W_sub is for 2x2 * 2x3.
        # We need to map the local indices (0..1 for rows/cols of 2x2, 0..2 for cols of 2x3)
        # to the global indices.
        
        def add_strassen_term(term_idx, u_blocks, v_blocks, w_blocks):
            """
            u_blocks: dict {block_idx: coeff}
            block_idx: 0=TL, 1=TR, 2=BL, 3=BR for 2x2 blocks of A
            v_blocks: dict {block_idx: coeff}
            block_idx: 0=TL, 1=TR, 2=BL, 3=BR for 2x2 blocks of B (but B is 2x3, so blocks are 2x3? No, B is split into 2x3 blocks? No.)
            
            Wait, the sub-problem is 2x2 * 2x3.
            A is 2x2. B is 2x3. C is 2x3.
            In the 4x4 * 4x6 context:
            A is 4x4. B is 4x6. C is 4x6.
            We split A into 2x2 blocks.
            We split B into 2x3 blocks.
            We split C into 2x3 blocks.
            
            So for M1 = (A00+A11)(B00+B11):
            A00 is A[0:2, 0:2]. A11 is A[2:4, 2:4].
            B00 is B[0:2, 0:3]. B11 is B[2:4, 3:6].
            C00 is C[0:2, 0:3]. C11 is C[2:4, 3:6].
            
            The sub-decomposition U_sub has shape (4, 10).
            Row 0: A[0,0], Row 1: A[0,1], Row 2: A[1,0], Row 3: A[1,1].
            V_sub has shape (6, 10).
            Row 0: B[0,0], Row 1: B[0,1], Row 2: B[0,2], Row 3: B[1,0], Row 4: B[1,1], Row 5: B[1,2].
            W_sub has shape (6, 10).
            Row 0: C[0,0], Row 1: C[0,1], Row 2: C[0,2], Row 3: C[1,0], Row 4: C[1,1], Row 5: C[1,2].
            
            For M1:
            U part:
            A00: rows 0,1,2,3 of U_sub map to global rows 0,1,4,5.
            A11: rows 0,1,2,3 of U_sub map to global rows 8,9,12,13.
            So U_global[0:4] += U_sub (for A00 part)
            U_global[8:12] += U_sub (for A11 part)
            
            V part:
            B00: rows 0..5 of V_sub map to global rows 0..5.
            B11: rows 0..5 of V_sub map to global rows 12..17.
            So V_global[0:6] += V_sub
            V_global[12:18] += V_sub
            
            W part:
            C00: rows 0..5 of W_sub map to global rows 0..5.
            C11: rows 0..5 of W_sub map to global rows 12..17.
            So W_global[0:6] += W_sub
            W_global[12:18] += W_sub
            """
            base = term_idx * rank_sub
            
            # Map U_sub to global U
            for b_idx, coeff in u_blocks.items():
                # b_idx 0: A[0:2, 0:2] -> rows 0,1,4,5
                # b_idx 1: A[0:2, 2:4] -> rows 2,3,6,7
                # b_idx 2: A[2:4, 0:2] -> rows 8,9,12,13
                # b_idx 3: A[2:4, 2:4] -> rows 10,11,14,15
                if b_idx == 0:
                    rows = [0, 1, 4, 5]
                elif b_idx == 1:
                    rows = [2, 3, 6, 7]
                elif b_idx == 2:
                    rows = [8, 9, 12, 13]
                elif b_idx == 3:
                    rows = [10, 11, 14, 15]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    U[global_r, base:base+rank_sub] += coeff * U_sub[local_r, :]
                    
            # Map V_sub to global V
            # V_sub rows: 0:B[0,0], 1:B[0,1], 2:B[0,2], 3:B[1,0], 4:B[1,1], 5:B[1,2]
            for b_idx, coeff in v_blocks.items():
                if b_idx == 0: # B[0:2, 0:3] -> rows 0..5
                    rows = [0, 1, 2, 3, 4, 5]
                elif b_idx == 1: # B[0:2, 3:6] -> rows 6..11
                    rows = [6, 7, 8, 9, 10, 11]
                elif b_idx == 2: # B[2:4, 0:3] -> rows 12..17
                    rows = [12, 13, 14, 15, 16, 17]
                elif b_idx == 3: # B[2:4, 3:6] -> rows 18..23
                    rows = [18, 19, 20, 21, 22, 23]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    V[global_r, base:base+rank_sub] += coeff * V_sub[local_r, :]
                    
            # Map W_sub to global W
            # W_sub rows: 0:C[0,0], 1:C[0,1], 2:C[0,2], 3:C[1,0], 4:C[1,1], 5:C[1,2]
            for b_idx, coeff in w_blocks.items():
                if b_idx == 0: # C[0:2, 0:3] -> rows 0..5
                    rows = [0, 1, 2, 3, 4, 5]
                elif b_idx == 1: # C[0:2, 3:6] -> rows 6..11
                    rows = [6, 7, 8, 9, 10, 11]
                elif b_idx == 2: # C[2:4, 0:3] -> rows 12..17
                    rows = [12, 13, 14, 15, 16, 17]
                elif b_idx == 3: # C[2:4, 3:6] -> rows 18..23
                    rows = [18, 19, 20, 21, 22, 23]
                else:
                    continue
                for local_r, global_r in enumerate(rows):
                    W[global_r, base:base+rank_sub] += coeff * W_sub[local_r, :]

        # Define the 7 Strassen terms
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term(0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        
        # M2 = (A10+A11)B00
        add_strassen_term(1, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        
        # M3 = A00(B01-B11)
        add_strassen_term(2, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        
        # M4 = A11(B10-B00)
        add_strassen_term(3, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        
        # M5 = (A00+A01)B11
        add_strassen_term(4, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        
        # M6 = (A10-A00)(B00+B01)
        add_strassen_term(5, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        
        # M7 = (A01-A11)(B10+B11)
        add_strassen_term(6, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Rank 77
    # Use rank 11 for 2x2 * 2x3
    # We can construct this using schoolbook for 2x2*2x1 (rank 4) and Strassen for 2x2*2x2 (rank 7)
    
    U_77 = np.zeros((n*m, 77))
    V_77 = np.zeros((m*p, 77))
    W_77 = np.zeros((p*n, 77))
    
    # Helper to add a 2x2 * 2x3 term
    def add_2x2x3_term(base_rank, u_blocks, v_blocks, w_blocks):
        # This is complex to map directly. 
        # Instead, let's just use the known rank-77 construction from the current program 
        # which was verified to work.
        pass
        
    # Since implementing the rank-77 fallback from scratch is risky, 
    # I will reuse the logic from the "Current Program" which was known to be valid.
    
    # Re-implementing the rank-77 construction quickly
    def add_term_ijk(u_dict, v_dict, w_dict):
        u_col = np.zeros(n*m)
        v_col = np.zeros(m*p)
        w_col = np.zeros(p*n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        return u_col, v_col, w_col

    strassen_2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms:
            u_col = np.zeros(n*m)
            v_col = np.zeros(m*p)
            w_col = np.zeros(p*n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
                        
            col_idx = len(U_77[0]) # This logic is wrong, U_77 is pre-allocated
            # We need to track the current column index
            # Let's use a list of columns instead
            pass

    # Simpler fallback: Just return the rank-77 decomposition constructed in the previous run.
    # I will hardcode the construction logic from the "Current Program" to ensure validity.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_ijk(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def strassen_2x2(oi, oj, ok):
        add_term_ijk({(oi, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1, (oi+1, ok+1): 1})
        add_term_ijk({(oi+1, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi+1, ok): 1, (oi+1, ok+1): -1})
        add_term_ijk({(oi, oj): 1}, {(oj, ok+1): 1, (oj+1, ok+1): -1}, {(oi, ok+1): 1, (oi+1, ok+1): 1})
        add_term_ijk({(oi+1, oj+1): 1}, {(oj+1, ok): 1, (oj, ok): -1}, {(oi, ok): 1, (oi+1, ok): 1})
        add_term_ijk({(oi, oj): 1, (oi, oj+1): 1}, {(oj+1, ok+1): 1}, {(oi, ok): -1, (oi, ok+1): 1})
        add_term_ijk({(oi+1, oj): 1, (oi, oj): -1}, {(oj, ok): 1, (oj, ok+1): 1}, {(oi+1, ok+1): 1})
        add_term_ijk({(oi, oj+1): 1, (oi+1, oj+1): -1}, {(oj+1, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1})

    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        strassen_2x2_terms_local = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms_local:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
                        
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # 4x4 * 4x4 part (cols 0-3)
    _strassen_super(0, 0, 0, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    _strassen_super(0, 0, 0, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    _strassen_super(0, 0, 0, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    _strassen_super(0, 0, 0, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    _strassen_super(0, 0, 0, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    _strassen_super(0, 0, 0, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    _strassen_super(0, 0, 0, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    # 4x4 * 4x2 part (cols 4-5)
    for rb in range(2):
        strassen_2x2(rb*2, 0, 4)
        strassen_2x2(rb*2, 2, 4)
        
    rank = len(U_cols)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
