import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    def build_target():
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        return target
    
    target = build_target()
    
    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)
    
    def get_strassen_66():
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W
    
    def systematic_prune(U, V, W):
        """Use linear dependency analysis to find removable columns."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        current_rank = Ua.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline - 10.0:
                break
            
            # Build system: check if column c's contribution is in span of others
            A_rows = []
            b_vals = []
            
            for i in range(n*m):
                for k in range(m*p):
                    for l in range(p*n):
                        lhs = Ua[i, c] * Va[k, c] * Wa[l, c]
                        if abs(lhs) < 1e-15:
                            # Check if all others also zero at this position
                            others_zero = True
                            for l_idx in range(current_rank):
                                if l_idx == c:
                                    continue
                                if abs(Ua[i, l_idx] * Va[k, l_idx] * Wa[l, l_idx]) > 1e-15:
                                    others_zero = False
                                    break
                            if others_zero:
                                continue
                        
                        row = np.zeros(current_rank - 1)
                        idx = 0
                        for l_idx in range(current_rank):
                            if l_idx == c:
                                continue
                            row[idx] = Ua[i, l_idx] * Va[k, l_idx] * Wa[l, l_idx]
                            idx += 1
                        A_rows.append(row)
                        b_vals.append(lhs)
            
            if not A_rows:
                keep = [l for l in range(current_rank) if l != c]
                U_new = Ua[:, keep]
                V_new = Va[:, keep]
                W_new = Wa[:, keep]
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new, current_rank - 1
                continue
            
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            try:
                alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                residual = np.max(np.abs(A @ alpha - b))
            except:
                continue
            
            if residual < 1e-9:
                keep = [l for l in range(current_rank) if l != c]
                U_new = Ua[:, keep]
                V_new = Va[:, keep]
                W_new = Wa[:, keep]
                if verify(U_new, V_new, W_new):
                    return U_new, V_new, W_new, current_rank - 1
        
        return Ua, Va, Wa, current_rank
    
    def greedy_prune(U, V, W, max_iter=100, seed=None):
        """Greedy random column removal."""
        if seed is not None:
            random.seed(seed)
        
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        current_rank = Ua.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            if current_rank <= 0:
                break
            
            removed_any = False
            cols_to_try = list(range(current_rank))
            random.shuffle(cols_to_try)
            
            for col in cols_to_try:
                if time.time() > deadline - 10.0:
                    break
                
                mask = [i for i in range(current_rank) if i != col]
                U_new = Ua[:, mask]
                V_new = Va[:, mask]
                W_new = Wa[:, mask]
                
                R = np.einsum('ic,kc,lc->ikl', U_new, V_new, W_new)
                
                if np.allclose(R, target, atol=1e-9):
                    Ua, Va, Wa = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        return Ua, Va, Wa, current_rank
    
    # Start with rank-66 decomposition
    U66, V66, W66 = get_strassen_66()
    
    if not verify(U66, V66, W66):
        # Fallback: return as-is (shouldn't happen)
        return {"U": U66, "V": V66, "W": W66}
    
    best_U, best_V, best_W = U66, V66, W66
    best_rank = 66
    
    # Step 1: Systematic linear dependency pruning
    U_s, V_s, W_s, r_s = systematic_prune(U66, V66, W66)
    if r_s < best_rank:
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = r_s
    
    # Step 2: Greedy random restarts from best found so far
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        seed_val = restart * 7919 + 104729
        U_g, V_g, W_g, r_g = greedy_prune(best_U, best_V, best_W, max_iter=100, seed=seed_val)
        
        if r_g < best_rank:
            if verify(U_g, V_g, W_g):
                best_U, best_V, best_W = U_g, V_g, W_g
                best_rank = r_g
    
    # Step 3: Another systematic pass on the greedily pruned result
    if best_rank < 66:
        U_s2, V_s2, W_s2, r_s2 = systematic_prune(best_U, best_V, best_W)
        if r_s2 < best_rank:
            best_U, best_V, best_W = U_s2, V_s2, W_s2
            best_rank = r_s2
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
