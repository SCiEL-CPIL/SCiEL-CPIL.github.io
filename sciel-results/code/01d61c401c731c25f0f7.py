"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 24
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def check_half_integer(val, tol=1e-9):
        return abs(val * 2 - round(val * 2)) < tol
    
    def verify_solution(U, V, W, r, tol=1e-9):
        # Check half-integer constraint
        for c in range(r):
            for idx in range(nm):
                if not check_half_integer(U[idx][c]):
                    return False
            for idx in range(mp):
                if not check_half_integer(V[idx][c]):
                    return False
            for idx in range(pn):
                if not check_half_integer(W[idx][c]):
                    return False
        # Check exactness
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to construct a rank-18 decomposition using the additive structure
    # The 2x3x4 tensor can be viewed as two 1x3x4 tensors (one for each row of A)
    # Each 1x3x4 multiplication has known decompositions.
    # We attempt a structured rank-18 decomposition.
    
    def try_structured_rank_18():
        """
        Attempt to construct a rank-18 decomposition.
        The tensor [2,3,4] = [1,3,4] ⊕ [1,3,4].
        If we can decompose [1,3,4] with rank 9, total rank is 18.
        
        For 1x3x4: A is 1x3, B is 3x4, C is 1x4.
        C_k = sum_j A_j * B_jk
        This is a bilinear map: (A_0, A_1, A_2) x (B_00, B_01, B_02, B_03, B_10, ..., B_23) -> (C_0, C_1, C_2, C_3)
        
        The tensor is sum_{j,k} e_j^U ⊗ e_{jk}^V ⊗ e_k^W where e_j is in R^3, e_{jk} in R^{12} (but really only the j-th block of 4), e_k in R^4.
        
        Known: rank of [1,3,4] is 12 (schoolbook). Can we do better?
        Actually, for [1, m, p], the rank is m*p if m=1 or p=1, but for general m,p it's at most m*p.
        Wait, [1,3,4] is just a set of 3 linear forms times 4 coefficients each.
        C_k = A_0 B_0k + A_1 B_1k + A_2 B_2k.
        This is rank 12 in the tensor sense? No, the tensor is in U^* ⊗ V^* ⊗ W^*.
        dim U = 3, dim V = 12, dim W = 4.
        The tensor is T = sum_{j=0}^2 sum_{k=0}^3 e_j^* ⊗ e_{jk}^* ⊗ e_k^*.
        
        Can we reduce the rank?
        This is equivalent to the rank of the matrix flattening.
        Flattening U x (V x W): 3 x 48 matrix. Rank 3.
        Flattening V x (U x W): 12 x 12 matrix.
        The matrix M_{jk, k'} is 1 if k'=k, 0 else? No.
        M_{(j,k), (j',k')} = delta_{j,j'} delta_{k,k'}.
        So it's a block diagonal matrix with 3 identity blocks of size 4x4. Rank 12.
        So the tensor rank is at least 12.
        Schoolbook is 12. So rank([1,3,4]) = 12.
        Therefore rank([2,3,4]) >= 12+12 = 24?
        No, the lower bound is max over flattening ranks.
        For [2,3,4]:
        U dim 6, V dim 12, W dim 8.
        Flattening U x (V x W): 6 x 96. Rank <= 6.
        Flattening V x (U x W): 12 x 48. Rank <= 12.
        Flattening W x (U x V): 8 x 72. Rank <= 8.
        So rank >= 12.
        Schoolbook is 24.
        So there is room for improvement down to 12.
        
        However, the directive suggests trying rank 18 first via a specific structure.
        Let's just try to optimize for rank 18 directly using the general method,
        but with a better initialization or more attempts if needed.
        """
        return None

    def error_func(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp : r * nm + r * mp + r * pn]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    err += (s - 1.0) ** 2
        return err
    
    # We will try ranks from 23 down to 12.
    # The directive mentions rank 18 specifically.
    # We'll start with 18 and go down if time permits, or up if 18 fails?
    # No, we want minimal rank. So we try 12, 13, ..., 23.
    # But 12 is the lower bound. Let's try 12 first?
    # Usually lower ranks are harder to find.
    # Let's try 18 as a "sweet spot" first, then 17, 16...
    # Or just scan from 23 down to 12, stopping when found.
    
    # To be efficient, let's try 18 first as suggested by the strategy.
    # If 18 fails, try 19, 20...
    # Actually, if we find a solution for rank r, we can try r-1.
    # So we should try 23, 22, ... 12.
    
    # However, the current code tries 23 down to 18.
    # Let's extend the range to 12.
    # And increase the number of attempts for lower ranks?
    # Or use a smarter optimizer?
    
    # Let's stick to the current approach but extend the range.
    # And maybe add a few more attempts for the harder lower ranks.
    
    ranks_to_try = list(range(23, 11, -1))
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline:
            break
        
        total_vars = r * (nm + mp + pn)
        
        # Number of attempts: more for lower ranks?
        # Let's say 10 attempts for r <= 15, 5 for r > 15.
        num_attempts = 10 if r <= 15 else 5
        
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            x0 = np.random.uniform(-1, 1, size=total_vars)
            
            try:
                res = minimize(error_func, x0, args=(r,), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp : r * nm + r * mp + r * pn]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Round to half-integers
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if verify_solution(U_list, V_list, W_list, r):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found solution for this rank, move to lower
    
    return best_result
# EVOLVE-BLOCK-END
