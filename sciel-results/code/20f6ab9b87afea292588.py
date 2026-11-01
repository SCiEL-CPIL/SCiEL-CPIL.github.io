"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                if a == a_prime and b == b_prime:
                    continue
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                if abs(t_val) > 1e-9:
                    return False
    return True

def _try_rank_r_optimization(target_rank, deadline):
    """Try to find a valid decomposition of target_rank using optimization."""
    n, m, p = 2, 2, 4
    size_U = n * m  # 4
    size_V = m * p  # 8
    size_W = p * n  # 8
    
    # Build target tensor
    T_target = np.zeros((size_U, size_V, size_W))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_target[i, j, k] = 1.0
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def objective(x):
        U = x[:size_U * target_rank].reshape((size_U, target_rank))
        V = x[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
        W = x[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
        T = compute_tensor(U, V, W)
        return np.sum((T - T_target) ** 2)
    
    base = _build_rank14()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    # Strategy 1: Start from rank-14, zero out random columns, optimize
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(attempt * 1000 + 42)
        
        # Randomly select which columns to keep (target_rank out of 14)
        if target_rank < 14:
            keep = np.random.choice(14, size=target_rank, replace=False)
        else:
            keep = np.arange(14)
        
        U_init = np.zeros((size_U, target_rank))
        V_init = np.zeros((size_V, target_rank))
        W_init = np.zeros((size_W, target_rank))
        
        for new_idx, old_idx in enumerate(keep):
            U_init[:, new_idx] = U_base[:, old_idx]
            V_init[:, new_idx] = V_base[:, old_idx]
            W_init[:, new_idx] = W_base[:, old_idx]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < 1e-12:
                x_opt = res.x
                U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                
                # Try rounding to nearest 0.5
                for scale in [2, 4, 1]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    # Strategy 2: Pure random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(10000 + restart * 7)
        U_init = np.random.randn(size_U, target_rank) * 0.3
        V_init = np.random.randn(size_V, target_rank) * 0.3
        W_init = np.random.randn(size_W, target_rank) * 0.3
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < 1e-12:
                x_opt = res.x
                U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                
                for scale in [2, 4, 1]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None

def _try_shared_strassen(deadline):
    """Try to find a rank-12 or rank-13 solution by sharing Strassen intermediates.
    
    Key insight: The two 2x2 blocks share the same A matrix. Strassen's 7 products
    for each block use 7 different linear combinations of A. If we can find a way
    to share some of these across blocks, we might reduce total rank.
    
    The 7 Strassen products for a 2x2*2x2 multiply are:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    For the right block, same A but different B columns. The A-side factors are:
    L1 = A00+A11, L2 = A10+A11, L3 = A00, L4 = A11, L5 = A00+A01, L6 = A10-A00, L7 = A01-A11
    
    These 7 A-combinations are the same for both blocks! So we can precompute them
    and reuse. But the B-side combinations differ.
    
    Actually, the rank is determined by the number of scalar multiplications, which
    is the number of (A_combo, B_combo) pairs. Since A_combos are shared but B_combos
    differ, we still need 14 multiplications unless we can further combine.
    
    Let me try a different approach: search for rank-12/13 directly.
    """
    n, m, p = 2, 2, 4
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    T_target = np.zeros((size_U, size_V, size_W))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T_target[a*m+b, b*p+c, c*n+a] = 1.0
    
    # Try rank 12 and 13 with more aggressive search
    for target_rank in [12, 13]:
        if time.time() > deadline - 10:
            break
        
        def objective(x):
            U = x[:size_U * target_rank].reshape((size_U, target_rank))
            V = x[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
            W = x[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
            T = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((T - T_target) ** 2)
        
        # Use the rank-14 solution as starting point, projecting down
        base = _build_rank14()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)
        
        # Try different projections: keep random subsets of columns
        for proj_attempt in range(40):
            if time.time() > deadline - 5:
                break
            
            seed = proj_attempt * 137 + 7
            np.random.seed(seed)
            
            if target_rank <= 14:
                keep = np.random.choice(14, size=target_rank, replace=False)
            else:
                keep = np.arange(14)
            
            U_init = np.zeros((size_U, target_rank))
            V_init = np.zeros((size_V, target_rank))
            W_init = np.zeros((size_W, target_rank))
            
            for ni, oi in enumerate(keep):
                U_init[:, ni] = U_base[:, oi]
                V_init[:, ni] = V_base[:, oi]
                W_init[:, ni] = W_base[:, oi]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
                if res.fun < 1e-14:
                    x_opt = res.x
                    U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                    V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                    W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                    
                    # Try multiple rounding scales
                    for scale in [2, 4, 8, 1]:
                        U_r = np.round(U_opt * scale) / scale
                        V_r = np.round(V_opt * scale) / scale
                        W_r = np.round(W_opt * scale) / scale
                        
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        
                        if _verify_decomposition(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
        
        # Random restarts
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(50000 + restart * 13)
            U_init = np.random.randn(size_U, target_rank) * 0.5
            V_init = np.random.randn(size_V, target_rank) * 0.5
            W_init = np.random.randn(size_W, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                              options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
                if res.fun < 1e-14:
                    x_opt = res.x
                    U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                    V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                    W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                    
                    for scale in [2, 4, 8, 1]:
                        U_r = np.round(U_opt * scale) / scale
                        V_r = np.round(V_opt * scale) / scale
                        W_r = np.round(W_opt * scale) / scale
                        
                        U_list = U_r.tolist()
                        V_list = V_r.tolist()
                        W_list = W_r.tolist()
                        
                        if _verify_decomposition(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
    
    return None

def _algebraic_search_rank12(deadline):
    """Use algebraic structure to search for rank-12 decomposition.
    
    The multiplication tensor for 2x2 * 2x4 has 8 nonzero entries in the output.
    We can view this as computing 8 bilinear forms. The question is whether 12
    scalar multiplications suffice.
    
    Let's parameterize: each rank-1 term is u_r ⊗ v_r ⊗ w_r where
    u_r ∈ R^4, v_r ∈ R^8, w_r ∈ R^8.
    
    The constraint is: for each (a,b,c), sum_r u_r[a*m+b] * v_r[b*p+c] * w_r[c*n+a] = 1
    and for all other (i,j,k), sum_r u_r[i] * v_r[j] * w_r[k] = 0.
    """
    n, m, p = 2, 2, 4
    size_U = n * m
    size_V = m * p
    size_W = p * n
    target_rank = 12
    
    T_target = np.zeros((size_U, size_V, size_W))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T_target[a*m+b, b*p+c, c*n+a] = 1.0
    
    def objective(x):
        U = x[:size_U * target_rank].reshape((size_U, target_rank))
        V = x[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
        W = x[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - T_target) ** 2)
    
    # Start from rank-14 and try to compress
    base = _build_rank14()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    # Try keeping 12 of 14 columns with various combinations
    import itertools
    combos = list(itertools.combinations(range(14), 12))
    
    for combo in combos:
        if time.time() > deadline - 5:
            break
        
        U_init = np.zeros((size_U, target_rank))
        V_init = np.zeros((size_V, target_rank))
        W_init = np.zeros((size_W, target_rank))
        
        for ni, oi in enumerate(combo):
            U_init[:, ni] = U_base[:, oi]
            V_init[:, ni] = V_base[:, oi]
            W_init[:, ni] = W_base[:, oi]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if res.fun < 1e-14:
                x_opt = res.x
                U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                
                for scale in [2, 4, 8, 1]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with the known rank-14 solution
    best_result = _build_rank14()
    
    # Verify the base solution
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    
    # Try to find rank-12 solution (most aggressive)
    if time.time() < deadline - 30:
        result = _algebraic_search_rank12(deadline)
        if result:
            return result
    
    # Try shared Strassen approach for rank 12-13
    if time.time() < deadline - 20:
        result = _try_shared_strassen(deadline)
        if result:
            return result
    
    # Try rank 13
    if time.time() < deadline - 15:
        result = _try_rank_r_optimization(13, deadline)
        if result:
            return result
    
    # Try rank 12
    if time.time() < deadline - 10:
        result = _try_rank_r_optimization(12, deadline)
        if result:
            return result
    
    # Return the best verified solution
    return best_result

# EVOLVE-BLOCK-END
