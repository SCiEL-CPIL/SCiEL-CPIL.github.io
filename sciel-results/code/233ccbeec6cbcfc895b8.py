"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    
    # Precompute target tensor
    T_target = np.zeros((U_rows, V_rows, W_rows))
    for (a, b, c) in triples:
        T_target[a, b, c] = 1.0
    
    # Schoolbook solution
    rank_school = n * m * p  # 144
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_recon - T_target)
        return np.max(diff) < tol
    
    def check_fast(U, V, W, tol=1e-6):
        """Faster approximate check."""
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        rng = np.random.RandomState(42)
        for _ in range(1000):
            a = rng.randint(0, U_rows)
            b = rng.randint(0, V_rows)
            c = rng.randint(0, W_rows)
            if (a, b, c) in triple_set:
                continue
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val) > tol:
                return False
        return True
    
    def als_decompose(target_rank, max_iters=150, seed=42, init_scale=0.1):
        """ALS for tensor decomposition with better convergence."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(U_rows, target_rank) * init_scale
        V = rng.randn(V_rows, target_rank) * init_scale
        W = rng.randn(W_rows, target_rank) * init_scale
        
        prev_err = float('inf')
        stagnation_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            B_tensor = np.einsum('br,cr->bcr', V, W)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-6 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            BT = B @ T_target.reshape(U_rows, -1).T
            U_new = (G_inv @ BT).T
            
            # Update V
            C_tensor = np.einsum('ar,cr->acr', U_new, W)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            T_V = np.transpose(T_target, (1, 0, 2))
            Gv = C @ C.T + 1e-6 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            CT = C @ T_V.reshape(V_rows, -1).T
            V_new = (Gv_inv @ CT).T
            
            # Update W
            D_tensor = np.einsum('ar,br->abr', U_new, V_new)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            T_W = np.transpose(T_target, (2, 0, 1))
            Gw = D @ D.T + 1e-6 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            DT = D @ T_W.reshape(W_rows, -1).T
            W_new = (Gw_inv @ DT).T
            
            # Error check
            if it % 5 == 0:
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U_new[a, :] * V_new[b, :], W_new[c, :])
                    err += (val - 1.0) ** 2
                # Also check off-diagonal
                rng2 = np.random.RandomState(123)
                for _ in range(200):
                    a = rng2.randint(0, U_rows)
                    b = rng2.randint(0, V_rows)
                    c = rng2.randint(0, W_rows)
                    if (a, b, c) not in triple_set:
                        val = np.dot(U_new[a, :] * V_new[b, :], W_new[c, :])
                        err += val ** 2
                
                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-14:
                    stagnation_count += 1
                    if stagnation_count > 3:
                        break
                else:
                    stagnation_count = 0
                prev_err = err
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    def polish_decomposition(U, V, W, max_iters=50):
        """High-precision polish pass using alternating least squares with tighter convergence."""
        if time.time() > deadline - 3:
            return U, V, W
        
        target_rank = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Update U with tighter regularization
            B_tensor = np.einsum('br,cr->bcr', V, W)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T + 1e-8 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            BT = B @ T_target.reshape(U_rows, -1).T
            U = (G_inv @ BT).T
            
            # Update V
            C_tensor = np.einsum('ar,cr->acr', U, W)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            T_V = np.transpose(T_target, (1, 0, 2))
            Gv = C @ C.T + 1e-8 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            CT = C @ T_V.reshape(V_rows, -1).T
            V = (Gv_inv @ CT).T
            
            # Update W
            D_tensor = np.einsum('ar,br->abr', U, V)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            T_W = np.transpose(T_target, (2, 0, 1))
            Gw = D @ D.T + 1e-8 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            DT = D @ T_W.reshape(W_rows, -1).T
            W = (Gw_inv @ DT).T
        
        return U, V, W
    
    def try_round_and_verify(U, V, W, half_int=True):
        """Try rounding to half-integers and verify exactness."""
        if half_int:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
        else:
            U_r = np.round(U)
            V_r = np.round(V)
            W_r = np.round(W)
        
        if check_exact(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None
    
    # Strategy: Try multiple ranks with multiple seeds and initializations
    # First, try to find any solution below schoolbook rank
    ranks_to_try = list(range(100, 40, -1))
    
    found = False
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # Multiple seeds with different initializations
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            # Vary initialization scale
            init_scales = [0.05, 0.1, 0.2]
            init_scale = init_scales[seed % len(init_scales)]
            
            result = als_decompose(target_rank, max_iters=100, seed=seed, init_scale=init_scale)
            if result is None:
                continue
            
            U, V, W = result
            
            # Check fast
            if check_fast(U, V, W, tol=1e-6):
                # Polish
                U, V, W = polish_decomposition(U, V, W, max_iters=30)
                
                # Try rounding
                rounded = try_round_and_verify(U, V, W, half_int=True)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
                
                # Also try integer rounding
                rounded_int = try_round_and_verify(U, V, W, half_int=False)
                if rounded_int is not None:
                    U_r, V_r, W_r = rounded_int
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If we found something, return it
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If no lower rank found, try to at least improve the schoolbook solution
    # by checking if we can get a cleaner decomposition
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
