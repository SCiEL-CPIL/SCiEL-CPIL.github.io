import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    min_rank = max(n * m, m * p, p * n)
    
    def try_rank(R, max_iters=30, restarts=10):
        for _ in range(restarts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                A_VW = np.einsum('br,cr->bc', V, W)
                M_U = np.zeros((m * p * p * n, R))
                for r in range(R):
                    M_U[:, r] = np.einsum('br,cr->bc', V[:, r], W[:, r]).flatten()
                T_flat = T.reshape(n * m, m * p * p * n)
                try:
                    U = T_flat @ np.linalg.pinv(M_U.T)
                except:
                    return None
                
                M_V = np.zeros((n * m * p * n, R))
                for r in range(R):
                    M_V[:, r] = np.einsum('ar,cr->ac', U[:, r], W[:, r]).flatten()
                T_flat_V = T.reshape(m * p, n * m * p * n)
                try:
                    V = T_flat_V @ np.linalg.pinv(M_V.T)
                except:
                    return None
                
                M_W = np.zeros((n * m * m * p, R))
                for r in range(R):
                    M_W[:, r] = np.einsum('ar,br->ab', U[:, r], V[:, r]).flatten()
                T_flat_W = T.reshape(p * n, n * m * m * p)
                try:
                    W = T_flat_W @ np.linalg.pinv(M_W.T)
                except:
                    return None
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T - T_approx)
            if error < 1e-6:
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.linalg.norm(T - T_check) < 1e-6:
                    return U_int, V_int, W_int
        
        return None

    # Try ranks in descending order of likelihood or just a few
    # We want to find the MINIMAL rank.
    # So we should try from low to high?
    # But low ranks are harder to find.
    # If we try 32 and fail, we try 33, etc.
    # But we have limited time.
    # Let's try a few specific ranks.
    
    # If we find a rank, we can try to go lower.
    # But for now, let's just try to find ANY better rank.
    
    # Heuristic: The rank is likely close to the lower bound or schoolbook.
    # Let's try 32, 40, 50, 60, 70, 80, 90, 95.
    
    ranks_to_try = [32, 40, 50, 60, 70, 80, 90, 95]
    # Actually, if we find 95, we should check 94?
    # The list is fixed.
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=20, restarts=5)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            # Continue to see if we can find lower?
            # The loop continues to lower R if we iterate in increasing order.
            # My list is increasing.
            # So if we find 95, we won't try 32.
            # We should iterate from low to high?
            # If we find 32, we are done.
            # If we don't find 32, we try 40.
            # So iterating from low to high is correct for finding the minimum.
            
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
