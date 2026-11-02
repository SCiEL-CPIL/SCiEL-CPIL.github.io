import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build the target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0

    # Unfolds
    T_1 = T.reshape(num_u, num_v * num_w)
    T_2 = T.transpose(1, 2, 0).reshape(num_v, num_w * num_u)
    T_3 = T.transpose(2, 0, 1).reshape(num_w, num_u * num_v)
    
    start_time = time.time()
    deadline = 145.0

    def check_solution(U, V, W, tol=1e-6):
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        max_err = np.max(np.abs(T_calc - T))
        return max_err < tol

    def try_round_and_verify(U, V, W):
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_solution(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # Try half-integer rounding
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_solution(U_half, V_half, W_half):
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        return None

    def als_decompose(R, U_init, V_init, W_init, max_iters=100, tol=1e-8):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None, np.inf
            
            # Update U
            K_u = np.einsum('vr, wr -> vw r', V, W).reshape(num_v * num_w, R)
            try:
                U, _, _, _ = np.linalg.lstsq(K_u.T, T_1.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
            
            # Update V
            K_v = np.einsum('wr, ur -> wu r', W, U).reshape(num_w * num_u, R)
            try:
                V, _, _, _ = np.linalg.lstsq(K_v.T, T_2.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
                
            # Update W
            K_w = np.einsum('ur, vr -> uv r', U, V).reshape(num_u * num_v, R)
            try:
                W, _, _, _ = np.linalg.lstsq(K_w.T, T_3.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
            
            T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
            err = np.linalg.norm(T_calc - T)
            
            if err < tol:
                return (U, V, W), err
                
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        final_err = np.linalg.norm(T_calc - T)
        return (U, V, W), final_err

    def strassen_init_333():
        """Create Strassen-like initialization for 3x3x3 part, extended to 3x3x7."""
        # Strassen decomposition for 3x3x3 has rank 7
        # We'll create a structured initialization that exploits the tensor structure
        # For 3x3x7, we can think of it as 7 independent 3x3x1 problems combined
        # Use a Kronecker-like structure
        
        R = 7
        U = np.zeros((num_u, R))
        V = np.zeros((num_v, R))
        W = np.zeros((num_w, R))
        
        # Strassen for 3x3: 7 products
        # This is a heuristic initialization, not the actual decomposition
        # We'll use random but structured init
        for r in range(R):
            # Random but with block structure
            for i in range(n):
                for j in range(m):
                    U[i*m+j, r] = np.random.choice([-1, 0, 1])
            for j in range(m):
                for k in range(p):
                    V[j*p+k, r] = np.random.choice([-1, 0, 1])
            for k in range(p):
                for i in range(n):
                    W[k*n+i, r] = np.random.choice([-1, 0, 1])
        
        return U, V, W

    def structured_init(R, seed):
        """Create structured initialization for rank R."""
        np.random.seed(seed)
        U = np.zeros((num_u, R))
        V = np.zeros((num_v, R))
        W = np.zeros((num_w, R))
        
        # Use a mix of structured and random
        for r in range(R):
            if r < 7:
                # Strassen-like for first 7
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = np.random.choice([-1, 0, 1])
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = np.random.choice([-1, 0, 1])
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, r] = np.random.choice([-1, 0, 1])
            elif r < 14:
                # Second block
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = np.random.randn()
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = np.random.randn()
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, r] = np.random.randn()
            else:
                # Random
                U[:, r] = np.random.randn(num_u)
                V[:, r] = np.random.randn(num_v)
                W[:, r] = np.random.randn(num_w)
        
        return U, V, W

    def try_rank(R, max_restarts=20):
        """Try to find rank-R decomposition."""
        best_result = None
        best_err = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            seed = 42 + R * 100 + restart
            U, V, W = structured_init(R, seed)
            
            result, err = als_decompose(R, U, V, W, max_iters=50)
            if result is not None:
                if err < best_err:
                    best_err = err
                    best_result = result
                
                if err < 1e-6:
                    rounded = try_round_and_verify(*result)
                    if rounded is not None:
                        return rounded
        
        return None

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try ranks from reasonable lower bound
    # C(3,3,7) is at least C(3,3,3) = 7, but realistically higher
    # Try a range
    for R in range(15, 45):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(R, max_restarts=15)
        if result is not None:
            return result
    
    # Fallback
    return schoolbook()
# EVOLVE-BLOCK-END
