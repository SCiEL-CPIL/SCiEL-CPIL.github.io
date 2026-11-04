import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = 2, 3, 5
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10

    def verify(U, V, W, R):
        """
        Verify exact decomposition.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        T_recon = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    T_recon[i, j, k] = s
        
        return np.all(T_recon == 1)

    def build_strassen_28():
        """
        Construct a rank-28 decomposition using Strassen's algorithm structure.
        We split the 2x3 matrix into 2x2 and 2x1 blocks, and 3x5 into 2x5 and 1x5.
        
        Let A = [[A11, A12, A13], [A21, A22, A23]] (2x3)
        Let B = [[B11, B12, B13, B14, B15], [B21, B22, B23, B24, B25], [B31, B32, B33, B34, B35]] (3x5)
        
        We can view this as multiplying:
        C = A * B
        
        One efficient way is to use Strassen for the 2x2 part (cols 1-2 of A, rows 1-2 of B) 
        and handle the rest. However, a standard rank-28 algorithm for 2x3 * 3x5 is known.
        
        Let's use a specific known construction or derive one.
        Actually, for 2x3 * 3x5, rank 28 is achievable via Winograd or similar.
        
        Let's try to construct it explicitly using the "peeling" logic from the prompt's inspiration.
        If we can't find a hardcoded one, we can use a general solver for small ranks if time permits,
        but rank 28 is large for a general solver.
        
        Alternative: Use the known fact that R(2,3,5) <= 28.
        Let's try to generate a valid rank-28 decomposition using a heuristic search 
        but with a very structured initialization based on block diagonals.
        
        Actually, a safer bet for the "baseline" is to use the schoolbook rank 30 
        and then attempt to reduce it. But the prompt asks for a strategy to try rank 27 
        by peeling from rank 28.
        
        Let's implement a rank-28 construction.
        We can decompose the multiplication into:
        1. 2x2 * 2x2 part (using Strassen rank 7) -> handles C[0:2, 0:2]
        2. 2x2 * 2x1 part (using rank 4 or 5?)
        3. 2x1 * 2x2 part
        4. 2x1 * 2x1 part
        
        This block approach is tricky to get exact rank 28 without overlap.
        
        Let's use a different approach:
        We know that R(2,3,5) is at most 28.
        We will use a randomized search but seeded with a "structured" random start 
        that is close to a valid decomposition, or we use a known decomposition.
        
        Since I don't have the explicit 28x(6,15,10) matrices memorized, I will use 
        a fast randomized search to find rank 28, which is much easier than rank 27 
        because the landscape is less rugged. Then I will attempt the peeling step.
        
        However, finding rank 28 via random search in 110s is risky.
        
        Let's look at the "Inspiration 1" which achieved 1.0714 (Rank 28).
        It likely used a specific construction.
        
        Let's try to construct Rank 28 using the following logic:
        The tensor T_{2,3,5} can be embedded in T_{3,3,5} or similar? No.
        
        Let's use a simple, robust method:
        1. Try to find Rank 28 using a fast local search with {-1, 0, 1} values.
           Since 28 is close to the lower bound, it should be findable.
           Actually, the lower bound is roughly (2*3*5)^(2/3) * something? No, LB is ~ 30/sqrt(3) ? 
           Actually, for 2x3x5, the rank is known to be 28.
           
        Let's implement a fast search for Rank 28.
        """
        
        def search_rank_fast(R, allowed_values, max_iter=2000, restarts=10):
            best_err = float('inf')
            best_U = None
            best_V = None
            best_W = None
            
            for restart in range(restarts):
                if time.time() > deadline:
                    break
                    
                U = np.array([random.choice(allowed_values) for _ in range(dim_u * R)])
                V = np.array([random.choice(allowed_values) for _ in range(dim_v * R)])
                W = np.array([random.choice(allowed_values) for _ in range(dim_w * R)])
                
                current_err = compute_error_fast(U, V, W, R)
                
                if current_err == 0:
                    return U.reshape(dim_u, R), V.reshape(dim_v, R), W.reshape(dim_w, R)
                
                if current_err < best_err:
                    best_err = current_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                        
                    var_idx = random.randint(0, R * (dim_u + dim_v + dim_w) - 1)
                    
                    if var_idx < dim_u * R:
                        mat = U; idx = var_idx
                    elif var_idx < dim_u * R + dim_v * R:
                        mat = V; idx = var_idx - dim_u * R
                    else:
                        mat = W; idx = var_idx - dim_u * R - dim_v * R
                    
                    old_val = mat[idx]
                    new_val = random.choice(allowed_values)
                    if new_val == old_val: continue
                    
                    mat[idx] = new_val
                    new_err = compute_error_fast(U, V, W, R)
                    
                    if new_err == 0:
                        return U.reshape(dim_u, R), V.reshape(dim_v, R), W.reshape(dim_w, R)
                    
                    if new_err < current_err:
                        current_err = new_err
                        if current_err < best_err:
                            best_err = current_err
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                    else:
                        mat[idx] = old_val
                        
            if best_err == 0:
                return best_U.reshape(dim_u, R), best_V.reshape(dim_v, R), best_W.reshape(dim_w, R)
            return None

        def compute_error_fast(U, V, W, R):
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                        diff = s - 1
                        err += diff * diff
            return err

        # Try to find Rank 28
        res_28 = search_rank_fast(28, [-1, 0, 1], max_iter=500, restarts=5)
        
        if res_28:
            U28, V28, W28 = res_28
            if verify(U28, V28, W28, 28):
                # We have a valid rank 28.
                # Now, the STRATEGY: Attempt to peel to rank 27.
                # The idea: Assume the rank 28 decomposition has a redundant term.
                # Or, more generally, try to find a rank 27 decomposition directly 
                # using the rank 28 as a seed for a local search that allows 
                # "merging" or "removing" a term.
                
                # A more direct approach for Rank 27:
                # Run a local search for Rank 27, but initialize it using 
                # the first 27 terms of the Rank 28 solution, and allow 
                # the values to adjust.
                
                U27_init = U28[:, :27].copy()
                V27_init = V28[:, :27].copy()
                W27_init = W28[:, :27].copy()
                
                # Flatten for search
                U_flat = U27_init.flatten()
                V_flat = V27_init.flatten()
                W_flat = W27_init.flatten()
                
                current_err = compute_error_fast(U_flat, V_flat, W_flat, 27)
                
                # If error is small, we might be close.
                # Perform a local search from this seed.
                # We allow values in {-1, 0, 1, 2, -2} to allow for adjustments.
                
                best_err = current_err
                best_U = U_flat.copy()
                best_V = V_flat.copy()
                best_W = W_flat.copy()
                
                if current_err == 0:
                    return {"U": U27_init.tolist(), "V": V27_init.tolist(), "W": W27_init.tolist()}
                
                max_iter_peel = 2000
                for _ in range(max_iter_peel):
                    if time.time() > deadline:
                        break
                        
                    # Random variable change
                    total_vars = 27 * (dim_u + dim_v + dim_w)
                    var_idx = random.randint(0, total_vars - 1)
                    
                    if var_idx < dim_u * 27:
                        mat = U_flat; idx = var_idx
                    elif var_idx < dim_u * 27 + dim_v * 27:
                        mat = V_flat; idx = var_idx - dim_u * 27
                    else:
                        mat = W_flat; idx = var_idx - dim_u * 27 - dim_v * 27
                    
                    old_val = mat[idx]
                    # Wider range for peeling
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val: continue
                    
                    mat[idx] = new_val
                    new_err = compute_error_fast(U_flat, V_flat, W_flat, 27)
                    
                    if new_err == 0:
                        U_final = U_flat.reshape(dim_u, 27)
                        V_final = V_flat.reshape(dim_v, 27)
                        W_final = W_flat.reshape(dim_w, 27)
                        if verify(U_final, V_final, W_final, 27):
                            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    
                    if new_err < current_err:
                        current_err = new_err
                        if current_err < best_err:
                            best_err = current_err
                            best_U = U_flat.copy()
                            best_V = V_flat.copy()
                            best_W = W_flat.copy()
                    else:
                        mat[idx] = old_val
                        
                # If we didn't find 27, return the valid 28
                return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
                
        # Fallback: Try Rank 27 directly with random restarts
        # This is less likely to succeed than peeling from 28, but worth a shot if time remains
        if time.time() < deadline - 10:
            res_27 = search_rank_fast(27, [-1, 0, 1], max_iter=1000, restarts=5)
            if res_27:
                U27, V27, W27 = res_27
                if verify(U27, V27, W27, 27):
                    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                    
        # Fallback: Schoolbook Rank 30
        rank = 30
        U = [[0.0] * rank for _ in range(dim_u)]
        V = [[0.0] * rank for _ in range(dim_v)]
        W = [[0.0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

# Note: I need to import random
import random

# Re-define construct to ensure random is in scope or imported at top
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 3, 5
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    def verify(U, V, W, R):
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T_recon = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    T_recon[i, j, k] = s
        return np.all(T_recon == 1)

    def compute_error_fast(U, V, W, R):
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    diff = s - 1
                    err += diff * diff
        return err

    def search_rank_fast(R, allowed_values, max_iter=500, restarts=5):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
                
            U = np.array([random.choice(allowed_values) for _ in range(dim_u * R)])
            V = np.array([random.choice(allowed_values) for _ in range(dim_v * R)])
            W = np.array([random.choice(allowed_values) for _ in range(dim_w * R)])
            
            current_err = compute_error_fast(U, V, W, R)
            
            if current_err == 0:
                return U.reshape(dim_u, R), V.reshape(dim_v, R), W.reshape(dim_w, R)
            
            if current_err < best_err:
                best_err = current_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                    
                var_idx = random.randint(0, R * (dim_u + dim_v + dim_w) - 1)
                
                if var_idx < dim_u * R:
                    mat = U; idx = var_idx
                elif var_idx < dim_u * R + dim_v * R:
                    mat = V; idx = var_idx - dim_u * R
                else:
                    mat = W; idx = var_idx - dim_u * R - dim_v * R
                
                old_val = mat[idx]
                new_val = random.choice(allowed_values)
                if new_val == old_val: continue
                
                mat[idx] = new_val
                new_err = compute_error_fast(U, V, W, R)
                
                if new_err == 0:
                    return U.reshape(dim_u, R), V.reshape(dim_v, R), W.reshape(dim_w, R)
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    mat[idx] = old_val
                    
        if best_err == 0:
            return best_U.reshape(dim_u, R), best_V.reshape(dim_v, R), best_W.reshape(dim_w, R)
        return None

    # 1. Try to find Rank 28
    res_28 = search_rank_fast(28, [-1, 0, 1], max_iter=300, restarts=10)
    
    if res_28:
        U28, V28, W28 = res_28
        if verify(U28, V28, W28, 28):
            # 2. Attempt to peel to Rank 27
            U27_init = U28[:, :27].copy()
            V27_init = V28[:, :27].copy()
            W27_init = W28[:, :27].copy()
            
            U_flat = U27_init.flatten()
            V_flat = V27_init.flatten()
            W_flat = W27_init.flatten()
            
            current_err = compute_error_fast(U_flat, V_flat, W_flat, 27)
            
            if current_err == 0:
                return {"U": U27_init.tolist(), "V": V27_init.tolist(), "W": W27_init.tolist()}
            
            best_err = current_err
            best_U = U_flat.copy()
            best_V = V_flat.copy()
            best_W = W_flat.copy()
            
            # Local search from seed
            max_iter_peel = 1000
            for _ in range(max_iter_peel):
                if time.time() > deadline:
                    break
                    
                total_vars = 27 * (dim_u + dim_v + dim_w)
                var_idx = random.randint(0, total_vars - 1)
                
                if var_idx < dim_u * 27:
                    mat = U_flat; idx = var_idx
                elif var_idx < dim_u * 27 + dim_v * 27:
                    mat = V_flat; idx = var_idx - dim_u * 27
                else:
                    mat = W_flat; idx = var_idx - dim_u * 27 - dim_v * 27
                
                old_val = mat[idx]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val: continue
                
                mat[idx] = new_val
                new_err = compute_error_fast(U_flat, V_flat, W_flat, 27)
                
                if new_err == 0:
                    U_final = U_flat.reshape(dim_u, 27)
                    V_final = V_flat.reshape(dim_v, 27)
                    W_final = W_flat.reshape(dim_w, 27)
                    if verify(U_final, V_final, W_final, 27):
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                
                if new_err < current_err:
                    current_err = new_err
                    if current_err < best_err:
                        best_err = current_err
                        best_U = U_flat.copy()
                        best_V = V_flat.copy()
                        best_W = W_flat.copy()
                else:
                    mat[idx] = old_val
                    
            # Return Rank 28 if 27 not found
            return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
            
    # Fallback: Try Rank 27 directly
    if time.time() < deadline - 5:
        res_27 = search_rank_fast(27, [-1, 0, 1], max_iter=500, restarts=5)
        if res_27:
            U27, V27, W27 = res_27
            if verify(U27, V27, W27, 27):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
                
    # Fallback: Schoolbook Rank 30
    rank = 30
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
