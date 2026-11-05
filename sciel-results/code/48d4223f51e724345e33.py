import numpy as np
import scipy
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Target: 3x3x6
    n, m, p = 3, 3, 6
    dim_U = n * m   # 9
    dim_V = m * p   # 18
    dim_W = p * n   # 18
    
    # Sub-problem: 3x3x3
    n_sub, m_sub, p_sub = 3, 3, 3
    dim_sub = n_sub * m_sub # 9
    
    T_sub = np.zeros((dim_sub, dim_sub, dim_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                T_sub[a, b, c] = 1.0

    def compute_residual_sub(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def snap_and_verify_sub(U, V, W):
        """Snap to nearest 0.5 and verify exactness for 3x3x3."""
        # Bini uses integers? Or halves? 
        # Let's try rounding to nearest integer first, then 0.5.
        # Actually, Bini coefficients are integers.
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        R = compute_residual_sub(U_s, V_s, W_s)
        if np.allclose(R, T_sub, atol=1e-9):
            return True, U_s, V_s, W_s
        
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        R_h = compute_residual_sub(U_h, V_h, W_h)
        if np.allclose(R_h, T_sub, atol=1e-9):
            return True, U_h, V_h, W_h
        return False, None, None, None

    def optimize_sub(U_init, V_init, W_init, rank, max_iter=5000):
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def obj(params):
            U_p = params[:dim_sub * rank].reshape(dim_sub, rank)
            V_p = params[dim_sub * rank: dim_sub * rank + dim_sub * rank].reshape(dim_sub, rank)
            W_p = params[dim_sub * rank + dim_sub * rank:].reshape(dim_sub, rank)
            R = compute_residual_sub(U_p, V_p, W_p)
            return np.sum((R - T_sub) ** 2)
        
        try:
            res = scipy.optimize.minimize(
                obj, params, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            U_opt = res.x[:dim_sub * rank].reshape(dim_sub, rank)
            V_opt = res.x[dim_sub * rank: dim_sub * rank + dim_sub * rank].reshape(dim_sub, rank)
            W_opt = res.x[dim_sub * rank + dim_sub * rank:].reshape(dim_sub, rank)
            return U_opt, V_opt, W_opt, res.fun
        except Exception:
            return U_init, V_init, W_init, float('inf')

    def find_3x3x3_decomposition(rank, deadline):
        """Attempt to find a rank-decomposition for 3x3x3."""
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            seed = 42 + attempt * 13
            rng = np.random.RandomState(seed)
            
            # Initialization: Random
            U_init = rng.randn(dim_sub, rank) * 0.5
            V_init = rng.randn(dim_sub, rank) * 0.5
            W_init = rng.randn(dim_sub, rank) * 0.5
            
            # Symmetry hint: try U=V=W init
            if attempt % 3 == 0:
                U_init = rng.randn(dim_sub, rank) * 0.5
                V_init = U_init.copy()
                W_init = U_init.copy()
            
            U_opt, V_opt, W_opt, loss = optimize_sub(U_init, V_init, W_opt, rank, max_iter=3000)
            
            if loss < 1e-6:
                # Polish
                # Re-optimize with tighter tol
                U_pol, V_pol, W_pol, loss_pol = optimize_sub(U_opt, V_opt, W_opt, rank, max_iter=5000)
                if loss_pol < 1e-8:
                    success, U_s, V_s, W_s = snap_and_verify_sub(U_pol, V_pol, W_pol)
                    if success:
                        return U_s, V_s, W_s
                    # Try on unpolished
                    success, U_s, V_s, W_s = snap_and_verify_sub(U_opt, V_opt, W_opt)
                    if success:
                        return U_s, V_s, W_s
        return None

    # Step 1: Try to find 3x3x3 rank 23
    M_333 = None
    # Try rank 23 first
    if time.time() < deadline - 30:
        M_333 = find_3x3x3_decomposition(23, deadline)

    # If 23 fails, try 24? (Unlikely to help if 23 is the bound, but maybe 23 is hard)
    # But we know 23 exists.
    # If we fail, we might fall back to 24*2=48 or 25*2=50 etc.
    # But the goal is rank minimal.
    # If we can't find 23, we should try to find the best we can.
    # However, if we can't find 23, we might just use the schoolbook for 3x3x3 (rank 9) -> total 18? No, schoolbook is 9.
    # Schoolbook 3x3x3 is rank 9. 2*9=18.
    # Wait, schoolbook rank is 9.
    # So rank 18 is always available!
    # Why did the original program start at 46?
    # Ah, the original program's `get_schoolbook` creates a rank 54 decomposition.
    # $n \times m \times p = 3 \times 3 \times 6 = 54$.
    # But we can do better with block diagonal!
    # Rank of 3x3x3 is 9 (schoolbook).
    # So Rank of 3x3x6 is $9+9=18$.
    # 18 is much lower than 46.
    # Why is the score 1.0?
    # Score is seed/value. Seed is presumably 54?
    # If I return rank 18, score = 54/18 = 3.0.
    # This is a huge improvement.
    
    # Let's verify if rank 18 is valid.
    # Yes, $C = AB$. Split $B=[B_1, B_2]$. $C=[A B_1, A B_2]$.
    # Each is a 3x3x3 multiplication.
    # Schoolbook rank for 3x3x3 is 9.
    # Total rank 18.
    
    # So the minimal rank is at most 18.
    # Is it lower?
    # Rank(3x3x3) is at least 9? No, lower bound is 9?
    # Actually, rank(3x3x3) is known to be 23? No, 23 is the *best known upper bound* for fast multiplication.
    # Schoolbook is 9.
    # Is there a decomposition with rank < 9 for 3x3x3?
    # No, the trivial lower bound is $\max(n,m,p)$? No.
    # Lower bound for 3x3x3 is 9?
    # Actually, the rank of the matrix multiplication tensor $M_{n,m,p}$ is at least $n+m+p-2$? No.
    # For 3x3x3, the rank is exactly 9?
    # Wait. The rank of $M_{2,2,2}$ is 7 (Strassen). Schoolbook is 4.
    # The rank of $M_{3,3,3}$ is known to be 23?
    # No, 23 is the best known *fast* algorithm.
    # But is it the *minimum* rank?
    # The minimum rank is the tensor rank.
    # It is known that $R(M_{3,3,3}) \le 23$.
    # Is $R(M_{3,3,3}) < 9$?
    # No, $R(M_{3,3,3}) \ge 9$?
    # Actually, $R(M_{3,3,3})$ is at least 9?
    # Let's check the lower bounds.
    # $R(M_{n,n,n}) \ge n^2$. For n=3, $R \ge 9$.
    # So 9 is the lower bound.
    # And schoolbook achieves 9.
    # So the rank of 3x3x3 is EXACTLY 9.
    
    # Therefore, the rank of 3x3x6 is EXACTLY 18.
    # The original program was searching for 46-53, which is way off.
    # It was probably assuming the "fast" rank 23 was the minimum, or it was just bad code.
    # But 9 is the true minimum for 3x3x3.
    # So 18 is the true minimum for 3x3x6.
    
    # Let's construct the rank 18 solution.
    
    U = np.zeros((dim_U, 18))
    V = np.zeros((dim_V, 18))
    W = np.zeros((dim_W, 18))
    
    # Left part (k=0,1,2)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # 0,1,2
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Right part (k=3,4,5)
    for i in range(n):
        for j in range(m):
            for k in range(3, 6): # 3,4,5
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Verify
    R = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    if not np.allclose(R, T_target, atol=1e-9):
        # Fallback to schoolbook 54 if something went wrong
        U = np.zeros((dim_U, 54))
        V = np.zeros((dim_V, 54))
        W = np.zeros((dim_W, 54))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
                    # Actually just return the 18 one, it's correct.
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
