import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def larsen_rank_analysis():
        """
        Use Larsen's framework to determine minimal rank.
        For matrix multiplication A(n×m) * B(m×p) = C(n×p),
        the tensor is T_{i,j,k} = 1 if indices match, else 0.
        
        We solve for the minimal rank r such that there exist
        U(nm×r), V(mp×r), W(pn×r) with:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        This is equivalent to finding the tensor rank. For rectangular
        matrices, we can use the "slice rank" or border rank techniques.
        
        Key insight: For m=3, p=10, n=2, the structure suggests:
        - j=0 row: 20 independent terms (n*p = 20)
        - j=1,2: can use Strassen on 2×2 blocks
        
        Let's compute the theoretical minimum via linear algebra.
        """
        # Build the multiplication tensor T as a vector of length n*m*p
        # T[i*m + j, j*p + k, k*n + i] = 1 for all valid (i,j,k)
        
        # The tensor can be viewed as a matrix of size (nm) x (mp*pn) 
        # or we can use the "flattening" approach.
        
        # For rank r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # This means for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Strategy: Fix W to be identity-like, then the problem becomes
        # a bilinear form minimization.
        
        # For our specific case, let's use the structural approach:
        # j=0: needs n*p = 20 terms
        # j=1,2: 5 blocks of 2×2, each needs 7 terms via Strassen
        # Total: 20 + 5*7 = 55
        
        # Can we do better? Let's check if we can merge some terms.
        # The key question: can we share factors between j=0 and j=1,2?
        
        # Answer: Generally no, because the index structures are different.
        # j=0 has U[i*3+0, r], V[0*10+k, r], W[k*2+i, r]
        # j=1 has U[i*3+1, r], V[1*10+k, r], W[k*2+i, r]
        # These use different U and V rows, so they're independent.
        
        # However, within j=1,2 we can potentially do better than Strassen
        # if we exploit the specific structure.
        
        # Let's set up the optimization problem for rank r < 55
        # and see if it's feasible.
        
        return 55  # Conservative estimate; optimization will try to reduce
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank

    def compute_error(U, V, W, r):
        """Fast vectorized error computation"""
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m + j, :]
                for k in range(p):
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max

    def verify(U, V, W, r, tol=1e-8):
        err = compute_error(U, V, W, r)
        return err < tol, err

    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-8):
        """Faster ALS with vectorized updates using normal equations"""
        U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
        best_err = float('inf')
        b_p = np.ones(p)
        b_n = np.ones(n)
        b_m = np.ones(m)
        
        # Precompute index arrays for speed
        u_indices = np.array([(i, j) for i in range(n) for j in range(m)])
        v_indices = np.array([(j, k) for j in range(m) for k in range(p)])
        w_indices = np.array([(k, i) for k in range(p) for i in range(n)])
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U rows
            for i, j in u_indices:
                idx = i * m + j
                A = V_cur[j*p:(j+1)*p, :] * W_cur[np.arange(p)*n + i, :].T
                try:
                    u_sol = np.linalg.solve(A.T @ A + 1e-10*np.eye(target_rank), A.T @ b_p)
                except np.linalg.LinAlgError:
                    u_sol, _, _, _ = np.linalg.lstsq(A, b_p, rcond=None)
                U_cur[idx, :] = u_sol
            
            # Update V rows
            for j, k in v_indices:
                idx = j * p + k
                A = U_cur[np.arange(n)*m + j, :].T * W_cur[k*n:(k+1)*n, :]
                try:
                    v_sol = np.linalg.solve(A.T @ A + 1e-10*np.eye(target_rank), A.T @ b_n)
                except np.linalg.LinAlgError:
                    v_sol, _, _, _ = np.linalg.lstsq(A, b_n, rcond=None)
                V_cur[idx, :] = v_sol
            
            # Update W rows
            for k, i in w_indices:
                idx = k * n + i
                A = U_cur[np.arange(m)*0 + i*m + np.arange(m), :].T * V_cur[k*p:(k+1)*p, :]
                try:
                    w_sol = np.linalg.solve(A.T @ A + 1e-10*np.eye(target_rank), A.T @ b_m)
                except np.linalg.LinAlgError:
                    w_sol, _, _, _ = np.linalg.lstsq(A, b_m, rcond=None)
                W_cur[idx, :] = w_sol
            
            # Check error every few iterations
            if iteration % 3 == 0 or iteration == max_iters - 1:
                err_val = compute_error(U_cur, V_cur, W_cur, target_rank)
                if err_val < best_err:
                    best_err = err_val
                    if err_val < tol:
                        break
                if err_val < 1e-10:
                    break
        
        final_err = compute_error(U_cur, V_cur, W_cur, target_rank)
        best_err = min(best_err, final_err)
        
        return U_cur, V_cur, W_cur, best_err

    def gradient_descent(U_init, V_init, W_init, target_rank, max_iters=200, lr=0.01, tol=1e-8):
        """Gradient descent with adaptive learning rate for smoother convergence"""
        U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
        best_err = float('inf')
        best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
        lr_cur = lr
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U_cur)
            grad_V = np.zeros_like(V_cur)
            grad_W = np.zeros_like(W_cur)
            
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        # Target is 1
                        current_val = np.dot(U_cur[u_idx, :], V_cur[v_idx, :] * W_cur[w_idx, :])
                        error = current_val - 1.0
                        # Gradient w.r.t U
                        grad_U[u_idx, :] += error * (V_cur[v_idx, :] * W_cur[w_idx, :])
                        # Gradient w.r.t V
                        grad_V[v_idx, :] += error * (U_cur[u_idx, :] * W_cur[w_idx, :])
                        # Gradient w.r.t W
                        grad_W[w_idx, :] += error * (U_cur[u_idx, :] * V_cur[v_idx, :])
            
            # Normalize gradients
            g_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if g_norm > 0:
                U_cur -= lr_cur * grad_U / g_norm
                V_cur -= lr_cur * grad_V / g_norm
                W_cur -= lr_cur * grad_W / g_norm
            
            # Check error every 5 iterations
            if iteration % 5 == 0 or iteration == max_iters - 1:
                err_val = compute_error(U_cur, V_cur, W_cur, target_rank)
                if err_val < best_err:
                    best_err = err_val
                    best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
                    if err_val < tol:
                        break
                    # Adaptive learning rate
                    if err_val < best_err * 0.9:
                        lr_cur = min(lr_cur * 1.5, 0.1)
                    else:
                        lr_cur = max(lr_cur * 0.5, 0.001)
                if err_val < 1e-10:
                    break
        
        return best_U, best_V, best_W, best_err

    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to reduce rank using iterative optimization
    if time.time() < deadline - 20:
        for target_rank in range(rank_best - 1, 40, -1):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_restarts = 15 if target_rank >= 50 else 8
            
            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(trial * 7919 + target_rank * 13)
                
                # Strategy 1: take rank-55 solution, truncate, add noise
                if trial % 2 == 0:
                    U_init = U_best[:, :target_rank].copy()
                    V_init = V_best[:, :target_rank].copy()
                    W_init = W_best[:, :target_rank].copy()
                    noise_scale = 0.03
                    U_init += noise_scale * np.random.randn(nm, target_rank)
                    V_init += noise_scale * np.random.randn(mp, target_rank)
                    W_init += noise_scale * np.random.randn(pn, target_rank)
                else:
                    # Strategy 2: merge columns via linear combination
                    perm = np.random.permutation(rank_best)
                    keep = perm[:target_rank]
                    U_init = U_best[:, keep].copy()
                    V_init = V_best[:, keep].copy()
                    W_init = W_best[:, keep].copy()
                    noise_scale = 0.02
                    U_init += noise_scale * np.random.randn(nm, target_rank)
                    V_init += noise_scale * np.random.randn(mp, target_rank)
                    W_init += noise_scale * np.random.randn(pn, target_rank)
                
                # Run ALS optimization
                U_cur, V_cur, W_cur, best_err = als_optimize(
                    U_init, V_init, W_init, target_rank, max_iters=120
                )
                
                if best_err < 1e-7:
                    U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                    found = True
                    break
                elif best_err < 1e-4 and time.time() < deadline - 12:
                    # Polish with gradient descent for smoother convergence
                    U_cur2, V_cur2, W_cur2, best_err2 = gradient_descent(
                        U_cur, V_cur, W_cur, target_rank, max_iters=100, lr=0.01, tol=1e-9
                    )
                    if best_err2 < 1e-7:
                        U_best, V_best, W_best, rank_best = U_cur2, V_cur2, W_cur2, target_rank
                        found = True
                        break
            
            if found:
                continue  # Try next lower rank
            else:
                break  # Can't reduce further
    
    # High-precision polish pass: use exact local solve with tighter tolerance
    if time.time() < deadline - 8:
        U_polished, V_polished, W_polished, err_polished = als_optimize(
            U_best, V_best, W_best, rank_best, max_iters=200, tol=1e-12
        )
        if err_polished < err:
            U_best, V_best, W_best = U_polished, V_polished, W_polished
    
    # Final verification and cleanup
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
