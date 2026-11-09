import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        # Efficient reconstruction using einsum
        # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def round_to_half_integer(x):
        """Round to nearest integer or half-integer."""
        return np.round(x * 2) / 2.0

    def optimize_rank(r, max_restarts=20):
        """
        Try to find a rank-r decomposition using optimization.
        """
        if time.time() > deadline:
            return None
        
        # Flatten U, V, W into a single vector for scipy
        # Size: r * (nm + mp + pn)
        dim = r * (nm + mp + pn)
        
        def unpack(x):
            U = x[:r*nm].reshape(nm, r)
            V = x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = x[r*(nm+mp):].reshape(pn, r)
            return U, V, W
        
        def loss_and_grad(x):
            U, V, W = unpack(x)
            
            # Compute tensor reconstruction
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            loss = np.sum(error**2)
            
            # Gradient computation
            # grad_U[a, r] = sum_{b,c} error[a,b,c] V[b,r] W[c,r]
            # grad_V[b, r] = sum_{a,c} error[a,b,c] U[a,r] W[c,r]
            # grad_W[c, r] = sum_{a,b} error[a,b,c] U[a,r] V[b,r]
            
            # Efficient gradient calculation
            # We can use einsum again
            # grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            # grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            # grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # We do NOT include the half-integer penalty in the gradient for the main minimization
            # to allow the optimizer to find the continuous minimum first.
            # We will handle the discretization separately.
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return loss, grad

        best_result = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale is important. Schoolbook entries are 1.
            # Let's initialize with small random values around 0
            np.random.seed(42 + restart * 13 + r)
            x0 = np.random.randn(dim) * 0.5
            
            try:
                # Use L-BFGS-B for large problems
                res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            except Exception:
                continue
            
            if res.fun < best_loss:
                U, V, W = unpack(res.x)
                
                # Round to nearest half-integer
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                # Check if rounded solution is exact
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                
                # If not exact, try a local discrete search around the rounded values
                # This is a simple hill-climbing on the discrete lattice
                U_cur = U_rounded.copy()
                V_cur = V_rounded.copy()
                W_cur = W_rounded.copy()
                
                # Calculate current error
                T_rec_cur = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                err_cur = np.sum((T_rec_cur - T)**2)
                
                # Try a few random local moves
                for step in range(10):
                    if time.time() > deadline:
                        break
                    # Pick a random matrix and index
                    mat_choice = np.random.choice(3)
                    if mat_choice == 0:
                        idx = np.random.choice(U_cur.size)
                        row, col = np.unravel_index(idx, U_cur.shape)
                        # Try changing by +/- 0.5
                        for delta in [-0.5, 0.5]:
                            orig_val = U_cur[row, col]
                            U_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                U_cur[row, col] = orig_val
                    elif mat_choice == 1:
                        idx = np.random.choice(V_cur.size)
                        row, col = np.unravel_index(idx, V_cur.shape)
                        for delta in [-0.5, 0.5]:
                            orig_val = V_cur[row, col]
                            V_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                V_cur[row, col] = orig_val
                    else:
                        idx = np.random.choice(W_cur.size)
                        row, col = np.unravel_index(idx, W_cur.shape)
                        for delta in [-0.5, 0.5]:
                            orig_val = W_cur[row, col]
                            W_cur[row, col] = orig_val + delta
                            T_rec_test = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                            err_test = np.sum((T_rec_test - T)**2)
                            if err_test < err_cur - 1e-12:
                                err_cur = err_test
                                if check_decomposition(U_cur, V_cur, W_cur):
                                    return (U_cur, V_cur, W_cur)
                            else:
                                W_cur[row, col] = orig_val
                                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = (U_rounded, V_rounded, W_rounded)
        
        return None

    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from 62 down to 50. 
    # Why? Because finding a rank 62 solution is easier than 50.
    # If we find 62, we can then try 61, etc. 
    # But to be efficient, let's just scan 50-62.
    # If we find any, we return it.
    # To ensure minimality, we should technically go low to high.
    # But low ranks are much harder.
    # Given time constraints, we might miss a rank 55 if we spend too much time on 50.
    # Let's try 62, 61, 60... down to 50.
    # If we find 62, we stop? No, we want MINIMAL.
    # So if we find 62, we should continue to 61, 60...
    # But we don't know if 61 exists.
    # Strategy:
    # 1. Try 62. If found, set best=62.
    # 2. Try 61. If found, set best=61.
    # 3. ...
    # 4. Try 50. If found, set best=50.
    # This is expensive.
    # Alternative: Randomly sample ranks from 50-62.
    
    # Let's try a few specific ranks known to be close to schoolbook.
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        result = optimize_rank(r, max_restarts=5) # Fewer restarts per rank to cover more ranks
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            # Do NOT break immediately. We want the MINIMAL rank.
            # But if we find 62, we still need to check 61...
            # However, checking 61 might take too long.
            # If we find 62, it's a good improvement over 63.
            # Let's continue to lower ranks if time permits.
            
    U, V, W = best_result
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
