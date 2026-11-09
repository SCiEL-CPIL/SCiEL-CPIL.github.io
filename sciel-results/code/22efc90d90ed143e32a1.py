import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
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
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def round_to_half_integer(x):
        """Round to nearest integer or half-integer."""
        return np.round(x * 2) / 2.0

    def optimize_rank(r, max_restarts=50):
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
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = T_rec - T
            loss = np.sum(error**2)
            
            # Add penalty to encourage integer/half-integer values
            # Penalty: sum (2*x - round(2*x))^2
            penalty = 0.0
            for mat in [U, V, W]:
                for val in mat:
                    rounded = np.round(val * 2)
                    penalty += (val * 2 - rounded)**2
            
            total_loss = loss + 0.1 * penalty # Weight for penalty
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                # Gradient of loss w.r.t U[:, r_idx]
                # d/dU[a,r] sum_{b,c} (T_rec[a,b,c] - T[a,b,c])^2
                # = 2 * sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                
                # Efficient computation:
                # error * outer(V[:, r_idx], W[:, r_idx])
                # Sum over b, c for each a
                
                # grad_U[:, r_idx] = np.tensordot(error, np.outer(V[:, r_idx], W[:, r_idx]), axes=([1,2],[0,1]))
                # This is equivalent to:
                temp = error * np.einsum('b,c->bc', V[:, r_idx], W[:, r_idx])[None, :, :]
                grad_U[:, r_idx] = np.sum(temp, axis=(1, 2))
                
                temp = error * np.einsum('a,c->ac', U[:, r_idx], W[:, r_idx])[None, :, :]
                grad_V[:, r_idx] = np.sum(temp, axis=(0, 2))
                
                temp = error * np.einsum('a,b->ab', U[:, r_idx], V[:, r_idx])[None, :, :]
                grad_W[:, r_idx] = np.sum(temp, axis=(0, 1))
            
            # Gradient of penalty
            # d/dx (2x - round(2x))^2 = 2 * (2x - round(2x)) * 2 = 4 * (2x - round(2x))
            # So grad_penalty = 0.1 * 4 * (2*x - round(2*x))
            
            grad_U_pen = 0.4 * (2 * U - np.round(2 * U))
            grad_V_pen = 0.4 * (2 * V - np.round(2 * V))
            grad_W_pen = 0.4 * (2 * W - np.round(2 * W))
            
            grad_U += grad_U_pen
            grad_V += grad_V_pen
            grad_W += grad_W_pen
            
            # Flatten gradient
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            return total_loss, grad
        
        best_result = None
        best_loss = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + restart * 13)
            
            # Initialize with small random values
            x0 = np.random.randn(dim) * 0.1
            
            # Try to optimize
            try:
                res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception:
                continue
            
            if res.success or res.fun < best_loss:
                U, V, W = unpack(res.x)
                
                # Round to nearest half-integer
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                # Check if rounded solution is exact
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return (U_rounded, V_rounded, W_rounded)
                
                # Also check if the unrounded solution is close enough to be "exact" in float
                # (Though the problem asks for integer/half-integer, so rounding is key)
                
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = (U_rounded, V_rounded, W_rounded)
        
        # If we didn't find an exact solution, return None
        # We could try to return the best approximate, but the task requires exact.
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(nm, mp, pn) = 21
    # Known lower bounds for matrix multiplication are higher.
    # Let's try 30 to 63.
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks in increasing order? Or decreasing?
    # We want the MINIMAL rank. So we should try low ranks first.
    # But low ranks are harder to find.
    # Let's try a range.
    
    ranks_to_try = list(range(30, 64))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        result = optimize_rank(r, max_restarts=20)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break # Found the minimal rank in this sequence
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
