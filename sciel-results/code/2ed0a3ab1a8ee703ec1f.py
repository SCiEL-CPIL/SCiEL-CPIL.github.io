"""Matrix multiplication 2x3 x 3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin (150s kill, 20s buffer)
    
    # Build the target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    # Precompute: for each a, which (b,c) pairs are targets
    targets_by_a = {a: [] for a in range(dim_u)}
    targets_by_b = {b: [] for b in range(dim_v)}
    targets_by_c = {c: [] for c in range(dim_w)}
    for (a, b, c) in target_indices:
        targets_by_a[a].append((b, c))
        targets_by_b[b].append((a, c))
        targets_by_c[c].append((a, b))
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    def compute_residual(U, V, W):
        """Compute the residual for the target entries."""
        res = 0.0
        for (a, b, c) in target_indices:
            val = np.sum(U[a] * V[b] * W[c])
            res += (val - 1.0) ** 2
        return res
    
    def verify_full(U, V, W, tol=1e-8):
        """Full verification of the decomposition."""
        # Check all target entries
        for (a, b, c) in target_indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        # Check all non-target entries are zero
        target_set = set(target_indices)
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    if (a, b, c) not in target_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > tol:
                            return False
        return True
    
    def round_to_half(X):
        return np.round(X * 2) / 2.0
    
    def als_solve_U(V, W, rank):
        """Solve for U given V and W using alternating least squares."""
        U = np.zeros((dim_u, rank))
        for a in range(dim_u):
            if not targets_by_a[a]:
                continue
            # For this a, we need: sum_d U[a,d] * V[b,d] * W[c,d] = 1 for each (b,c)
            # This is: U[a] @ (V[b] * W[c]) = 1
            # Stack all equations: M @ U[a] = 1 where M has rows V[b]*W[c]
            rows = []
            for (b, c) in targets_by_a[a]:
                rows.append(V[b] * W[c])
            if not rows:
                continue
            M = np.array(rows)  # shape (num_targets_for_a, rank)
            rhs = np.ones(M.shape[0])
            # Solve M @ U[a] = rhs using least squares
            try:
                U[a] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                U[a] = np.zeros(rank)
        return U
    
    def als_solve_V(U, W, rank):
        """Solve for V given U and W using alternating least squares."""
        V = np.zeros((dim_v, rank))
        for b in range(dim_v):
            if not targets_by_b[b]:
                continue
            rows = []
            for (a, c) in targets_by_b[b]:
                rows.append(U[a] * W[c])
            if not rows:
                continue
            M = np.array(rows)
            rhs = np.ones(M.shape[0])
            try:
                V[b] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                V[b] = np.zeros(rank)
        return V
    
    def als_solve_W(U, V, rank):
        """Solve for W given U and V using alternating least squares."""
        W = np.zeros((dim_w, rank))
        for c in range(dim_w):
            if not targets_by_c[c]:
                continue
            rows = []
            for (a, b) in targets_by_c[c]:
                rows.append(U[a] * V[b])
            if not rows:
                continue
            M = np.array(rows)
            rhs = np.ones(M.shape[0])
            try:
                W[c] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                W[c] = np.zeros(rank)
        return W
    
    def optimize_rank_als(rank, max_iter=50, seed=42):
        """Try to find a rank-r decomposition using ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(dim_u, rank) * 0.1
        V = rng.randn(dim_v, rank) * 0.1
        W = rng.randn(dim_w, rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Alternate solving for U, V, W
            U = als_solve_U(V, W, rank)
            V = als_solve_V(U, W, rank)
            W = als_solve_W(U, V, rank)
            
            loss = compute_residual(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
        
        return best_U, best_V, best_W, best_loss
    
    def polish(U, V, W, rank, max_iter=200):
        """Polish the solution with gradient descent at small learning rate."""
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            lr = 0.005 / (1 + iteration * 0.05)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in target_indices:
                val = np.sum(U[a] * V[b] * W[c])
                err = val - 1.0
                grad_U[a] += 2 * err * V[b] * W[c]
                grad_V[b] += 2 * err * U[a] * W[c]
                grad_W[c] += 2 * err * U[a] * V[b]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        return U, V, W
    
    # Try ranks from schoolbook down to a reasonable minimum
    # For 2x3 x 3x9, we expect rank around 30-40
    min_rank_to_try = max(20, schoolbook_rank - 30)
    
    for rank in range(schoolbook_rank, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline:
                break
            
            U, V, W, loss = optimize_rank_als(rank, max_iter=40, seed=trial * 1000 + rank)
            
            if loss < 1e-6:
                # Good solution found, polish it
                U, V, W = polish(U, V, W, rank, max_iter=100)
                loss = compute_residual(U, V, W)
                
                if loss < 1e-8:
                    # Round to half-integers
                    U_r = round_to_half(U)
                    V_r = round_to_half(V)
                    W_r = round_to_half(W)
                    
                    if verify_full(U_r, V_r, W_r, tol=1e-6):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                        break
                    else:
                        # Check if unrounded is valid and close to half-integers
                        if verify_full(U, V, W, tol=1e-8):
                            U_close = np.allclose(U, U_r, atol=0.01)
                            V_close = np.allclose(V, V_r, atol=0.01)
                            W_close = np.allclose(W, W_r, atol=0.01)
                            if U_close and V_close and W_close:
                                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                                best_rank = rank
                                found = True
                                break
        
        if found:
            continue
        else:
            # If we couldn't find for this rank, stop (higher ranks are harder)
            # Actually, continue trying lower ranks as they might be easier
            pass
    
    return best_result
# EVOLVE-BLOCK-END
