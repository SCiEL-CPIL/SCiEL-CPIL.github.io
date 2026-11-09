import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-minimal (or best found) decomposition for 3x3 by 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 63
    
    def make_schoolbook():
        """Return the standard schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
        r = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check consistent indices sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check inconsistent indices sum to 0
        # We can do this by checking the full tensor or sampling.
        # For safety, check all inconsistent indices?
        # There are (nm*mp*pn) - nmp indices.
        # 9*21*21 - 63 = 3969 - 63 = 3906.
        # This is fast enough with numpy.
        
        # Construct full tensor S
        # S[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # We can compute this via einsum or loops.
        # Given R is small (~60), and sizes are small, we can just loop r and add outer products?
        # Or use the fast loss calculation logic.
        
        # Let's use the fast check:
        # Sum of squares of all entries should be Sum of squares of support entries (if zeros are 0).
        # Actually, just compute the loss.
        loss = compute_loss(U_arr, V_arr, W_arr)
        return loss < tol
    
    def compute_loss(U, V, W):
        """Compute the loss: sum(S - T)^2 over all indices."""
        R = U.shape[1]
        # Gram matrices
        GU = U.T @ U
        GV = V.T @ V
        GW = W.T @ W
        
        # Sum of squares of all entries
        total_sq = np.sum(GU * GV * GW)
        
        # Sum of support entries
        support_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # S[i,j,k] = dot(U[u], V[v]*W[w])
                    support_sum += np.dot(U[u_idx], V[v_idx] * W[w_idx])
        
        # Loss = total_sq - 2 * support_sum + nmp
        return total_sq - 2.0 * support_sum + n * m * p

    def compute_grads(U, V, W):
        """Compute gradients of the loss w.r.t U, V, W."""
        R = U.shape[1]
        GU = U.T @ U
        GV = V.T @ V
        GW = W.T @ W
        
        # H matrices
        H_VW = GV * GW
        H_UW = GU * GW
        H_UV = GU * GV
        
        # Gradient of total_sq term
        grad_U_sq = 2.0 * U @ H_VW.T
        grad_V_sq = 2.0 * V @ H_UW.T
        grad_W_sq = 2.0 * W @ H_UV.T
        
        # Gradient of support_sum term
        # d/dU[a,r] (2 * sum_s S_s) = -2 * sum_{s: u_s=a} V[v_s, r] W[w_s, r]
        grad_U_sup = np.zeros_like(U)
        grad_V_sup = np.zeros_like(V)
        grad_W_sup = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # For U[u_idx, r], the term is V[v_idx, r] * W[w_idx, r]
                    grad_U_sup[u_idx] -= 2.0 * (V[v_idx] * W[w_idx])
                    grad_V_sup[v_idx] -= 2.0 * (U[u_idx] * W[w_idx])
                    grad_W_sup[w_idx] -= 2.0 * (U[u_idx] * V[v_idx])
                    
        return grad_U_sq + grad_U_sup, grad_V_sq + grad_V_sup, grad_W_sq + grad_W_sup

    def try_random_rank(target_rank, num_trials=5, max_iters=5000, lr=0.01):
        """Try to find a decomposition of the given rank using random initialization and gradient descent."""
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(trial * 7 + 12345)
            
            # Initialize with small random values
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                loss = compute_loss(U, V, W)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if best_loss < 1e-9:
                    break
                
                grad_U, grad_V, grad_W = compute_grads(U, V, W)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate?
                # If loss increases, reduce lr
                # Simple: just keep it fixed for now.
            
            # Check best found
            if best_loss < 1e-7:
                # Snap to nearest half-integer
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                
                # Verify snapped version
                if compute_loss(U_snap, V_snap, W_snap) < 1e-9:
                    # Convert to list
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
        
        return None

    best_result = make_schoolbook()
    best_rank = rank_school
    
    # Try ranks lower than schoolbook
    target_ranks = [62, 61, 60, 59, 58, 57, 56]
    
    for tr in target_ranks:
        if time.time() > deadline:
            break
        if tr >= best_rank:
            continue
        result = try_random_rank(tr, num_trials=10, max_iters=2000, lr=0.05)
        if result is not None:
            best_result = result
            best_rank = tr
            break  # Found a valid lower rank, use it
    
    if best_rank == rank_school:
        return make_schoolbook()
    else:
        return best_result

# EVOLVE-BLOCK-END
