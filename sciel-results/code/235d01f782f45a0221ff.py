import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 10-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j, j * p + k, k * n + i)
                T[idx] = 1.0
                nz_indices.append(idx)
    nz_indices = np.array(nz_indices, dtype=int)  # shape (63, 3)
    nz_a = nz_indices[:, 0]
    nz_b = nz_indices[:, 1]
    nz_c = nz_indices[:, 2]
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        # Check at non-zero positions
        rec_nz = np.zeros(len(nz_a))
        for r_idx in range(r):
            rec_nz += U[nz_a, r_idx] * V[nz_b, r_idx] * W[nz_c, r_idx]
        if np.max(np.abs(rec_nz - 1.0)) > tol:
            return False
        # Check a sample of zero positions
        all_positions = np.array(np.unravel_index(np.arange(nm * mp * pn), (nm, mp, pn)))
        # Find positions not in nz_indices
        nz_set = set(map(tuple, nz_indices))
        zero_positions = [pos for pos in all_positions if pos not in nz_set]
        # Sample up to 100 zero positions
        np.random.seed(0)
        sample_idx = np.random.choice(len(zero_positions), min(100, len(zero_positions)), replace=False)
        for si in sample_idx:
            a, b, c = zero_positions[si]
            val = 0.0
            for r_idx in range(r):
                val += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
            if abs(val) > tol:
                return False
        return True
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 63
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
    
    best_result = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    best_rank = rank_sb
    
    def try_gradient_descent(r, max_iters=300, lr=0.05, seed=42):
        """Try to find a rank-r decomposition using gradient descent with sparse evaluation."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute loss at non-zero positions only
            rec_nz = np.zeros(len(nz_a))
            for r_idx in range(r):
                rec_nz += U[nz_a, r_idx] * V[nz_b, r_idx] * W[nz_c, r_idx]
            error_nz = rec_nz - 1.0
            loss = 0.5 * np.dot(error_nz, error_nz)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            # Compute gradients at non-zero positions
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                err = error_nz
                vb = V[nz_b, r_idx]
                wc = W[nz_c, r_idx]
                ua = U[nz_a, r_idx]
                
                contrib_U = err * vb * wc
                np.add.at(grad_U, (nz_a, r_idx), contrib_U)
                
                contrib_V = err * ua * wc
                np.add.at(grad_V, (nz_b, r_idx), contrib_V)
                
                contrib_W = err * ua * vb
                np.add.at(grad_W, (nz_c, r_idx), contrib_W)
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.995
        
        # Final check
        if check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    # Try ranks from 62 down to 50
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        result = try_gradient_descent(r, max_iters=200, lr=0.05, seed=42)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
