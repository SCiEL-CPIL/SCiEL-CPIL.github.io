# EVOLVE-BLOCK-START
import numpy as np


def _shannon_entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_ratios(P):
    """Compute D and M for a distribution P[y, x]."""
    n = 10
    # x and y indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)  # X[y,x]=x, Y[y,x]=y
    
    # D: distribution of x - y
    diff = X - Y  # shape (n,n), values in -9..9
    diff_vals = diff.ravel()
    diff_probs = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            idx = diff[i, j] + (n - 1)
            diff_probs[idx] += P[i, j]
    D = _shannon_entropy(diff_probs)
    
    # M: max of H_0, H_1, H_2
    # H_0: distribution of x
    x_vals = X.ravel()
    x_probs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            x_probs[j] += P[i, j]
    H0 = _shannon_entropy(x_probs)
    
    # H_1: distribution of x + y
    sum1 = X + Y  # values 0..18
    sum1_probs = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            sum1_probs[i + j] += P[i, j]
    H1 = _shannon_entropy(sum1_probs)
    
    # H_2: distribution of x + 2y
    sum2 = X + 2 * Y  # values 0..27
    sum2_probs = np.zeros(3 * n - 1)
    for i in range(n):
        for j in range(n):
            sum2_probs[i + 2 * j] += P[i, j]
    H2 = _shannon_entropy(sum2_probs)
    
    M = max(H0, H1, H2)
    return D, M


def _compute_ratios_fast(P):
    """Faster computation using numpy operations."""
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')  # X[i,j] = i (y-index), Y[i,j]=j (x-index)
    # Wait, P[y,x], so P[i,j] where i=y, j=x
    # X = x coordinate = j, Y = y coordinate = i
    
    # Let me redo: P is indexed as P[y, x], so P[i, j] = prob(y=i, x=j)
    # x = j, y = i
    x_coords = np.arange(n)  # 0..9
    y_coords = np.arange(n)  # 0..9
    
    # x - y: for cell (y=i, x=j), value = j - i
    diff = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            diff[i, j] = j - i  # x - y
    
    # x: value = j
    x_val = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            x_val[i, j] = j
    
    # x + y: value = j + i
    sum1_val = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sum1_val[i, j] = j + i
    
    # x + 2y: value = j + 2*i
    sum2_val = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sum2_val[i, j] = j + 2 * i
    
    # Compute entropies
    def _entropy_from_values(values, probs):
        """values: (n,n) array of integer values, probs: (n,n) array of probabilities"""
        unique_vals, inverse = np.unique(values, return_inverse=True)
        n_unique = len(unique_vals)
        dist = np.zeros(n_unique)
        for k in range(n_unique):
            dist[k] = np.sum(probs[inverse == k])
        return _shannon_entropy(dist)
    
    D = _entropy_from_values(diff, P)
    H0 = _entropy_from_values(x_val, P)
    H1 = _entropy_from_values(sum1_val, P)
    H2 = _entropy_from_values(sum2_val, P)
    M = max(H0, H1, H2)
    return D, M


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid using greedy submodular maximization.
    """
    n = 10
    total_cells = n * n
    
    # Precompute value arrays
    diff_vals = np.zeros((n, n), dtype=np.int64)
    x_vals = np.zeros((n, n), dtype=np.int64)
    sum1_vals = np.zeros((n, n), dtype=np.int64)
    sum2_vals = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):  # y
        for j in range(n):  # x
            diff_vals[i, j] = j - i + (n - 1)  # offset to make non-negative: 0..18
            x_vals[i, j] = j  # 0..9
            sum1_vals[i, j] = j + i  # 0..18
            sum2_vals[i, j] = j + 2 * i  # 0..27
    
    num_diff = 2 * n - 1  # 19
    num_x = n  # 10
    num_sum1 = 2 * n - 1  # 19
    num_sum2 = 3 * n - 1  # 28
    
    def _compute_D_M(P):
        """Compute D and M efficiently."""
        # D: x - y
        diff_dist = np.zeros(num_diff)
        for k in range(num_diff):
            mask = (diff_vals == k)
            diff_dist[k] = np.sum(P[mask])
        D = _shannon_entropy(diff_dist)
        
        # H0: x
        x_dist = np.zeros(num_x)
        for k in range(num_x):
            mask = (x_vals == k)
            x_dist[k] = np.sum(P[mask])
        H0 = _shannon_entropy(x_dist)
        
        # H1: x + y
        sum1_dist = np.zeros(num_sum1)
        for k in range(num_sum1):
            mask = (sum1_vals == k)
            sum1_dist[k] = np.sum(P[mask])
        H1 = _shannon_entropy(sum1_dist)
        
        # H2: x + 2y
        sum2_dist = np.zeros(num_sum2)
        for k in range(num_sum2):
            mask = (sum2_vals == k)
            sum2_dist[k] = np.sum(P[mask])
        H2 = _shannon_entropy(sum2_dist)
        
        M = max(H0, H1, H2)
        return D, M
    
    # Greedy submodular maximization
    # Start with empty support, add cells one by one
    # We'll use a fixed number of cells (support size) and assign equal weight to each
    
    best_P = np.ones((n, n), dtype=np.float64) / total_cells
    best_ratio = _compute_D_M(best_P)[0] / max(_compute_D_M(best_P)[1], 1e-15)
    
    # Try different support sizes
    for support_size in range(5, total_cells + 1):
        # Greedily add cells to maximize D - lambda * M
        # We'll use a penalty approach: maximize D - 0.5 * M
        P = np.zeros((n, n), dtype=np.float64)
        added = np.zeros((n, n), dtype=bool)
        
        for step in range(support_size):
            best_gain = -np.inf
            best_cell = None
            best_D = 0
            best_M = 0
            
            # Try adding each unadded cell
            for i in range(n):
                for j in range(n):
                    if added[i, j]:
                        continue
                    
                    # Tentatively add this cell with equal weight
                    P_new = P.copy()
                    P_new[i, j] = 1.0 / (step + 1)
                    # Also redistribute existing weights
                    for ii in range(n):
                        for jj in range(n):
                            if added[ii, jj]:
                                P_new[ii, jj] = 1.0 / (step + 1)
                    
                    D_new, M_new = _compute_D_M(P_new)
                    if M_new < 1e-12:
                        gain = D_new
                    else:
                        gain = D_new / M_new
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_cell = (i, j)
                        best_D = D_new
                        best_M = M_new
            
            if best_cell is None:
                break
            
            # Commit to adding this cell
            added[best_cell[0], best_cell[1]] = True
            # Reset P and assign equal weights to all added cells
            P = np.zeros((n, n), dtype=np.float64)
            for ii in range(n):
                for jj in range(n):
                    if added[ii, jj]:
                        P[ii, jj] = 1.0 / support_size  # Final weight will be 1/support_size
                        # But during greedy, we use 1/(step+1) for the gain calculation
            # Actually, let's just track the set and compute at the end
            # For now, update P with equal weights for greedy evaluation
            P = np.zeros((n, n), dtype=np.float64)
            for ii in range(n):
                for jj in range(n):
                    if added[ii, jj]:
                        P[ii, jj] = 1.0 / (step + 1)
        
        # Final distribution: equal weight on all support cells
        P_final = np.zeros((n, n), dtype=np.float64)
        num_added = np.sum(added)
        if num_added == 0:
            continue
        for ii in range(n):
            for jj in range(n):
                if added[ii, jj]:
                    P_final[ii, jj] = 1.0 / num_added
        
        D_final, M_final = _compute_D_M(P_final)
        if M_final < 1e-9:
            continue
        ratio_final = D_final / M_final
        
        if ratio_final > best_ratio:
            best_ratio = ratio_final
            best_P = P_final.copy()
    
    # Clip and normalize
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total < 1e-9:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / total_cells
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
