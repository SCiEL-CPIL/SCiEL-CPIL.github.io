# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _greedy_support_search(idx_a, idx_b, idx_c, idx_d, deadline):
    """
    Greedy submodular maximization heuristic to find a sparse support set S.
    Objective: Maximize D/M.
    Heuristic: Iteratively add the cell that maximizes marginal gain in D
    while penalizing increases in M.
    """
    n = 10
    total_cells = n * n
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    # Start with an empty support (all weights 0)
    # We will build a binary mask of the support
    # Then we will assign weights (initially uniform, then refined)
    
    # To make the greedy step meaningful, we need to evaluate the objective
    # for a candidate support.
    # Let's maintain a set of active cells.
    # Start with a small random set or a specific structural seed.
    
    # Seed 1: Cross (x=0 or y=0)
    # Seed 2: Random
    # We'll run multiple greedy searches from different seeds.
    
    best_val = -1e9
    best_support = None
    
    # Precompute bin indices for each cell
    # cell -> (bin_a, bin_b, bin_c, bin_d)
    cell_bins = np.zeros((total_cells, 4), dtype=np.int64)
    cell_bins[:, 0] = idx_a
    cell_bins[:, 1] = idx_b
    cell_bins[:, 2] = idx_c
    cell_bins[:, 3] = idx_d
    
    def eval_support(support_set):
        if len(support_set) == 0:
            return -1e9
        # Uniform weights on support
        w = np.ones(len(support_set)) / len(support_set)
        # Aggregate to bins
        bins_a = cell_bins[support_set, 0]
        bins_b = cell_bins[support_set, 1]
        bins_c = cell_bins[support_set, 2]
        bins_d = cell_bins[support_set, 3]
        
        qa = np.bincount(bins_a, weights=w, minlength=la)
        qb = np.bincount(bins_b, weights=w, minlength=lb)
        qc = np.bincount(bins_c, weights=w, minlength=lc)
        qd = np.bincount(bins_d, weights=w, minlength=ld)
        
        D = _entropy(qc)
        M = max(_entropy(qa), _entropy(qb), _entropy(qd))
        if M < 1e-12:
            return -1e9
        return D / M

    # Define a few seeds
    seeds = []
    
    # Seed 1: Cross
    support = set()
    for i in range(n):
        support.add(i * n + 0) # x=0, y=i (P[y, x]) -> index y*n + x
        support.add(0 * n + i) # y=0, x=i
    seeds.append(list(support))
    
    # Seed 2: Anti-Cross
    support = set()
    for i in range(n):
        support.add(i * n + (n - 1 - i)) # x+y = n-1
    seeds.append(list(support))
    
    # Seed 3: Random 15 points
    rng = np.random.default_rng(123)
    for _ in range(5):
        support = list(rng.choice(total_cells, 15, replace=False))
        seeds.append(support)
        
    # Seed 4: Grid (x in {0, 4, 9} or y in {0, 4, 9})
    support = set()
    for i in range(n):
        for j in range(n):
            if i in [0, 4, 9] or j in [0, 4, 9]:
                support.add(j * n + i)
    seeds.append(list(support))
    
    # Greedy Expansion
    max_size = 40 # Don't make it too dense, sparsity helps M stay low
    target_time = deadline - 10.0
    
    for seed in seeds:
        if time.time() > target_time:
            break
            
        current_support = set(seed)
        current_val = eval_support(current_support)
        
        # Iteratively add cells
        for _ in range(max_size - len(current_support)):
            if time.time() > target_time:
                break
                
            best_gain = -np.inf
            best_cell = None
            
            # Try adding each cell not in support
            # Optimization: only try a subset if too slow? 100 cells is small.
            for cell in range(total_cells):
                if cell in current_support:
                    continue
                
                new_support = current_support | {cell}
                val = eval_support(new_support)
                
                # Penalty for increasing M? The ratio D/M handles it.
                # But we might want to encourage sparsity.
                # If val > current_val + threshold, consider.
                
                if val > best_gain:
                    best_gain = val
                    best_cell = cell
            
            if best_cell is not None and best_gain > current_val:
                current_support.add(best_cell)
                current_val = best_gain
            else:
                # No improvement, stop expanding this seed
                break
        
        # After greedy expansion, we have a support set.
        # Now we need to optimize the weights on this support.
        # Since the problem asks for a DIFFERENT strategy, and the current one uses L-BFGS,
        # we can use a simple coordinate ascent or even just uniform weights if we have time.
        # But uniform weights might not be optimal.
        # Let's do a few steps of gradient ascent on the weights for this support.
        
        if current_val > best_val:
            best_val = current_val
            best_support = list(current_support)
            
    # Construct P from best_support
    P = np.zeros((n, n))
    if best_support:
        P.ravel()[best_support] = 1.0
        P /= P.sum()
        
    # Refine weights on the support using a few iterations of gradient ascent
    # This is a "local refinement" step, distinct from the global L-BFGS in the seed.
    # We use a simple power iteration style update: w_i <- w_i * exp(eta * grad_i)
    
    if best_support and time.time() < deadline - 20.0:
        sup = np.array(best_support)
        w = np.ones(len(sup)) / len(sup)
        
        bins_a = cell_bins[sup, 0]
        bins_b = cell_bins[sup, 1]
        bins_c = cell_bins[sup, 2]
        bins_d = cell_bins[sup, 3]
        
        for _ in range(50):
            if time.time() > deadline - 20.0:
                break
                
            qa = np.bincount(bins_a, weights=w, minlength=la)
            qb = np.bincount(bins_b, weights=w, minlength=lb)
            qc = np.bincount(bins_c, weights=w, minlength=lc)
            qd = np.bincount(bins_d, weights=w, minlength=ld)
            
            D = _entropy(qc)
            Ha = _entropy(qa)
            Hb = _entropy(qb)
            Hd = _entropy(qd)
            M = max(Ha, Hb, Hd)
            
            if M < 1e-12:
                break
                
            val = D / M
            
            # Gradient of D wrt w
            # dH/dq = -log2(q) - 1/ln(2)
            # dq_i/dw_j = 1 if j in bin i
            # grad_D_j = sum_{i: j in bin i} (dH/dq_i) * (1)
            
            grad_D = np.zeros(len(sup))
            grad_M = np.zeros(len(sup))
            
            # D
            mask_c = qc > 1e-300
            dq_c = np.zeros_like(qc)
            dq_c[mask_c] = -(np.log2(qc[mask_c]) + 1.0 / np.log(2.0))
            # Scatter to support
            np.add.at(grad_D, bins_c, dq_c[bins_c])
            
            # M
            # Identify which H is max
            if Ha == M:
                mask_a = qa > 1e-300
                dq_a = np.zeros_like(qa)
                dq_a[mask_a] = -(np.log2(qa[mask_a]) + 1.0 / np.log(2.0))
                np.add.at(grad_M, bins_a, dq_a[bins_a])
            elif Hb == M:
                mask_b = qb > 1e-300
                dq_b = np.zeros_like(qb)
                dq_b[mask_b] = -(np.log2(qb[mask_b]) + 1.0 / np.log(2.0))
                np.add.at(grad_M, bins_b, dq_b[bins_b])
            else:
                mask_d = qd > 1e-300
                dq_d = np.zeros_like(qd)
                dq_d[mask_d] = -(np.log2(qd[mask_d]) + 1.0 / np.log(2.0))
                np.add.at(grad_M, bins_d, dq_d[bins_d])
                
            # Objective: Maximize D/M
            # Grad = (M * grad_D - D * grad_M) / M^2
            grad_obj = (M * grad_D - D * grad_M) / (M * M)
            
            # Update w
            # w <- w * exp(eta * grad_obj)
            eta = 0.5
            w = w * np.exp(eta * grad_obj)
            w = np.clip(w, 1e-10, None)
            w /= w.sum()
            
            # Check improvement
            qa = np.bincount(bins_a, weights=w, minlength=la)
            qb = np.bincount(bins_b, weights=w, minlength=lb)
            qc = np.bincount(bins_c, weights=w, minlength=lc)
            qd = np.bincount(bins_d, weights=w, minlength=ld)
            D_new = _entropy(qc)
            M_new = max(_entropy(qa), _entropy(qb), _entropy(qd))
            if M_new < 1e-12:
                break
            val_new = D_new / M_new
            if val_new < val:
                # Revert? Or just stop?
                break
        
        # Assign final weights
        P.ravel()[sup] = w
        P /= P.sum()
        
        # Final check
        val_final = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
        if val_final > best_val:
            best_val = val_final

    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Use the new greedy strategy
    P = _greedy_support_search(idx_a, idx_b, idx_c, idx_d, deadline)
    
    # Fallback if something went wrong
    if not np.all(np.isfinite(P)) or P.sum() < 1e-9:
        P = np.full((n, n), 1.0 / 100.0)
        
    # Ensure validity
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
    P = P / s
    
    # Verify M > 1e-9
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
        
    return P.astype(np.float64)

# EVOLVE-BLOCK-END
