# EVOLVE-BLOCK-START
import numpy as np
import time


def _shannon_entropy(probs):
    """Compute Shannon entropy in bits for a probability vector."""
    probs = np.array(probs)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _compute_ratios(P, n=10):
    """
    Compute all projection entropies and the D/M ratio for distribution P.
    Returns (D, M, ratio) where D = H(x-y), M = max(H(x), H(x+y), H(x+2y)).
    """
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    
    # x - y: values range from -(n-1) to (n-1)
    vals_dy = X - Y  # shape (n, n), values in [-(n-1), n-1]
    dy_min = -(n - 1)
    dy_max = n - 1
    dy_counts = np.zeros(dy_max - dy_min + 1)
    for v in range(dy_max - dy_min + 1):
        mask = (vals_dy == dy_min + v)
        dy_counts[v] = P[mask].sum()
    D = _shannon_entropy(dy_counts)
    
    # x: values range from 0 to n-1
    x_counts = np.zeros(n)
    for x in range(n):
        x_counts[x] = P[:, x].sum()
    H0 = _shannon_entropy(x_counts)
    
    # x + y: values range from 0 to 2*(n-1)
    vals_xy = X + Y
    xy_min = 0
    xy_max = 2 * (n - 1)
    xy_counts = np.zeros(xy_max - xy_min + 1)
    for v in range(xy_max - xy_min + 1):
        mask = (vals_xy == xy_min + v)
        xy_counts[v] = P[mask].sum()
    H1 = _shannon_entropy(xy_counts)
    
    # x + 2*y: values range from 0 to (n-1) + 2*(n-1) = 3*(n-1)
    vals_x2y = X + 2 * Y
    x2y_min = 0
    x2y_max = 3 * (n - 1)
    x2y_counts = np.zeros(x2y_max - x2y_min + 1)
    for v in range(x2y_max - x2y_min + 1):
        mask = (vals_x2y == x2y_min + v)
        x2y_counts[v] = P[mask].sum()
    H2 = _shannon_entropy(x2y_counts)
    
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return D, M, 0.0
    return D, M, D / M


def _greedy_set_cover(n=10, max_cells=100, deadline=None):
    """
    Greedy set-cover algorithm to find sparse support set maximizing D/M.
    Iteratively adds the cell that gives the best marginal improvement in D/M.
    """
    # Start with empty set
    selected = set()
    P = np.zeros((n, n))
    
    # Precompute cell coordinates
    cells = [(y, x) for y in range(n) for x in range(n)]
    
    # We want to maximize D/M. Since M is in the denominator, we want
    # high D (spread in x-y) and low M (concentrated in x, x+y, x+2y).
    # 
    # Strategy: greedily add cells that maximize the marginal gain in D/M.
    # But D/M is not submodular, so we use a heuristic.
    #
    # Alternative: maximize D - lambda*M for some lambda, which is closer to submodular.
    # Let's try maximizing D - 0.5*M as a proxy.
    
    best_ratio = 0.0
    best_P = P.copy()
    
    for iteration in range(max_cells):
        if deadline is not None and time.time() > deadline:
            break
        
        best_gain = -np.inf
        best_cell = None
        best_new_ratio = 0.0
        
        for (y, x) in cells:
            if (y, x) in selected:
                continue
            
            P_trial = P.copy()
            P_trial[y, x] = 1.0
            total = P_trial.sum()
            if total <= 1e-9:
                continue
            P_trial = P_trial / total
            
            D, M, ratio = _compute_ratios(P_trial, n)
            
            if ratio > best_new_ratio:
                best_new_ratio = ratio
                best_gain = ratio - best_ratio
                best_cell = (y, x)
        
        if best_cell is None:
            break
        
        # Add the best cell
        selected.add(best_cell)
        P[best_cell] = 1.0
        total = P.sum()
        P = P / total
        
        D, M, ratio = _compute_ratios(P, n)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()
        
        # Check if we should stop (diminishing returns)
        if best_gain < 1e-6:
            break
    
    return best_P, best_ratio


def _random_restart_greedy(n=10, num_restarts=50, deadline=None):
    """
    Multiple random restarts of greedy set-cover with different initial seeds.
    """
    best_P = None
    best_ratio = 0.0
    
    np.random.seed(42)
    
    for restart in range(num_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initial subset of cells
        init_size = np.random.randint(1, 20)
        init_cells = np.random.choice(n * n, size=init_size, replace=False)
        
        P_init = np.zeros((n, n))
        for idx in init_cells:
            y, x = divmod(idx, n)
            P_init[y, x] = 1.0
        
        total = P_init.sum()
        if total > 1e-9:
            P_init = P_init / total
        
        # Run greedy from this initial point
        selected = set()
        for idx in init_cells:
            y, x = divmod(idx, n)
            selected.add((y, x))
        
        P = P_init.copy()
        D, M, ratio = _compute_ratios(P, n)
        current_ratio = ratio
        
        # Continue greedy from here
        for iteration in range(50):
            if deadline is not None and time.time() > deadline:
                break
            
            best_gain = -np.inf
            best_cell = None
            best_new_ratio = current_ratio
            
            cells = [(y, x) for y in range(n) for x in range(n) if (y, x) not in selected]
            
            for (y, x) in cells:
                P_trial = P.copy()
                P_trial[y, x] = 1.0
                total = P_trial.sum()
                if total <= 1e-9:
                    continue
                P_trial = P_trial / total
                
                D, M, r = _compute_ratios(P_trial, n)
                if r > best_new_ratio:
                    best_new_ratio = r
                    best_gain = r - current_ratio
                    best_cell = (y, x)
            
            if best_cell is None:
                break
            
            selected.add(best_cell)
            P[best_cell] = 1.0
            total = P.sum()
            P = P / total
            
            D, M, ratio = _compute_ratios(P, n)
            current_ratio = ratio
            
            if best_gain < 1e-6:
                break
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_P = P.copy()
    
    return best_P, best_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization using discrete greedy set-cover.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin
    
    n = 10
    
    # Phase 1: Try structured initializations (diagonals, etc.)
    best_P = np.ones((n, n)) / (n * n)
    best_ratio = _compute_ratios(best_P, n)[2]
    
    # Diagonal x+y=c
    for c in range(2 * n - 1):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = c - x
            if 0 <= y < n:
                P_init[y, x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratios(P_init, n)[2]
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Diagonal x+2y=c
    for c in range(3 * n):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = (c - x) / 2.0
            if y == int(y) and 0 <= int(y) < n:
                P_init[int(y), x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratios(P_init, n)[2]
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Anti-diagonal x-y=c
    for c in range(-(n - 1), n):
        P_init = np.zeros((n, n))
        for x in range(n):
            y = x - c
            if 0 <= y < n:
                P_init[y, x] = 1.0
        if P_init.sum() > 0:
            P_init /= P_init.sum()
            r = _compute_ratios(P_init, n)[2]
            if r > best_ratio:
                best_ratio = r
                best_P = P_init.copy()
    
    # Phase 2: Greedy set-cover from best structured init
    if time.time() < deadline - 30:
        P_greedy, ratio_greedy = _greedy_set_cover(n, max_cells=100, deadline=deadline - 30)
        if ratio_greedy > best_ratio:
            best_ratio = ratio_greedy
            best_P = P_greedy
    
    # Phase 3: Random restarts with greedy
    if time.time() < deadline - 10:
        P_restart, ratio_restart = _random_restart_greedy(n, num_restarts=100, deadline=deadline - 10)
        if P_restart is not None and ratio_restart > best_ratio:
            best_ratio = ratio_restart
            best_P = P_restart
    
    # Final validation
    P_out = np.clip(best_P, 0.0, None)
    total = P_out.sum()
    if total <= 1e-9:
        P_out = np.ones((n, n)) / (n * n)
    else:
        P_out = P_out / total
    
    return P_out


# EVOLVE-BLOCK-END
