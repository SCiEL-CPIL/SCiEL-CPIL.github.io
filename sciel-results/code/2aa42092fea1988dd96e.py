# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _area_and_subgradient(x, n=64, M=20000):
    """
    Compute area and subgradient of the union area w.r.t. x.
    The subgradient is derived from the fact that area is a function of the
    sorted L values, and d(area)/d(x_i) = sum over y where i is the "active"
    triangle in the union boundary.
    
    For the union of intervals [L_i, L_i + w], the area at height y is:
    union_length = w + sum_{consecutive pairs in sorted order} min(gap, w)
    
    The subgradient w.r.t. x_i is computed by tracking which triangles are
    at the boundary of the union.
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    # Sort L at each height
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    # Union length: w + sum of min(gap, w) for consecutive pairs
    # The subgradient w.r.t. L_sorted[j, m] for the j-th smallest triangle:
    # - If j == 0 (leftmost): contributes -1 if this triangle extends the union to the left
    #   Actually, let's think more carefully.
    #
    # union_length(y) = max_i(L_i + w) - min_i(L_i) + corrections for gaps > w
    # More precisely: union_length = w + sum_{j=1}^{n-1} min(gap_j, w)
    # where gap_j = L_sorted[j] - L_sorted[j-1]
    #
    # d(union_length)/d(L_sorted[j]) for the sorted values:
    # For the leftmost (j=0): -1 (moving it left increases union by 1 per unit)
    #   Actually no. Let's think of it as:
    #   union = (L_sorted[n-1] + w) - L_sorted[0] - sum_{j=1}^{n-1} max(gap_j - w, 0)
    #   = w + L_sorted[n-1] - L_sorted[0] - sum_{j=1}^{n-1} max(gap_j - w, 0)
    #
    # d(union)/d(L_sorted[0]) = -1 + max(gap_1 - w, 0)'s derivative w.r.t. L_sorted[0]
    #   gap_1 = L_sorted[1] - L_sorted[0], so d(gap_1)/d(L_sorted[0]) = -1
    #   d(max(gap_1 - w, 0))/d(L_sorted[0]) = -1 if gap_1 > w, else 0
    #   So d(union)/d(L_sorted[0]) = -1 + (1 if gap_1 > w else 0)
    #
    # For interior j (1 <= j <= n-2):
    #   gap_j = L_sorted[j] - L_sorted[j-1], gap_{j+1} = L_sorted[j+1] - L_sorted[j]
    #   d(union)/d(L_sorted[j]) = -d(max(gap_j-w,0))/d(L_sorted[j]) - d(max(gap_{j+1}-w,0))/d(L_sorted[j])
    #   = -(-1 if gap_j > w else 0) - (1 if gap_{j+1} > w else 0)
    #   = (1 if gap_j > w else 0) - (1 if gap_{j+1} > w else 0)
    #
    # For the rightmost (j=n-1):
    #   gap_{n-1} = L_sorted[n-1] - L_sorted[n-2]
    #   d(union)/d(L_sorted[n-1]) = 1 - (1 if gap_{n-1} > w else 0)
    
    # Compute subgradients for sorted positions
    n_tri = n
    grad_sorted = np.zeros((n_tri, M), dtype=np.float64)
    
    # Leftmost
    gap1 = gaps[0, :]
    grad_sorted[0, :] = -1.0 + (gap1 > w[0, :]).astype(np.float64)
    
    # Interior
    for j in range(1, n_tri - 1):
        grad_sorted[j, :] = (gaps[j-1, :] > w[0, :]).astype(np.float64) - (gaps[j, :] > w[0, :]).astype(np.float64)
    
    # Rightmost
    gap_last = gaps[-1, :]
    grad_sorted[n_tri - 1, :] = 1.0 - (gap_last > w[0, :]).astype(np.float64)
    
    # Map back to original indices
    grad_L = np.zeros((n_tri, M), dtype=np.float64)
    np.put_along_axis(grad_L, idx, grad_sorted, axis=0)
    
    # d(area)/d(x_i) = mean over y of grad_L[i, m] * a_i
    # because L_i = x_i + a_i * y, so d(L_i)/d(x_i) = 1
    # Actually d(L_i)/d(x_i) = 1, so grad_x[i] = mean_m(grad_L[i, m])
    grad_x = np.mean(grad_L, axis=1)
    
    # Compute area
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    
    return area, grad_x


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _softmin_area_and_grad(x, n=64, M=20000, beta=100.0):
    """
    Compute a smooth approximation of the union area using soft-min.
    The union length at height y can be written as:
    union = max(L + w) - min(L) - sum of "excess gaps"
    
    We use a soft version:
    soft_min(L) = -(1/beta) * log(sum(exp(-beta * L)))
    soft_max(L) = (1/beta) * log(sum(exp(beta * L)))
    
    For the gap corrections, we use a soft version of max(gap - w, 0).
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    # Soft min and max
    # soft_min(L) = -(1/beta) * log(sum(exp(-beta * L)))
    neg_beta_L = -beta * L
    neg_beta_L -= neg_beta_L.max(axis=0, keepdims=True)  # for numerical stability
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(neg_beta_L), axis=0))
    
    beta_L = beta * L
    beta_L -= beta_L.max(axis=0, keepdims=True)
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta_L), axis=0))
    
    # For the gap corrections, we need sum of max(gap_j - w, 0)
    # This is harder to make smooth. Let's use a different approach.
    # 
    # Alternative: The union area can be computed as:
    # area = integral of [soft_max(L + w) - soft_min(L) - penalty_for_gaps] dy
    #
    # But the gap penalty is tricky. Let's use the exact sorting approach
    # but with a smoothed version of the min function.
    #
    # Actually, let's just use the exact area for the objective and compute
    # the gradient via the subgradient method.
    
    # Fall back to exact computation
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    # Use soft version of min(gap, w)
    # min(a, b) = (a + b - |a - b|) / 2
    # |a - b| can be approximated by sqrt((a-b)^2 + eps)
    eps = 1e-8
    soft_min_gap = 0.5 * (gaps + w[None, :] - np.sqrt((gaps - w[None, :])**2 + eps))
    
    union_lengths = w + soft_min_gap.sum(axis=0)
    area = np.mean(union_lengths)
    
    # Gradient via finite differences would be too slow.
    # Use the subgradient from the exact formulation.
    _, grad_x = _area_and_subgradient(x, n, M)
    
    return area, grad_x


def _gradient_descent(x0, n=64, M=20000, deadline=None, max_iter=5000, lr_init=0.01, lr_decay=0.999):
    """
    Gradient descent with the subgradient of the union area.
    """
    x = x0.copy()
    best_x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    
    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        area, grad = _area_and_subgradient(x, n, M)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        x_new = x - lr * grad
        
        area_new = _compute_area_fast(x_new, n, M)
        
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
            x = x_new
        else:
            # Reduce learning rate
            lr *= 0.95
        
        lr *= lr_decay
    
    return best_x, best_area


def _simulated_annealing(x0, n=64, M=20000, deadline=None, T_start=0.1, T_end=0.001, alpha=0.9995, max_iter=50000):
    """
    Simulated annealing with large jumps that change the ordering structure.
    """
    x = x0.copy()
    best_x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    current_area = best_area
    
    rng = np.random.default_rng(12345)
    
    T = T_start
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Generate a perturbation: move 1-3 coordinates by a random amount
        n_move = rng.integers(1, 4)
        indices = rng.choice(n, size=n_move, replace=False)
        
        # Scale of jump depends on temperature
        scale = T * 2.0
        delta = rng.normal(0, scale, size=n_move)
        
        x_new = x.copy()
        x_new[indices] += delta
        
        area_new = _compute_area_fast(x_new, n, M)
        
        # Metropolis criterion
        if area_new < current_area:
            x = x_new
            current_area = area_new
            if area_new < best_area:
                best_area = area_new
                best_x = x_new.copy()
        else:
            delta_area = area_new - current_area
            if delta_area < 1e-12:
                x = x_new
                current_area = area_new
            elif rng.random() < np.exp(-delta_area / max(T, 1e-15)):
                x = x_new
                current_area = area_new
        
        T = max(T * alpha, T_end)
    
    return best_x, best_area


def _local_refinement(x, n=64, M=20000, deadline=None, max_rounds=100):
    """
    Aggressive coordinate descent with adaptive step sizes.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    L_base = a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    def _area_with_xi(xi_val, xi_idx):
        L = x[:, None] + L_base
        L[xi_idx, :] = xi_val + L_base[xi_idx, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
        return np.mean(union_lengths)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 15:
                break
            
            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area
            
            for step in step_sizes:
                for direction in [1, -1]:
                    new_xi = current_xi + direction * step
                    new_area = _area_with_xi(new_xi, i)
                    if new_area < best_local_area - 1e-15:
                        best_local_area = new_area
                        best_xi = new_xi
            
            if best_xi != current_xi:
                x[i] = best_xi
                best_area = best_local_area
                improved = True
        
        if not improved:
            break
    
    return x, best_area


def _subgradient_polish(x, n=64, M=20000, deadline=None, max_iter=2000, lr_init=0.005):
    """
    Subgradient-based polishing with adaptive step sizes.
    """
    x = x.copy()
    best_x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    
    lr = lr_init
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 20:
            break
        
        area, grad = _area_and_subgradient(x, n, M)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        x_new = x - lr * grad / grad_norm
        
        area_new = _compute_area_fast(x_new, n, M)
        
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
            x = x_new
            lr = min(lr * 1.05, 0.02)
        else:
            lr *= 0.9
        
        if lr < 1e-6:
            break
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a combination of:
    1. Simulated annealing to explore the combinatorial landscape
    2. Subgradient-based optimization for smooth refinement
    3. Coordinate descent for final polishing
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    n = 64
    M = 20000
    
    # Start from Keich construction
    x_keich = _keich_initial(n)
    
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    # Phase 1: Simulated annealing with multiple restarts
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:
            break
        
        if restart == 0:
            x0 = x_keich.copy()
        elif restart == 1:
            x0 = x_keich + rng.normal(0, 0.05, size=n)
        elif restart == 2:
            x0 = x_keich + rng.normal(0, 0.1, size=n)
        elif restart == 3:
            x0 = rng.uniform(-0.5, 0.5, size=n)
        else:
            x0 = best_x + rng.normal(0, 0.02, size=n)
        
        try:
            sa_x, sa_area = _simulated_annealing(
                x0, n, M, deadline=deadline,
                T_start=0.15, T_end=0.0005, alpha=0.9997, max_iter=30000
            )
            if sa_area < best_area:
                best_area = sa_area
                best_x = sa_x.copy()
        except Exception:
            continue
    
    # Phase 2: Subgradient polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            poly_x, poly_area = _subgradient_polish(
                best_x, n, M, deadline=deadline, max_iter=3000, lr_init=0.005
            )
            if poly_area < best_area:
                best_area = poly_area
                best_x = poly_x.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            refined_x, refined_area = _local_refinement(
                best_x, n, M, deadline=deadline, max_rounds=150
            )
            if refined_area < best_area:
                best_area = refined_area
                best_x = refined_x.copy()
        except Exception:
            pass
    
    # Phase 4: Another round of SA from the refined solution if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            sa_x2, sa_area2 = _simulated_annealing(
                best_x, n, M, deadline=deadline,
                T_start=0.05, T_end=0.0001, alpha=0.9998, max_iter=20000
            )
            if sa_area2 < best_area:
                best_area = sa_area2
                best_x = sa_x2.copy()
                
            if time.time() < deadline - 20:
                refined_x2, refined_area2 = _local_refinement(
                    best_x, n, M, deadline=deadline, max_rounds=100
                )
                if refined_area2 < best_area:
                    best_area = refined_area2
                    best_x = refined_x2.copy()
                    
                if time.time() < deadline - 15:
                    poly_x2, poly_area2 = _subgradient_polish(
                        best_x, n, M, deadline=deadline, max_iter=1000, lr_init=0.003
                    )
                    if poly_area2 < best_area:
                        best_area = poly_area2
                        best_x = poly_x2.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
