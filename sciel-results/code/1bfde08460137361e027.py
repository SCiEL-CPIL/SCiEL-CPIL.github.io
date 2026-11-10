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


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
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
    
    # Gradient: d(area)/d(x_i)
    # The union length at each y is w + sum_j min(gap_j, w)
    # d(union)/d(x_i) = sum_j d(min(gap_j, w))/d(x_i)
    # If gap_j < w (overlap), d(min(gap_j, w))/d(x_i) = d(gap_j)/d(x_i)
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(gap_j)/d(x_i) = 1 if i is the (j+1)-th in sorted order, -1 if i is the j-th, 0 otherwise
    
    active = gaps < w[None, :]  # (n-1, M)
    
    # For each position in sorted order, determine which original index it came from
    # idx[k, m] is the original index of the k-th smallest L at height m
    # So for gap j (between sorted position j and j+1), 
    # d(gap_j)/d(x_i) = 1 if idx[j+1, m] == i, -1 if idx[j, m] == i
    
    # Build gradient using bincount-style accumulation
    grad = np.zeros((n, M))
    # For each gap j, at each m:
    #   if active[j, m]: grad[idx[j+1, m], m] += 1, grad[idx[j, m], m] -= 1
    
    # Vectorized: use np.add.at
    for j in range(n - 1):
        mask = active[j, :]  # (M,)
        if not np.any(mask):
            continue
        # For active gaps at this j:
        # upper = idx[j+1, mask], lower = idx[j, mask]
        upper_idx = idx[j + 1, mask]
        lower_idx = idx[j, mask]
        np.add.at(grad, (upper_idx, np.where(mask)[0]), 1.0)
        np.add.at(grad, (lower_idx, np.where(mask)[0]), -1.0)
    
    grad = grad.mean(axis=1)  # average over M heights
    
    return area, grad


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


def _coordinate_descent_polish(x, n=64, M=20000, deadline=None, max_rounds=50):
    """
    Coordinate descent polish with adaptive step sizes.
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
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
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


def _simulated_annealing(x0, n=64, M=20000, deadline=None, T_start=0.1, T_end=0.001, alpha=0.9995, max_iter=50000):
    """
    Simulated annealing with large jumps that change the ordering structure.
    This explores the combinatorial landscape of gap configurations.
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
            # Accept with probability exp(-delta/T)
            delta_area = area_new - current_area
            if delta_area < 1e-12:
                x = x_new
                current_area = area_new
            elif rng.random() < np.exp(-delta_area / max(T, 1e-15)):
                x = x_new
                current_area = area_new
        
        # Cool down
        T = max(T * alpha, T_end)
    
    return best_x, best_area


def _local_refinement(x, n=64, M=20000, deadline=None, max_rounds=100):
    """
    Aggressive coordinate descent with adaptive step sizes.
    """
    return _coordinate_descent_polish(x, n, M, deadline=deadline, max_rounds=max_rounds)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses gradient-based optimization with multi-start, followed by
    simulated annealing for exploration and coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    n = 64
    M = 20000
    
    # Start from Keich construction
    x_keich = _keich_initial(n)
    
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    def _fun_and_jac(x):
        area, grad = _compute_area_and_grad(x, n, M)
        return area, grad
    
    # Phase 1: Gradient-based multi-start optimization
    rng = np.random.default_rng(42)
    
    # Try multiple starting points with gradient-based optimization
    n_starts = 8
    for s in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 180:
            break
        
        if s == 0:
            x0 = x_keich.copy()
        elif s == 1:
            x0 = x_keich + rng.normal(0, 0.03, size=n)
        elif s == 2:
            x0 = x_keich + rng.normal(0, 0.06, size=n)
        elif s == 3:
            x0 = x_keich + rng.normal(0, 0.1, size=n)
        elif s == 4:
            x0 = best_x + rng.normal(0, 0.01, size=n)
        elif s == 5:
            x0 = best_x + rng.normal(0, 0.03, size=n)
        elif s == 6:
            x0 = rng.uniform(-0.3, 0.3, size=n)
        else:
            x0 = rng.uniform(-0.5, 0.5, size=n)
        
        try:
            result = minimize(
                _fun_and_jac,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            if result.success or result.fun < best_area:
                if result.fun < best_area:
                    best_area = result.fun
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Phase 2: Coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            refined_x, refined_area = _coordinate_descent_polish(
                best_x, n, M, deadline=deadline, max_rounds=100
            )
            if refined_area < best_area:
                best_area = refined_area
                best_x = refined_x.copy()
        except Exception:
            pass
    
    # Phase 3: Simulated annealing for exploration if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        try:
            sa_x, sa_area = _simulated_annealing(
                best_x, n, M, deadline=deadline,
                T_start=0.1, T_end=0.0005, alpha=0.9997, max_iter=20000
            )
            if sa_area < best_area:
                best_area = sa_area
                best_x = sa_x.copy()
        except Exception:
            pass
    
    # Phase 4: Final gradient-based refinement + coordinate descent
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(
                _fun_and_jac,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            refined_x, refined_area = _coordinate_descent_polish(
                best_x, n, M, deadline=deadline, max_rounds=50
            )
            if refined_area < best_area:
                best_area = refined_area
                best_x = refined_x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
