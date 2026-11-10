# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area_and_subgrad(x, n=64, n_m=20000):
    """Compute area and subgradient of area w.r.t. x.
    
    Returns (area, subgrad) where subgrad is a vector of shape (n,).
    """
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w = (1.0 - np.arange(n_m, dtype=np.float64) / n_m) / n  # w at each y_m
    y = (np.arange(n_m) + 0.5) / n_m  # y_m values
    
    # L_i at each height: L[i, m] = x[i] + a[i] * y[m]
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, n_m)
    
    # Sort L at each height
    order = np.argsort(L, axis=0)  # shape (n, n_m)
    L_sorted = np.take_along_axis(L, order, axis=0)  # shape (n, n_m)
    
    # Compute gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)  # shape (n-1, n_m)
    
    # Union length at each height: w + sum(min(gap, w))
    min_gaps = np.minimum(gaps, w[None, :])  # shape (n-1, n_m)
    union_len = w[None, :] + min_gaps.sum(axis=0)  # shape (n_m,)
    
    area = union_len.mean()
    
    # Subgradient: for each triangle i, d(area)/d(x_i)
    # x_i appears in L_i at all heights. In the sorted order, L_i contributes
    # to the gap above it and the gap below it (if not at boundary).
    # 
    # For triangle i at height m:
    # - If it's the smallest (rank 0): it contributes +1 to the gap below (but no gap below)
    #   Actually, it's the leftmost, so it only affects the gap to the next one.
    #   d(union)/d(L_i) = 1 if the gap to the next is < w (i.e., min is active), else 0
    #   Wait, let me think more carefully.
    #
    # union_len = w + sum_j min(gap_j, w)
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # 
    # If triangle i is at rank r at height m:
    # - It appears in gap_{r-1} (as the right element) if r > 0: contributes -1 to gap_{r-1}
    # - It appears in gap_r (as the left element) if r < n-1: contributes +1 to gap_r
    #
    # d(min(gap, w))/d(gap) = 1 if gap < w, else 0
    #
    # So d(union)/d(L_i) = [gap_r < w ? 1 : 0] - [gap_{r-1} < w ? 1 : 0]
    # where gap_{-1} and gap_{n-1} don't exist (treat as not active)
    
    subgrad_L = np.zeros((n, n_m), dtype=np.float64)
    
    # For each rank r (0 to n-1), find which triangle is at that rank
    # For rank r > 0: the triangle at rank r contributes -1 if gap_{r-1} < w
    # For rank r < n-1: the triangle at rank r contributes +1 if gap_r < w
    
    # gap_j < w condition
    gap_active = (gaps < w[None, :]).astype(np.float64)  # shape (n-1, n_m)
    
    # For each triangle i, at each height m, find its rank
    # rank[i, m] = position of triangle i in the sorted order at height m
    rank = np.zeros((n, n_m), dtype=np.int64)
    for r in range(n):
        rank[order[r, :], :] = r
    
    # Now compute subgradient contribution
    # For triangle i at height m with rank r:
    # contrib = (r < n-1 and gap_r < w) - (r > 0 and gap_{r-1} < w)
    
    # Vectorized approach:
    # +1 contribution: triangle i is at rank r < n-1 and gap_r < w
    # -1 contribution: triangle i is at rank r > 0 and gap_{r-1} < w
    
    # For +1: for each r in 0..n-2, triangles at rank r get +1 where gap_r < w
    for r in range(n - 1):
        mask = (rank == r)  # shape (n, n_m)
        subgrad_L += mask * gap_active[r, :][None, :]
    
    # For -1: for each r in 1..n-1, triangles at rank r get -1 where gap_{r-1} < w
    for r in range(1, n):
        mask = (rank == r)  # shape (n, n_m)
        subgrad_L -= mask * gap_active[r - 1, :][None, :]
    
    # subgrad w.r.t. x_i: since L_i = x_i + a_i * y, and d(L_i)/d(x_i) = 1
    # So subgrad_x[i] = mean over m of subgrad_L[i, m]
    subgrad_x = subgrad_L.mean(axis=1)
    
    return area, subgrad_x


def _compute_area_only(x, n=64, n_m=20000):
    """Compute area only (faster, no subgradient)."""
    a = (np.arange(n) + 1) / n
    w = (1.0 - np.arange(n_m, dtype=np.float64) / n_m) / n
    y = (np.arange(n_m) + 0.5) / n_m
    
    L = x[:, None] + a[:, None] * y[None, :]
    order = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, order, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    min_gaps = np.minimum(gaps, w[None, :])
    union_len = w[None, :] + min_gaps.sum(axis=0)
    return union_len.mean()


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Start from Keich construction
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Compute initial area
    best_area = _compute_area_only(x)
    best_x = x.copy()
    
    # Subgradient descent with adaptive step size
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    # Use a moderate number of quadrature points for speed during optimization
    n_m_opt = 5000
    
    step_size = 0.001
    best_step = step_size
    patience = 0
    max_patience = 50
    
    a = (np.arange(n) + 1) / n
    y_opt = (np.arange(n_m_opt) + 0.5) / n_m_opt
    w_opt = (1.0 - np.arange(n_m_opt) / n_m_opt) / n
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        # Compute area and subgradient
        L = x[:, None] + a[:, None] * y_opt[None, :]
        order = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, order, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        min_gaps = np.minimum(gaps, w_opt[None, :])
        union_len = w_opt[None, :] + min_gaps.sum(axis=0)
        area = union_len.mean()
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
            patience = 0
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.01)
        else:
            patience += 1
            if patience > max_patience:
                # Decrease step size
                step_size *= 0.9
                patience = 0
                if step_size < 1e-8:
                    break
        
        # Compute subgradient
        gap_active = (gaps < w_opt[None, :]).astype(np.float64)
        
        # Build rank array
        rank = np.zeros((n, n_m_opt), dtype=np.int64)
        for r in range(n):
            rank[order[r, :], :] = r
        
        subgrad_L = np.zeros((n, n_m_opt), dtype=np.float64)
        for r in range(n - 1):
            mask = (rank == r)
            subgrad_L += mask * gap_active[r, :][None, :]
        for r in range(1, n):
            mask = (rank == r)
            subgrad_L -= mask * gap_active[r - 1, :][None, :]
        
        subgrad_x = subgrad_L.mean(axis=1)
        
        # Normalize subgradient
        sg_norm = np.linalg.norm(subgrad_x)
        if sg_norm > 1e-12:
            subgrad_x = subgrad_x / sg_norm
        
        # Update
        x = x - step_size * subgrad_x
        
        # Keep x finite
        if not np.all(np.isfinite(x)):
            x = best_x.copy()
            step_size *= 0.5
    
    # Final evaluation with full quadrature
    final_area = _compute_area_only(best_x, n_m=20000)
    
    # Try a few more refinements with full quadrature if time permits
    if time.time() < deadline - 30:
        for it in range(500):
            if time.time() > deadline:
                break
            
            area, sg = _compute_area_and_subgrad(best_x, n=64, n_m=20000)
            if area < best_area:
                best_area = area
            
            sg_norm = np.linalg.norm(sg)
            if sg_norm < 1e-12:
                break
            
            sg = sg / sg_norm
            candidate = best_x - step_size * sg
            
            cand_area = _compute_area_only(candidate, n_m=20000)
            if cand_area < best_area:
                best_x = candidate
                best_area = cand_area
                step_size = min(step_size * 1.02, 0.005)
            else:
                step_size *= 0.95
                if step_size < 1e-10:
                    break
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
