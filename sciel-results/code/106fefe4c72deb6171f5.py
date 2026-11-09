# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _score(centers, sides, tol=1e-7):
    """Score = sum of sides. Higher is better."""
    if not _is_valid(centers, sides, tol):
        return -np.inf
    return np.sum(sides)


def _optimize_shelf_partition(squares_per_shelf):
    """
    Analytically optimize a shelf-based packing.
    For each shelf k with m_k squares, the optimal side is min(h_k, 1/m_k).
    We want to maximize sum_k m_k * min(h_k, 1/m_k) subject to sum h_k <= 1.
    
    This is a piecewise linear problem. The optimal solution has each shelf
    either at its "square" height h_k = 1/m_k (where width constraint binds)
    or using all remaining height. We try all subsets of shelves that are
    "width-limited" and compute the optimal allocation.
    """
    n_shelves = len(squares_per_shelf)
    n_squares = sum(squares_per_shelf)
    if n_squares != 10:
        return None, None, -np.inf
    
    best_value = -np.inf
    best_h = None
    
    # Try all subsets of shelves that are width-limited (h_k = 1/m_k)
    # For a given subset S of width-limited shelves, the remaining height
    # (1 - sum_{k in S} 1/m_k) is distributed among shelves not in S.
    # For shelves not in S, they use all their height, so s_k = h_k.
    # We need to allocate remaining height to maximize sum of h_k for k not in S.
    # Since s_k = h_k for those, we just want to maximize sum h_k = remaining.
    # But each h_k must be >= 0, and the total must equal remaining.
    # The objective contribution from non-S shelves is sum h_k = remaining.
    # So any allocation works for the objective, but we need h_k >= 0.
    # Actually, for non-S shelves, s_k = h_k, so contribution is h_k.
    # Total objective = sum_{k in S} m_k * (1/m_k) + sum_{k not in S} h_k
    #                  = |S| + (1 - sum_{k in S} 1/m_k)
    #                  = 1 + |S| - sum_{k in S} 1/m_k
    
    # Wait, that's not right. Let me reconsider.
    # For shelf k with m_k squares:
    #   If h_k <= 1/m_k: s_k = h_k, contribution = m_k * h_k
    #   If h_k > 1/m_k: s_k = 1/m_k, contribution = m_k * (1/m_k) = 1
    # 
    # So the objective is sum_k f_k(h_k) where f_k(h) = m_k * min(h, 1/m_k)
    # This is a concave piecewise linear function.
    # The maximum of sum of concave functions subject to linear constraint
    # can be found by water-filling or by trying all breakpoints.
    # 
    # The breakpoints are h_k = 1/m_k. We try all 2^n_shelves subsets S
    # where shelves in S have h_k = 1/m_k (width-limited) and shelves not in S
    # have h_k < 1/m_k (height-limited).
    # 
    # For a given S:
    #   Fixed height for S: H_S = sum_{k in S} 1/m_k
    #   Remaining height: R = 1 - H_S
    #   For k not in S: h_k < 1/m_k, and sum_{k not in S} h_k = R
    #   Objective from S: |S| (since each contributes 1)
    #   Objective from not S: sum_{k not in S} m_k * h_k
    #   To maximize sum m_k * h_k subject to sum h_k = R and 0 <= h_k < 1/m_k:
    #     This is a linear program. The optimal solution puts all height
    #     into the shelf with the largest m_k, up to its capacity 1/m_k,
    #     then the next largest, etc.
    
    for mask in range(1 << n_shelves):
        S = set()
        for k in range(n_shelves):
            if mask & (1 << k):
                S.add(k)
        
        H_S = sum(1.0 / squares_per_shelf[k] for k in S)
        if H_S > 1.0 + 1e-12:
            continue
        
        R = 1.0 - H_S
        
        # Allocate R to shelves not in S, respecting h_k < 1/m_k
        # Greedy: allocate to shelves with largest m_k first
        not_S = [k for k in range(n_shelves) if k not in S]
        not_S.sort(key=lambda k: -squares_per_shelf[k])
        
        h = np.zeros(n_shelves)
        for k in S:
            h[k] = 1.0 / squares_per_shelf[k]
        
        remaining = R
        obj_not_S = 0.0
        for k in not_S:
            m_k = squares_per_shelf[k]
            cap = 1.0 / m_k  # h_k must be < cap
            alloc = min(remaining, cap)
            h[k] = alloc
            remaining -= alloc
            obj_not_S += m_k * alloc
            if remaining <= 1e-12:
                break
        
        if remaining > 1e-10:
            # Can't allocate all remaining height (shouldn't happen if H_S <= 1)
            # Actually this can happen if all not-S shelves are at capacity
            # In that case, we have excess height that doesn't help.
            # But wait, if all shelves are at capacity, then H_S + sum_{not S} 1/m_k <= 1
            # which means we're in a different regime. Let's just skip.
            continue
        
        total_obj = float(len(S)) + obj_not_S
        
        if total_obj > best_value + 1e-12:
            best_value = total_obj
            best_h = h.copy()
    
    if best_h is None:
        return None, None, -np.inf
    
    # Construct centers and sides
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    y_offset = 0.0
    idx = 0
    for i, m in enumerate(squares_per_shelf):
        h_k = best_h[i]
        s_k = min(h_k, 1.0/m)
        x_offset = (1.0 - m * s_k) / 2.0
        for j in range(m):
            cx = x_offset + j * s_k + s_k / 2.0
            cy = y_offset + h_k / 2.0
            centers[idx, 0] = cx
            centers[idx, 1] = cy
            sides[idx] = s_k
            idx += 1
        y_offset += h_k
    
    return centers, sides, best_value


def _generate_shelf_partitions(n=10):
    """Generate diverse shelf partitions."""
    partitions = []
    # 1 shelf
    partitions.append([10])
    # 2 shelves
    partitions.append([5, 5])
    partitions.append([6, 4])
    partitions.append([7, 3])
    partitions.append([8, 2])
    partitions.append([9, 1])
    # 3 shelves
    partitions.append([4, 3, 3])
    partitions.append([3, 3, 4])
    partitions.append([5, 3, 2])
    partitions.append([2, 3, 5])
    partitions.append([4, 4, 2])
    partitions.append([2, 4, 4])
    partitions.append([6, 2, 2])
    partitions.append([2, 6, 2])
    partitions.append([7, 2, 1])
    partitions.append([1, 7, 2])
    partitions.append([3, 4, 3])
    partitions.append([4, 3, 3])
    partitions.append([5, 2, 3])
    partitions.append([3, 2, 5])
    partitions.append([6, 3, 1])
    partitions.append([1, 3, 6])
    # 4 shelves
    partitions.append([3, 3, 2, 2])
    partitions.append([2, 2, 3, 3])
    partitions.append([4, 2, 2, 2])
    partitions.append([2, 4, 2, 2])
    partitions.append([5, 2, 2, 1])
    partitions.append([1, 2, 2, 5])
    partitions.append([3, 3, 3, 1])
    partitions.append([1, 3, 3, 3])
    partitions.append([4, 3, 2, 1])
    partitions.append([1, 2, 3, 4])
    partitions.append([2, 3, 3, 2])
    partitions.append([3, 2, 3, 2])
    partitions.append([5, 3, 1, 1])
    partitions.append([1, 1, 3, 5])
    partitions.append([6, 2, 1, 1])
    partitions.append([1, 1, 2, 6])
    partitions.append([4, 4, 1, 1])
    partitions.append([1, 1, 4, 4])
    # 5 shelves
    partitions.append([2, 2, 2, 2, 2])
    partitions.append([3, 2, 2, 2, 1])
    partitions.append([1, 2, 2, 2, 3])
    partitions.append([4, 2, 2, 1, 1])
    partitions.append([1, 1, 2, 2, 4])
    partitions.append([3, 3, 2, 1, 1])
    partitions.append([1, 1, 2, 3, 3])
    partitions.append([5, 2, 1, 1, 1])
    partitions.append([1, 1, 1, 2, 5])
    partitions.append([4, 3, 1, 1, 1])
    partitions.append([1, 1, 1, 3, 4])
    # 6 shelves
    partitions.append([2, 2, 2, 1, 1, 1])
    partitions.append([1, 1, 1, 2, 2, 2])
    partitions.append([3, 2, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 2, 3])
    partitions.append([4, 1, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 1, 4])
    # 7 shelves
    partitions.append([2, 1, 1, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 1, 1, 2])
    partitions.append([3, 1, 1, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 1, 1, 3])
    # 8 shelves
    partitions.append([2, 1, 1, 1, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 1, 1, 1, 2])
    # 9 shelves
    partitions.append([2, 1, 1, 1, 1, 1, 1, 1, 1])
    partitions.append([1, 1, 1, 1, 1, 1, 1, 1, 2])
    # 10 shelves
    partitions.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    return partitions


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 620  # seconds, with safety margin
    
    best_centers = None
    best_sides = None
    best_value = -np.inf
    
    # Try all shelf partitions
    partitions = _generate_shelf_partitions(n)
    
    for part in partitions:
        if time.time() - start_time > deadline - 5:
            break
        c, s, val = _optimize_shelf_partition(part)
        if c is not None and _is_valid(c, s, tol):
            if val > best_value:
                best_value = val
                best_centers = c.copy()
                best_sides = s.copy()
                
    # Fallback: grid if no valid solution found
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
