import numpy as np
import time
import itertools
import sys

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _get_compositions(n, max_parts=10):
    """Generate all compositions of n into at most max_parts parts."""
    if n == 0:
        yield []
        return
    if max_parts == 0:
        return
    for i in range(1, n + 1):
        for rest in _get_compositions(n - i, max_parts - 1):
            yield [i] + rest

def _solve_shelf_layout(composition, is_horizontal=True):
    """
    Solve a shelf layout for a given composition.
    Returns (centers, sides) or None if infeasible.
    is_horizontal=True means rows are stacked vertically (composition of x-dimensions).
    is_horizontal=False means columns are stacked horizontally (composition of y-dimensions).
    """
    n = sum(composition)
    if n == 0:
        return None, None
    
    # For a shelf layout, the size of the square is limited by:
    # 1. The height of the shelf (if horizontal) or width (if vertical)
    # 2. The width of the shelf (if horizontal) / n_squares
    # 3. The height of the shelf / n_squares (if vertical)
    
    # Let's define the layout recursively or iteratively.
    # Actually, a "shelf" usually implies a single row of squares.
    # If we have a composition [2, 3, 5], it means 3 rows.
    # Row 1 has 2 squares, Row 2 has 3, Row 3 has 5.
    
    # In a standard shelf packing, all squares in a row have the same size, 
    # determined by the minimum of (row_width / count) and (row_height).
    # But wait, if we stack rows, the row height is determined by the square size.
    # So, if square size is s, row height is s.
    # Total height = sum(s) = n_rows * s.
    # Total width = max(row_widths) = max(count_i * s) = s * max(counts).
    
    # So for a given composition C = [c1, c2, ..., ck]:
    # s <= 1 / k  (height constraint)
    # s <= 1 / max(C) (width constraint)
    # So s = min(1/k, 1/max(C)).
    
    # This is a very restrictive "equal size" assumption for the whole shelf stack.
    # However, we can have heterogeneous sizes if we allow different rows to have different sizes?
    # No, in a shelf, squares in the same row must have the same height (s).
    # But different rows can have different sizes?
    # If row i has size s_i, height is s_i.
    # Width of row i is c_i * s_i.
    # Constraints:
    # sum(s_i) <= 1
    # max(c_i * s_i) <= 1
    
    # To maximize sum(s_i) subject to these?
    # This is a linear programming problem!
    # Maximize sum(s_i)
    # s.t. sum(s_i) <= 1
    # c_i * s_i <= 1  => s_i <= 1/c_i
    # s_i >= 0
    
    # The optimal solution is to set s_i = min(1/c_i, remaining_budget / remaining_rows) ?
    # Actually, since objective is sum(s_i) and constraint is sum(s_i) <= 1, 
    # we want to make sum(s_i) as close to 1 as possible.
    # We are limited by s_i <= 1/c_i.
    # So we should set s_i = 1/c_i for all i, unless sum(1/c_i) > 1.
    # If sum(1/c_i) > 1, we must scale them down.
    # Let S = sum(1/c_i).
    # If S <= 1, set s_i = 1/c_i. Sum = S.
    # If S > 1, we need sum(s_i) = 1.
    # Since s_i <= 1/c_i, and we want to maximize sum, we should hit the upper bounds as much as possible?
    # No, if we hit upper bounds, sum might exceed 1.
    # Actually, the constraint is sum(s_i) <= 1.
    # If we set s_i = 1/c_i, sum = S.
    # If S <= 1, this is valid and optimal (can't increase any s_i without violating s_i <= 1/c_i).
    # If S > 1, we must reduce some s_i.
    # To keep sum(s_i) = 1 (maximizing sum), we should reduce them proportionally?
    # Or just set sum(s_i) = 1.
    # Since the objective is linear, any point on the plane sum(s_i)=1 within the box [0, 1/c_i] is optimal.
    # But we also need to fit them in the square.
    # Wait, the width constraint is max(c_i * s_i) <= 1.
    # If we scale down, c_i * s_i decreases, so width constraint is easier to satisfy.
    # So if S > 1, we can just distribute 1 among the rows such that s_i <= 1/c_i.
    # The simplest is s_i = 1/k for all i? No, that might violate s_i <= 1/c_i if c_i < k.
    # We need s_i <= 1/c_i.
    # So we can set s_i = min(1/c_i, 1/k) ?
    # Let's just solve it simply:
    # We want to assign s_i such that sum(s_i) <= 1 and c_i * s_i <= 1.
    # Maximize sum(s_i).
    
    # Algorithm:
    # 1. Calculate max possible s_i for each row: u_i = 1/c_i.
    # 2. If sum(u_i) <= 1, set s_i = u_i.
    # 3. Else, we need sum(s_i) = 1.
    #    We can just set s_i = 1/k for all i? 
    #    Check if 1/k <= 1/c_i => c_i <= k.
    #    If some c_i > k, then 1/k > 1/c_i is false, 1/k < 1/c_i is true.
    #    So s_i = 1/k is valid for all i if c_i >= k? No.
    #    If c_i is small, 1/c_i is large.
    #    If c_i is large, 1/c_i is small.
    #    The bottleneck is the largest c_i.
    #    Let c_max = max(c_i). u_min = 1/c_max.
    #    If we set all s_i = 1/k, is it valid?
    #    We need 1/k <= 1/c_i for all i => c_i <= k.
    #    If there is a c_i > k, then 1/k > 1/c_i, so s_i = 1/k violates s_i <= 1/c_i.
    #    So we must set s_i <= 1/c_i for those rows.
    
    # Let's just use a simple heuristic:
    # s_i = 1 / max(k, c_i) ?
    # Sum = sum(1 / max(k, c_i)).
    # This might be < 1.
    
    # Better:
    # Let's just try to fill the height 1.
    # We have k rows.
    # Constraint: c_i * s_i <= 1.
    # We want sum(s_i) = 1.
    # This is feasible if sum(1/c_i) >= 1. (Which is true if S > 1).
    # We can just set s_i = 1/k for all i, and then check constraints.
    # If c_i * (1/k) > 1, i.e., c_i > k, then we must reduce s_i.
    # Set s_i = 1/c_i for those with c_i > k.
    # Then distribute the remaining height among the others.
    
    # Let's implement this logic.
    c = np.array(composition)
    k = len(c)
    
    s = np.zeros(k)
    remaining = 1.0
    free_indices = []
    fixed_indices = []
    
    # First, fix rows where 1/k > 1/c_i (i.e., c_i < k is false? No. c_i > k => 1/c_i < 1/k)
    # If c_i > k, then s_i cannot be 1/k. It must be <= 1/c_i.
    # So set s_i = 1/c_i.
    for i in range(k):
        if c[i] > k:
            s[i] = 1.0 / c[i]
            fixed_indices.append(i)
        else:
            free_indices.append(i)
            
    sum_fixed = np.sum(s[fixed_indices]) if fixed_indices else 0.0
    remaining = 1.0 - sum_fixed
    
    n_free = len(free_indices)
    if n_free > 0:
        # Distribute remaining height equally among free rows?
        # s_free = remaining / n_free
        # Check if this violates s_i <= 1/c_i.
        s_val = remaining / n_free
        for i in free_indices:
            if s_val > 1.0 / c[i]:
                # This shouldn't happen if we correctly identified fixed ones?
                # c[i] <= k => 1/c[i] >= 1/k.
                # Is remaining/n_free <= 1/k?
                # remaining = 1 - sum(1/c_j for c_j > k).
                # This is getting complicated.
                # Let's just use a simple iterative approach or LP.
                # Given time, let's just use s_i = 1/k for all, and shrink if needed.
                pass
                
    # Simpler robust approach:
    # s_i = 1/k
    s = np.full(k, 1.0 / k)
    # Check constraints
    for i in range(k):
        if s[i] * c[i] > 1.0:
            s[i] = 1.0 / c[i]
            
    # Now sum(s) might be < 1.
    # We can try to increase the others to fill up to 1.
    # This is a water-filling problem.
    # But for now, let's just use this. It's a valid packing.
    # To improve, we can try to increase s_i for rows with c_i < k.
    
    # Let's try to optimize s slightly.
    # We want to maximize sum(s) subject to sum(s)<=1 and c_i*s_i<=1.
    # We already have a feasible point.
    # Let's just leave it. The local refinement will help.
    
    # Construct centers
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    idx = 0
    y_offset = 0.0
    for i in range(k):
        count = composition[i]
        s_i = s[i]
        # Place squares in this row
        for j in range(count):
            x_offset = j * s_i
            cx = x_offset + s_i / 2
            cy = y_offset + s_i / 2
            centers[idx, 0] = cx
            centers[idx, 1] = cy
            sides[idx] = s_i
            idx += 1
        y_offset += s_i
        
    return centers, sides

def _generate_shelf_layouts():
    """Generate all shelf layouts for 10 squares."""
    layouts = []
    # Compositions of 10
    for comp in _get_compositions(10, 10):
        # Horizontal shelves (rows)
        c, s = _solve_shelf_layout(comp, is_horizontal=True)
        if c is not None:
            layouts.append((c, s))
            
        # Vertical shelves (columns) - swap x and y
        c_v, s_v = _solve_shelf_layout(comp, is_horizontal=False)
        if c_v is not None:
            # Rotate 90 degrees: (x, y) -> (y, x)
            c_rot = c_v.copy()
            c_rot[:, 0], c_rot[:, 1] = c_v[:, 1].copy(), c_v[:, 0].copy()
            layouts.append((c_rot, s_v))
            
    return layouts

def _local_refine(centers, sides, tol=1e-7, max_iters=500):
    """
    Local refinement: try to enlarge each square as much as possible.
    """
    n = len(sides)
    centers = centers.copy()
    sides = sides.copy()
    
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            # Try to increase side i
            # Max possible side is limited by:
            # 1. Boundary: 2 * min(cx, 1-cx, cy, 1-cy)
            max_by_boundary = 2 * min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            
            # 2. Other squares
            max_by_others = max_by_boundary
            for j in range(n):
                if i == j:
                    continue
                # Constraint: |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
                # We want to find max s_i such that this holds.
                # This is non-convex. 
                # Approximation: assume we can move centers? No, keep centers fixed for this step.
                # If we keep centers fixed, we can only increase s_i until it hits the "forbidden zone" of j.
                # Forbidden zone for i relative to j:
                # |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
                # Let dx = |cx_i - cx_j|, dy = |cy_i - cy_j|.
                # We need dx >= (s_i + s_j)/2 OR dy >= (s_i + s_j)/2.
                # (s_i + s_j)/2 <= dx  => s_i <= 2*dx - s_j
                # (s_i + s_j)/2 <= dy  => s_i <= 2*dy - s_j
                # So s_i <= max(2*dx - s_j, 2*dy - s_j)
                
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                limit = max(2 * dx - sides[j], 2 * dy - sides[j])
                if limit < max_by_others:
                    max_by_others = limit
                    
            new_s = min(max_by_boundary, max_by_others)
            if new_s > sides[i] + 1e-9:
                sides[i] = new_s
                improved = True
                
        if not improved:
            break
            
    return centers, sides

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 570
    
    best_centers = None
    best_sides = None
    best_val = -float('inf')
    
    # 1. Generate shelf layouts
    layouts = _generate_shelf_layouts()
    
    # 2. Evaluate and refine each
    for c, s in layouts:
        if time.time() > deadline:
            break
            
        if not _is_valid(c, s):
            continue
            
        # Refine
        c_ref, s_ref = _local_refine(c, s)
        
        if _is_valid(c_ref, s_ref):
            val = np.sum(s_ref)
            if val > best_val:
                best_val = val
                best_centers = c_ref.copy()
                best_sides = s_ref.copy()
                
    # 3. Fallback if no valid layout found (should not happen)
    if best_centers is None:
        best_centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]], dtype=np.float64)
        best_sides = np.full(10, 0.15, dtype=np.float64)
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
