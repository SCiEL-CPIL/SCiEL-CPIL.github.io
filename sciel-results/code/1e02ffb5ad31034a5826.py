"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog

N = 21
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _validate(circles, tol=1e-9):
    """Check that circles are valid: disjoint interiors, inside rectangle with perimeter <= 4."""
    if len(circles) != N:
        return False
    # Check radii positive
    if np.any(circles[:, 2] <= 0):
        return False
    # Check circles are within some rectangle of perimeter <= 4
    # The bounding box of the circles
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    xmin = np.min(x - r)
    xmax = np.max(x + r)
    ymin = np.min(y - r)
    ymax = np.max(y + r)
    
    width = xmax - xmin
    height = ymax - ymin
    perimeter = 2 * (width + height)
    
    if perimeter > 4 + tol:
        return False
    
    # Check disjoint interiors: for each pair, distance >= sum of radii
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _solve_lp(positions):
    """Given fixed positions, solve LP for maximum sum of radii with non-overlap and boundary constraints."""
    n = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    
    # We need to find the rectangle. Let's assume the rectangle is [0, W] x [0, H] with 2(W+H)=4, so W+H=2.
    # But the rectangle can be positioned anywhere. Let's use the bounding box approach.
    # Actually, let's fix the rectangle to be [0, W] x [0, H] and scale/translate positions.
    # Better: let's optimize over W (aspect ratio) too, but for LP we fix W.
    
    # For a given aspect ratio, let W = w, H = 2 - w
    # Constraints:
    # r_i >= 0
    # x_i - r_i >= 0  =>  r_i <= x_i
    # x_i + r_i <= W  =>  r_i <= W - x_i
    # y_i - r_i >= 0  =>  r_i <= y_i
    # y_i + r_i <= H  =>  r_i <= H - y_i
    # r_i + r_j <= d_ij  for all pairs (i,j)
    
    # We'll try several aspect ratios and pick the best.
    best_r = None
    best_sum = -1
    
    # Try a range of aspect ratios
    for w in np.linspace(0.3, 1.7, 20):
        h = 2.0 - w
        
        # Check if positions fit in this rectangle (with some margin)
        # Scale positions to fit in [0, w] x [0, h]
        # Actually, we should position the circles optimally. Let's just check if the bounding box fits.
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        
        # The positions need to be in [0, w] x [0, h] after some translation
        # For simplicity, let's translate so min is 0
        dx = xmin
        dy = ymin
        x_shifted = x - dx
        y_shifted = y - dy
        
        if x_shifted.max() > w or y_shifted.max() > h:
            continue
        
        # Now solve LP with these shifted positions
        # Variables: r_0, ..., r_{n-1}
        # Maximize sum(r)
        c = -np.ones(n)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # Boundary constraints: r_i <= x_i_shifted, r_i <= w - x_i_shifted, r_i <= y_i_shifted, r_i <= h - y_i_shifted
        for i in range(n):
            A_ub.append(np.eye(n)[i])
            b_ub.append(x_shifted[i])
            A_ub.append(np.eye(n)[i])
            b_ub.append(w - x_shifted[i])
            A_ub.append(np.eye(n)[i])
            b_ub.append(y_shifted[i])
            b_ub.append(h - y_shifted[i])
        
        # Pairwise constraints: r_i + r_j <= d_ij
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((x_shifted[i] - x_shifted[j])**2 + (y_shifted[i] - y_shifted[j])**2)
                if d <= 0:
                    continue
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: r_i >= 0
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                r = res.x
                total = np.sum(r)
                if total > best_sum:
                    best_sum = total
                    best_r = r.copy()
        except Exception:
            continue
    
    return best_r, best_sum


def _refine_positions(radii, w, h, max_iters=50):
    """Iteratively adjust positions to maximize minimum slack, then re-solve LP."""
    n = len(radii)
    x = np.zeros(n)
    y = np.zeros(n)
    
    # Initial positions: grid-like
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    for i in range(n):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
    
    best_positions = x.copy(), y.copy()
    best_sum = -1
    
    for iteration in range(max_iters):
        # Solve LP for radii given current positions
        r, total = _solve_lp(np.column_stack([x, y]))
        if r is not None and total > best_sum:
            best_sum = total
            best_positions = x.copy(), y.copy()
        
        # Try to improve positions using gradient-like steps
        # For each circle, try small perturbations
        improved = False
        for i in range(n):
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    x_new = x.copy()
                    y_new = y.copy()
                    x_new[i] += dx
                    y_new[i] += dy
                    # Keep in bounds
                    x_new[i] = np.clip(x_new[i], 0, w)
                    y_new[i] = np.clip(y_new[i], 0, h)
                    
                    r_new, total_new = _solve_lp(np.column_stack([x_new, y_new]))
                    if r_new is not None and total_new > total + 1e-6:
                        x = x_new
                        y = y_new
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_positions, best_sum


def construct():
    start_time = time.time()
    
    # Strategy: Use simulated annealing on positions with LP-based radius optimization
    # This is a structural change from the seed's fixed grid approach.
    
    best_circles = None
    best_sum = 0
    
    # Try multiple aspect ratios
    aspect_ratios = np.linspace(0.4, 1.6, 13)
    
    for w in aspect_ratios:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        h = 2.0 - w
        
        # Initialize with a grid
        n = N
        cols = int(np.ceil(np.sqrt(n * w / h)))
        rows = int(np.ceil(n / cols))
        
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            col = i % cols
            row = i // rows if rows > 0 else 0
            x[i] = (col + 0.5) * w / cols
            y[i] = (row + 0.5) * h / rows if rows > 0 else 0.5 * h
        
        # Simulated annealing
        temp = 0.05
        cooling = 0.995
        current_sum = -1
        current_x = x.copy()
        current_y = y.copy()
        current_r = None
        
        # Get initial radii
        current_r, current_sum = _solve_lp(np.column_stack([current_x, current_y]))
        if current_r is None:
            continue
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = np.column_stack([current_x, current_y, current_r])
        
        iters = 0
        max_iters = 2000
        
        while iters < max_iters:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            iters += 1
            
            # Random perturbation
            i = np.random.randint(n)
            dx = np.random.uniform(-temp, temp)
            dy = np.random.uniform(-temp, temp)
            
            x_new = current_x.copy()
            y_new = current_y.copy()
            x_new[i] += dx
            y_new[i] += dy
            x_new[i] = np.clip(x_new[i], 0, w)
            y_new[i] = np.clip(y_new[i], 0, h)
            
            r_new, sum_new = _solve_lp(np.column_stack([x_new, y_new]))
            
            if r_new is not None:
                if sum_new > current_sum:
                    current_x = x_new
                    current_y = y_new
                    current_r = r_new
                    current_sum = sum_new
                    
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = np.column_stack([current_x, current_y, current_r])
                else:
                    # Accept with probability exp((sum_new - current_sum) / temp)
                    prob = np.exp((sum_new - current_sum) / temp)
                    if np.random.rand() < prob:
                        current_x = x_new
                        current_y = y_new
                        current_r = r_new
                        current_sum = sum_new
            
            temp *= cooling
        
        # Local refinement
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        refined_x, refined_y, refined_sum = _refine_positions(current_r, w, h, max_iters=20)
        if refined_sum > best_sum:
            best_sum = refined_sum
            r_final, _ = _solve_lp(np.column_stack([refined_x, refined_y]))
            if r_final is not None:
                best_circles = np.column_stack([refined_x, refined_y, r_final])
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and adjustment
    if _validate(best_circles):
        return best_circles
    
    # If validation fails, shrink radii slightly
    circles = best_circles.copy()
    circles[:, 2] *= 0.99
    if _validate(circles):
        return circles
    
    # Last resort: uniform small circles
    r = 0.05
    x = np.linspace(0.1, 1.9, N)
    y = np.full(N, 0.5)
    return np.column_stack([x, y, np.full(N, r)])


if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"Valid: {_validate(result)}")
