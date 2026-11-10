# EVOLVE-BLOCK-START
import numpy as np
import time

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 570.0  # safety margin
    n = 10

    def _is_feasible_vec(centers, sides, tol=TOL):
        """
        Vectorized feasibility check.
        Returns True if all squares are inside [0,1]^2 and pairwise non-overlapping.
        """
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Check bounds
        if np.any(s <= 0):
            return False
        if np.any(cx - s / 2.0 < -tol) or np.any(cx + s / 2.0 > 1.0 + tol):
            return False
        if np.any(cy - s / 2.0 < -tol) or np.any(cy + s / 2.0 > 1.0 + tol):
            return False
        
        # Check overlaps
        # Overlap condition for i, j: |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
        # We can compute this for all pairs using broadcasting.
        cx_diff = np.abs(cx[:, None] - cx[None, :])
        cy_diff = np.abs(cy[:, None] - cy[None, :])
        s_sum = (s[:, None] + s[None, :]) / 2.0 - tol
        
        # We only care about i < j, so we can use the upper triangle
        # But it's easier to just check all and ignore diagonal (where diff=0, s_sum>0 usually, but diagonal is self so not an overlap)
        # Actually, for i=j, cx_diff=0, cy_diff=0. s_sum = s_i - tol. If s_i > tol, it would flag as overlap with itself?
        # No, the condition is for i != j.
        # Let's mask out the diagonal.
        mask = ~np.eye(n, dtype=bool)
        
        overlap_x = cx_diff < s_sum
        overlap_y = cy_diff < s_sum
        
        # If any pair (i != j) has overlap_x AND overlap_y, it's infeasible
        if np.any(overlap_x & overlap_y & mask):
            return False
            
        return True

    def _score(centers, sides):
        if not _is_feasible_vec(centers, sides):
            return -np.inf
        return np.sum(sides)

    def _make_layout_30():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        # Row 0: 4 squares of 0.25
        s1 = 0.25
        for j in range(4):
            centers[idx, 0] = j * s1 + s1 / 2
            centers[idx, 1] = s1 / 2
            sides[idx] = s1
            idx += 1
        # Row 1: 3 squares of 1/3
        s2 = 1.0 / 3.0
        y1 = s1 + s2 / 2
        for j in range(3):
            centers[idx, 0] = j * s2 + s2 / 2
            centers[idx, 1] = y1
            sides[idx] = s2
            idx += 1
        # Row 2: 3 squares of 1/3
        s3 = 1.0 / 3.0
        y2 = s1 + s2 + s3 / 2
        for j in range(3):
            centers[idx, 0] = j * s3 + s3 / 2
            centers[idx, 1] = y2
            sides[idx] = s3
            idx += 1
        return centers, sides

    def _make_layout_32():
        # Try 2 rows of 2 (size 0.5) and 1 row of 6? No, 6*0.166=1.
        # 2*0.5 + 2*0.5 + 6*0.166 = 1 + 1 + 1 = 3.0.
        # Let's try a different mix.
        # 1 square of 0.5 (bottom left)
        # 1 square of 0.5 (bottom right)
        # Top row: 8 squares? 1/8 = 0.125.
        # Height: 0.5 + 0.125 = 0.625.
        # Sum: 1 + 8*0.125 = 2.0. Worse.
        
        # How about:
        # 2 squares of 0.4
        # 2 squares of 0.4
        # 6 squares of 1/6 ~ 0.166
        # Height: 0.4+0.4+0.166 = 0.966.
        # Sum: 0.8 + 0.8 + 1.0 = 2.6. Worse.
        
        # Let's try 3 rows: 2, 3, 5?
        # 2 of 0.5, 3 of 0.33, 5 of 0.2.
        # Height: 0.5 + 0.33 + 0.2 = 1.03 > 1.
        # Shrink 0.5 to 0.45? 0.45+0.33+0.2 = 0.98.
        # Sum: 0.9 + 1.0 + 1.0 = 2.9. Worse.
        
        # It seems 3.0 is hard to beat with simple rows.
        # Let's just return the 3.0 layout for now, or a slight variation.
        # Variation: Shift rows slightly to allow larger squares?
        # If row 1 and 2 are staggered, can we make them larger?
        # No, width is the constraint.
        
        return _make_layout_30()

    def _stochastic_search(centers_init, sides_init, max_iters=20000):
        best_c = centers_init.copy()
        best_s = sides_init.copy()
        best_val = _score(best_c, best_s)
        if not np.isfinite(best_val):
            return best_c, best_s, best_val
            
        # Initial step size
        step = 0.05
        temp = 0.05
        it = 0
        while it < max_iters:
            if time.time() > DEADLINE - 10:
                break
            it += 1
            
            # Pick a random square to perturb
            i = np.random.randint(n)
            
            # Generate random perturbation
            # Perturb center and side
            dcx = np.random.uniform(-step, step)
            dcy = np.random.uniform(-step, step)
            ds = np.random.uniform(-step, step)
            
            new_c = best_c.copy()
            new_s = best_s.copy()
            
            new_c[i, 0] += dcx
            new_c[i, 1] += dcy
            new_s[i] += ds
            
            # Clip to valid range roughly to avoid obvious infeasibility
            # But we check feasibility anyway.
            if new_s[i] <= 0 or new_s[i] > 1.0:
                continue
            if new_c[i, 0] < 0 or new_c[i, 0] > 1.0:
                continue
            if new_c[i, 1] < 0 or new_c[i, 1] > 1.0:
                continue
                
            if _is_feasible_vec(new_c, new_s):
                new_val = np.sum(new_s)
                if new_val > best_val:
                    best_c = new_c
                    best_s = new_s
                    best_val = new_val
                    # Accept improvement, maybe increase step slightly? Or keep same.
                    # step = min(0.1, step * 1.05) 
                else:
                    # Simulated annealing style acceptance
                    # delta = new_val - best_val (negative)
                    # Accept with prob exp(delta / temp)
                    if np.random.rand() < np.exp((new_val - best_val) / temp):
                        best_c = new_c
                        best_s = new_s
                        best_val = new_val
            else:
                # Rejection
                pass
                
            # Cool down
            temp *= 0.999
            if temp < 1e-5:
                temp = 1e-5
                step *= 0.95
                if step < 1e-5:
                    break

        return best_c, best_s, best_val

    # Main logic
    seeds = []
    
    # Seed 1: The 3.0 grid
    c1, s1 = _make_layout_30()
    seeds.append((c1, s1))
    
    # Seed 2: Random perturbation of Seed 1
    c2, s2 = _make_layout_30()
    for i in range(n):
        c2[i, 0] += np.random.uniform(-0.01, 0.01)
        c2[i, 1] += np.random.uniform(-0.01, 0.01)
        s2[i] += np.random.uniform(-0.01, 0.01)
    # Ensure feasibility for seed 2
    if not _is_feasible_vec(c2, s2):
        # Fallback to seed 1 if perturbation broke it
        c2, s2 = c1, s1
    seeds.append((c2, s2))
    
    # Seed 3: Another analytical layout?
    # Try 4 rows: 2, 2, 3, 3?
    # 2*0.5 + 2*0.5 + 3*0.33 + 3*0.33 = 1+1+1+1 = 4?
    # Height: 0.5+0.5+0.33+0.33 = 1.66 > 1. No.
    # 2*0.4 + 2*0.4 + 3*0.2 + 3*0.2 = 0.8+0.8+0.6+0.6 = 2.8.
    # Height: 0.4+0.4+0.2+0.2 = 1.2 > 1.
    # 2*0.3 + 2*0.3 + 3*0.2 + 3*0.2 = 0.6+0.6+0.6+0.6 = 2.4.
    # Height: 0.3+0.3+0.2+0.2 = 1.0.
    # Sum 2.4 < 3.0.
    
    # So 3.0 is the best grid.
    
    best_centers, best_sides = seeds[0]
    best_score = _score(best_centers, best_sides)
    
    np.random.seed(42)
    
    for c_seed, s_seed in seeds:
        if time.time() > DEADLINE - 60:
            break
        c_res, s_res, val_res = _stochastic_search(c_seed, s_seed)
        if val_res > best_score:
            best_centers, best_sides, best_score = c_res, s_res, val_res

    # Final check
    if not _is_feasible_vec(best_centers, best_sides):
        best_centers, best_sides = _make_layout_30()
        best_score = _score(best_centers, best_sides)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
