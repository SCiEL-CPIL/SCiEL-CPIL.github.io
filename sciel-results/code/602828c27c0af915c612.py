import time
import numpy as np

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 600  # Safety margin

    def is_feasible(centers, sides):
        """Check if the configuration is valid."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        if np.any(centers - sides[:, None] / 2 < -tol) or np.any(centers + sides[:, None] / 2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def greedy_expand(centers, sides, max_passes=10):
        """
        Iteratively try to expand each square's side as much as possible
        while maintaining feasibility. We do this by fixing the center
        and finding the max side that fits given neighbors and boundaries.
        Then we might adjust the center slightly if it helps, but primarily
        we just expand.
        
        Actually, a better approach:
        For each square i:
          1. Determine the max possible side s_new such that square i with side s_new
             centered at current center[i] does not overlap any other square j
             (with their current sides) and stays in bounds.
          2. Set sides[i] = s_new.
        
        This is valid because increasing a square's side only makes it harder to fit,
        but we are checking feasibility for that specific size. If we find the max size
        that fits, we set it. We repeat this for all squares in a few passes.
        
        Note: This greedy approach might get stuck in local optima if a square needs
        to shrink to allow another to grow, but for a good initial tiling, it should
        only grow or stay stable.
        """
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for i in range(n):
                cx, cy = centers[i]
                # Max side limited by boundaries
                max_s_bound = 2 * min(cx, 1 - cx, cy, 1 - cy) + 2 * tol
                if max_s_bound > 1.0:
                    max_s_bound = 1.0
                
                # Max side limited by neighbors
                max_s_neighbor = max_s_bound
                for j in range(n):
                    if i == j:
                        continue
                    cx_j, cy_j = centers[j]
                    s_j = sides[j]
                    
                    # Overlap condition: |cx - cx_j| < (s + s_j)/2 - tol AND |cy - cy_j| < (s + s_j)/2 - tol
                    # We want to find max s such that NOT (overlap)
                    # i.e., |cx - cx_j| >= (s + s_j)/2 - tol OR |cy - cy_j| >= (s + s_j)/2 - tol
                    
                    dx = abs(cx - cx_j)
                    dy = abs(cy - cy_j)
                    
                    # If dx is large enough, no constraint from this neighbor in x
                    # Condition: dx >= (s + s_j)/2 - tol  =>  s <= 2*dx - s_j + 2*tol
                    # If dy is large enough, no constraint from this neighbor in y
                    # Condition: dy >= (s + s_j)/2 - tol  =>  s <= 2*dy - s_j + 2*tol
                    
                    # The constraint is active if BOTH dx < (s+s_j)/2 - tol AND dy < (s+s_j)/2 - tol
                    # We want to avoid this.
                    # So we need s such that either:
                    # s <= 2*dx - s_j + 2*tol  OR  s <= 2*dy - s_j + 2*tol
                    # This is equivalent to s <= max(2*dx - s_j + 2*tol, 2*dy - s_j + 2*tol)
                    # Wait, no.
                    # If s > 2*dx - s_j + 2*tol, then dx < (s+s_j)/2 - tol.
                    # If s > 2*dy - s_j + 2*tol, then dy < (s+s_j)/2 - tol.
                    # Overlap happens if BOTH conditions are true.
                    # So we must have:
                    # NOT (s > 2*dx - s_j + 2*tol AND s > 2*dy - s_j + 2*tol)
                    # Which is:
                    # s <= 2*dx - s_j + 2*tol  OR  s <= 2*dy - s_j + 2*tol
                    # This means s can be as large as max(2*dx - s_j + 2*tol, 2*dy - s_j + 2*tol)
                    # BUT only if that value is positive. If both are negative, it's impossible?
                    # No, if both are negative, it means even s=0 overlaps?
                    # If s=0, t = -tol. |dx| < -tol is false. So s=0 is always safe.
                    # The formula s <= max(...) works.
                    
                    limit_x = 2 * dx - s_j + 2 * tol
                    limit_y = 2 * dy - s_j + 2 * tol
                    limit = max(limit_x, limit_y)
                    
                    if limit < max_s_neighbor:
                        max_s_neighbor = limit
                
                # Ensure side is at least current side? No, we want to maximize.
                # But we shouldn't shrink if not needed? Actually, if we shrink, we might allow others to grow.
                # However, greedy expansion usually only grows.
                # Let's just set to max possible, but not less than current?
                # If current is feasible, max_s_neighbor should be >= current side.
                # Due to floating point, it might be slightly less.
                # Let's take max(current_side, max_s_neighbor) but capped by boundary?
                # Actually, if max_s_neighbor < current_side, it means current is infeasible?
                # Or just numerical error.
                # Let's just set to max_s_neighbor if it's > current_side, else keep current_side.
                # But wait, if we keep current_side, we don't improve.
                # Let's try to set to max_s_neighbor. If it's smaller, we might break feasibility if it was tight.
                # Safer: only update if max_s_neighbor > sides[i] + 1e-9.
                
                if max_s_neighbor > sides[i] + 1e-9:
                    sides[i] = max_s_neighbor
                    changed = True
                elif max_s_neighbor < sides[i] - 1e-9:
                    # This shouldn't happen if we started feasible and only grew.
                    # But if it does, shrink to avoid infeasibility.
                    sides[i] = max_s_neighbor
                    changed = True
            
            if not changed:
                break
        
        return centers, sides

    # Construct the optimal tiling:
    # 2 large squares of side 0.5
    # 8 small squares of side 0.25
    # Layout:
    # Large 1: [0, 0.5] x [0, 0.5] -> center (0.25, 0.25)
    # Large 2: [0.5, 1] x [0.5, 1] -> center (0.75, 0.75)
    # Small squares fill the rest:
    # Top-left quadrant [0, 0.5] x [0.5, 1]: 4 small squares of 0.25
    #   (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)
    # Bottom-right quadrant [0.5, 1] x [0, 0.5]: 4 small squares of 0.25
    #   (0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375)
    
    centers = np.array([
        [0.25, 0.25], # Large 1
        [0.75, 0.75], # Large 2
        [0.125, 0.625], # Small 1
        [0.375, 0.625], # Small 2
        [0.125, 0.875], # Small 3
        [0.375, 0.875], # Small 4
        [0.625, 0.125], # Small 5
        [0.875, 0.125], # Small 6
        [0.625, 0.375], # Small 7
        [0.875, 0.375], # Small 8
    ])
    sides = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    
    # Verify initial feasibility
    if not is_feasible(centers, sides):
        # Fallback to grid if something went wrong (shouldn't happen)
        centers = np.column_stack((np.linspace(0.1, 0.9, 5).repeat(2), 
                                   np.tile(np.linspace(0.1, 0.9, 2), 5)))
        sides = np.full(n, 0.2)
        return centers, sides
    
    # Try to expand
    centers_exp, sides_exp = greedy_expand(centers.copy(), sides.copy(), max_passes=5)
    
    if is_feasible(centers_exp, sides_exp):
        if np.sum(sides_exp) > np.sum(sides):
            centers, sides = centers_exp, sides_exp
    
    return centers, sides
