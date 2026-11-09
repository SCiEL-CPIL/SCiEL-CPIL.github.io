# EVOLVE-BLOCK-START
import numpy as np
import random

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Uses a Grid-based Local Search with Hill Climbing and Neighbor Shoving.
    """
    n = 10
    tol = 1e-7
    
    def is_valid(centers, sides):
        """Check if the configuration is valid (within bounds and no overlaps)."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        
        # Check bounds
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
            
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def objective(centers, sides):
        """Calculate the sum of side lengths. Returns -inf if invalid."""
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)

    def push_apart(i, j, centers, sides):
        """
        Attempt to separate squares i and j if they overlap.
        Returns True if separated, False if unable.
        Modifies centers in place.
        """
        si = sides[i]
        sj = sides[j]
        t = (si + sj) / 2 + tol # Need separation >= si/2 + sj/2
        
        cx_i, cy_i = centers[i]
        cx_j, cy_j = centers[j]
        
        dx = cx_i - cx_j
        dy = cy_i - cy_j
        
        # Calculate required push in x and y
        # If |dx| < si/2 + sj/2 and |dy| < si/2 + sj/2, they overlap.
        # We need to increase |dx| to >= si/2 + sj/2 OR increase |dy| to >= si/2 + sj/2.
        
        overlap_x = max(0, (si + sj) / 2 - abs(dx))
        overlap_y = max(0, (si + sj) / 2 - abs(dy))
        
        if overlap_x == 0 and overlap_y == 0:
            return True # Already separated
        
        # Choose the axis with smaller overlap to push
        if overlap_x < overlap_y:
            # Push along x
            if dx >= 0:
                # i is to the right of j
                push = overlap_x / 2 + tol
                centers[i, 0] += push
                centers[j, 0] -= push
            else:
                # i is to the left of j
                push = overlap_x / 2 + tol
                centers[i, 0] -= push
                centers[j, 0] += push
        else:
            # Push along y
            if dy >= 0:
                # i is above j
                push = overlap_y / 2 + tol
                centers[i, 1] += push
                centers[j, 1] -= push
            else:
                # i is below j
                push = overlap_y / 2 + tol
                centers[i, 1] -= push
                centers[j, 1] += push
                
        # Check if bounds are violated after push
        valid = True
        for k in [i, j]:
            cx, cy = centers[k]
            s = sides[k]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                valid = False
        
        if not valid:
            # Revert if invalid
            centers[i] = [cx_i, cy_i]
            centers[j] = [cx_j, cy_j]
            return False
            
        return True

    def enforce_bounds(centers, sides):
        """Clip centers to ensure squares stay within [0,1]x[0,1]."""
        for i in range(n):
            s = sides[i]
            half = s / 2
            centers[i, 0] = np.clip(centers[i, 0], half, 1.0 - half)
            centers[i, 1] = np.clip(centers[i, 1], half, 1.0 - half)

    def local_search(centers, sides, max_iterations=1000):
        """
        Perform local search by trying to grow squares and resolve conflicts.
        """
        centers = centers.copy()
        sides = sides.copy()
        
        best_score = objective(centers, sides)
        best_centers = centers.copy()
        best_sides = sides.copy()
        
        for _ in range(max_iterations):
            improved = False
            
            # Try to grow each square
            for i in range(n):
                original_side = sides[i]
                # Try increasing size slightly
                new_side = original_side * 1.01 + 0.001
                if new_side > 1.0:
                    continue
                
                # Save state
                temp_centers = centers.copy()
                temp_sides = sides.copy()
                temp_sides[i] = new_side
                
                # Enforce bounds for the grown square
                enforce_bounds(temp_centers, temp_sides)
                
                # Resolve overlaps
                success = True
                # Iterate a few times to resolve cascading overlaps
                for _ in range(10):
                    resolved = True
                    for j in range(n):
                        if j == i:
                            continue
                        # Check if i and j overlap
                        t = (temp_sides[i] + temp_sides[j]) / 2 - tol
                        dx = abs(temp_centers[i, 0] - temp_centers[j, 0])
                        dy = abs(temp_centers[i, 1] - temp_centers[j, 1])
                        
                        if dx < t and dy < t:
                            if not push_apart(i, j, temp_centers, temp_sides):
                                success = False
                                break
                            resolved = False
                    if success and resolved:
                        break
                    if not success:
                        break
                
                if success:
                    # Check if all pairs are valid
                    if is_valid(temp_centers, temp_sides):
                        score = objective(temp_centers, temp_sides)
                        if score > best_score:
                            best_score = score
                            best_centers = temp_centers.copy()
                            best_sides = temp_sides.copy()
                            centers = temp_centers.copy()
                            sides = temp_sides.copy()
                            improved = True
            
            # Try to move each square slightly to allow better packing?
            # For now, growth is the main driver.
            
            if not improved:
                break
                
        return best_centers, best_sides, best_score

    def init_grid():
        """
        Initialize with a valid grid packing.
        3x3 grid of squares size 1/3 = 9 squares.
        We need 10.
        Let's start with 10 squares in a 4x3 grid? 1.0/4 = 0.25.
        4 cols, 3 rows = 12 spots. Pick 10.
        Size 0.25. Sum = 2.5.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 4x3 grid
        # Indices 0-9
        # Row 0: 0, 1, 2, 3
        # Row 1: 4, 5, 6, 7
        # Row 2: 8, 9, (skip 10, 11)
        
        for i in range(n):
            row = i // 4
            col = i % 4
            sides[i] = 0.25
            centers[i, 0] = (col + 0.5) * 0.25
            centers[i, 1] = (row + 0.5) * 0.25
            
        return centers, sides

    def init_random_valid():
        """
        Generate a random valid configuration by placing squares one by one
        in available space.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # Start with a reasonable size
        current_size = 0.3
        
        for i in range(n):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                s = current_size * (0.8 + 0.4 * random.random())
                if s > 0.5: s = 0.5
                cx = random.uniform(s/2, 1 - s/2)
                cy = random.uniform(s/2, 1 - s/2)
                
                # Check against existing squares
                valid = True
                for j in range(i):
                    t = (s + sides[j]) / 2 - tol
                    dx = abs(cx - centers[j, 0])
                    dy = abs(cy - centers[j, 1])
                    if dx < t and dy < t:
                        valid = False
                        break
                
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
            
            if not placed:
                # Fallback: place in a corner with small size
                s = 0.1
                cx = s/2 + i * 0.05
                cy = s/2 + i * 0.05
                if cx + s/2 > 1: cx = 1 - s/2
                if cy + s/2 > 1: cy = 1 - s/2
                centers[i] = [cx, cy]
                sides[i] = s
                
        return centers, sides

    # Main execution
    np.random.seed(42)
    random.seed(42)
    
    best_global_score = -np.inf
    best_global_centers = None
    best_global_sides = None
    
    # Run multiple restarts with different initializations
    for restart in range(5):
        if restart == 0:
            centers, sides = init_grid()
        else:
            centers, sides = init_random_valid()
        
        # Run local search
        c, s, score = local_search(centers, sides, max_iterations=500)
        
        if score > best_global_score:
            best_global_score = score
            best_global_centers = c
            best_global_sides = s
            
        # Perturb best solution for next restart? 
        # Or just try different random inits.
        
    # Final verification and cleanup
    if best_global_centers is None:
        # Fallback to grid
        best_global_centers, best_global_sides = init_grid()
        
    # Ensure finite and positive
    best_global_sides = np.maximum(best_global_sides, 1e-6)
    best_global_centers = np.clip(best_global_centers, 0, 1)
    
    # Final pass to fix any tiny overlaps or bound violations
    for i in range(n):
        half = best_global_sides[i] / 2
        best_global_centers[i, 0] = np.clip(best_global_centers[i, 0], half, 1.0 - half)
        best_global_centers[i, 1] = np.clip(best_global_centers[i, 1], half, 1.0 - half)
        
    # One last check for overlaps, push if needed
    for _ in range(10):
        for i in range(n):
            for j in range(i+1, n):
                t = (best_global_sides[i] + best_global_sides[j]) / 2 - tol
                dx = abs(best_global_centers[i, 0] - best_global_centers[j, 0])
                dy = abs(best_global_centers[i, 1] - best_global_centers[j, 1])
                if dx < t and dy < t:
                    push_apart(i, j, best_global_centers, best_global_sides)
                    enforce_bounds(best_global_centers, best_global_sides)

    # Verify final result
    if not is_valid(best_global_centers, best_global_sides):
        # If invalid, return a known valid grid
        centers, sides = init_grid()
        return centers, sides
        
    return best_global_centers, best_global_sides

# EVOLVE-BLOCK-END
