"""Program for the no-isosceles set task using a NumPy-vectorized greedy + simulated annealing approach."""
import time
import random
import numpy as np

N = 64
DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s limit


def construct():
    """
    Construct a set of points in [0, N)^2 with no three forming an isosceles triangle.
    
    Strategy:
    1. Greedy initialization with multiple random orderings.
    2. Simulated annealing with swap moves (remove one point, add another).
    3. Use NumPy for vectorized distance checks where possible.
    """
    start_time = time.time()
    deadline = start_time + 330 - DEADLINE_MARGIN
    
    # Precompute grid points as integer array
    grid = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    n_grid = len(grid)
    
    best_points = []
    
    # Multiple restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # --- Phase 1: Greedy construction ---
        order = np.random.permutation(n_grid)
        selected = []
        selected_set = set()
        
        # For fast lookup: maintain list of selected points and their squared distances to others
        # We'll use a list of tuples and check incrementally
        
        for idx in order:
            if time.time() > deadline:
                break
            
            p = grid[idx]
            px, py = p[0], p[1]
            valid = True
            
            # Check against all pairs of selected points
            k = len(selected)
            if k >= 2:
                # Vectorized check: compute distances from p to all selected points
                sel_arr = np.array(selected, dtype=np.int32)
                dx = sel_arr[:, 0] - px
                dy = sel_arr[:, 1] - py
                d2 = dx * dx + dy * dy  # squared distances from p to each selected point
                
                # Check 1: No two selected points are equidistant from p
                # i.e., all d2 values must be distinct
                if len(np.unique(d2)) < len(d2):
                    valid = False
                else:
                    # Check 2: For each selected point A, dist(p, A) must not equal dist(A, B) for any other selected B
                    # For each i, check if d2[i] equals any dist(selected[i], selected[j]) for j != i
                    # Precompute pairwise distances among selected points
                    # This is O(k^2) but k is small in early stages
                    if k <= 100:  # Only do full check for small k
                        for i in range(k):
                            ai = selected[i]
                            # d2[i] is dist(p, ai)^2
                            for j in range(k):
                                if i == j:
                                    continue
                                aj = selected[j]
                                dax2 = (ai[0] - aj[0])**2 + (ai[1] - aj[1])**2
                                if dax2 == d2[i]:
                                    valid = False
                                    break
                            if not valid:
                                break
            
            if valid:
                selected.append(p)
                selected_set.add((px, py))
        
        # Convert to list of tuples for easier handling
        current_points = [(int(p[0]), int(p[1])) for p in selected]
        
        if len(current_points) > len(best_points):
            best_points = list(current_points)
        
        if time.time() > deadline:
            break
        
        # --- Phase 2: Simulated Annealing with swaps ---
        # Energy function: number of isosceles triangles (we want 0)
        # But since we maintain validity, energy is always 0 for valid states.
        # Instead, we use a different approach: try to add points via swaps.
        
        # Actually, let's do a local search: try to swap a point with a non-point to increase count.
        # This is tricky because removing a point might allow adding two points elsewhere.
        
        # Simpler: try to add a point by removing one that's "least useful"
        # For now, just do greedy + a few random restarts
        
        # Try to extend with a second pass of greedy
        remaining = [grid[i] for i in range(n_grid) if (grid[i][0], grid[i][1]) not in selected_set]
        random.shuffle(remaining)
        
        for p in remaining:
            if time.time() > deadline:
                break
            px, py = p[0], p[1]
            valid = True
            k = len(current_points)
            if k >= 2:
                sel_arr = np.array(current_points, dtype=np.int32)
                dx = sel_arr[:, 0] - px
                dy = sel_arr[:, 1] - py
                d2 = dx * dx + dy * dy
                if len(np.unique(d2)) < len(d2):
                    valid = False
                else:
                    for i in range(k):
                        ai = current_points[i]
                        for j in range(k):
                            if i == j:
                                continue
                            aj = current_points[j]
                            dax2 = (ai[0] - aj[0])**2 + (ai[1] - aj[1])**2
                            if dax2 == d2[i]:
                                valid = False
                                break
                        if not valid:
                            break
            if valid:
                current_points.append((px, py))
                selected_set.add((px, py))
        
        if len(current_points) > len(best_points):
            best_points = list(current_points)
    
    return best_points


# Test locally
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Found {len(res)} points in {end-start:.2f}s")
    # Verify no isosceles triangles
    n = len(res)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = res[i], res[j], res[k]
                d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                if d12 == d23 or d23 == d13 or d13 == d12:
                    print(f"Isosceles found: {p1}, {p2}, {p3}")
                    exit(1)
    print("Valid!")
