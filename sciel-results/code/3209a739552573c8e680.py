import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to check validity and compute score
    def evaluate_solution(sol):
        x = sol[:, 0]
        y = sol[:, 1]
        r = sol[:, 2]
        
        if np.any(r <= 0):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
            
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        req = r[:, None] + r[None, :]
        
        i, j = np.triu_indices(N, k=1)
        if np.any(dist[i, j] < req[i, j] - 1e-9):
            return -np.inf
            
        return np.sum(r)

    def greedy_place(grid_size, seed):
        """
        Place circles one by one on a grid.
        For each circle, find the grid position that maximizes the radius it can take.
        """
        np.random.seed(seed)
        
        # Grid points
        # Use a finer grid for better precision
        # 50x50 = 2500 points. 26 circles. 26*2500 = 65k checks. Fast.
        # But we need to check against existing circles.
        # Let's use 40x40 for speed.
        
        # Add some random jitter to grid points to avoid symmetry traps?
        # Or just use the grid.
        
        # Actually, let's generate candidate centers.
        # A dense grid of potential centers.
        # If a circle is placed at (x,y) with radius r, it occupies [x-r, x+r] x [y-r, y+r].
        # For it to fit in [0,1]^2, x must be in [r, 1-r] and y in [r, 1-r].
        # So if r is small, x can be anywhere.
        
        # Strategy:
        # Maintain a list of placed circles (x, y, r).
        # For the next circle, iterate over candidate centers (x_c, y_c).
        # For each candidate, calculate the maximum radius r_max such that:
        # 1. x_c - r_max >= 0, x_c + r_max <= 1 => r_max <= min(x_c, 1-x_c)
        # 2. y_c - r_max >= 0, y_c + r_max <= 1 => r_max <= min(y_c, 1-y_c)
        # 3. For all placed circles (x_j, y_j, r_j): dist((x_c,y_c), (x_j,y_j)) >= r_max + r_j
        #    => r_max <= dist((x_c,y_c), (x_j,y_j)) - r_j
        
        # So r_max = min( boundary_constraint, min_j(dist_j - r_j) )
        # We want to maximize r_max.
        
        # Candidate centers:
        # A grid of points.
        # Let's use a 30x30 grid for candidates.
        # That's 900 candidates.
        # For each candidate, check against up to 25 placed circles.
        # 900 * 25 = 22,500 distance calculations per circle.
        # 26 circles * 22,500 = 585,000 operations. Very fast.
        
        # To improve, we can refine the best candidate.
        
        candidates_x = np.linspace(0, 1, 30)
        candidates_y = np.linspace(0, 1, 30)
        
        # Add center and corners? No, corners are covered by grid.
        # Maybe add a finer grid? 50x50 = 2500. 2500*25*26 = 1.6M. Still fast.
        # Let's use 40x40.
        
        candidates_x = np.linspace(0, 1, 40)
        candidates_y = np.linspace(0, 1, 40)
        
        # Shuffle candidate order to avoid bias?
        # No, we pick the BEST candidate.
        
        placed = []
        
        for _ in range(N):
            best_r = -1
            best_x = None
            best_y = None
            
            # Vectorized calculation for all candidates
            # cx, cy: (40, 40) or (1600,)
            cx, cy = np.meshgrid(candidates_x, candidates_y)
            cx = cx.ravel()
            cy = cy.ravel()
            
            # Boundary constraints
            r_bound = np.minimum(np.minimum(cx, 1 - cx), np.minimum(cy, 1 - cy))
            
            # Constraints from placed circles
            if len(placed) > 0:
                # placed: list of (x, y, r)
                px = np.array([p[0] for p in placed])
                py = np.array([p[1] for p in placed])
                pr = np.array([p[2] for p in placed])
                
                # Distances: (num_candidates, num_placed)
                # cx: (C,), px: (P,)
                # dx: (C, P)
                dx = cx[:, None] - px[None, :]
                dy = cy[:, None] - py[None, :]
                dists = np.sqrt(dx**2 + dy**2)
                
                # Required distance: dists >= r_new + r_j
                # r_new <= dists - r_j
                # r_new <= min_j(dists - r_j)
                # dists - r_j: (C, P)
                # min over j (axis=1)
                r_placed_constraint = np.min(dists - pr[None, :], axis=1)
                
                # Final radius is min of boundary and placed constraints
                r_final = np.minimum(r_bound, r_placed_constraint)
            else:
                r_final = r_bound
            
            # Find the best candidate
            # We want max r_final
            # Note: r_final can be negative if a candidate is inside a placed circle.
            # We only care about positive radii.
            
            max_r = np.max(r_final)
            
            if max_r <= 1e-6:
                # No valid placement?
                # This shouldn't happen unless the square is full.
                # But if it does, break.
                break
                
            # Get the index of the max
            idx = np.argmax(r_final)
            best_r = r_final[idx]
            best_x = cx[idx]
            best_y = cy[idx]
            
            # Refine the position?
            # The grid might be too coarse.
            # Let's do a local search around (best_x, best_y).
            # Try shifting x and y slightly.
            # Or just accept the grid point.
            # Given the 40x40 grid, the error is small.
            
            placed.append((best_x, best_y, best_r))
            
            # Optional: Remove very small circles?
            # If best_r is very small, maybe we should have placed a different circle?
            # Greedy doesn't backtrack.
            
        if len(placed) < N:
            return None
            
        sol = np.array(placed)
        return sol

    def refine_positions(sol, max_iters=100):
        """
        Refine positions to ensure validity and potentially allow larger radii?
        No, radii are fixed. We just want to ensure no overlaps.
        Actually, if we have overlaps, we need to fix them.
        But greedy should produce non-overlapping circles.
        However, due to floating point or grid discretization, there might be tiny overlaps.
        Also, we might be able to move circles to create more space, but radii are fixed.
        So this is just a "fix overlaps" step.
        
        Wait, if I fix overlaps by moving circles, I might push them out of bounds.
        So I need to be careful.
        
        Actually, let's just check validity. If invalid, try to fix.
        """
        x = sol[:, 0].copy()
        y = sol[:, 1].copy()
        r = sol[:, 2].copy()
        
        # Simple coordinate descent to resolve overlaps
        # If dist(i,j) < r_i + r_j, move i and j apart.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx**2 + dy**2)
                    req = r[i] + r[j]
                    
                    if dist < req - 1e-9:
                        # Overlap
                        # Move them apart along the vector connecting them
                        if dist < 1e-9:
                            # Random direction
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        # Move each by half the overlap
                        overlap = req - dist
                        move = overlap / 2.0
                        
                        x[i] += dx * move
                        y[i] += dy * move
                        x[j] -= dx * move
                        y[j] -= dy * move
                        
                        moved = True
            
            # Clamp to boundaries
            # If x - r < 0, x = r
            # If x + r > 1, x = 1 - r
            x = np.maximum(x, r)
            x = np.minimum(x, 1 - r)
            y = np.maximum(y, r)
            y = np.minimum(y, 1 - r)
            
            if not moved:
                break
                
        sol_refined = np.column_stack((x, y, r))
        return sol_refined

    best_score = -np.inf
    best_solution = None
    
    # Try multiple seeds for the greedy algorithm
    # The order of placement matters? No, we pick the best spot for each.
    # But the "best spot" depends on what's already there.
    # So the sequence of circles (which one is placed first) matters.
    # In my greedy, I just place circle 1, then 2, etc.
    # But circle 1 and 2 are identical in size (initially).
    # So the specific index doesn't matter, only the set.
    # However, the "best spot" for circle 1 is the center (largest radius).
    # Then circle 2 goes to a corner or side.
    # This is deterministic.
    # To get diversity, I should randomize the candidate grid or the order.
    
    # Let's randomize the grid slightly.
    for seed in range(50):
        if time.time() > deadline:
            break
            
        # Perturb the grid
        # Instead of fixed linspace, add noise?
        # Or just use the same grid.
        # Since the choice is "max radius", it's deterministic for a fixed grid.
        # To break symmetry, I can rotate the grid or use a different grid size.
        
        # Let's vary the grid size.
        grid_size = 20 + (seed % 10) * 5  # 20, 25, 30...
        
        # Re-implement greedy with variable grid size
        # This is inefficient to re-implement.
        # Let's just use a fixed grid but randomize the *order* of evaluation?
        # No, we pick the global max.
        
        # Okay, let's just use the same grid.
        # The result will be the same for all seeds.
        # I need to introduce randomness.
        
        # Idea: Randomize the initial "bias" for radii?
        # No.
        
        # Idea: Randomly select the "next" circle to place?
        # They are all identical.
        
        # Idea: Randomize the grid points.
        
        # Let's modify greedy_place to accept a grid.
        
        pass # Placeholder for now, will implement in code below

    # Actual implementation of the loop with diverse grids
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        # Create a perturbed grid
        base_size = 30 + (seed % 5) * 5
        xs = np.linspace(0, 1, base_size)
        ys = np.linspace(0, 1, base_size)
        
        # Add random jitter to grid points
        # This ensures different "best" spots are found
        np.random.seed(seed)
        jitter = 0.01
        xs = xs + np.random.uniform(-jitter, jitter, base_size)
        ys = ys + np.random.uniform(-jitter, jitter, base_size)
        
        # Clip to [0,1]
        xs = np.clip(xs, 0, 1)
        ys = np.clip(ys, 0, 1)
        
        # Greedy placement
        placed = []
        
        for _ in range(N):
            best_r = -1
            best_x = None
            best_y = None
            
            cx, cy = np.meshgrid(xs, ys)
            cx = cx.ravel()
            cy = cy.ravel()
            
            r_bound = np.minimum(np.minimum(cx, 1 - cx), np.minimum(cy, 1 - cy))
            
            if len(placed) > 0:
                px = np.array([p[0] for p in placed])
                py = np.array([p[1] for p in placed])
                pr = np.array([p[2] for p in placed])
                
                dx = cx[:, None] - px[None, :]
                dy = cy[:, None] - py[None, :]
                dists = np.sqrt(dx**2 + dy**2)
                
                r_placed_constraint = np.min(dists - pr[None, :], axis=1)
                r_final = np.minimum(r_bound, r_placed_constraint)
            else:
                r_final = r_bound
            
            max_r = np.max(r_final)
            
            if max_r <= 1e-6:
                break
                
            idx = np.argmax(r_final)
            best_r = r_final[idx]
            best_x = cx[idx]
            best_y = cy[idx]
            
            placed.append((best_x, best_y, best_r))
            
        if len(placed) < N:
            continue
            
        sol = np.array(placed)
        
        # Refine
        sol_refined = refine_positions(sol, max_iters=50)
        
        score = evaluate_solution(sol_refined)
        
        if score > best_score:
            best_score = score
            best_solution = sol_refined.copy()
            
    # Fallback
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        
    return best_solution
