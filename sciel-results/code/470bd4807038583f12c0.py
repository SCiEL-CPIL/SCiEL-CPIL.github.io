import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with a grid
    side = int(np.ceil(np.sqrt(N)))
    # For N=26, side=6. 6x6=36 cells. We use the first 26.
    # To make it more symmetric, we can center the grid or use a specific layout.
    # A simple grid is fine for initialization.
    
    # Let's use a 6x5 grid roughly, or just 6x6 taking first 26.
    # Actually, a 5x5 grid is 25 circles. 26 needs one more.
    # Let's stick to a 6x6 grid, taking the first 26.
    # Positions: (i+0.5)/side, (j+0.5)/side
    # Radii: 0.5/side
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    for i in range(N):
        row = i // side
        col = i % side
        xs[i] = (col + 0.5) / side
        ys[i] = (row + 0.5) / side
        rs[i] = 0.5 / side
    
    # To make the initial packing slightly denser or different, we could randomize slightly,
    # but a grid is stable.
    
    best_score = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()
    
    # Optimization Loop
    growth_factor = 1.005  # Start with a conservative growth
    relax_iterations = 50
    
    while time.time() < deadline:
        # 1. Relaxation: Remove overlaps and boundary violations
        for _ in range(relax_iterations):
            moved = False
            
            # Boundary constraints
            for i in range(N):
                if xs[i] < rs[i]:
                    xs[i] = rs[i]
                    moved = True
                elif xs[i] > 1.0 - rs[i]:
                    xs[i] = 1.0 - rs[i]
                    moved = True
                
                if ys[i] < rs[i]:
                    ys[i] = rs[i]
                    moved = True
                elif ys[i] > 1.0 - rs[i]:
                    ys[i] = 1.0 - rs[i]
                    moved = True
            
            # Pairwise overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-6
                        # Overlap amount
                        overlap = min_dist - dist
                        # Push apart
                        # We push each circle by half the overlap, along the line connecting centers
                        if dist < 1e-9:
                            # Random direction if perfectly overlapping
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        
                        push = overlap * 0.5 * 1.05 # Slight over-push to break symmetry
                        xs[i] -= dx * push
                        ys[i] -= dy * push
                        xs[j] += dx * push
                        ys[j] += dy * push
                        moved = True
            
            if not moved:
                break
        
        # Check if current state is valid (it should be after relaxation)
        # But relaxation might have pushed circles out of bounds again?
        # The boundary check is done inside the loop, so they should be in bounds.
        # However, pushing one circle might push another out of bounds.
        # So we should re-check validity before scoring.
        
        valid = True
        for i in range(N):
            if xs[i] < -1e-9 or xs[i] > 1.0 + 1e-9 or ys[i] < -1e-9 or ys[i] > 1.0 + 1e-9:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    if dx*dx + dy*dy < (rs[i] + rs[j])**2 - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_score = np.sum(rs)
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
                
            # 2. Growth
            # Try to grow
            new_rs = rs * growth_factor
            # Check if growing causes immediate boundary violation
            # If any circle center is at boundary, it can't grow.
            # We just try to grow, and then relax.
            # If relaxation fails to fix overlaps, we revert.
            
            # Simple check: if any radius is too large to fit in square even at center
            # Max possible radius is 0.5.
            if np.max(new_rs) > 0.5:
                # Can't grow further for the largest circle
                # Try smaller growth? Or stop?
                # Let's try to reduce growth factor
                growth_factor *= 0.9
                if growth_factor < 1.0001:
                    break
                continue
            
            rs = new_rs
            
            # Relax again to fix overlaps caused by growth
            # We reuse the relaxation logic above, but we need to ensure it converges.
            # If it doesn't converge, we revert.
            
            # Let's do a limited number of relaxation steps for the growth phase
            converged = False
            for _ in range(relax_iterations):
                moved = False
                # Boundary
                for i in range(N):
                    if xs[i] < rs[i]:
                        xs[i] = rs[i]
                        moved = True
                    elif xs[i] > 1.0 - rs[i]:
                        xs[i] = 1.0 - rs[i]
                        moved = True
                    if ys[i] < rs[i]:
                        ys[i] = rs[i]
                        moved = True
                    elif ys[i] > 1.0 - rs[i]:
                        ys[i] = 1.0 - rs[i]
                        moved = True
                
                # Pairs
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[j] - xs[i]
                        dy = ys[j] - ys[i]
                        dist_sq = dx*dx + dy*dy
                        min_dist = rs[i] + rs[j]
                        
                        if dist_sq < min_dist * min_dist:
                            dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-6
                            overlap = min_dist - dist
                            if dist < 1e-9:
                                angle = np.random.rand() * 2 * np.pi
                                dx = np.cos(angle)
                                dy = np.sin(angle)
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                            
                            push = overlap * 0.5 * 1.05
                            xs[i] -= dx * push
                            ys[i] -= dy * push
                            xs[j] += dx * push
                            ys[j] += dy * push
                            moved = True
                            
                if not moved:
                    converged = True
                    break
            
            if not converged:
                # Revert growth
                rs = rs / growth_factor
                # Reduce growth factor
                growth_factor *= 0.95
                if growth_factor < 1.0001:
                    break
            else:
                # Successful growth, maybe increase growth factor slightly?
                # Or keep it constant. Keeping it constant is safer.
                pass
        else:
            # State is invalid, revert growth?
            # This shouldn't happen if relaxation works, but just in case.
            # Revert to last known good state?
            # For simplicity, if invalid, revert growth.
            rs = rs / growth_factor
            growth_factor *= 0.95
            if growth_factor < 1.0001:
                break

    # Return the best configuration found
    return np.column_stack((best_xs, best_ys, best_rs))
