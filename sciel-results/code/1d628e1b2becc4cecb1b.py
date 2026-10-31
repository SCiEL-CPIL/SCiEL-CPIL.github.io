import numpy as np
import time

N = 26

def construct():
    # New Approach: Greedy Local Search with Overlap Relaxation
    # 1. Initialize with a grid.
    # 2. Iteratively try to increase radius of a random circle.
    # 3. If overlap occurs, resolve by pushing circles apart (relaxation).
    # 4. Accept move only if resulting configuration is valid (inside box, no overlaps).
    # 5. Occasionally perturb positions to escape local minima.
    
    def initial_solution():
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Start with small radii to ensure validity
                r = 0.5 / side * 0.5
                xs[count] = x
                ys[count] = y
                rs[count] = r
                count += 1
                if count == N:
                    break
        
        return xs, ys, rs

    def is_valid(xs, ys, rs):
        # Check boundaries
        if np.any(xs < rs - 1e-6) or np.any(xs > 1 - rs + 1e-6):
            return False
        if np.any(ys < rs - 1e-6) or np.any(ys > 1 - rs + 1e-6):
            return False
        
        # Check overlaps
        # Vectorized check is tricky for all pairs, but N=26 is small.
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                sum_r = rs[i] + rs[j]
                if dist_sq < sum_r*sum_r - 1e-6:
                    return False
        return True

    def resolve_overlaps(xs, ys, rs, max_iter=100):
        # Returns True if resolved without breaking boundaries significantly
        # We push circles apart and towards center if hitting boundary
        for _ in range(max_iter):
            moved = False
            # Handle overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        if dist < 1e-9:
                            # Random direction if coincident
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dist = np.sqrt(dx*dx + dy*dy)
                        
                        # Push apart
                        # We want to move them so dist becomes min_dist
                        # Move each by half the overlap
                        overlap = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        
                        xs[i] += ux * overlap
                        ys[i] += uy * overlap
                        xs[j] -= ux * overlap
                        ys[j] -= uy * overlap
                        moved = True
            
            # Handle boundaries
            for i in range(N):
                # x bounds
                if xs[i] < rs[i]:
                    xs[i] = rs[i]
                    moved = True
                elif xs[i] > 1 - rs[i]:
                    xs[i] = 1 - rs[i]
                    moved = True
                # y bounds
                if ys[i] < rs[i]:
                    ys[i] = rs[i]
                    moved = True
                elif ys[i] > 1 - rs[i]:
                    ys[i] = 1 - rs[i]
                    moved = True
            
            if not moved:
                break
        
        return is_valid(xs, ys, rs)

    xs, ys, rs = initial_solution()
    
    # Ensure initial is valid
    if not is_valid(xs, ys, rs):
        # Shrink radii until valid
        for _ in range(100):
            rs *= 0.9
            if is_valid(xs, ys, rs):
                break
    
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()
    best_sum_r = np.sum(rs)
    
    start_time = time.time()
    time_limit = 110
    
    # Adaptive step size for radius growth
    delta_r = 0.005
    
    while time.time() - start_time < time_limit:
        # Choose a random move type
        # 90% chance: Try to grow a random circle
        # 10% chance: Try to move a random circle (jiggle)
        
        if np.random.rand() < 0.9:
            i = np.random.randint(N)
            # Save state
            old_xs = xs.copy()
            old_ys = ys.copy()
            old_rs = rs.copy()
            
            # Try to increase radius
            rs[i] += delta_r
            
            # Resolve overlaps
            valid = resolve_overlaps(xs, ys, rs)
            
            if valid:
                sum_r = np.sum(rs)
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
                # Accept the move
            else:
                # Revert
                xs = old_xs
                ys = old_ys
                rs = old_rs
        else:
            # Jiggle move: Perturb position, keep radius same
            i = np.random.randint(N)
            old_xs = xs.copy()
            old_ys = ys.copy()
            old_rs = rs.copy()
            
            # Small random displacement
            xs[i] += np.random.uniform(-0.05, 0.05)
            ys[i] += np.random.uniform(-0.05, 0.05)
            
            # Resolve overlaps (and boundaries)
            valid = resolve_overlaps(xs, ys, rs)
            
            if valid:
                # Accept even if sum_r doesn't change, as it might help future growth
                # But we must ensure we didn't break validity (resolve_overlaps ensures this)
                pass
            else:
                # Revert
                xs = old_xs
                ys = old_ys
                rs = old_rs
        
        # Occasionally adjust delta_r
        if np.random.rand() < 0.01:
            delta_r *= np.random.uniform(0.5, 2.0)
            delta_r = np.clip(delta_r, 0.001, 0.05)

    return np.column_stack((best_xs, best_ys, best_rs))
