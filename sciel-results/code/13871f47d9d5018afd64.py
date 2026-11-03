import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    def validate(circles):
        """Check if circles are valid: disjoint interiors, within bounds"""
        if len(circles) != N:
            return False
        for i in range(N):
            x, y, r = circles[i]
            if r <= 0:
                return False
            # Check bounds (need to find bounding box)
        # Find bounding box
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        w = xmax - xmin
        h = ymax - ymin
        if 2 * (w + h) > 4.0 + 1e-6:
            return False
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = circles[i,0] - circles[j,0]
                dy = circles[i,1] - circles[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i,2] + circles[j,2] - 1e-6:
                    return False
        return True
    
    def get_score(circles):
        """Sum of radii"""
        return np.sum(circles[:, 2])
    
    def resolve_collisions(circles, w, h, iters=100):
        """Resolve collisions and boundary violations"""
        circles = circles.copy()
        for _ in range(iters):
            if time.time() > deadline:
                break
            moved = False
            # Resolve boundary violations
            for i in range(N):
                x, y, r = circles[i]
                # Keep within [0, w] x [0, h]
                if x - r < 0:
                    circles[i, 0] = r
                    moved = True
                if x + r > w:
                    circles[i, 0] = w - r
                    moved = True
                if y - r < 0:
                    circles[i, 1] = r
                    moved = True
                if y + r > h:
                    circles[i, 1] = h - r
                    moved = True
            
            # Resolve pairwise collisions
            for i in range(N):
                for j in range(i+1, N):
                    dx = circles[j,0] - circles[i,0]
                    dy = circles[j,1] - circles[i,1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = circles[i,2] + circles[j,2]
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart equally
                        circles[i,0] -= nx * overlap * 0.5
                        circles[i,1] -= ny * overlap * 0.5
                        circles[j,0] += nx * overlap * 0.5
                        circles[j,1] += ny * overlap * 0.5
                        moved = True
                    elif dist <= 1e-10:
                        # Random small push
                        circles[i,0] -= 0.01
                        circles[j,0] += 0.01
                        moved = True
            if not moved:
                break
        return circles
    
    def optimize(circles, w, h, max_iters=5000):
        """Iteratively grow radii and resolve collisions"""
        circles = circles.copy()
        current_r = np.min(circles[:, 2])
        target_r = current_r * 1.01  # Start with small growth
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to increase radii
            new_r = min(target_r, (w + h) / (2 * N) * 0.5)  # Upper bound estimate
            if new_r <= current_r + 1e-10:
                # Try larger target
                target_r *= 1.001
                continue
            
            # Set all radii to new_r
            for i in range(N):
                circles[i, 2] = new_r
            
            # Resolve collisions
            circles = resolve_collisions(circles, w, h, iters=50)
            
            # Check if valid
            if validate(circles):
                current_r = new_r
                target_r = new_r * 1.005  # Small growth factor
            else:
                # Rollback
                for i in range(N):
                    circles[i, 2] = current_r
                target_r = (target_r + current_r) / 2
        
        return circles
    
    # Try multiple initial configurations
    best_result = None
    best_score = 0
    
    # Aspect ratios to try
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Initialize with grid
        # Find best grid for this aspect ratio
        best_init = None
        best_init_score = 0
        
        for m in range(3, 8):  # columns
            for n in range(3, 8):  # rows
                if m * n < N:
                    continue
                # Place N circles in m x n grid
                cells = []
                count = 0
                for i in range(m):
                    for j in range(n):
                        if count >= N:
                            break
                        x = (i + 0.5) * w / m
                        y = (j + 0.5) * h / n
                        r = min(w / m, h / n) / 2 - 1e-4
                        if r > 0:
                            cells.append([x, y, r])
                            count += 1
                if len(cells) == N:
                    cells = np.array(cells)
                    score = np.sum(cells[:, 2])
                    if score > best_init_score:
                        best_init_score = score
                        best_init = cells
        
        if best_init is None:
            continue
        
        # Optimize
        result = optimize(best_init, w, h, max_iters=2000)
        score = get_score(result)
        
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result is not None and validate(best_result):
        return best_result
    
    # Fallback to seed
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
