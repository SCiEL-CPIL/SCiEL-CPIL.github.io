import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_radii(x, y):
        """Compute max feasible radius for each circle given positions."""
        r = np.full(N, 0.5)
        # Boundary constraints
        r = np.minimum(r, np.minimum(x, 1.0 - x))
        r = np.minimum(r, np.minimum(y, 1.0 - y))
        # Pairwise constraints: r_i + r_j <= dist_ij
        # For each pair, we need r_i <= dist_ij - r_j and r_j <= dist_ij - r_i
        # Iterate to convergence
        for _ in range(20):
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    r[i] = min(r[i], 0.0)
                    r[j] = min(r[j], 0.0)
                else:
                    # r_i + r_j <= d => r_i <= d - r_j, r_j <= d - r_i
                    new_ri = min(r[i], d - r[j])
                    new_rj = min(r[j], d - r[i])
                    r[i] = max(0.0, new_ri)
                    r[j] = max(0.0, new_rj)
        return r
    
    def voronoi_centroid(x, y, r):
        """Compute Voronoi cell centroid for each circle using iterative clipping."""
        centroids = np.zeros((N, 2))
        for i in range(N):
            # Start with the bounding box [0,1]^2
            # Approximate Voronoi cell by clipping against perpendicular bisectors
            # Use a simple approach: sample points and find nearest neighbor regions
            # For efficiency, use the fact that in a packing, the "center of mass" 
            # of the available space is what we want.
            # Simple approach: move toward the center of the box, away from neighbors
            cx = x[i]
            cy = y[i]
            
            # Compute "push" directions from neighbors
            total_fx = 0.0
            total_fy = 0.0
            weight_sum = 0.0
            
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    continue
                # Weight by how close they are (closer = more influence)
                # The Voronoi edge is at midpoint; if we're past it, we should move back
                mid_x = (x[i] + x[j]) / 2.0
                mid_y = (y[i] + y[j]) / 2.0
                # Distance from i to the midpoint
                dist_to_mid = np.sqrt((x[i]-mid_x)**2 + (y[i]-mid_y)**2)
                # If circle i is large relative to distance, it "owns" more space
                # Simple heuristic: weight = 1/d^2
                w = 1.0 / (d*d + 1e-12)
                # Direction from neighbor to i
                fx = dx / d * w
                fy = dy / d * w
                total_fx += fx
                total_fy += fy
                weight_sum += w
            
            # Boundary forces: push away from walls
            if x[i] < r[i]:
                total_fx += (r[i] - x[i]) * 10.0
            if 1.0 - x[i] < r[i]:
                total_fx -= (r[i] - (1.0 - x[i])) * 10.0
            if y[i] < r[i]:
                total_fy += (r[i] - y[i]) * 10.0
            if 1.0 - y[i] < r[i]:
                total_fy -= (r[i] - (1.0 - y[i])) * 10.0
            
            if weight_sum > 1e-12:
                cx = x[i] + total_fx / weight_sum * 0.1
                cy = y[i] + total_fy / weight_sum * 0.1
            
            # Clip to valid region
            cx = np.clip(cx, r[i], 1.0 - r[i])
            cy = np.clip(cy, r[i], 1.0 - r[i])
            
            centroids[i] = [cx, cy]
        
        return centroids
    
    def push_apart(x, y, r, n_iters=50):
        """Iteratively push overlapping circles apart."""
        for _ in range(n_iters):
            moved = False
            for k in range(len(i_idx)):
                i, j = i_idx[k], j_idx[k]
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-12:
                    # Push apart
                    if d < 1e-12:
                        # Random direction
                        angle = random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        d = 1.0
                    else:
                        dx /= d
                        dy /= d
                    
                    overlap = (min_d - d) / 2.0 + 1e-8
                    x[i] += dx * overlap
                    y[i] += dy * overlap
                    x[j] -= dx * overlap
                    y[j] -= dy * overlap
                    moved = True
            
            # Clip to bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            if not moved:
                break
        return x, y
    
    def score(x, y):
        r = compute_radii(x, y)
        return np.sum(r)
    
    # Initialize with a few different starting configurations
    best_score = -1.0
    best_x = None
    best_y = None
    best_r = None
    
    # Starting configuration 1: Jittered grid
    side = 5
    x = np.zeros(N)
    y = np.zeros(N)
    indices = 0
    for i in range(side):
        for j in range(side):
            if indices < N:
                x[indices] = (i + 0.5) / side
                y[indices] = (j + 0.5) / side
                indices += 1
    x += np.random.uniform(-0.02, 0.02, N)
    y += np.random.uniform(-0.02, 0.02, N)
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    
    # Starting configuration 2: Hexagonal-like
    x2 = np.zeros(N)
    y2 = np.zeros(N)
    row = 0
    col = 0
    for idx in range(N):
        x2[idx] = 0.1 + col * 0.15
        y2[idx] = 0.1 + row * 0.13 + (0.07 if row % 2 == 1 else 0)
        col += 1
        if col >= 5:
            col = 0
            row += 1
    x2 += np.random.uniform(-0.01, 0.01, N)
    y2 += np.random.uniform(-0.01, 0.01, N)
    x2 = np.clip(x2, 0.05, 0.95)
    y2 = np.clip(y2, 0.05, 0.95)
    
    # Starting configuration 3: Random
    x3 = np.random.uniform(0.1, 0.9, N)
    y3 = np.random.uniform(0.1, 0.9, N)
    
    # Starting configuration 4: Ring
    x4 = np.zeros(N)
    y4 = np.zeros(N)
    x4[0] = 0.5
    y4[0] = 0.5
    for idx in range(1, N):
        angle = 2 * np.pi * (idx - 1) / (N - 1)
        radius = 0.3 if idx < 8 else 0.45
        x4[idx] = 0.5 + radius * np.cos(angle)
        y4[idx] = 0.5 + radius * np.sin(angle)
    x4 = np.clip(x4, 0.05, 0.95)
    y4 = np.clip(y4, 0.05, 0.95)
    
    inits = [(x, y), (x2, y2), (x3, y3), (x4, y4)]
    
    for init_idx, (init_x, init_y) in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        cur_x = init_x.copy()
        cur_y = init_y.copy()
        cur_r = compute_radii(cur_x, cur_y)
        
        # Ensure initial validity by shrinking
        for _ in range(50):
            cur_r = compute_radii(cur_x, cur_y)
            # Check pairwise
            dx = cur_x[i_idx] - cur_x[j_idx]
            dy = cur_y[i_idx] - cur_y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            if np.max(cur_r[i_idx] + cur_r[j_idx] - dist) < 1e-9:
                break
            cur_r *= 0.95
        
        cur_score = np.sum(cur_r)
        
        # Main iterative loop: alternate between radius computation and center movement
        for step in range(2000):
            if time.time() > deadline - 15.0:
                break
            
            # Step 1: Compute max feasible radii
            cur_r = compute_radii(cur_x, cur_y)
            
            # Step 2: Move centers toward Voronoi centroids
            centroids = voronoi_centroid(cur_x, cur_y, cur_r)
            
            # Blend: move a fraction toward centroid
            alpha = 0.1
            new_x = cur_x + alpha * (centroids[:, 0] - cur_x)
            new_y = cur_y + alpha * (centroids[:, 1] - cur_y)
            
            # Clip to bounds
            new_x = np.clip(new_x, 0, 1)
            new_y = np.clip(new_y, 0, 1)
            
            # Step 3: Push apart any overlaps caused by movement
            new_x, new_y = push_apart(new_x, new_y, cur_r, n_iters=20)
            
            # Recompute radii after push
            new_r = compute_radii(new_x, new_y)
            
            # Ensure validity
            for _ in range(20):
                dx = new_x[i_idx] - new_x[j_idx]
                dy = new_y[i_idx] - new_y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                if np.max(new_r[i_idx] + new_r[j_idx] - dist) < 1e-9:
                    break
                new_r *= 0.98
                new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            new_score = np.sum(new_r)
            
            if new_score > cur_score:
                cur_x = new_x
                cur_y = new_y
                cur_r = new_r
                cur_score = new_score
            
            # Occasionally add small perturbation to escape local minima
            if step % 200 == 100:
                perturb = 0.01
                test_x = cur_x + np.random.uniform(-perturb, perturb, N)
                test_y = cur_y + np.random.uniform(-perturb, perturb, N)
                test_x = np.clip(test_x, 0, 1)
                test_y = np.clip(test_y, 0, 1)
                test_r = compute_radii(test_x, test_y)
                test_score = np.sum(test_r)
                if test_score > cur_score:
                    cur_x = test_x
                    cur_y = test_y
                    cur_r = test_r
                    cur_score = test_score
        
        # Update best
        if cur_score > best_score:
            best_score = cur_score
            best_x = cur_x.copy()
            best_y = cur_y.copy()
            best_r = cur_r.copy()
    
    # Final polish: more aggressive local search on the best configuration
    if best_x is not None and time.time() < deadline - 10.0:
        cur_x = best_x.copy()
        cur_y = best_y.copy()
        cur_r = best_r.copy()
        cur_score = best_score
        
        # Fine-tune with smaller steps
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            cur_r = compute_radii(cur_x, cur_y)
            centroids = voronoi_centroid(cur_x, cur_y, cur_r)
            
            alpha = 0.05
            new_x = cur_x + alpha * (centroids[:, 0] - cur_x)
            new_y = cur_y + alpha * (centroids[:, 1] - cur_y)
            new_x = np.clip(new_x, 0, 1)
            new_y = np.clip(new_y, 0, 1)
            
            new_x, new_y = push_apart(new_x, new_y, cur_r, n_iters=10)
            new_r = compute_radii(new_x, new_y)
            
            for _ in range(10):
                dx = new_x[i_idx] - new_x[j_idx]
                dy = new_y[i_idx] - new_y[j_idx]
                dist = np.sqrt(dx*dx + dy*dy)
                if np.max(new_r[i_idx] + new_r[j_idx] - dist) < 1e-9:
                    break
                new_r *= 0.99
                new_r = np.minimum(new_r, np.minimum(new_x, 1-new_x))
                new_r = np.minimum(new_r, np.minimum(new_y, 1-new_y))
            
            new_score = np.sum(new_r)
            if new_score > cur_score:
                cur_x = new_x
                cur_y = new_y
                cur_r = new_r
                cur_score = new_score
        
        if cur_score > best_score:
            best_score = cur_score
            best_x = cur_x
            best_y = cur_y
            best_r = cur_r
    
    # Final validity check
    bx = best_x
    by = best_y
    br = best_r
    
    br = np.minimum(br, np.minimum(bx, 1-bx))
    br = np.minimum(br, np.minimum(by, 1-by))
    
    for _ in range(50):
        dx = bx[i_idx] - bx[j_idx]
        dy = by[i_idx] - by[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        violations = br[i_idx] + br[j_idx] - dist
        if np.max(violations) < 1e-10:
            break
        br *= 0.999
        br = np.minimum(br, np.minimum(bx, 1-bx))
        br = np.minimum(br, np.minimum(by, 1-by))
    
    final_state = np.column_stack([bx, by, br])
    return final_state
