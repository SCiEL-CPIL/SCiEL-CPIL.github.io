"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def generate_hex_grid(spacing):
        # Generate hex grid centered at 0
        max_coord = 0.5
        n_x = int(2 * max_coord / spacing) + 2
        n_y = int(2 * max_coord / (spacing * np.sqrt(3) / 2)) + 2
        
        points = []
        for j in range(-n_y, n_y + 1):
            for i in range(-n_x, n_x + 1):
                x = i * spacing + (j % 2) * spacing / 2
                y = j * spacing * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
        return np.array(points)

    def random_init(spacing=0.07):
        grid_points = generate_hex_grid(spacing)
        if len(grid_points) < N:
            # Fallback to square grid if hex is too sparse (shouldn't happen with 0.07)
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # Randomly select N points
        indices = np.random.choice(len(grid_points), N, replace=False)
        return grid_points[indices]

    def min_distance(points):
        min_d = 1.0
        # Pairs
        for i in range(N):
            pi = points[i]
            for j in range(i+1, N):
                pj = points[j]
                d = np.linalg.norm(pi - pj)
                if d < min_d:
                    min_d = d
        # Boundaries
        for i in range(N):
            pi = points[i]
            d_x = min(pi[0] + 0.5, 0.5 - pi[0])
            d_y = min(pi[1] + 0.5, 0.5 - pi[1])
            d_b = 2 * min(d_x, d_y)
            if d_b < min_d:
                min_d = d_b
        return min_d

    def relax(points, r, max_iters=500):
        points = points.copy()
        for _ in range(max_iters):
            changed = False
            # Pairwise
            for i in range(N):
                pi = points[i]
                for j in range(i+1, N):
                    pj = points[j]
                    d_vec = pi - pj
                    dist = np.linalg.norm(d_vec)
                    if dist < 2*r:
                        if dist < 1e-10:
                            d_vec = np.random.normal(0, 0.01, 2)
                            dist = np.linalg.norm(d_vec)
                        d_vec /= dist
                        overlap = (2*r - dist) / 2
                        points[i] += d_vec * overlap
                        points[j] -= d_vec * overlap
                        changed = True
            # Boundaries
            for i in range(N):
                pi = points[i]
                if pi[0] < -0.5 + r:
                    points[i][0] = -0.5 + r
                    changed = True
                elif pi[0] > 0.5 - r:
                    points[i][0] = 0.5 - r
                    changed = True
                if pi[1] < -0.5 + r:
                    points[i][1] = -0.5 + r
                    changed = True
                elif pi[1] > 0.5 - r:
                    points[i][1] = 0.5 - r
                    changed = True
            if not changed:
                break
        return points

    best_points = None
    best_r = 0.0

    # Try multiple restarts with different random initializations
    for restart in range(20):
        if time.time() > deadline:
            break

        points = random_init()
        r_curr = min_distance(points)
        
        if best_points is None or r_curr > best_r:
            best_points = points
            best_r = r_curr

        # Optimization loop
        while time.time() < deadline:
            r_target = r_curr * 1.01
            success = False
            for attempt in range(10):
                if time.time() > deadline:
                    break
                relaxed_points = relax(points, r_target)
                r_new = min_distance(relaxed_points)
                
                if r_new >= r_target:
                    points = relaxed_points
                    r_curr = r_new
                    success = True
                    if r_curr > best_r:
                        best_points = points
                        best_r = r_curr
                    break
                else:
                    # Perturb to escape local minimum
                    points = relaxed_points + np.random.normal(0, 0.005, (N, 2))
                    points = np.clip(points, -0.5, 0.5)
            
            if not success:
                # Stuck, break to restart
                break

    return best_points
# EVOLVE-BLOCK-END
