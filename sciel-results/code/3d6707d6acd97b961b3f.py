import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def generate_hex_lattice(spacing):
        points = []
        max_coord = 1.5
        i = int(-max_coord / spacing)
        while i * spacing < max_coord:
            j = int(-max_coord / spacing)
            while j * spacing < max_coord:
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                points.append([x, y])
                j += 1
            i += 1
        return np.array(points)

    points = None
    for spacing in [0.1, 0.12, 0.15, 0.2]:
        pts = generate_hex_lattice(spacing)
        radii = np.linalg.norm(pts, axis=1)
        pts = pts[radii < 1.0]
        if len(pts) >= N:
            points = pts
            break
            
    if points is None:
        points = np.random.rand(N, 2) * 2 - 1
        radii = np.linalg.norm(points, axis=1)
        mask = radii < 1.0
        while np.sum(mask) < N:
            new_pts = np.random.rand(N, 2) * 2 - 1
            new_radii = np.linalg.norm(new_pts, axis=1)
            new_mask = new_radii < 1.0
            points = np.vstack([points, new_pts[new_mask]])
            mask = np.linalg.norm(points, axis=1) < 1.0
        points = points[mask][:N]
        
    radii = np.linalg.norm(points, axis=1)
    indices = np.argsort(radii)[:N]
    x0 = points[indices].copy()

    def calc_score(x):
        x = x.reshape(N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        radii = np.linalg.norm(x, axis=1)
        max_radius = np.max(radii)
        return min(min_dist / 2.0, 1.0 - max_radius)

    def local_search(x, steps=2000, step_size=0.005):
        x = x.copy()
        for step in range(steps):
            if time.time() > deadline:
                break
                
            forces = np.zeros_like(x)
            
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            dists[dists == 0] = 1e-6
            
            forces_matrix = diff / dists[:, :, np.newaxis]**3
            np.fill_diagonal(forces_matrix[:, :, 0], 0) # This doesn't work for 3D array diagonal easily
            # Better:
            forces_matrix[np.eye(N, dtype=bool)] = 0
            
            forces += np.sum(forces_matrix, axis=1)
            
            radii = np.linalg.norm(x, axis=1)
            # Avoid division by zero
            radii[radii == 0] = 1e-6
            excess = np.maximum(0, radii - (1.0 - 0.05))
            forces -= 10.0 * excess[:, np.newaxis] * (x / radii[:, np.newaxis])
            
            x += step_size * forces
            
            radii = np.linalg.norm(x, axis=1)
            for i in range(N):
                if radii[i] > 1.0:
                    x[i] = x[i] / radii[i]
                    
        return x

    x_opt = local_search(x0, steps=2000, step_size=0.005)
    best_score = calc_score(x_opt)

    for restart in range(5):
        if time.time() > deadline - 10:
            break
        x_pert = x0 + np.random.randn(N, 2) * 0.01
        radii = np.linalg.norm(x_pert, axis=1)
        for i in range(N):
            if radii[i] > 1.0:
                x_pert[i] = x_pert[i] / radii[i]
                
        x_try = local_search(x_pert, steps=1000, step_size=0.005)
        score_try = calc_score(x_try)
        
        if score_try > best_score:
            best_score = score_try
            x_opt = x_try
            
    return x_opt.reshape(N, 2)
