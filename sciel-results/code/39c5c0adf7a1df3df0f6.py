import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s
    
    # 1. Initialization: Hexagonal lattice
    def hex_init(n):
        # Create a grid of points
        pts = []
        r_est = 1.0 / np.sqrt(n)
        spacing = 2.0 * r_est
        # Generate enough points
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                pts.append((x, y))
        pts = np.array(pts)
        # Filter to unit disk
        dists = np.linalg.norm(pts, axis=1)
        pts = pts[dists < 0.9]
        # Shuffle to avoid symmetry bias in initial relaxation
        np.random.shuffle(pts)
        # Select N points, preferring those closer to center? Or just random?
        # Random selection from a dense cloud is okay.
        if len(pts) >= N:
            idx = np.random.choice(len(pts), N, replace=False)
            return pts[idx]
        else:
            # Fallback: random uniform
            return np.random.rand(N, 2) * 2 - 1

    pos = hex_init(N)
    best_r = 0.0
    best_pos = pos.copy()
    
    # Start with a small radius
    r = 0.05
    
    # Relaxation function
    def relax(positions, radius, iterations=200, alpha=0.5):
        p = positions.copy()
        for _ in range(iterations):
            displacements = np.zeros_like(p)
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = p[i] - p[j]
                    d = np.linalg.norm(d_vec)
                    if d < 2 * radius and d > 1e-10:
                        # Overlap amount
                        overlap = 2 * radius - d
                        # Unit vector
                        u = d_vec / d
                        # Push apart
                        force = overlap * alpha
                        displacements[i] += u * force
                        displacements[j] -= u * force
                    elif d < 1e-10:
                        # Coincident, push randomly
                        displacements[i] += np.random.rand(2) * 0.1
                        displacements[j] -= np.random.rand(2) * 0.1
            
            # Boundary repulsion
            for i in range(N):
                dist = np.linalg.norm(p[i])
                limit = 1.0 - radius
                if dist > limit and dist > 1e-10:
                    overlap = dist - limit
                    u = p[i] / dist
                    force = overlap * alpha
                    displacements[i] -= u * force
            
            p += displacements
            
            # Optional: Hard clip to prevent flying out (though force should handle it)
            # dists = np.linalg.norm(p, axis=1)
            # mask = dists > 1.0
            # if np.any(mask):
            #     p[mask] /= dists[mask][:, None] * (1.0 + 1e-6)
            
        return p

    def is_valid(positions, radius):
        # Check boundary
        dists = np.linalg.norm(positions, axis=1)
        if np.any(dists > 1.0 - radius + 1e-9):
            return False
        # Check pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2 * radius - 1e-9:
                    return False
        return True

    # Main loop
    while time.time() - start_time < time_limit:
        # Try to increase radius
        r_new = r + 0.0005
        
        # Relax at new radius
        pos_new = relax(pos, r_new, iterations=100)
        
        # Check validity
        if is_valid(pos_new, r_new):
            pos = pos_new
            r = r_new
            best_r = r
            best_pos = pos.copy()
        else:
            # Try to recover by relaxing at current r with more iterations or perturbation
            # If we can't even maintain current r, we are stuck.
            # Try a perturbation
            pos_perturbed = pos + np.random.randn(N, 2) * 0.01
            pos_recovered = relax(pos_perturbed, r, iterations=200)
            if is_valid(pos_recovered, r):
                pos = pos_recovered
            else:
                # If we can't maintain current r, break or reduce r?
                # Usually, if we can't maintain r, we shouldn't have accepted it.
                # But numerical errors might creep in.
                # Let's try one more time with stronger relaxation
                pos_strong = relax(pos, r, iterations=500, alpha=1.0)
                if is_valid(pos_strong, r):
                    pos = pos_strong
                else:
                    # Stuck, try to decrease r slightly to escape? No, we want max.
                    # Break to avoid wasting time
                    break
            
            # If we failed to increase, we can try a few more times with perturbation
            # But to save time, we might just break if we fail multiple times in a row
            # For simplicity, we break on first failure to increase, assuming we are near local max.
            # To improve, we could implement a "cooling" of step size.
            break

    # Final polish: Try to maximize r further with smaller steps
    # The loop above breaks on first failure. We can add a secondary loop for fine-tuning.
    # But the first loop already increments by 0.0005.
    # Let's do a few more fine steps if time permits.
    while time.time() - start_time < time_limit - 5.0:
        r_new = r + 0.0001
        pos_new = relax(pos, r_new, iterations=50)
        if is_valid(pos_new, r_new):
            pos = pos_new
            r = r_new
            best_r = r
            best_pos = pos.copy()
        else:
            break

    return best_pos
