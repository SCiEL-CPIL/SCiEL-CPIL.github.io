import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    def calc_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(np.log(1.0 / dist[mask]))
        return energy

    def optimize(pts_init, max_time):
        pts = pts_init.copy()
        energy = calc_energy(pts)
        best_energy = energy
        best_pts = pts.copy()
        
        # Adam parameters
        lr0 = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        t = 0
        stagnation = 0
        diverged = False

        while time.time() < max_time:
            # Compute gradient
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq = np.maximum(dist_sq, 1e-12)
            
            # Force F_i = sum_j (x_i - x_j) / dist_sq_ij
            forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
            
            # Project to tangent space
            dot_products = np.sum(forces * pts, axis=1)
            grad = forces - dot_products[:, np.newaxis] * pts
            
            # Adam update
            t += 1
            # Decay learning rate
            lr = lr0 * (0.999 ** t)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            update = m_hat / (np.sqrt(v_hat) + eps)
            
            new_pts = pts - lr * update
            norms = np.sqrt(np.sum(new_pts**2, axis=1))
            new_pts = new_pts / norms[:, np.newaxis]
            
            new_energy = calc_energy(new_energy if False else new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                pts = new_pts
                stagnation = 0
            else:
                stagnation += 1
                # If energy gets significantly worse, revert to best and reset optimizer
                if new_energy > best_energy * 1.1 or stagnation > 200:
                    pts = best_pts.copy()
                    m = np.zeros_like(pts)
                    v = np.zeros_like(pts)
                    t = 0
                    stagnation = 0
                    # Reduce learning rate base if we keep diverging
                    # lr0 *= 0.9 # Optional, might be too aggressive
                    if lr0 < 1e-8:
                        break

        return best_pts, best_energy

    # Strategy: Multiple restarts to escape local minima
    best_overall_energy = float('inf')
    best_overall_pts = None

    # 1. Fibonacci Spiral (Good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    t_start = time.time()
    # Give the first run the majority of the time
    limit1 = 70.0
    pts1, e1 = optimize(fib_pts, t_start + limit1)
    if e1 < best_overall_energy:
        best_overall_energy = e1
        best_overall_pts = pts1

    # 2. Random Start
    if time.time() < deadline - 10:
        np.random.seed(42)
        rand_pts = np.random.randn(N, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        t_start = time.time()
        limit2 = 40.0
        pts2, e2 = optimize(rand_pts, t_start + limit2)
        if e2 < best_overall_energy:
            best_overall_energy = e2
            best_overall_pts = pts2

    # 3. Another Random Start
    if time.time() < deadline - 5:
        np.random.seed(123)
        rand_pts3 = np.random.randn(N, 3)
        rand_pts3 /= np.linalg.norm(rand_pts3, axis=1, keepdims=True)
        t_start = time.time()
        limit3 = 20.0
        pts3, e3 = optimize(rand_pts3, t_start + limit3)
        if e3 < best_overall_energy:
            best_overall_energy = e3
            best_overall_pts = pts3

    if best_overall_pts is None:
        # Fallback to Fibonacci if something went wrong
        return fib_pts

    return best_overall_pts
