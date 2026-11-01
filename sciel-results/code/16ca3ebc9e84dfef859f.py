import numpy as np
import time
import math

N = 150

def _halton_sphere(n):
    """Generate points using Halton sequence projected to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        theta = 2.0 * math.pi * x
        phi = math.acos(2.0 * y - 1.0)
        px = math.sin(phi) * math.cos(theta)
        py = math.sin(phi) * math.sin(theta)
        pz = math.cos(phi)
        points.append([px, py, pz])
    
    pts = np.array(points)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral on sphere."""
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(max(0.0, 1.0 - z*z))
        phi = golden_angle * i
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        points.append([x, y, z])
    pts = np.array(points)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initial configuration: Fibonacci spiral is usually better for spread
    pts = _fibonacci_sphere(N)
    
    def calc_energy(pts):
        """Vectorized log energy computation."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        log_d2 = np.log(dist_sq)
        np.fill_diagonal(log_d2, 0.0)
        energy = -0.25 * np.sum(log_d2)
        return energy

    def gradient(pts):
        """Compute Riemannian gradient of log energy on sphere."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        np.fill_diagonal(dist_sq, 1.0)
        # Euclidean gradient
        forces = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * pts
        return proj_forces

    best_energy = calc_energy(pts)
    best_pts = pts.copy()
    
    # Riemannian Gradient Descent with Line Search
    c1 = 1e-4
    max_line_search = 30
    base_step = 0.01
    
    while time.time() < deadline - 2.0:
        grad = gradient(pts)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            break
            
        v = -grad
        
        alpha = base_step
        current_energy = calc_energy(pts)
        accepted = False
        new_pts = pts
        new_energy = current_energy
        
        for _ in range(max_line_search):
            new_pts = pts + alpha * v
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            new_pts = new_pts / norms
            
            new_energy = calc_energy(new_pts)
            
            directional_derivative = np.sum(grad * v)
            
            if new_energy <= current_energy + c1 * alpha * directional_derivative:
                accepted = True
                break
            
            alpha *= 0.5
            
        if accepted:
            pts = new_pts
            best_pts = new_pts
            best_energy = new_energy
            # Increase step size if we didn't shrink much
            if alpha > base_step * 0.5:
                base_step = min(base_step * 1.05, 0.1)
        else:
            base_step *= 0.5
            if base_step < 1e-8:
                break

    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms

    return best_pts
