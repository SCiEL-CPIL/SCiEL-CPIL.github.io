"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    def check_validity(x, w, h, tol=1e-7):
        """Check if configuration is valid. Returns True if valid."""
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9:
                return False
            if xi - ri < -tol or xi + ri > w + tol or yi - ri < -tol or yi + ri > h + tol:
                return False
        # Check disjointness using Delaunay for efficiency
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        
        # Use Delaunay to find close pairs
        points = np.column_stack([xs, ys])
        try:
            tri = Delaunay(points)
            # Get edges from triangulation
            edges = set()
            for simplex in tri.simplices:
                for k in range(3):
                    i, j = simplex[k], simplex[(k+1) % 3]
                    if i < j:
                        edges.add((i, j))
                    else:
                        edges.add((j, i))
        except Exception:
            # Fallback to all pairs
            edges = set()
            for i in range(N):
                for j in range(i+1, N):
                    edges.add((i, j))
        
        for (i, j) in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - tol)**2:
                return False
        return True
    
    def get_violations(x, w, h):
        """Get list of violating pairs and their violation amounts."""
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        violations = []
        
        # Check bounds
        for i in range(N):
            xi, yi, ri = xs[i], ys[i], rs[i]
            if ri < 1e-9:
                violations.append((i, -1, ri - 1e-9))
            if xi - ri < 0:
                violations.append((i, -2, ri - xi))
            if xi + ri > w:
                violations.append((i, -3, xi + ri - w))
            if yi - ri < 0:
                violations.append((i, -4, ri - yi))
            if yi + ri > h:
                violations.append((i, -5, yi + ri - h))
        
        # Check circle-circle using Delaunay
        points = np.column_stack([xs, ys])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for k in range(3):
                    i, j = simplex[k], simplex[(k+1) % 3]
                    if i < j:
                        edges.add((i, j))
                    else:
                        edges.add((j, i))
        except Exception:
            edges = set()
            for i in range(N):
                for j in range(i+1, N):
                    edges.add((i, j))
        
        for (i, j) in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                violations.append((i, j, min_dist - dist))
        
        return violations
    
    def sa_optimize(x0, w, h, max_iter=20000, t0=1.0, t_end=0.001):
        """Simulated annealing optimization."""
        x = x0.copy()
        current_energy = _energy(x, w, h)
        best_x = x.copy()
        best_energy = current_energy
        
        n_iter = max_iter
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            # Temperature schedule
            t = t0 * (t_end / t0) ** (it / n_iter)
            
            # Generate perturbation
            x_new = x.copy()
            
            # Perturb a random circle
            idx = random.randint(0, N-1)
            base = 3 * idx
            
            # Randomly decide what to perturb
            choice = random.random()
            if choice < 0.4:
                # Perturb position
                scale = 0.05 * (t / t0 + 0.1)
                x_new[base] += random.gauss(0, scale)
                x_new[base+1] += random.gauss(0, scale)
            elif choice < 0.8:
                # Perturb radius
                scale = 0.02 * (t / t0 + 0.1)
                x_new[base+2] += random.gauss(0, scale)
            else:
                # Perturb both
                scale = 0.03 * (t / t0 + 0.1)
                x_new[base] += random.gauss(0, scale)
                x_new[base+1] += random.gauss(0, scale)
                x_new[base+2] += random.gauss(0, scale * 0.5)
            
            # Clamp to bounds
            for i in range(N):
                b = 3 * i
                x_new[b] = np.clip(x_new[b], 0, w)
                x_new[b+1] = np.clip(x_new[b+1], 0, h)
                x_new[b+2] = np.clip(x_new[b+2], 1e-4, min(w, h) / 2.0)
            
            new_energy = _energy(x_new, w, h)
            delta_e = new_energy - current_energy
            
            # Accept or reject
            if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-10)):
                x = x_new
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_x = x_new.copy()
        
        return best_x
    
    def _energy(x, w, h):
        """Energy function: sum of violations squared + small term to discourage tiny radii."""
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        
        energy = 0.0
        
        # Boundary violations
        for i in range(N):
            xi, yi, ri = xs[i], ys[i], rs[i]
            if ri < 1e-4:
                energy += (1e-4 - ri)**2
            if xi - ri < 0:
                energy += (ri - xi)**2
            if xi + ri > w:
                energy += (xi + ri - w)**2
            if yi - ri < 0:
                energy += (ri - yi)**2
            if yi + ri > h:
                energy += (yi + ri - h)**2
        
        # Circle-circle violations using Delaunay
        points = np.column_stack([xs, ys])
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for k in range(3):
                    i, j = simplex[k], simplex[(k+1) % 3]
                    if i < j:
                        edges.add((i, j))
                    else:
                        edges.add((j, i))
        except Exception:
            edges = set()
            for i in range(N):
                for j in range(i+1, N):
                    edges.add((i, j))
        
        for (i, j) in edges:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                energy += (min_dist - dist)**2
        
        # Small penalty for very small radii to encourage growth
        energy += 0.01 * np.sum(np.maximum(0, 0.05 - rs)**2)
        
        return energy
    
    def grow_radii(x, w, h, max_iter=500):
        """Try to grow radii while maintaining feasibility."""
        x = x.copy()
        rs = x[2::3]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to grow a random circle's radius
            idx = random.randint(0, N-1)
            old_r = x[3*idx + 2]
            new_r = old_r * 1.01 + 0.001  # 1% growth plus small increment
            
            # Check if new radius fits
            x[3*idx + 2] = new_r
            if not check_validity(x, w, h, tol=1e-6):
                x[3*idx + 2] = old_r
            else:
                # Keep the growth
                pass
        
        return x
    
    def polish(x, w, h):
        """Final polish using local search."""
        x = x.copy()
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Try small moves to reduce energy
            improved = False
            for idx in range(N):
                base = 3 * idx
                old_val = x[base:base+3].copy()
                
                # Try small perturbations
                for delta in [0.001, -0.001, 0.005, -0.005]:
                    x[base] += delta
                    if check_validity(x, w, h, tol=1e-6):
                        # Check if it helps (reduces energy)
                        new_e = _energy(x, w, h)
                        old_e = _energy(old_val, w, h) if False else None  # Simplified
                        improved = True
                        break
                    x[base] = old_val[0]
                
                x[base:base+3] = old_val
            if not improved:
                break
        
        return x
    
    # Main loop over aspect ratios
    widths = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize with grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        
        x0 = np.zeros(3*N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3*i] = (col + 0.5) * w / a
            x0[3*i+1] = (row + 0.5) * h / b
            x0[3*i+2] = min(w/a, h/b) * 0.45
        
        # Run SA
        x_opt = sa_optimize(x0, w, h, max_iter=15000)
        
        # Try to grow radii
        x_opt = grow_radii(x_opt, w, h, max_iter=300)
        
        # Final check and update
        if check_validity(x_opt, w, h):
            val = np.sum(x_opt[2::3])
            if val > best_value:
                best_value = val
                best_result = x_opt.copy()
        
        # Also try a different initialization: hexagonal packing
        if time.time() < deadline:
            # Hexagonal initialization
            x0_hex = np.zeros(3*N)
            r_est = min(w, h) / (2 * np.sqrt(N)) * 0.8
            dx = 2 * r_est * 0.95
            dy = 2 * r_est * 0.85
            count = 0
            row = 0
            while count < N:
                for col in range(int(w / dx) + 1):
                    if count >= N:
                        break
                    x_pos = (col + 0.5) * dx
                    y_pos = (row + 0.5) * dy
                    if x_pos < w and y_pos < h:
                        x0_hex[3*count] = x_pos
                        x0_hex[3*count+1] = y_pos
                        x0_hex[3*count+2] = r_est
                        count += 1
                row += 1
            
            if count == N:
                x_opt2 = sa_optimize(x0_hex, w, h, max_iter=10000)
                x_opt2 = grow_radii(x_opt2, w, h, max_iter=200)
                if check_validity(x_opt2, w, h):
                    val2 = np.sum(x_opt2[2::3])
                    if val2 > best_value:
                        best_value = val2
                        best_result = x_opt2.copy()
    
    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
