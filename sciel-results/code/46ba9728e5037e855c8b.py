import numpy as np
from scipy.special import ellipk, ellipe
import time

N = 26

def construct():
    # Schwarz-Christoffel map from unit disk to unit square
    # f(z) = C * ∫₀^z (1 - t⁴)^(-1/2) dt
    # The constant C is chosen so that f(1) = 1 (maps to corner)
    # f(1) = C * ∫₀¹ (1 - t⁴)^(-1/2) dt = C * (1/4) * B(1/4, 1/2)
    # where B is the beta function
    # B(1/4, 1/2) = Γ(1/4)Γ(1/2)/Γ(3/4)
    
    from scipy.special import gamma
    beta_val = gamma(1/4) * gamma(1/2) / gamma(3/4)
    integral_val = beta_val / 4.0
    C = 1.0 / integral_val  # So that f(1) = 1
    
    def sc_map(z):
        """Map from unit disk to unit square using Schwarz-Christoffel"""
        # z is complex
        # f(z) = C * ∫₀^z (1 - t⁴)^(-1/2) dt
        # We'll use numerical integration
        # For real z in [0,1]: f(z) = C * ∫₀^z (1 - t⁴)^(-1/2) dt
        # For complex z, we need to integrate along a path
        
        # Simplified: use the fact that for our purposes, we can use
        # a series expansion or numerical quadrature
        
        # Actually, for practical purposes, let's use a simpler approximation
        # The map is approximately: f(z) ≈ z + z³/6 + ... near 0
        # But near the boundary, it's more complex
        
        # Let's use a different approach: generate points in the disk,
        # then use the inverse map to place them in the square
        
        # Actually, let's just use the inverse: map from square to disk
        # This is easier for initialization
        
        return z  # Placeholder
    
    # Better approach: Use the inverse map. Given a point in the square,
    # find the corresponding point in the disk.
    # This is also complex.
    
    # Let me use a practical approach:
    # 1. Generate a good packing in the disk
    # 2. Use a simple affine or polynomial map to transform to the square
    # 3. Refine
    
    # Actually, let's use the fact that the Schwarz-Christoffel map has the property
    # that it maps the boundary of the disk to the boundary of the square.
    # We can parametrize the boundary of the disk and map it to the square boundary.
    
    # For interior points, we can use bilinear interpolation or a simple transformation.
    
    # Practical approach: Use the map f(z) = z * (1 + z^2) / (1 + z^2) ... no.
    
    # Let's use a simpler conformal-like map:
    # Map disk to square using: x = Re(z) * (1 - |z|^2/2), y = Im(z) * (1 - |z|^2/2)
    # This isn't conformal but preserves the topology.
    
    # Better: Use the Koebe function or a Möbius transformation.
    
    # I'll use a practical heuristic:
    # 1. Pack 26 circles in a unit disk (radius 0.5 to keep them inside)
    # 2. Scale and shift to fit in the unit square
    # 3. Optimize
    
    # Step 1: Pack circles in a disk of radius R=0.5
    # Use hexagonal packing
    R = 0.5
    disk_points = []
    s = 0.15  # spacing
    h = s * np.sqrt(3) / 2
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if x*x + y*y <= (R - 0.05)**2:  # Keep away from boundary
                disk_points.append((x, y))
    
    # Select 26 points
    if len(disk_points) >= N:
        np.random.seed(42)
        indices = np.random.permutation(len(disk_points))[:N]
        xs_disk = np.array([p[0] for p in disk_points[indices]])
        ys_disk = np.array([p[1] for p in disk_points[indices]])
    else:
        # Fallback: random in disk
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, R*0.8, N)
        xs_disk = radii * np.cos(angles)
        ys_disk = radii * np.sin(angles)
    
    # Step 2: Map from disk to square
    # Use a simple transformation that maps the disk to the square
    # The Schwarz-Christoffel map is complex, so use an approximation:
    # For each point, normalize by the maximum extent
    
    # Simple map: scale so that the disk fits in the square
    # Disk radius R=0.5, square side 1
    # Scale factor: 1.0 / R = 2.0
    # But this maps the disk boundary to the square boundary only at 4 points
    
    # Better: Use the fact that the SC map is approximately linear near the center
    # and has specific behavior at the boundary
    
    # Practical map: x_sq = x_disk / R * 0.5, y_sq = y_disk / R * 0.5
    # This maps the disk to a square of side 1 centered at origin, then shift to [0,1]^2
    
    xs = xs_disk / R * 0.5 + 0.5
    ys = ys_disk / R * 0.5 + 0.5
    
    # Initial radii: small
    rs = np.full(N, 0.03)
    
    # Optimization: Simulated annealing
    def get_score(xs, ys, rs):
        for i in range(N):
            if rs[i] > xs[i] + 1e-9 or rs[i] > 1 - xs[i] + 1e-9 or rs[i] > ys[i] + 1e-9 or rs[i] > 1 - ys[i] + 1e-9:
                return -np.inf
            if rs[i] < 0:
                return -np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return -np.inf
        return np.sum(rs)
    
    current_xs, current_ys, current_rs = xs.copy(), ys.copy(), rs.copy()
    current_score = get_score(current_xs, current_ys, current_rs)
    
    if current_score == -np.inf:
        current_rs *= 0.5
        current_score = get_score(current_xs, current_ys, current_rs)
        if current_score == -np.inf:
            current_rs = np.full(N, 0.01)
            current_score = get_score(current_xs, current_ys, current_rs)
    
    best_xs, best_ys, best_rs = current_xs.copy(), current_ys.copy(), current_rs.copy()
    best_score = current_score
    
    T = 0.03
    T_min = 0.0001
    alpha = 0.997
    steps = 8000
    
    for step in range(steps):
        new_xs = current_xs.copy()
        new_ys = current_ys.copy()
        new_rs = current_rs.copy()
        
        i = np.random.randint(N)
        
        dx = np.random.normal(0, T * 0.08)
        dy = np.random.normal(0, T * 0.08)
        new_xs[i] += dx
        new_ys[i] += dy
        
        dr = np.random.normal(0, T * 0.04)
        new_rs[i] += dr
        new_rs[i] = max(0.001, min(new_rs[i], 0.5))
        
        new_score = get_score(new_xs, new_ys, new_rs)
        
        if new_score > current_score:
            current_xs, current_ys, current_rs = new_xs, new_ys, new_rs
            current_score = new_score
            if current_score > best_score:
                best_xs, best_ys, best_rs = new_xs.copy(), new_ys.copy(), new_rs.copy()
                best_score = current_score
        else:
            if new_score == -np.inf:
                delta = -50
            else:
                delta = new_score - current_score
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_xs, current_ys, current_rs = new_xs, new_ys, new_rs
                current_score = new_score
                if current_score > best_score:
                    best_xs, best_ys, best_rs = new_xs.copy(), new_ys.copy(), new_rs.copy()
                    best_score = current_score
                    
        T *= alpha
        if T < T_min:
            T = T_min
    
    # Final validity check
    final_score = get_score(best_xs, best_ys, best_rs)
    if final_score == -np.inf:
        for _ in range(10):
            best_rs *= 0.9
            if get_score(best_xs, best_ys, best_rs) != -np.inf:
                break
        if get_score(best_xs, best_ys, best_rs) == -np.inf:
            best_rs = np.full(N, 0.01)
    
    result = np.zeros((N, 3))
    result[:, 0] = best_xs
    result[:, 1] = best_ys
    result[:, 2] = best_rs
    
    return result
