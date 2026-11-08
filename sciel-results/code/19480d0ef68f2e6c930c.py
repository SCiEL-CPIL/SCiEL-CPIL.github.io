import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        return x, y
    
    # Generate honeycomb positions up to distance 2
    positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            d = (abs(q) + abs(r) + abs(q + r)) // 2
            if d <= 2:
                positions.append((q, r, d))
    
    positions.sort(key=lambda x: (x[2], x[0], x[1]))
    
    selected = positions[:11]
    inner_positions = [axial_to_cart(q, r) for q, r, _ in selected]
    
    def compute_required_side(points, theta, cx=0.0, cy=0.0):
        inradius_factor = np.sqrt(3) / 2
        max_proj = 0.0
        for i in range(6):
            angle = theta + i * np.pi / 3
            nx, ny = np.cos(angle), np.sin(angle)
            for px, py in points:
                proj = (px - cx) * nx + (py - cy) * ny
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
        return max_proj / inradius_factor
    
    best_theta = 0.0
    best_side = float('inf')
    
    # Coarse search
    for theta in np.linspace(0, np.pi/3, 300):
        side = compute_required_side(inner_positions, theta)
        if side < best_side:
            best_side = side
            best_theta = theta
    
    # Fine search
    if time.time() < deadline - 15:
        for theta in np.linspace(best_theta - 0.003, best_theta + 0.003, 1000):
            side = compute_required_side(inner_positions, theta)
            if side < best_side:
                best_side = side
                best_theta = theta
    
    # Optimize center
    if time.time() < deadline - 10:
        best_cx, best_cy = 0.0, 0.0
        for dx in np.linspace(-0.3, 0.3, 41):
            for dy in np.linspace(-0.3, 0.3, 41):
                side = compute_required_side(inner_positions, best_theta, dx, dy)
                if side < best_side:
                    best_side = side
                    best_cx, best_cy = dx, dy
    
    inner = [[x, y, 0.0] for x, y in inner_positions]
    
    return inner, (best_cx, best_cy), best_side, best_theta * 180 / np.pi
