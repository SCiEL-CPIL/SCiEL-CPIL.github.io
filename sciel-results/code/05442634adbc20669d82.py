"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    sqrt3 = np.sqrt(3.0)
    
    def make_honeycomb(row_counts):
        """Generate center positions for honeycomb packing."""
        centers = []
        for i, count in enumerate(row_counts):
            y = i * 1.5
            x_offset = (i % 2) * (sqrt3 / 2.0)
            for j in range(count):
                x = (j - (count - 1) / 2.0) * sqrt3 + x_offset
                centers.append((x, y))
        return np.array(centers)
    
    def min_enclosing_hex_side(centers, angle_rad):
        """Compute minimum side length of enclosing regular hexagon with given rotation, centered at origin."""
        max_proj = 0.0
        for k in range(6):
            theta = angle_rad + k * np.pi / 3.0
            nx = np.cos(theta)
            ny = np.sin(theta)
            proj = centers[:, 0] * nx + centers[:, 1] * ny
            max_proj = max(max_proj, float(np.max(proj)))
        return max_proj * 2.0 / sqrt3
    
    # Try different honeycomb arrangements
    arrangements = [
        [2, 3, 3, 2, 1],
        [3, 4, 3, 1],
        [1, 3, 4, 3],
        [2, 4, 3, 2],
        [3, 3, 3, 2],
        [2, 3, 4, 2],
        [1, 2, 4, 3, 1],
        [2, 2, 3, 3, 1],
        [3, 2, 3, 3],
        [1, 3, 3, 3, 1],
        [4, 3, 3, 1],
        [1, 3, 3, 4],
        [3, 3, 4, 1],
        [1, 4, 3, 3],
        [2, 3, 3, 3],
        [3, 3, 3, 2],
    ]
    
    best_side = float('inf')
    best_result = None
    
    for arr in arrangements:
        if sum(arr) != N or len(arr) > 6:
            continue
        
        centers = make_honeycomb(arr)
        
        # Try different rotation angles for the outer hexagon
        for angle_deg in range(0, 60, 3):  # 20 angles
            angle = np.radians(angle_deg)
            side = min_enclosing_hex_side(centers, angle)
            if side < best_side:
                best_side = side
                best_result = (centers, angle_deg)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, best_angle = best_result
    
    inner = []
    for cx, cy in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    return inner, (0.0, 0.0), float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
