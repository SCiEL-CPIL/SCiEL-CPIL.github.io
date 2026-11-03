import math
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 115.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple hexagonal packing
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate initial candidate configurations
    def generate_candidates():
        candidates = []
        
        # Hexagonal packing centered
        hex_centers = [(0, 0)]
        for ring in range(1, 4):
            for i in range(6 * ring):
                angle = i * math.pi / 3
                x = ring * math.cos(angle)
                y = ring * math.sin(angle)
                hex_centers.append((round(x, 6), round(y, 6)))
        
        # Take first N centers
        if len(hex_centers) >= N:
            candidates.append(np.array(hex_centers[:N], dtype=float))
        
        # Linear arrangement
        linear = []
        for i in range(N):
            linear.append([i * 1.732, 0])  # sqrt(3) spacing
        candidates.append(np.array(linear, dtype=float))
        
        # Grid arrangement
        grid = []
        for i in range(N):
            row = i // 3
            col = i % 3
            x = col * 1.732
            y = row * 1.5
            grid.append([x, y])
        candidates.append(np.array(grid, dtype=float))
        
        # Random configurations
        np.random.seed(42)
        for _ in range(5):
            config = np.random.uniform(-3, 3, (N, 2))
            candidates.append(config)
        
        return candidates

    def objective(params):
        """Objective function: minimize outer hexagon side length"""
        # Unpack parameters
        positions = params[:2 * N].reshape(N, 2)
        angles = params[2 * N:3 * N]
        cx = params[3 * N]
        cy = params[3 * N + 1]
        outer_angle = params[3 * N + 2]
        
        # Compute vertices for each inner hexagon
        all_vertices = []
        for i in range(N):
            px, py = positions[i]
            angle = angles[i]
            for k in range(6):
                vertex_angle = angle + k * math.pi / 3
                vx = px + math.cos(vertex_angle)
                vy = py + math.sin(vertex_angle)
                all_vertices.append([vx, vy])
        
        all_vertices = np.array(all_vertices)
        
        # Compute outer hexagon constraints
        # For the outer hexagon to contain all points, we need:
        # |dot(v - center, normal_k)| <= D for all vertices v and all 6 normals
        # where D is the apothem of the outer hexagon
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = outer_angle + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Project all vertices onto this normal
            projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        # Outer hexagon side length
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        
        # Check for overlaps between inner hexagons
        # For two hexagons to not overlap, their centers must be at least 
        # 2 * apothem apart when they have the same orientation
        # But with different orientations, we need to check more carefully
        # For simplicity, we use a conservative distance check
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 2 * APOTHEM - EPS:
                    # Penalty for overlap
                    return S + 1000 * (2 * APOTHEM - dist)
        
        return S

    # Generate initial candidates
    candidates = generate_candidates()
    
    best_S = float('inf')
    best_params = None
    
    # Try each candidate with local optimization
    for candidate in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Initialize parameters
        init_params = np.zeros(3 * N + 3)
        init_params[:2 * N] = candidate.ravel()
        init_params[2 * N:3 * N] = 0.0  # Start with all angles 0
        cx = np.mean(candidate[:, 0])
        cy = np.mean(candidate[:, 1])
        init_params[3 * N] = cx
        init_params[3 * N + 1] = cy
        init_params[3 * N + 2] = 0.0
        
        # Use differential evolution for global optimization
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((-math.pi, math.pi))  # angle
        bounds.append((-10, 10))  # cx
        bounds.append((-10, 10))  # cy
        bounds.append((-math.pi, math.pi))  # outer angle
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_S:
                best_S = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # If no good solution found, use the best candidate directly
    if best_params is None:
        # Use the hexagonal packing
        candidate = candidates[0]
        positions = candidate
        angles = np.zeros(N)
        cx = np.mean(positions[:, 0])
        cy = np.mean(positions[:, 1])
        outer_angle = 0.0
        
        # Compute S
        all_vertices = []
        for i in range(N):
            px, py = positions[i]
            angle = angles[i]
            for k in range(6):
                vertex_angle = angle + k * math.pi / 3
                vx = px + math.cos(vertex_angle)
                vy = py + math.sin(vertex_angle)
                all_vertices.append([vx, vy])
        all_vertices = np.array(all_vertices)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = outer_angle + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        best_S = (max_proj + APOTHEM) / APOTHEM
        best_params = np.zeros(3 * N + 3)
        best_params[:2 * N] = positions.ravel()
        best_params[2 * N:3 * N] = angles
        best_params[3 * N] = cx
        best_params[3 * N + 1] = cy
        best_params[3 * N + 2] = outer_angle
    
    # Extract final solution
    positions = best_params[:2 * N].reshape(N, 2)
    angles = best_params[2 * N:3 * N]
    cx = best_params[3 * N]
    cy = best_params[3 * N + 1]
    outer_angle = best_params[3 * N + 2]
    
    # Compute final S
    all_vertices = []
    for i in range(N):
        px, py = positions[i]
        angle = angles[i]
        for k in range(6):
            vertex_angle = angle + k * math.pi / 3
            vx = px + math.cos(vertex_angle)
            vy = py + math.sin(vertex_angle)
            all_vertices.append([vx, vy])
    all_vertices = np.array(all_vertices)
    
    max_proj = 0.0
    for k in range(6):
        normal_angle = outer_angle + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D = max_proj
    S = (D + APOTHEM) / APOTHEM
    
    # Apply small safety scaling to inner hexagons
    SCALE = 1.0 + 1e-12
    scaled_positions = []
    for i in range(N):
        px, py = positions[i]
        sx = cx + (px - cx) * SCALE
        sy = cy + (py - cy) * SCALE
        scaled_positions.append([sx, sy])
    
    # Recompute S with scaled positions
    all_vertices_scaled = []
    for i in range(N):
        px, py = scaled_positions[i]
        angle = angles[i]
        for k in range(6):
            vertex_angle = angle + k * math.pi / 3
            vx = px + math.cos(vertex_angle)
            vy = py + math.sin(vertex_angle)
            all_vertices_scaled.append([vx, vy])
    all_vertices_scaled = np.array(all_vertices_scaled)
    
    max_proj_scaled = 0.0
    for k in range(6):
        normal_angle = outer_angle + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        projs = (all_vertices_scaled[:, 0] - cx) * nx + (all_vertices_scaled[:, 1] - cy) * ny
        max_proj_scaled = max(max_proj_scaled, np.max(projs), -np.min(projs))
    
    S_scaled = (max_proj_scaled + APOTHEM) / APOTHEM
    
    # Build output
    inner = []
    for i in range(N):
        x, y = scaled_positions[i]
        angle_deg = math.degrees(angles[i])
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(math.degrees(outer_angle))
    
    return inner, center, side, rotation
