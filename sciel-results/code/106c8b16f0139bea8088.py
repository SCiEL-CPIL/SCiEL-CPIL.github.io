import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3
    
    # Outer hexagon normals
    phi = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(phi), np.sin(phi)))
    
    def get_vertices(x, y, theta):
        angles = np.deg2rad(theta) + np.pi/3 * np.arange(6)
        return np.column_stack((x + np.cos(angles), y + np.sin(angles)))
        
    def calc_required_S(positions, angles):
        max_proj = 0.0
        for i in range(N):
            verts = get_vertices(positions[i][0], positions[i][1], angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        return (2.0 / sqrt3) * max_proj

    def calc_overlap_penalty(positions):
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
        return penalty

    def objective(positions, angles):
        S = calc_required_S(positions, angles)
        pen = calc_overlap_penalty(positions)
        return S + 100.0 * pen  # High penalty weight

    best_S = float('inf')
    best_pos = None
    best_ang = None
    
    # Initialization: Hexagonal Cluster
    # 1 center, 6 at r=sqrt(3), 5 at r=2*sqrt(3)
    init_pos = np.zeros((N, 2))
    init_ang = np.zeros(N)
    
    # Center
    init_pos[0] = [0, 0]
    
    # Ring 1
    r1 = sqrt3
    for i in range(1, 7):
        theta = 2 * np.pi * (i-1) / 6
        init_pos[i] = [r1 * np.cos(theta), r1 * np.sin(theta)]
        
    # Ring 2 (5 hexagons)
    r2 = 2 * sqrt3
    # Place them at angles 0, 60, 120, 180, 240
    for i in range(7, 12):
        theta = 2 * np.pi * (i-7) / 5 # Spread them out? Or specific angles?
        # Better to place them in the gaps of ring 1?
        # Gaps are at 30, 90, 150, 210, 270, 330.
        # Let's place them at 0, 72, 144, 216, 288
        theta = 2 * np.pi * (i-7) / 5
        init_pos[i] = [r2 * np.cos(theta), r2 * np.sin(theta)]
        
    current_pos = init_pos.copy()
    current_ang = init_ang.copy()
    
    # Local Search
    step_size = 0.2
    best_local_obj = objective(current_pos, current_ang)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart > 0:
            # Random restart
            current_pos = init_pos + np.random.normal(0, 1.0, (N, 2))
            current_ang = np.random.uniform(0, 360, N)
            step_size = 0.5
            best_local_obj = objective(current_pos, current_ang)
            
        no_improve = 0
        while time.time() < deadline and no_improve < 100:
            improved = False
            for _ in range(20):
                # Perturb one or two hexagons
                idx = np.random.randint(0, N)
                trial_pos = current_pos.copy()
                trial_ang = current_ang.copy()
                
                trial_pos[idx] += np.random.normal(0, step_size, 2)
                trial_ang[idx] = (trial_ang[idx] + np.random.normal(0, step_size*10)) % 360
                
                trial_obj = objective(trial_pos, trial_ang)
                
                if trial_obj < best_local_obj:
                    current_pos = trial_pos
                    current_ang = trial_ang
                    best_local_obj = trial_obj
                    improved = True
                    no_improve = 0
                    step_size = max(0.01, step_size * 0.95)
                else:
                    no_improve += 1
            
            if not improved:
                no_improve += 10 # Fast forward to restart
                
        # Check if valid
        pen = calc_overlap_penalty(current_pos)
        if pen < 1e-6:
            S_val = calc_required_S(current_pos, current_ang)
            if S_val < best_S:
                best_S = S_val
                best_pos = current_pos.copy()
                best_ang = current_ang.copy()
                
    if best_pos is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
