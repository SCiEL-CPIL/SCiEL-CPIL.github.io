import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 120.0 # 120s limit, keep margin
    
    # Hexagon vertices for unit hexagon, flat-top orientation
    # Side length 1
    v_base = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    # Directions and constants for outer hexagon containment check
    # Outer hexagon: Flat top, side S, centered at (0,0)
    # Inequalities:
    # 1. x <= S          -> n=(1,0), k=1
    # 2. -x <= S         -> n=(-1,0), k=1
    # 3. y <= S*sqrt(3)/2-> n=(0,1), k=sqrt(3)/2
    # 4. -y <= S*sqrt(3)/2-> n=(0,-1), k=sqrt(3)/2
    # 5. sqrt(3)x + y <= S*sqrt(3) -> n=(sqrt(3),1), k=sqrt(3)
    # 6. -sqrt(3)x + y <= S*sqrt(3) -> n=(-sqrt(3),1), k=sqrt(3)
    # 7. -sqrt(3)x - y <= S*sqrt(3) -> n=(-sqrt(3),-1), k=sqrt(3)
    # 8. sqrt(3)x - y <= S*sqrt(3)  -> n=(sqrt(3),-1), k=sqrt(3)
    # Wait, I listed 8 again. Let's stick to the 6 unique sides derived earlier.
    # Side 1 (Right): x <= S. n=(1,0), k=1
    # Side 2 (Top-Right): sqrt(3)x + y <= S*sqrt(3). n=(sqrt(3),1), k=sqrt(3)
    # Side 3 (Top): y <= S*sqrt(3)/2. n=(0,1), k=sqrt(3)/2
    # Side 4 (Top-Left): -sqrt(3)x + y <= S*sqrt(3). n=(-sqrt(3),1), k=sqrt(3)
    # Side 5 (Left): -x <= S. n=(-1,0), k=1
    # Side 6 (Bottom-Left): -sqrt(3)x - y <= S*sqrt(3). n=(-sqrt(3),-1), k=sqrt(3)
    # Side 7 (Bottom): -y <= S*sqrt(3)/2. n=(0,-1), k=sqrt(3)/2
    # Side 8 (Bottom-Right): sqrt(3)x - y <= S*sqrt(3). n=(sqrt(3),-1), k=sqrt(3)
    # 
    # My previous count of 6 was wrong. A hexagon has 6 sides.
    # Let's re-verify the vertices and sides.
    # V0: (S, 0)
    # V1: (S/2, S*sqrt(3)/2)
    # V2: (-S/2, S*sqrt(3)/2)
    # V3: (-S, 0)
    # V4: (-S/2, -S*sqrt(3)/2)
    # V5: (S/2, -S*sqrt(3)/2)
    #
    # Side V0-V1: Line through (S,0) and (S/2, S*sqrt(3)/2).
    #   Slope: (S*sqrt(3)/2) / (-S/2) = -sqrt(3).
    #   Eq: y - 0 = -sqrt(3)(x - S) => sqrt(3)x + y = S*sqrt(3).
    #   Normal: (sqrt(3), 1). Const: S*sqrt(3).
    #
    # Side V1-V2: Line through (S/2, S*sqrt(3)/2) and (-S/2, S*sqrt(3)/2).
    #   Horizontal. y = S*sqrt(3)/2.
    #   Normal: (0, 1). Const: S*sqrt(3)/2.
    #
    # Side V2-V3: Line through (-S/2, S*sqrt(3)/2) and (-S, 0).
    #   Slope: (-S*sqrt(3)/2) / (-S/2) = sqrt(3).
    #   Eq: y - 0 = sqrt(3)(x + S) => -sqrt(3)x + y = S*sqrt(3).
    #   Normal: (-sqrt(3), 1). Const: S*sqrt(3).
    #
    # Side V3-V4: Line through (-S, 0) and (-S/2, -S*sqrt(3)/2).
    #   Slope: (-S*sqrt(3)/2) / (S/2) = -sqrt(3).
    #   Eq: y - 0 = -sqrt(3)(x + S) => sqrt(3)x + y = -S*sqrt(3).
    #   Wait. Interior is "above" this line?
    #   Center (0,0) gives 0. -S*sqrt(3) is negative.
    #   So inequality is: -sqrt(3)x - y <= S*sqrt(3)?
    #   Let's check normal. Vector from V3 to V4 is (S/2, -S*sqrt(3)/2).
    #   Outward normal is (sqrt(3), 1)? No.
    #   Let's use the property that the 6 normals are at 0, 60, 120, 180, 240, 300 degrees?
    #   No, for flat-top, normals are at 0, 60... ?
    #   Side V0-V1 normal is at angle 30 deg?
    #   Vector (sqrt(3), 1) has angle arctan(1/sqrt(3)) = 30 deg.
    #   Side V1-V2 normal is at 90 deg. (0,1).
    #   Side V2-V3 normal is at 150 deg. (-sqrt(3), 1).
    #   Side V3-V4 normal is at 210 deg. (-sqrt(3), -1).
    #   Side V4-V5 normal is at 270 deg. (0, -1).
    #   Side V5-V0 normal is at 330 deg. (sqrt(3), -1).
    #
    # So the 6 normals are:
    # n0 = (1, 0) ? No, side V0 is a vertex, not a side.
    # The sides are between vertices.
    # My normals:
    # 1. (sqrt(3), 1) / 2 -> angle 30. Const S*sqrt(3).
    # 2. (0, 1) -> angle 90. Const S*sqrt(3)/2.
    # 3. (-sqrt(3), 1) / 2 -> angle 150. Const S*sqrt(3).
    # 4. (-sqrt(3), -1) / 2 -> angle 210. Const S*sqrt(3).
    # 5. (0, -1) -> angle 270. Const S*sqrt(3)/2.
    # 6. (sqrt(3), -1) / 2 -> angle 330. Const S*sqrt(3).
    
    # Let's normalize normals to unit length for cleaner math, or just use the raw vectors and constants.
    # Using raw vectors:
    # d0 = v . (sqrt(3), 1) <= S*sqrt(3)
    # d1 = v . (0, 1) <= S*sqrt(3)/2
    # d2 = v . (-sqrt(3), 1) <= S*sqrt(3)
    # d3 = v . (-sqrt(3), -1) <= S*sqrt(3)
    # d4 = v . (0, -1) <= S*sqrt(3)/2
    # d5 = v . (sqrt(3), -1) <= S*sqrt(3)
    
    # S >= max(
    #   max(d0)/sqrt(3),
    #   max(d1)/(sqrt(3)/2),
    #   max(d2)/sqrt(3),
    #   max(d3)/sqrt(3),
    #   max(d4)/(sqrt(3)/2),
    #   max(d5)/sqrt(3)
    # )

    norms = np.array([
        [math.sqrt(3), 1],
        [0, 1],
        [-math.sqrt(3), 1],
        [-math.sqrt(3), -1],
        [0, -1],
        [math.sqrt(3), -1]
    ])
    consts = np.array([
        math.sqrt(3),
        math.sqrt(3)/2,
        math.sqrt(3),
        math.sqrt(3),
        math.sqrt(3)/2,
        math.sqrt(3)
    ])

    def get_vertices(center, angle=0.0):
        # Rotate v_base by angle
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return v_base @ R.T + center

    def calc_S(positions, angles):
        # positions: (N, 2)
        # angles: (N,)
        max_projs = np.zeros(6)
        for i in range(N):
            verts = get_vertices(positions[i], angles[i])
            # Project onto normals
            projs = verts @ norms.T # (6,)
            max_projs = np.maximum(max_projs, projs)
        
        S_req = np.max(max_projs / consts)
        return S_req

    def calc_overlap_penalty(positions, angles):
        # Simplified: Assume aligned hexagons for speed in SA.
        # If angles are not 0, this is an approximation.
        # Penalty for dist < sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                # If we allow rotation, min dist varies.
                # For aligned, min dist is sqrt(3).
                # If we assume angles are 0, this is exact.
                if d < math.sqrt(3):
                    penalty += (math.sqrt(3) - d)**2
        return penalty

    # Initial Setup
    # 7 hexagons in center
    pos = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Center
    pos[0] = [0.0, 0.0]
    
    # 6 neighbors
    for i in range(1, 7):
        ang = i * math.pi / 3
        pos[i] = [math.cos(ang) * math.sqrt(3), math.sin(ang) * math.sqrt(3)]
        
    # 4 extra: Place them in a second ring, e.g., at distance 3*sqrt(3)/2?
    # Or just randomize near the cluster
    for i in range(7, N):
        pos[i] = np.random.uniform(-2, 2, 2)
        
    # Optimize
    best_pos = pos.copy()
    best_angles = angles.copy()
    best_S = calc_S(pos, angles)
    best_score = best_S + 100.0 * calc_overlap_penalty(pos, angles)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.99
    
    while time.time() - start_time < timeout - 5.0:
        if T < T_min:
            break
            
        # Propose move
        idx = np.random.randint(N)
        new_pos = pos.copy()
        new_angles = angles.copy()
        
        new_pos[idx] += np.random.normal(0, 0.5, 2)
        # Optionally perturb angle
        # new_angles[idx] += np.random.normal(0, 0.1)
        
        S_new = calc_S(new_pos, new_angles)
        P_new = calc_overlap_penalty(new_pos, new_angles)
        score_new = S_new + 100.0 * P_new
        
        delta = score_new - best_score
        if delta < 0 or np.random.rand() < math.exp(-delta / T):
            pos = new_pos
            angles = new_angles
            best_score = score_new
            if P_new == 0: # Only update best_S if valid? Or just track min S for valid
                if S_new < best_S:
                    best_S = S_new
                    best_pos = pos.copy()
                    best_angles = angles.copy()
        T *= cooling

    # Final cleanup: ensure no overlaps in best_pos
    # If best_pos has overlaps, we might need to run a few more steps or just return it if S is good enough
    # The evaluator checks validity. If invalid, score 0.
    # We must ensure the returned config is VALID.
    
    # Check validity of best_pos
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < math.sqrt(3) - 1e-9:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: Return the initial 7-hex cluster + 4 placed far away?
        # Or just return the best found even if slightly invalid? No, score 0.
        # Let's try to fix overlaps by pushing apart
        # Simple iterative push
        for _ in range(100):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = best_pos[i] - best_pos[j]
                    d = np.linalg.norm(d_vec)
                    if d < math.sqrt(3):
                        d_vec = d_vec / (d + 1e-9)
                        push = (math.sqrt(3) - d) / 2
                        best_pos[i] += d_vec * push
                        best_pos[j] -= d_vec * push
                        moved = True
            if not moved: break
        
        # Re-check
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < math.sqrt(3) - 1e-9:
                    valid = False
                    break
                if not valid: break
        
        if not valid:
            # Last resort: Linear arrangement
            best_pos = np.array([[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
            best_angles = np.zeros(N)
            best_S = calc_S(best_pos, best_angles)

    # Calculate final S for the returned configuration
    final_S = calc_S(best_pos, best_angles)
    
    # Construct output
    inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(best_pos, best_angles)]
    return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
