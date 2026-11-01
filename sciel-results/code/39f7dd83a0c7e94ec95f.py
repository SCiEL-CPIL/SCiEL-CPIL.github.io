import numpy as np
import time
import math

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0 # Safety margin

def get_hex_vertices(center, angle):
    x, y = center
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        verts.append((x + math.cos(a), y + math.sin(a)))
    return np.array(verts)

def check_overlap_penalty(verts_list):
    """
    Returns the minimum overlap depth between any pair of hexagons.
    0.0 if no overlaps.
    """
    min_overlap = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            separated = False
            local_min_overlap = float('inf')
            
            # Check axes from v1
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9: continue
                normal /= norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                else:
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < local_min_overlap:
                        local_min_overlap = overlap
            
            if separated:
                continue
                
            # Check axes from v2
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9: continue
                normal /= norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                else:
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < local_min_overlap:
                        local_min_overlap = overlap
            
            if not separated:
                if local_min_overlap < min_overlap:
                    min_overlap = local_min_overlap
                    
    if min_overlap == float('inf'):
        return 0.0
    return min_overlap

def compute_L(positions, angles):
    max_proj = 0.0
    normals = []
    for i in range(6):
        a = math.pi / 6 + i * math.pi / 3
        normals.append(np.array([math.cos(a), math.sin(a)]))
        
    for i in range(N):
        x, y = positions[i]
        theta = angles[i]
        for j in range(6):
            va = theta + j * math.pi / 3
            vx = x + math.cos(va)
            vy = y + math.sin(va)
            v = np.array([vx, vy])
            for n in normals:
                proj = np.dot(v, n)
                if abs(proj) > max_proj:
                    max_proj = abs(proj)
    return 2.0 * max_proj / math.sqrt(3)

def construct():
    # Initial configuration: Flower of 7 + 5 extra
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    for i in range(6):
        a = i * math.pi / 3
        positions.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
        angles.append(0.0)
        
    for i in range(5):
        a = (i + 0.5) * 2 * math.pi / 5
        # Place them further out to avoid initial overlap
        positions.append((2.5 * math.sqrt(3) * math.cos(a), 2.5 * math.sqrt(3) * math.sin(a)))
        angles.append(0.0)
        
    while len(positions) < N:
        positions.append((0.0, 0.0))
        angles.append(0.0)
    while len(positions) > N:
        positions.pop()
        angles.pop()
        
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    
    best_L = float('inf')
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    T = 0.5
    cooling = 0.9995
    start_time = time.time()
    
    # Pre-calculate initial penalty to ensure we start valid or close
    initial_verts = [get_hex_vertices(positions[i], angles[i]) for i in range(N)]
    initial_penalty = check_overlap_penalty(initial_verts)
    initial_L = compute_L(positions, angles)
    initial_cost = initial_L + 10.0 * initial_penalty
    best_L = initial_L if initial_penalty < 1e-6 else float('inf')
    if initial_penalty < 1e-6:
        best_positions = positions.copy()
        best_angles = angles.copy()
    
    while time.time() - start_time < TIME_LIMIT:
        idx = np.random.randint(N)
        
        new_positions = positions.copy()
        new_angles = angles.copy()
        
        # Move
        new_positions[idx, 0] += np.random.normal(0, 0.1)
        new_positions[idx, 1] += np.random.normal(0, 0.1)
        new_angles[idx] += np.random.normal(0, 0.1)
        
        # Calculate new L
        new_L = compute_L(new_positions, new_angles)
        
        # Calculate new penalty
        # Optimization: Only recompute vertices for idx?
        # No, we need all for overlap check.
        # But we can reuse the list and replace idx.
        # For simplicity, recompute all.
        new_verts = [get_hex_vertices(new_positions[i], new_angles[i]) for i in range(N)]
        new_penalty = check_overlap_penalty(new_verts)
        
        new_cost = new_L + 10.0 * new_penalty
        
        # Update best
        if new_penalty < 1e-6 and new_L < best_L:
            best_L = new_L
            best_positions = new_positions.copy()
            best_angles = new_angles.copy()
            
        # Acceptance
        # If we are currently invalid, we strongly prefer moves that reduce penalty.
        # If valid, we prefer moves that reduce L.
        
        curr_penalty = check_overlap_penalty([get_hex_vertices(positions[i], angles[i]) for i in range(N)])
        curr_L = compute_L(positions, angles)
        curr_cost = curr_L + 10.0 * curr_penalty
        
        if new_cost < curr_cost:
            positions = new_positions
            angles = new_angles
        else:
            delta = new_cost - curr_cost
            if np.random.rand() < math.exp(-delta / T):
                positions = new_positions
                angles = new_angles
                
        T *= cooling
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 2:
            break
            
    # Return best valid configuration
    # If best_L is inf, it means no valid config was found.
    # We should return the best config we have (which might be invalid).
    # But the evaluator scores 0 for invalid.
    # So we return the best valid one if found.
    
    return best_positions.tolist(), (0.0, 0.0), best_L, 0.0
