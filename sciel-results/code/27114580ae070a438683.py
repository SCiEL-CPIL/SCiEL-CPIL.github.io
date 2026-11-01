import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    # Normals for the outer hexagon (flat top)
    normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    
    # Vertices of a unit hexagon centered at origin, rotation 0
    verts_offset = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def calc_required_side(positions):
        # positions: (N, 2)
        # all_verts: (N*6, 2)
        all_verts = positions[:, np.newaxis, :] + verts_offset[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 2)
        
        max_proj = 0.0
        for n in normals:
            projs = all_verts @ n
            max_proj = max(max_proj, np.max(projs))
            
        return max_proj * 2.0 / np.sqrt(3.0)
        
    def check_overlap(pos1, pos2):
        d = np.linalg.norm(pos1 - pos2)
        return d < np.sqrt(3.0)
        
    def is_valid(positions):
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(positions[i], positions[j]):
                    return False
        return True
        
    best_score = float('inf')
    best_pos = None
    
    # Use multiple restarts
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialize positions
        # Strategy: Start with a random valid lattice subset
        lattice_vecs = [
            np.array([np.sqrt(3), 0]),
            np.array([np.sqrt(3)/2, 1.5])
        ]
        
        pts = [np.array([0.0, 0.0])]
        while len(pts) < N:
            base = pts[np.random.randint(len(pts))]
            dirs = [
                lattice_vecs[0],
                -lattice_vecs[0],
                lattice_vecs[1],
                -lattice_vecs[1],
                lattice_vecs[0] - lattice_vecs[1],
                lattice_vecs[1] - lattice_vecs[0]
            ]
            np.random.shuffle(dirs)
            added = False
            for d in dirs:
                new_pt = base + d
                # Check if new_pt is too close to any existing point
                valid_new = True
                for p in pts:
                    if np.linalg.norm(new_pt - p) < np.sqrt(3.0):
                        valid_new = False
                        break
                if valid_new:
                    pts.append(new_pt)
                    added = True
                    break
            if not added:
                # Fallback: random point
                pts.append(np.random.uniform(-3, 3, 2))
                
        positions = np.array(pts)
        
        # Ensure initial validity (might fail with random fallback)
        if not is_valid(positions):
            continue
            
        current_score = calc_required_side(positions)
        
        # Local Search
        step_size = 0.2
        for step in range(2000):
            if time.time() > deadline - 2:
                break
                
            idx = np.random.randint(N)
            orig_pos = positions[idx].copy()
            
            positions[idx] += np.random.normal(0, step_size, 2)
            
            if is_valid(positions):
                new_score = calc_required_side(positions)
                if new_score < current_score:
                    current_score = new_score
                else:
                    positions[idx] = orig_pos
            else:
                positions[idx] = orig_pos
                
            if step % 100 == 0:
                step_size *= 0.95
                
        if current_score < best_score:
            best_score = current_score
            best_pos = positions.copy()
            
    if best_pos is None:
        # Fallback to a known valid configuration
        # 7-flower + 4
        pts = [
            [0, 0],
            [np.sqrt(3), 0],
            [-np.sqrt(3), 0],
            [0, 1.5], # Wait, y should be sqrt(3)/2 * 2 = sqrt(3)? No.
            # Lattice y-step is 1.5.
            # Let's just use a simple grid
        ]
        # Fallback: Line
        pts = [[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_pos = np.array(pts)
        best_score = 40.0
        
    return [[p[0], p[1], 0.0] for p in best_pos], (0.0, 0.0), best_score, 0.0

