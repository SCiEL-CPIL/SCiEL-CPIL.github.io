import numpy as np
import time
from collections import deque

def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rx @ Ry @ Rz

def _rotation_matrix_inv(yaw_deg, pitch_deg, roll_deg):
    """Inverse of R = Rx(roll) Ry(pitch) Rz(yaw) is Rz(-yaw) Ry(-pitch) Rx(-roll)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    RzT = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    RyT = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    RxT = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    return RzT @ RyT @ RxT

def _corridor_contains_vec(px, py, pz):
    """Vectorized corridor check. px, py, pz are 1D arrays."""
    r1 = (px <= 0) & (py >= 0) & (py <= 1) & (pz >= 0) & (pz <= 1)
    r2 = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 5) & (pz >= 0) & (pz <= 1)
    r3 = (px >= 0) & (px <= 1) & (py >= 5) & (py <= 6) & (pz >= 0)
    return r1 | r2 | r3

def _pose_is_feasible(pose, grid_points):
    """Check if a single pose keeps all grid points in the corridor."""
    R_inv = _rotation_matrix_inv(pose[3], pose[4], pose[5])
    # Local coords of grid points in body frame
    local = (grid_points - pose[:3]) @ R_inv
    # local is (N,3): lx, ly, lz
    # The placed point is R * local + pos = grid_point, so we need grid_point in corridor
    # Actually: placed = R @ p + t, where p is the grid point in world... 
    # Wait: "point p is placed at R p + (x,y,z)"
    # So for each grid point p, placed = R @ p + t
    # We need placed in corridor.
    R = _rotation_matrix(pose[3], pose[4], pose[5])
    placed = grid_points @ R.T + pose[:3]
    return _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])

def _step_feasible(prev_pose, curr_pose, grid_points):
    """Check if the single step from prev to curr is feasible.
    The evaluator does: x move, y move, z move, then rotation about single axis.
    For the step to be valid, each sub-move must keep all grid points in corridor.
    """
    # X move: change x from prev[0] to curr[0], keep y,z,angles from prev
    p_x = prev_pose.copy()
    p_x[0] = curr_pose[0]
    R = _rotation_matrix(p_x[3], p_x[4], p_x[5])
    placed = grid_points @ R.T + p_x[:3]
    if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
        return False
    
    # Y move
    p_y = p_x.copy()
    p_y[1] = curr_pose[1]
    R = _rotation_matrix(p_y[3], p_y[4], p_y[5])
    placed = grid_points @ R.T + p_y[:3]
    if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
        return False
    
    # Z move
    p_z = p_y.copy()
    p_z[2] = curr_pose[2]
    R = _rotation_matrix(p_z[3], p_z[4], p_z[5])
    placed = grid_points @ R.T + p_z[:3]
    if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
        return False
    
    # Rotation: only one axis should change
    dyaw = curr_pose[3] - prev_pose[3]
    dpitch = curr_pose[4] - prev_pose[4]
    droll = curr_pose[5] - prev_pose[5]
    
    # Check which axis rotates
    rotating_yaw = abs(dyaw) > 1e-9 and abs(dpitch) < 1e-9 and abs(droll) < 1e-9
    rotating_pitch = abs(dpitch) > 1e-9 and abs(dyaw) < 1e-9 and abs(droll) < 1e-9
    rotating_roll = abs(droll) > 1e-9 and abs(dyaw) < 1e-9 and abs(dpitch) < 1e-9
    
    if not (rotating_yaw or rotating_pitch or rotating_roll):
        # No rotation or multiple axes - check final pose
        R = _rotation_matrix(curr_pose[3], curr_pose[4], curr_pose[5])
        placed = grid_points @ R.T + curr_pose[:3]
        return np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2]))
    
    # Sample along rotation arc
    if rotating_yaw:
        n_s = max(2, int(np.ceil(abs(dyaw) / 2.0)))
        for s in range(n_s + 1):
            t = s / n_s
            yaw_s = prev_pose[3] + t * dyaw
            R = _rotation_matrix(yaw_s, prev_pose[4], prev_pose[5])
            placed = grid_points @ R.T + p_z[:3]
            if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
                return False
    elif rotating_pitch:
        n_s = max(2, int(np.ceil(abs(dpitch) / 2.0)))
        for s in range(n_s + 1):
            t = s / n_s
            pitch_s = prev_pose[4] + t * dpitch
            R = _rotation_matrix(prev_pose[3], pitch_s, prev_pose[5])
            placed = grid_points @ R.T + p_z[:3]
            if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
                return False
    else:  # rotating_roll
        n_s = max(2, int(np.ceil(abs(droll) / 2.0)))
        for s in range(n_s + 1):
            t = s / n_s
            roll_s = prev_pose[5] + t * droll
            R = _rotation_matrix(prev_pose[3], prev_pose[4], roll_s)
            placed = grid_points @ R.T + p_z[:3]
            if not np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])):
                return False
    
    return True

def _densify_and_evaluate(path, grid_points):
    """Full evaluation: densify path, check each step, count surviving blocks."""
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    full_path = np.vstack([start, path, end])
    
    # Densify
    densified = [full_path[0]]
    for i in range(1, len(full_path)):
        p0 = full_path[i-1]
        p1 = full_path[i]
        pos_dist = np.linalg.norm(p1[:3] - p0[:3])
        ang_diffs = np.abs(p1[3:] - p0[3:])
        n_steps = max(1, int(np.ceil(pos_dist / 0.05)), int(np.ceil(np.max(ang_diffs) / 2.0)))
        for j in range(1, n_steps + 1):
            t = j / n_steps
            densified.append(p0 + t * (p1 - p0))
    densified = np.array(densified)
    
    # Check first pose
    R0 = _rotation_matrix(densified[0,3], densified[0,4], densified[0,5])
    placed0 = grid_points @ R0.T + densified[0,:3]
    alive = _corridor_contains_vec(placed0[:,0], placed0[:,1], placed0[:,2])
    
    if not np.any(alive):
        return 0.0
    
    # Process each step
    for step in range(1, len(densified)):
        if not np.any(alive):
            break
        prev = densified[step-1]
        curr = densified[step]
        
        # X move
        p_x = prev.copy()
        p_x[0] = curr[0]
        R = _rotation_matrix(p_x[3], p_x[4], p_x[5])
        placed = grid_points @ R.T + p_x[:3]
        alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
        
        if not np.any(alive):
            break
        
        # Y move
        p_y = p_x.copy()
        p_y[1] = curr[1]
        R = _rotation_matrix(p_y[3], p_y[4], p_y[5])
        placed = grid_points @ R.T + p_y[:3]
        alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
        
        if not np.any(alive):
            break
        
        # Z move
        p_z = p_y.copy()
        p_z[2] = curr[2]
        R = _rotation_matrix(p_z[3], p_z[4], p_z[5])
        placed = grid_points @ R.T + p_z[:3]
        alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
        
        if not np.any(alive):
            break
        
        # Rotation
        dyaw = curr[3] - prev[3]
        dpitch = curr[4] - prev[4]
        droll = curr[5] - prev[5]
        
        rotating_yaw = abs(dyaw) > 1e-9 and abs(dpitch) < 1e-9 and abs(droll) < 1e-9
        rotating_pitch = abs(dpitch) > 1e-9 and abs(dyaw) < 1e-9 and abs(droll) < 1e-9
        rotating_roll = abs(droll) > 1e-9 and abs(dyaw) < 1e-9 and abs(dpitch) < 1e-9
        
        if rotating_yaw:
            n_s = max(2, int(np.ceil(abs(dyaw) / 2.0)))
            for s in range(n_s + 1):
                t = s / n_s
                yaw_s = prev[3] + t * dyaw
                R = _rotation_matrix(yaw_s, prev[4], prev[5])
                placed = grid_points @ R.T + p_z[:3]
                alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
                if not np.any(alive):
                    break
        elif rotating_pitch:
            n_s = max(2, int(np.ceil(abs(dpitch) / 2.0)))
            for s in range(n_s + 1):
                t = s / n_s
                pitch_s = prev[4] + t * dpitch
                R = _rotation_matrix(prev[3], pitch_s, prev[5])
                placed = grid_points @ R.T + p_z[:3]
                alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
                if not np.any(alive):
                    break
        elif rotating_roll:
            n_s = max(2, int(np.ceil(abs(droll) / 2.0)))
            for s in range(n_s + 1):
                t = s / n_s
                roll_s = prev[5] + t * droll
                R = _rotation_matrix(prev[3], prev[4], roll_s)
                placed = grid_points @ R.T + p_z[:3]
                alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
                if not np.any(alive):
                    break
        else:
            R = _rotation_matrix(curr[3], curr[4], curr[5])
            placed = grid_points @ R.T + curr[:3]
            alive &= _corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2])
    
    if not np.any(alive):
        return 0.0
    
    # Count 4x4x4 blocks
    alive_3d = alive.reshape(81, 21, 21)
    block_count = 0
    for i in range(78):
        for j in range(18):
            for k in range(18):
                if np.all(alive_3d[i:i+4, j:j+4, k:k+4]):
                    block_count += 1
    
    return 0.05**3 * block_count

def _build_pose_grid():
    """Build a structured grid of candidate poses for graph search."""
    # Position grid: x in [0, 1.1], y in [0, 6.1], z in [-0.1, 7.1]
    # Angle grid: yaw in [0, 90], pitch in [0, 90], roll in [0, 90]
    # But this is too large. Use a coarser grid and refine.
    
    # For the corridor structure, the sofa needs to:
    # 1. Enter from x<0, y in [0,1], z in [0,1] with identity orientation
    # 2. Move to x in [0,1], y in [0,5], z in [0,1] 
    # 3. Turn 90 deg yaw to face y direction
    # 4. Move along y to y in [5,6]
    # 5. Turn 90 deg pitch and 90 deg roll to face up
    # 6. Move up in z to z=7
    
    # Key waypoints (coarse):
    waypoints = []
    
    # Phase 1: Enter corridor (x from -3 to 0.5, y~0.5, z~0.5, angles~0)
    for x in np.linspace(-3, 0.5, 20):
        waypoints.append([x, 0.5, 0.5, 0, 0, 0])
    
    # Phase 2: Move along x=0.5, y from 0.5 to 4.5, starting yaw rotation
    for y in np.linspace(0.5, 4.5, 20):
        for yaw in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            waypoints.append([0.5, y, 0.5, float(yaw), 0, 0])
    
    # Phase 3: At y~5, move to y=5.5, complete yaw=90
    for y in np.linspace(4.5, 5.5, 10):
        waypoints.append([0.5, y, 0.5, 90, 0, 0])
    
    # Phase 4: Move along y=5.5, x from 0.5 to 0.5 (stay), start pitch
    for x in np.linspace(0.5, 0.5, 5):
        for pitch in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            waypoints.append([x, 5.5, 0.5, 90, float(pitch), 0])
    
    # Phase 5: At pitch~90, start roll, move up in z
    for z in np.linspace(0.5, 7.0, 20):
        for roll in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            waypoints.append([0.5, 5.5, z, 90, 90, float(roll)])
    
    # Convert to array
    return np.array(waypoints, dtype=np.float64)

def _find_path_dijkstra(waypoints, grid_points, deadline):
    """Find path from start to end using Dijkstra on pose graph."""
    start = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])
    
    # Check which waypoints are individually feasible
    n_wp = len(waypoints)
    feasible = np.zeros(n_wp, dtype=bool)
    for i in range(n_wp):
        if time.time() > deadline:
            break
        try:
            R = _rotation_matrix(waypoints[i,3], waypoints[i,4], waypoints[i,5])
            placed = grid_points @ R.T + waypoints[i,:3]
            feasible[i] = np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2]))
        except:
            feasible[i] = False
    
    # Find start and end nearest feasible waypoints
    # Start is at (-3,0,0,0,0,0), end is at (0,6,7,90,90,90)
    # We need to add these to our graph
    
    all_poses = np.vstack([start, waypoints, end])
    n_total = len(all_poses)
    
    # Feasibility for start and end
    feasible_full = np.zeros(n_total, dtype=bool)
    # Start
    R = _rotation_matrix(0, 0, 0)
    placed = grid_points @ R.T + start[:3]
    feasible_full[0] = np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2]))
    # Waypoints
    feasible_full[1:n_wp+1] = feasible
    # End
    R = _rotation_matrix(90, 90, 90)
    placed = grid_points @ R.T + end[:3]
    feasible_full[n_total-1] = np.all(_corridor_contains_vec(placed[:,0], placed[:,1], placed[:,2]))
    
    # Build adjacency: connect feasible poses that are close enough
    # and where the step is feasible
    # For efficiency, only connect poses within a certain distance
    
    # Use a simple approach: for each feasible pose, find nearby feasible poses
    # and check if the step is valid
    
    # This is expensive, so limit the number of connections
    max_neighbors = 50
    
    # Dijkstra
    dist = np.full(n_total, np.inf)
    prev_node = np.full(n_total, -1, dtype=int)
    dist[0] = 0.0
    visited = np.zeros(n_total, dtype=bool)
    
    # Priority queue (simple list-based for now)
    import heapq
    heap = [(0.0, 0)]
    
    while heap:
        if time.time() > deadline:
            break
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        
        if u == n_total - 1:
            break
        
        if not feasible_full[u]:
            continue
        
        # Find neighbors
        # Compute distances to all other feasible poses
        diffs = all_poses - all_poses[u]
        pos_dists = np.linalg.norm(diffs[:, :3], axis=1)
        ang_dists = np.max(np.abs(diffs[:, 3:]), axis=1)
        
        # Candidate neighbors: close in position and angle
        candidates = np.where((pos_dists < 2.0) & (ang_dists < 30.0) & (np.arange(n_total) != u) & feasible_full)[0]
        
        # Sort by distance
        if len(candidates) > 0:
            cand_dists = pos_dists[candidates]
            order = np.argsort(cand_dists)
            candidates = candidates[order[:max_neighbors]]
        
        for v in candidates:
            if visited[v]:
                continue
            if time.time() > deadline:
                break
            # Check if step u->v is feasible
            try:
                if _step_feasible(all_poses[u], all_poses[v], grid_points):
                    new_dist = d + pos_dists[v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_node[v] = u
                        heapq.heappush(heap, (new_dist, v))
            except:
                pass
    
    # Reconstruct path
    if dist[n_total - 1] == np.inf:
        return None
    
    path_indices = []
    node = n_total - 1
    while node != -1:
        path_indices.append(node)
        node = prev_node[node]
    path_indices.reverse()
    
    # Convert to path (exclude start and end, they're added by evaluator)
    if len(path_indices) <= 2:
        return np.array([[]])
    
    path = all_poses[path_indices[1:-1]]
    return path

def moving_sofa_3d_path() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin
    
    # Grid points
    x_grid = np.linspace(-4, 0, 81)
    y_grid = np.linspace(0, 1, 21)
    z_grid = np.linspace(0, 1, 21)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Seed path (from current program)
    seed_path = np.array([
        [0.939609406, 0.00998329527, -0.00407296132, 0.0886163961, 0.0514178831, 0.789799574],
        [0.976853949, 0.0843900243, -0.00187236014, 2.22743402, 0.0275244655, 0.595653008],
        [1.01563377, 0.158600496, 9.83065911e-05, 4.35572642, -0.00995231395, 0.363970701],
        [1.08493385, 0.307755287, 0.00404501406, 8.62938219, -0.000435990906, -0.335077274],
        [1.12726319, 0.436994587, 0.0165792562, 12.3522097, -0.444355561, -1.09977761],
        [1.15035596, 0.509864783, 0.0246729498, 14.4926365, -0.703150883, -1.77573027],
        [1.20020002, 0.674818962, 0.0413920989, 19.2858257, -1.46824802, -3.24301616],
        [1.22921833, 0.78032244, 0.0508392229, 22.3604231, -2.05934965, -4.19183977],
        [1.25656232, 0.886955923, 0.0622851979, 25.4006424, -2.79233605, -4.94301221],
        [1.28161834, 0.9941859, 0.0713168622, 28.4982429, -3.53439006, -5.72810662],
        [1.30160821, 1.12543063, 0.0720451855, 32.0688854, -3.9280592, -5.56488257],
        [1.32438826, 1.32927434, 0.0724034864, 37.7041144, -4.47472162, -5.17490305],
        [1.33399008, 1.48086681, 0.0758177845, 41.9024902, -4.92138226, -4.72192269],
        [1.33698854, 1.59471234, 0.075956822, 45.1249666, -5.16592127, -4.32429483],
        [1.33554955, 1.70831674, 0.0733949585, 48.4117195, -5.24474942, -3.95318659],
        [1.33301955, 1.77848192, 0.0730557819, 50.4395679, -5.29391038, -3.60268508],
        [1.32299384, 1.89675169, 0.066224521, 54.0094721, -5.07170268, -3.10511721],
        [1.31609554, 1.95530642, 0.0645580036, 55.7906035, -4.95025879, -2.75163185],
        [1.29699464, 2.0647489, 0.0528964026, 59.2092384, -4.24840518, -2.08208301],
        [1.28467533, 2.12603596, 0.0465722033, 61.19788, -3.85394213, -1.77381137],
        [1.27612175, 2.16246336, 0.0429218547, 62.3960278, -3.60313312, -1.60391888],
        [1.25918263, 2.22238115, 0.0352027157, 64.3876548, -2.94645508, -1.14315788],
        [1.21195969, 2.37272154, 0.0185866337, 69.5501344, -1.44731306, -0.351817856],
        [1.17367747, 2.46947271, 0.00787738711, 73.3278162, -0.59130695, -0.204315921],
        [1.13343824, 2.56892736, 0.000240804007, 77.2294036, 0.0742535282, -0.0118797041],
        [1.10227764, 2.63702207, -0.000813164259, 80.2921591, 0.325638556, -0.00909491031],
        [1.08115946, 2.67872855, -0.0019446761, 82.2507337, 0.456149966, 0.00923364344],
        [1.06864394, 2.70464551, -0.00209793986, 83.4698923, 0.395788406, -0.00251260641],
        [0.993054246, 2.83888423, -0.00277991124, 90.0331465, 0.651004673, -0.0200027208],
        [0.992050998, 3.42584708, -0.00292617372, 90.0630824, 0.648340038, -0.0336076805],
        [0.990524577, 4.81296934, -0.00378757857, 90.0846127, 0.741107384, -0.0439029391],
        [0.988680148, 5.99738121, -0.00451717463, 90.104646, 0.883485061, -0.041030785],
        [0.957624286, 5.9528988, 0.0102741306, 90.1327839, 4.19080459, 1.12928446],
        [0.942862105, 5.99368058, 0.0162024942, 90.1288679, 5.76538787, 1.62560793],
        [0.931969027, 5.99412693, 0.020619344, 90.1263482, 6.90261077, 1.98875845],
        [0.904773557, 6.00941729, 0.0313747872, 90.128735, 9.61483109, 2.83144875],
        [0.8858989, 6.01830162, 0.0381113604, 90.135911, 11.4508584, 3.34953358],
        [0.863540853, 6.02619691, 0.04487657, 90.1598001, 13.5181743, 3.88721021],
        [0.836463078, 6.03765479, 0.0528267196, 90.1967295, 15.9298717, 4.48186113],
        [0.804731182, 6.04627379, 0.0625195903, 90.2336098, 18.6990514, 5.14505668],
        [0.788296376, 6.05202369, 0.0675622892, 90.2479959, 20.1264906, 5.47138803],
        [0.778474876, 6.05538925, 0.0706050032, 90.2577268, 20.9723911, 5.66341348],
        [0.770888811, 6.05788046, 0.0730068444, 90.2657676, 21.6229248, 5.80874653],
        [0.764648333, 6.0599328, 0.0749383027, 90.2726388, 22.1532161, 5.92642676],
        [0.748753549, 6.06467644, 0.0801606053, 90.2892425, 23.4927925, 6.21910882],
        [0.737683416, 6.06785126, 0.0838714978, 90.3001161, 24.4180728, 6.42046395],
        [0.726094096, 6.07124306, 0.0879934282, 90.3107738, 25.3815663, 6.631397],
        [0.708478913, 6.07645575, 0.094580537, 90.324703, 26.8427434, 6.95357494],
        [0.685450204, 6.08320452, 0.103688262, 90.3410972, 28.7527373, 7.37124273],
        [0.660511655, 6.09058578, 0.114647767, 90.341297, 30.8426719, 7.83842912],
        [0.631106763, 6.09898622, 0.127562353, 90.3399229, 33.2683226, 8.35622932],
        [0.60239252, 6.10718508, 0.141197449, 90.3187387, 35.6585372, 8.87446837],
        [0.574685996, 6.1149268, 0.154862481, 90.2890178, 37.982628, 9.36856002],
        [0.549279838, 6.12170874, 0.168073709, 90.2538308, 40.1148352, 9.81523017],
        [0.523487025, 6.12859076, 0.182443691, 90.2080219, 42.279737, 10.2799932],
        [0.498121776, 6.13520082, 0.197968844, 90.1482985, 44.4163048, 10.7608292],
        [0.476465121, 6.14082429, 0.2120163, 90.0844905, 46.2626926, 11.1825901],
        [0.456153588, 6.14607051, 0.226245551, 90.0101313, 48.012554, 11.601748],
        [0.433791775, 6.15190581, 0.243138907, 89.9157178, 49.954221, 12.084816],
        [0.404025852, 6.15902259, 0.266186906, 89.7734311, 52.548863, 12.7293228],
        [0.366005234, 6.16774759, 0.297031124, 89.5738271, 55.893128, 13.5723105],
        [0.321312876, 6.17691876, 0.335574819, 89.2922705, 59.8692011, 14.594953],
        [0.278427663, 6.18417441, 0.373123513, 88.9719858, 63.7515612, 15.5832162],
        [0.239534769, 6.19553929, 0.413872182, 88.6402231, 67.3322992, 16.6327601],
        [0.204472084, 6.19968082, 0.464675511, 88.3088859, 70.5963626, 17.9436724],
        [0.175394931, 6.21048495, 0.524148773, 88.0051881, 73.3241903, 19.4940659],
        [0.153758289, 6.22132345, 0.583751542, 87.7681793, 75.3731869, 21.0578654],
        [0.126442905, 6.23697046, 0.675335517, 87.4796317, 77.9661162, 23.4337293],
        [0.11557363, 6.25005689, 0.744980177, 87.365021, 79.008173, 25.2702993],
        [0.106313411, 6.26635361, 0.836966776, 87.2610122, 79.9079131, 27.7208667],
        [0.0989070592, 6.2783871, 0.912917517, 87.1845804, 80.6362714, 29.7408798],
        [0.0945573652, 6.28550694, 0.960719372, 87.1536509, 81.0361104, 31.0067456],
        [0.0902778512, 6.29266034, 1.01249647, 87.1182294, 81.451701, 32.3850431],
        [0.0852959027, 6.30150778, 1.08113362, 87.0519944, 81.9245444, 34.2396643],
        [0.083875265, 6.30582796, 1.11958943, 86.977003, 82.1005075, 35.3326516],
        [0.0825523794, 6.30959352, 1.15442156, 86.9227818, 82.2502751, 36.313756],
        [0.0813272459, 6.31280445, 1.18563003, 86.8893308, 82.3738474, 37.1785229],
        [0.0795266549, 6.31701618, 1.22936759, 86.8298912, 82.5573864, 38.4051678],
        [0.0760640114, 6.32390774, 1.30814197, 86.730576, 82.9155599, 40.6273839],
        [0.0733345584, 6.32845988, 1.38417722, 86.6469757, 83.1880754, 42.7737559],
        [0.0728754398, 6.33436689, 1.51928934, 85.9570928, 83.321603, 47.2189208],
        [0.0740241462, 6.33651625, 1.67086132, 85.1992673, 83.4017941, 52.295749],
        [0.0709732742, 6.32973557, 1.82119648, 84.427344, 83.8384848, 57.4161351],
        [0.0689950542, 6.32296873, 1.89917235, 84.1030756, 84.0687341, 60.0778089],
        [0.0654533403, 6.31403661, 1.97470221, 83.8065972, 84.4296311, 62.6673579],
        [0.0598040781, 6.30238868, 2.04639195, 83.482155, 84.9583795, 65.1956567],
        [0.0532315605, 6.28848325, 2.11585078, 83.1684777, 85.5618308, 67.7111146],
        [0.0467194205, 6.27295621, 2.18521241, 82.8588875, 86.1559337, 70.2420111],
        [0.0211915399, 6.21382682, 2.3716358, 82.5672227, 88.3741932, 76.9350513],
        [0.0119033827, 6.16998589, 2.48828327, 83.7295105, 89.1215994, 80.3298032],
        [0.0114904543, 6.16373556, 2.50403291, 83.888837, 89.1782177, 80.7917836],
        [0.00847440903, 6.15720704, 2.52053877, 84.0621014, 89.3189944, 81.2648114],
        [0.0054345193, 6.12168354, 2.60523315, 84.9795728, 89.7896324, 83.7983903],
        [0.00500718906, 6.08778028, 2.67536568, 85.9496099, 89.8305811, 86.1593636],
        [0.00496310192, 6.07475445, 2.70089071, 86.2819807, 89.9205737, 87.0030543],
        [0.00447449792, 6.05605429, 2.7350364, 86.757467, 89.802935, 88.1996353],
        [0.00297530192, 6.02087628, 2.79766649, 87.459475, 90.1346094, 90.5955514],
        [0.000953058079, 5.9970912, 2.84427868, 87.6265515, 90.27883, 92.3854717],
        [-0.00231375478, 5.99154793, 4.38250817, 87.3674843, 90.6652632, 92.7094979],
    ], dtype=np.float64)
    
    # Evaluate seed
    best_path = seed_path.copy()
    best_vol = _densify_and_evaluate(seed_path, grid_points)
    
    # Try Dijkstra-based path finding
    if time.time() < deadline - 120:
        try:
            waypoints = _build_pose_grid()
            dijkstra_path = _find_path_dijkstra(waypoints, grid_points, deadline - 60)
            if dijkstra_path is not None and len(dijkstra_path) >= 1:
                vol_dj = _densify_and_evaluate(dijkstra_path, grid_points)
                if vol_dj > best_vol:
                    best_vol = vol_dj
                    best_path = dijkstra_path.copy()
        except Exception:
            pass
    
    # Local search refinement on best path
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        path = best_path.copy()
        step_sizes = [0.02, 0.01, 0.005, 0.002]
        
        for ss in step_sizes:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                idx = rng.integers(0, len(path))
                dim = rng.integers(0, 6)
                candidate = path.copy()
                candidate[idx, dim] += rng.normal(0, ss)
                try:
                    vol = _densify_and_evaluate(candidate, grid_points)
                    if vol > best_vol:
                        best_vol = vol
                        best_path = candidate
                        path = candidate
                        improved = True
                except:
                    pass
    
    return best_path
