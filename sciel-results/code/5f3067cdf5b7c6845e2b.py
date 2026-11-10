# EVOLVE-BLOCK-START
import numpy as np
import time

def moving_sofa_3d_path() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin

    # Grid points
    gx = np.linspace(-4, 0, 81)
    gy = np.linspace(0, 1, 21)
    gz = np.linspace(0, 1, 21)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    grid_points = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=1)
    N_pts = len(grid_points)

    start_pose = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end_pose = np.array([0.0, 6.0, 7.0, 90.0, 90.0, 90.0])

    def rot_matrix(yaw, pitch, roll):
        yr = np.radians(yaw); pr = np.radians(pitch); rr = np.radians(roll)
        cy, sy = np.cos(yr), np.sin(yr)
        cp, sp = np.cos(pr), np.sin(pr)
        cr, sr = np.cos(rr), np.sin(rr)
        # R = Rx @ Ry @ Rz
        return np.array([
            [cr*cp, cr*sp*sy - sr*cy, cr*sp*cy + sr*sy],
            [sr*cp, sr*sp*sy + cr*cy, sr*sp*cy - cr*sy],
            [-sp,   cp*sy,            cp*cy]
        ])

    def in_corridor_vec(wx, wy, wz):
        r1 = (wx <= 0) & (wy >= 0) & (wy <= 1) & (wz >= 0) & (wz <= 1)
        r2 = (wx >= 0) & (wx <= 1) & (wy >= 0) & (wy <= 5) & (wz >= 0) & (wz <= 1)
        r3 = (wx >= 0) & (wx <= 1) & (wy >= 5) & (wy <= 6) & (wz >= 0)
        return r1 | r2 | r3

    def densify(path):
        full = np.vstack([start_pose, path, end_pose])
        max_d = 0.05; max_a = 2.0
        result = [full[0]]
        for i in range(len(full)-1):
            p1, p2 = full[i], full[i+1]
            d = np.linalg.norm(p1[:3]-p2[:3])
            ad = np.max(np.abs(p1[3:]-p2[3:]))
            n = max(1, int(np.ceil(d/max_d)), int(np.ceil(ad/max_a)))
            while d/n > max_d or ad/n > max_a:
                n += 1
            for j in range(1, n+1):
                t = j/n
                result.append(p1*(1-t) + p2*t)
        return np.array(result)

    def evaluate_fast(path):
        dense = densify(path)
        K = len(dense)
        survive = np.ones(N_pts, dtype=bool)
        
        # Check first pose
        R0 = rot_matrix(dense[0][3], dense[0][4], dense[0][5])
        pts0 = (R0 @ grid_points.T).T + dense[0][:3]
        survive = in_corridor_vec(pts0[:,0], pts0[:,1], pts0[:,2])
        if not np.any(survive): return 0
        
        for i in range(K-1):
            p1, p2 = dense[i], dense[i+1]
            
            # x move
            R_x = rot_matrix(p1[3], p1[4], p1[5])
            pts_x = (R_x @ grid_points.T).T + np.array([p2[0], p1[1], p1[2]])
            survive &= in_corridor_vec(pts_x[:,0], pts_x[:,1], pts_x[:,2])
            if not np.any(survive): return 0

            # y move
            pts_y = (R_x @ grid_points.T).T + np.array([p2[0], p2[1], p1[2]])
            survive &= in_corridor_vec(pts_y[:,0], pts_y[:,1], pts_y[:,2])
            if not np.any(survive): return 0

            # z move
            pts_z = (R_x @ grid_points.T).T + np.array([p2[0], p2[1], p2[2]])
            survive &= in_corridor_vec(pts_z[:,0], pts_z[:,1], pts_z[:,2])
            if not np.any(survive): return 0

            # rotation bounding box check
            n_rot = 3
            min_coords = np.array([np.inf]*3)
            max_coords = np.array([-np.inf]*3)
            for s in range(n_rot+1):
                t = s/n_rot
                yaw = p1[3]*(1-t) + p2[3]*t
                pitch = p1[4]*(1-t) + p2[4]*t
                roll = p1[5]*(1-t) + p2[5]*t
                R_s = rot_matrix(yaw, pitch, roll)
                pts_s = (R_s @ grid_points.T).T + p2[:3]
                min_coords = np.minimum(min_coords, pts_s.min(axis=0))
                max_coords = np.maximum(max_coords, pts_s.max(axis=0))
            
            r1_ok = (max_coords[0] <= 0) and (min_coords[1] >= 0) and (max_coords[1] <= 1) and (min_coords[2] >= 0) and (max_coords[2] <= 1)
            r2_ok = (min_coords[0] >= 0) and (max_coords[0] <= 1) and (min_coords[1] >= 0) and (max_coords[1] <= 5) and (min_coords[2] >= 0) and (max_coords[2] <= 1)
            r3_ok = (min_coords[0] >= 0) and (max_coords[0] <= 1) and (min_coords[1] >= 5) and (max_coords[1] <= 6) and (min_coords[2] >= 0)
            
            if not (r1_ok or r2_ok or r3_ok):
                return 0

        # Count 4x4x4 blocks
        sg = survive.reshape(81, 21, 21)
        cs = np.zeros((82, 22, 22))
        cs[1:,1:,1:] = sg
        cs = np.cumsum(np.cumsum(np.cumsum(cs, axis=0), axis=1), axis=2)
        
        blocks = cs[4:,4:,4:] - cs[:-4,4:,4:] - cs[4:,:-4,4:] - cs[4:,4:,:-4] + cs[:-4,:-4,4:] + cs[4:,:-4,:-4] + cs[:-4,4:,:-4] - cs[:-4,:-4,:-4]
        count = np.sum(blocks == 64)
        return count

    # Better initial path: medial axis inspired, centered in corridor
    # The corridor has three regions:
    # R1: x<=0, 0<=y<=1, 0<=z<=1 (entry)
    # R2: 0<=x<=1, 0<=y<=5, 0<=z<=1 (main)
    # R3: 0<=x<=1, 5<=y<=6, z>=0 (exit)
    
    # Generate a smooth path along the medial axis
    # Entry: center at y=0.5, z=0.5, moving in x from -3 to 0
    # Main: center at x=0.5, z=0.5, moving in y from 0 to 5
    # Exit: center at x=0.5, moving in y from 5 to 6, z from 0 to 7
    # Rotation: yaw 0->90, pitch 0->90, roll 0->90
    
    # Let's create a path with more waypoints for smoother interpolation
    waypoints = []
    
    # Entry phase: x from -3 to 0.5, y=0.5, z=0.5
    for x in np.linspace(-3, 0.5, 20):
        waypoints.append([x, 0.5, 0.5, 0, 0, 0])
    
    # Transition: rotate while moving
    # We need to go from (0.5, 0.5, 0.5) with (0,0,0) to eventually (0, 6, 7) with (90,90,90)
    # Main corridor: x in [0,1], y in [0,5], z in [0,1]
    # Move along y while keeping x=0.5, z=0.5
    for y in np.linspace(0.5, 5.0, 30):
        waypoints.append([0.5, y, 0.5, 0, 0, 0])
    
    # Exit: y from 5 to 6, z from 0.5 to 7
    # Need to rotate: yaw 0->90, pitch 0->90, roll 0->90
    # Do this gradually
    for i in range(40):
        t = i/39.0
        y = 5.0 + t*1.0
        z = 0.5 + t*6.5
        x = 0.5 - t*0.5  # move x from 0.5 to 0
        yaw = t*90
        pitch = t*90
        roll = t*90
        waypoints.append([x, y, z, yaw, pitch, roll])
    
    base_path = np.array(waypoints, dtype=np.float64)
    
    best_path = base_path.copy()
    best_score = evaluate_fast(base_path)
    
    if time.time() > deadline - 10:
        return best_path

    rng = np.random.default_rng(42)
    
    # Simulated annealing style optimization
    pos_scale = 0.05
    ang_scale = 1.0
    temp = 1.0
    cooling = 0.995
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 20:
        iteration += 1
        no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count >= max_no_improve:
            pos_scale *= 0.8
            ang_scale *= 0.8
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.1)
        
        new_path = best_path.copy()
        n_perturb = rng.integers(1, 4)
        for _ in range(n_perturb):
            idx = rng.integers(0, len(new_path))
            new_path[idx, :3] += rng.normal(0, pos_scale, 3)
            new_path[idx, 3:] += rng.normal(0, ang_scale, 3)
        
        score = evaluate_fast(new_path)
        
        # Accept if better or with probability based on temperature
        if score > best_score:
            best_path = new_path
            best_score = score
            no_improve_count = 0
        else:
            # Metropolis criterion
            delta = best_score - score
            if delta < temp:
                if rng.random() < np.exp(-delta / max(temp, 1e-10)):
                    best_path = new_path
                    best_score = score
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    # Final polish
    if time.time() < deadline - 5:
        polish_pos = 0.01
        polish_ang = 0.3
        for _ in range(100):
            if time.time() >= deadline - 5:
                break
            new_path = best_path.copy()
            idx = rng.integers(0, len(new_path))
            new_path[idx, :3] += rng.normal(0, polish_pos, 3)
            new_path[idx, 3:] += rng.normal(0, polish_ang, 3)
            score = evaluate_fast(new_path)
            if score > best_score:
                best_path = new_path
                best_score = score

    return best_path

# EVOLVE-BLOCK-END
