"""Hexagon packing via support-function overlap + simulated annealing + min enclosing hexagon."""
import numpy as np
import time
import math

N = 11
SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    # Precompute unit hexagon vertices (circumradius 1) at angle 0
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * (math.pi / 180.0)
    base_vx = np.cos(base_angles)
    base_vy = np.sin(base_angles)
    # Edge normals (outward) for unit hexagon at angle 0: directions at 30,90,150,210,270,330
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * (math.pi / 180.0)
    base_nx = np.cos(normal_angles)
    base_ny = np.sin(normal_angles)

    def rot_vecs(vx, vy, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return vx * c - vy * s, vx * s + vy * c

    def hex_support(cx, cy, theta):
        """Return support values h_i = max over vertices of v_i . n_i for the 6 edge normals at rotation theta."""
        nx, ny = rot_vecs(base_nx, base_ny, theta)
        vx, vy = rot_vecs(base_vx, base_vy, theta)
        # support = cx*nx + cy*ny + max(vx*nx + vy*ny) but max of rotated dot is just max over base since both rotated by same theta
        # Actually support in direction n = c . n + max_v (v . n). Since n and v both rotated by theta, v . n is invariant = base_v . base_n
        # So support_i = cx*nx_i + cy*ny_i + base_support_i
        base_support = np.max(base_vx * base_nx + base_vy * base_ny)  # this is just 1.0 for unit hex
        # Wait: max over 6 vertices of v . n where n is one of the normals. For regular hex, support in normal direction = 1.0 (circumradius)
        # Actually for a unit hexagon (circumradius 1), the support in the direction of an edge normal is exactly 1.0.
        # Let's just compute directly to be safe.
        dots = vx * nx + vy * ny
        max_dots = np.max(dots)  # should be 1.0
        return cx * nx + cy * ny + max_dots

    def overlap_test(pos1, ang1, pos2, ang2):
        """Return True if two unit hexagons overlap (interior intersection).
        Using separating axis: check if any edge normal of either hex separates them."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        # Normals for hex1
        nx1, ny1 = rot_vecs(base_nx, base_ny, ang1)
        # Normals for hex2
        nx2, ny2 = rot_vecs(base_nx, base_ny, ang2)
        all_nx = np.concatenate([nx1, nx2])
        all_ny = np.concatenate([ny1, ny2])
        # Support of hex1 in each normal direction
        vx1, vy1 = rot_vecs(base_vx, base_vy, ang1)
        s1 = pos1[0] * all_nx + pos1[1] * all_ny + np.max(vx1 * all_nx + vy1 * all_ny)
        # Support of hex2
        vx2, vy2 = rot_vecs(base_vx, base_vy, ang2)
        s2 = pos2[0] * all_nx + pos2[1] * all_ny + np.max(vx2 * all_nx + vy2 * all_ny)
        # If for any normal, s1 < -s2 (i.e., max1 < -min2), they are separated
        # min of hex in direction n = -max in direction -n. So separation condition: s1 + min2 < 0 where min2 = -s2_neg
        # Simpler: check if s1 < -s2 for any normal means separation? No.
        # Separation: exists normal n such that max(v1.n) < min(v2.n)
        # min(v2.n) = -max(v2.(-n))
        # Let's just compute: for each normal, check if s1 < -s2_opposite... 
        # Actually standard SAT: for each axis, if max1 < min2 or max2 < min1, separated.
        # min in direction n = -max in direction -n.
        # Since we have all 6 normals, -n is also in the set (opposite). So we can check:
        # For each normal i: if s1[i] < -s2[i+3] (where i+3 is opposite) OR s2[i] < -s1[i+3]
        # But easier: just check all 12 normals and see if any separates.
        # Let me just do the direct check: for each of the 12 normals, compute support of both hexes in that direction,
        # and check if they are separated.
        for i in range(12):
            nxi = all_nx[i]
            nyi = all_ny[i]
            # Support of hex1 in direction (nxi, nyi)
            sup1 = pos1[0]*nxi + pos1[1]*nyi + np.max(vx1*nxi + vy1*nyi)
            # Support of hex2 in direction (nxi, nyi)
            sup2 = pos2[0]*nxi + pos2[1]*nyi + np.max(vx2*nxi + vy2*nyi)
            # Support in opposite direction
            sup1_neg = pos1[0]*(-nxi) + pos1[1]*(-nyi) + np.max(vx1*(-nxi) + vy1*(-nyi))
            sup2_neg = pos2[0]*(-nxi) + pos2[1]*(-nyi) + np.max(vx2*(-nxi) + vy2*(-nyi))
            # Separated if sup1_neg < sup2 or sup2_neg < sup1
            if sup1_neg < sup2 or sup2_neg < sup1:
                return False  # separated, no overlap
        return True  # no separating axis found, overlap

    def min_enclosing_hex_side(positions, angles):
        """Compute minimum side length of a regular hexagon (any rotation) that contains all unit hexagons.
        For a given outer hexagon rotation phi, the required side is max over all inner vertices of the support in the 6 outer normals.
        We minimize over phi."""
        # Collect all vertices of all inner hexagons
        all_vx = []
        all_vy = []
        for i in range(N):
            vx, vy = rot_vecs(base_vx, base_vy, angles[i])
            all_vx.append(positions[i, 0] + vx)
            all_vy.append(positions[i, 1] + vy)
        all_vx = np.concatenate(all_vx)
        all_vy = np.concatenate(all_vy)
        
        def side_for_phi(phi):
            nx, ny = rot_vecs(base_nx, base_ny, phi)
            # Support of the set of points in each normal direction
            sup = all_vx * nx + all_vy * ny
            max_sup = np.max(sup)
            # The side length needed is max_sup (since support from center = side for unit normal)
            # But we can also shift the outer hexagon center. The minimal enclosing hexagon centered optimally:
            # side = max_i (sup_i - min_i) / 2 where min_i = -sup_{i+3}
            # Actually for a hexagon, the width in direction n_i is sup_i - min_i = sup_i + sup_{i+3} (opposite)
            # The minimal side length for a hexagon with normals at phi is max over i of (sup_i + sup_{i+3})/2
            # But we can choose the center to minimize the max. The optimal center makes the max support equal to min support for each pair.
            # So side = max_i (sup_i + sup_{i+3}) / 2
            opp = np.concatenate([sup[3:6], sup[0:3]])
            widths = sup + opp
            return np.max(widths) / 2.0
        
        # Minimize over phi: try a grid
        best = 1e18
        for k in range(60):
            phi = k * (math.pi / 30.0)  # 0 to 180 deg in 3 deg steps
            s = side_for_phi(phi)
            if s < best:
                best = s
        return best

    def total_overlap_count(positions, angles):
        """Count number of overlapping pairs."""
        cnt = 0
        for i in range(N):
            for j in range(i+1, N):
                if overlap_test(positions[i], angles[i], positions[j], angles[j]):
                    cnt += 1
        return cnt

    # Simulated annealing
    best_side = 1e18
    best_positions = None
    best_angles = None

    # Initialize: hexagonal lattice arrangement
    # For 11 hexagons, a compact arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Or just a linear arrangement first
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Try a few initial configurations
    configs = []
    
    # Config 1: linear row
    for i in range(N):
        positions[i, 0] = i * 2.0
        positions[i, 1] = 0.0
    configs.append((positions.copy(), angles.copy()))
    
    # Config 2: 2D hexagonal lattice
    # Centers at (i*2 + j*1, j*SQRT3) for appropriate i,j
    idx = 0
    for j in range(-2, 3):
        for i in range(-2, 3):
            if idx >= N:
                break
            positions[idx, 0] = i * 2.0 + j * 1.0
            positions[idx, 1] = j * SQRT3
            idx += 1
    configs.append((positions.copy(), angles.copy()))
    
    # Config 3: random
    positions = np.random.randn(N, 2) * 5.0
    angles = np.random.randn(N) * (math.pi / 6.0)
    configs.append((positions.copy(), angles.copy()))
    
    for cfg_idx, (pos_init, ang_init) in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
        
        pos = pos_init.copy()
        ang = ang_init.copy()
        # Center
        pos[:, 0] -= np.mean(pos[:, 0])
        pos[:, 1] -= np.mean(pos[:, 1])
        
        T = 2.0
        T_min = 0.01
        cool_rate = 0.999
        step_size = 0.5
        
        cur_side = min_enclosing_hex_side(pos, ang)
        cur_overlaps = total_overlap_count(pos, ang)
        cur_cost = cur_side + 10.0 * cur_overlaps
        
        for step in range(50000):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb a random hexagon
            i = np.random.randint(N)
            old_pos = pos[i].copy()
            old_ang = ang[i]
            
            pos[i, 0] += np.random.randn() * step_size
            pos[i, 1] += np.random.randn() * step_size
            ang[i] += np.random.randn() * step_size * 0.1
            
            new_side = min_enclosing_hex_side(pos, ang)
            new_overlaps = total_overlap_count(pos, ang)
            new_cost = new_side + 10.0 * new_overlaps
            
            delta = new_cost - cur_cost
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                cur_side = new_side
                cur_overlaps = new_overlaps
                cur_cost = new_cost
                if cur_overlaps == 0 and cur_side < best_side:
                    best_side = cur_side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
            else:
                pos[i] = old_pos
                ang[i] = old_ang
            
            T *= cool_rate
            if T < T_min:
                T = T_min
                step_size *= 0.95  # slow down
    
    # If we found a valid configuration, return it
    if best_positions is not None and best_overlaps == 0:
        inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i] * 180.0 / math.pi)] for i in range(N)]
        return inner, (0.0, 0.0), float(best_side), 0.0
    
    # Fallback: return a simple valid configuration
    # Linear arrangement with large enough outer hexagon
    inner = []
    for i in range(N):
        inner.append([float(i * 2.0), 0.0, 0.0])
    # Outer hexagon side must contain a row of N hexagons
    # Total width = (N-1)*2 + 2 = 2N
    # For flat-top hexagon, width across flats = SQRT3 * side
    # So side >= 2N / SQRT3
    side = 2.0 * N / SQRT3 + 1.0
    return inner, (0.0, 0.0), float(side), 0.0

# Fix: best_overlaps wasn't defined in the fallback path
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    base_angles = np.array([0, 60, 120, 180, 240, 300]) * (math.pi / 180.0)
    base_vx = np.cos(base_angles)
    base_vy = np.sin(base_angles)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * (math.pi / 180.0)
    base_nx = np.cos(normal_angles)
    base_ny = np.sin(normal_angles)

    def rot_vecs(vx, vy, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return vx * c - vy * s, vx * s + vy * c

    def overlap_test(pos1, ang1, pos2, ang2):
        vx1, vy1 = rot_vecs(base_vx, base_vy, ang1)
        vx2, vy2 = rot_vecs(base_vx, base_vy, ang2)
        nx1, ny1 = rot_vecs(base_nx, base_ny, ang1)
        nx2, ny2 = rot_vecs(base_nx, base_ny, ang2)
        all_nx = np.concatenate([nx1, nx2])
        all_ny = np.concatenate([ny1, ny2])
        for i in range(12):
            nxi = all_nx[i]
            nyi = all_ny[i]
            sup1 = pos1[0]*nxi + pos1[1]*nyi + np.max(vx1*nxi + vy1*nyi)
            sup2 = pos2[0]*nxi + pos2[1]*nyi + np.max(vx2*nxi + vy2*nyi)
            sup1_neg = pos1[0]*(-nxi) + pos1[1]*(-nyi) + np.max(vx1*(-nxi) + vy1*(-nyi))
            sup2_neg = pos2[0]*(-nxi) + pos2[1]*(-nyi) + np.max(vx2*(-nxi) + vy2*(-nyi))
            if sup1_neg < sup2 or sup2_neg < sup1:
                return False
        return True

    def min_enclosing_hex_side(positions, angles):
        all_vx = []
        all_vy = []
        for i in range(N):
            vx, vy = rot_vecs(base_vx, base_vy, angles[i])
            all_vx.append(positions[i, 0] + vx)
            all_vy.append(positions[i, 1] + vy)
        all_vx = np.concatenate(all_vx)
        all_vy = np.concatenate(all_vy)
        
        def side_for_phi(phi):
            nx, ny = rot_vecs(base_nx, base_ny, phi)
            sup = all_vx * nx + all_vy * ny
            opp = np.concatenate([sup[3:6], sup[0:3]])
            widths = sup + opp
            return np.max(widths) / 2.0
        
        best = 1e18
        for k in range(60):
            phi = k * (math.pi / 30.0)
            s = side_for_phi(phi)
            if s < best:
                best = s
        return best

    def total_overlap_count(positions, angles):
        cnt = 0
        for i in range(N):
            for j in range(i+1, N):
                if overlap_test(positions[i], angles[i], positions[j], angles[j]):
                    cnt += 1
        return cnt

    best_side = 1e18
    best_positions = None
    best_angles = None
    best_overlaps = 1e18

    configs = []
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = i * 2.0
        positions[i, 1] = 0.0
    configs.append((positions.copy(), angles.copy()))
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    idx = 0
    for j in range(-2, 3):
        for i in range(-2, 3):
            if idx >= N:
                break
            positions[idx, 0] = i * 2.0 + j * 1.0
            positions[idx, 1] = j * SQRT3
            idx += 1
    configs.append((positions.copy(), angles.copy()))
    
    positions = np.random.randn(N, 2) * 5.0
    angles = np.random.randn(N) * (math.pi / 6.0)
    configs.append((positions.copy(), angles.copy()))
    
    for cfg_idx, (pos_init, ang_init) in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
        
        pos = pos_init.copy()
        ang = ang_init.copy()
        pos[:, 0] -= np.mean(pos[:, 0])
        pos[:, 1] -= np.mean(pos[:, 1])
        
        T = 2.0
        T_min = 0.01
        cool_rate = 0.999
        step_size = 0.5
        
        cur_side = min_enclosing_hex_side(pos, ang)
        cur_overlaps = total_overlap_count(pos, ang)
        cur_cost = cur_side + 10.0 * cur_overlaps
        
        for step in range(50000):
            if time.time() > deadline - 2.0:
                break
            
            i = np.random.randint(N)
            old_pos = pos[i].copy()
            old_ang = ang[i]
            
            pos[i, 0] += np.random.randn() * step_size
            pos[i, 1] += np.random.randn() * step_size
            ang[i] += np.random.randn() * step_size * 0.1
            
            new_side = min_enclosing_hex_side(pos, ang)
            new_overlaps = total_overlap_count(pos, ang)
            new_cost = new_side + 10.0 * new_overlaps
            
            delta = new_cost - cur_cost
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                cur_side = new_side
                cur_overlaps = new_overlaps
                cur_cost = new_cost
                if cur_overlaps == 0 and cur_side < best_side:
                    best_side = cur_side
                    best_positions = pos.copy()
                    best_angles = ang.copy()
                    best_overlaps = 0
            else:
                pos[i] = old_pos
                ang[i] = old_ang
            
            T *= cool_rate
            if T < T_min:
                T = T_min
                step_size *= 0.95

    if best_positions is not None and best_overlaps == 0:
        inner = [[float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i] * 180.0 / math.pi)] for i in range(N)]
        return inner, (0.0, 0.0), float(best_side), 0.0
    
    inner = []
    for i in range(N):
        inner.append([float(i * 2.0), 0.0, 0.0])
    side = 2.0 * N / SQRT3 + 1.0
    return inner, (0.0, 0.0), float(side), 0.0
