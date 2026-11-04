"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    dx, dy = px - cx, py - cy
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    r = side * np.sqrt(3) / 2
    normal_angles = np.radians(30 + 60 * np.arange(6)) + angle_rad
    dist = np.sqrt(lx**2 + ly**2)
    if dist < 1e-12:
        return True
    theta = np.arctan2(ly, lx)
    cos_vals = np.cos(theta - normal_angles)
    h = r * np.max(cos_vals)
    return dist <= h + 1e-9

def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and min(p3[0],p4[0])-1e-9 <= p1[0] <= max(p3[0],p4[0])+1e-9 and min(p3[1],p4[1])-1e-9 <= p1[1] <= max(p3[1],p4[1])+1e-9: return True
    if d2 == 0 and min(p3[0],p4[0])-1e-9 <= p2[0] <= max(p3[0],p4[0])+1e-9 and min(p3[1],p4[1])-1e-9 <= p2[1] <= max(p3[1],p4[1])+1e-9: return True
    if d3 == 0 and min(p1[0],p2[0])-1e-9 <= p3[0] <= max(p1[0],p2[0])+1e-9 and min(p1[1],p2[1])-1e-9 <= p3[1] <= max(p1[1],p2[1])+1e-9: return True
    if d4 == 0 and min(p1[0],p2[0])-1e-9 <= p4[0] <= max(p1[0],p2[0])+1e-9 and min(p1[1],p2[1])-1e-9 <= p4[1] <= max(p1[1],p2[1])+1e-9: return True
    return False

def _hex_overlap(v1, v2, a1=0.0, a2=0.0):
    verts1 = _hex_vertices(v1[0], v1[1], 1.0, a1)
    verts2 = _hex_vertices(v2[0], v2[1], 1.0, a2)
    for v in verts1:
        if _point_in_hex(v[0], v[1], v2[0], v2[1], 1.0, a2):
            return True
    for v in verts2:
        if _point_in_hex(v[0], v[1], v1[0], v1[1], 1.0, a1):
            return True
    for i in range(6):
        e1 = [verts1[i], verts1[(i+1)%6]]
        for j in range(6):
            e2 = [verts2[j], verts2[(j+1)%6]]
            if _segments_intersect(e1, e2):
                return True
    return False

def _no_overlaps(positions, angles):
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if _hex_overlap(positions[i], positions[j], angles[i], angles[j]):
                return False
    return True

def _min_container_side(positions, angles):
    normal_angles = np.radians(30 + 60 * np.arange(6))
    max_extent = 0.0
    for i in range(len(positions)):
        verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for v in verts:
            for na in normal_angles:
                dot = v[0] * np.cos(na) + v[1] * np.sin(na)
                if dot > max_extent:
                    max_extent = dot
    return max_extent * 2 / np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster: 1 center + 6 first ring + 4 second ring
    positions = [(0.0, 0.0)]
    angles = [0.0]
    for k in range(6):
        a = np.radians(60 * k)
        positions.append((sqrt3 * np.cos(a), sqrt3 * np.sin(a)))
        angles.append(0.0)
    
    s = sqrt3
    second_ring = [
        (2*s, 0.0), (s, 1.5*s), (-s, 1.5*s), (-2*s, 0.0),
        (-s, -1.5*s), (s, -1.5*s)
    ]
    for pos in second_ring[:4]:
        positions.append(pos)
        angles.append(0.0)
    
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_side = _min_container_side(best_positions, best_angles)
    
    # Verify initial is valid
    if not _no_overlaps(best_positions, best_angles):
        # Fallback to simple linear
        best_positions = [(4.0*k, 0.0) for k in range(-(N//2), N-N//2)]
        best_angles = [0.0]*N
        best_side = 40.0
    
    np.random.seed(42)
    
    for it in range(500):
        if time.time() > deadline:
            break
        i = np.random.randint(0, N)
        dx = np.random.normal(0, 0.15)
        dy = np.random.normal(0, 0.15)
        da = np.random.normal(0, 8)
        
        new_pos = [p[:] for p in best_positions]
        new_ang = best_angles[:]
        new_pos[i][0] += dx
        new_pos[i][1] += dy
        new_ang[i] = (best_angles[i] + da) % 360
        
        if not _no_overlaps(new_pos, new_ang):
            continue
        ns = _min_container_side(new_pos, new_ang)
        if ns < best_side - 1e-7:
            best_positions = new_pos
            best_angles = new_ang
            best_side = ns
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

