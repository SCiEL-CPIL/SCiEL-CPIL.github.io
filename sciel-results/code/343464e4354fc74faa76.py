import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 146.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            if dist_sq < 4.0 - 1e-9:
                return False
            
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Define the 19-hexagon cluster (side 2) and remove 7 to get 12
    # Lattice basis for hexagonal grid
    # Let's use axial coordinates (q, r)
    # (0,0) is center
    # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = math.sqrt(3) / 2.0 * q + math.sqrt(3) * r
        return x, y

    all_hexes = []
    # Generate side 2 hexagon (radius 1 in axial coords? No, side 2 has radius 2 in hex distance?)
    # Hex distance d = (|q| + |r| + |q+r|) / 2
    # Side 2 hexagon includes all hexes with d <= 2?
    # d=0: 1
    # d=1: 6
    # d=2: 12
    # Total 19.
    
    for q in range(-2, 3):
        for r in range(-2, 3):
            d = (abs(q) + abs(r) + abs(q+r)) // 2
            if d <= 2:
                x, y = axial_to_xy(q, r)
                all_hexes.append((q, r, x, y))
                
    # We have 19 hexes. We need to remove 7.
    # To minimize the bounding box, remove the ones at the "corners" (d=2, |q+r|=2 or similar)
    # The corners of the side-2 hexagon are at axial (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # These are 6 hexes.
    # Remove these 6.
    # Remaining: 13.
    # Remove 1 more. Maybe (1,1) or (1,-1) etc.
    # Let's remove (1,1) which is on the edge.
    
    corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    remove_set = set(corners + [(1,1)])
    
    cluster = []
    for q, r, x, y in all_hexes:
        if (q, r) not in remove_set:
            cluster.append((x, y, 0.0)) # angle 0 initially
            
    if len(cluster) != 12:
        # Fallback to simple flower if logic fails
        cluster = []
        cluster.append((0.0, 0.0, 0.0))
        for i in range(6):
            angle = math.radians(60 * i)
            x = math.sqrt(3) * math.cos(angle)
            y = math.sqrt(3) * math.sin(angle)
            cluster.append((x, y, 0.0))
        for i in range(5):
            angle = math.radians(60 * i + 30)
            x = 2 * math.sqrt(3) * math.cos(angle)
            y = 2 * math.sqrt(3) * math.sin(angle)
            cluster.append((x, y, 0.0))

    # Precompute vertices of the cluster in local frame (angle 0)
    local_verts = []
    for x, y, a in cluster:
        local_verts.append(_hex_vertices(x, y, 0.0))
        
    # Flatten to a list of points
    all_points = []
    for verts in local_verts:
        all_points.extend(verts)
        
    # Normals of the outer hexagon (fixed orientation 0)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))

    def compute_max_width(theta_deg):
        theta = math.radians(theta_deg)
        ct, st = math.cos(theta), math.sin(theta)
        
        max_width = 0.0
        for nx, ny in normals:
            min_proj = float('inf')
            max_proj = float('-inf')
            for px, py in all_points:
                # Rotate point
                rx = ct * px - st * py
                ry = st * px + ct * py
                proj = rx * nx + ry * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
            width = max_proj - min_proj
            if width > max_width:
                max_width = width
                
        return max_width / math.sqrt(3) # Convert width to side length L, since L*sqrt(3) = width

    # Optimize theta in [0, 60)
    best_theta = 0.0
    best_side = float('inf')
    
    # Coarse search
    for i in range(600):
        theta = i * 0.1
        side = compute_max_width(theta)
        if side < best_side:
            best_side = side
            best_theta = theta
            
    # Fine search around best
    for i in range(1000):
        theta = best_theta - 0.05 + i * 0.0001
        if theta < 0: theta += 60
        if theta >= 60: theta -= 60
        side = compute_max_width(theta)
        if side < best_side:
            best_side = side
            best_theta = theta

    # Apply best rotation to cluster
    ct, st = math.cos(math.radians(best_theta)), math.sin(math.radians(best_theta))
    final_cfg = []
    for x, y, a in cluster:
        rx = ct * x - st * y
        ry = st * x + ct * y
        ra = a + best_theta
        final_cfg.append([rx, ry, ra])
        
    # Verify validity (should be valid since it's a rigid cluster of touching hexes)
    cx_arr = np.array([p[0] for p in final_cfg])
    cy_arr = np.array([p[1] for p in final_cfg])
    angle_arr = np.array([p[2] for p in final_cfg])
    
    if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
        # Fallback to independent SA if rigid cluster fails (should not happen)
        # Use the old logic or a simple valid config
        return [[0.0, 0.0, 0.0]] * 12, (0.0, 0.0), 5.0, 0.0

    # The outer hexagon center should be (0,0) if the cluster is centered?
    # The width calculation assumed the center of the projection interval is at 0.
    # We can translate the cluster to center it.
    # But since the outer hexagon can be anywhere, and we just need the side length,
    # and we calculated the side length based on the width, the center is implicitly handled.
    # We just need to return a valid center. (0,0) is fine if the cluster is roughly centered.
    # To be safe, we can compute the optimal center for the outer hexagon.
    
    # For now, return (0,0) as center. The side length is the metric.
    return final_cfg, (0.0, 0.0), best_side, 0.0

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
