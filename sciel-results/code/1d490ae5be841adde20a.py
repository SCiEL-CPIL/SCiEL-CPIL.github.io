"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def hex_vertex_angles(angle_deg):
        """Return the 6 vertex angles of a regular hexagon with given rotation."""
        base = np.deg2rad(angle_deg)
        return base + np.arange(6) * np.pi / 3.0
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side length and rotation."""
        dx, dy = px - cx, py - cy
        # Transform to hexagon's local frame
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        lx = dx * cos_t + dy * sin_t
        ly = -dx * sin_t + dy * cos_t
        
        # For a regular hexagon with side s, the vertices are at distance s from center
        # at angles 0, 60, 120, 180, 240, 300 degrees
        # The hexagon is the intersection of 6 half-planes
        # For each edge, the normal points inward, and the distance from center to edge is s*sqrt(3)/2
        inradius = side * math.sqrt(3) / 2.0
        
        for i in range(6):
            edge_angle = i * np.pi / 3.0
            nx = math.cos(edge_angle)
            ny = math.sin(edge_angle)
            # The edge is at distance inradius from center in direction (nx, ny)
            # Point must satisfy: lx*nx + ly*ny <= inradius
            if lx * nx + ly * ny > inradius + 1e-10:
                return False
        return True
    
    def hexagon_contains_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
        """Check if outer hexagon contains inner hexagon (all 6 vertices inside)."""
        # Get inner hexagon vertices
        inner_theta = np.deg2rad(inner_angle)
        for i in range(6):
            va = inner_theta + i * np.pi / 3.0
            vx = inner_cx + math.cos(va)
            vy = inner_cy + math.sin(va)
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                return False
        return True
    
    def compute_min_outer_side(inner_positions, inner_angles, outer_angle, outer_cx=0.0, outer_cy=0.0):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        # The outer hexagon must contain all vertices of all inner hexagons
        # For a given outer angle, the constraint is:
        # For each edge i of outer hexagon, max over all inner vertices of (dot product with edge normal) <= outer_inradius
        # outer_inradius = outer_side * sqrt(3)/2
        # So outer_side >= (max_dot) / (sqrt(3)/2) = 2 * max_dot / sqrt(3)
        
        outer_theta = np.deg2rad(outer_angle)
        max_dot = 0.0
        
        for (icx, icy, iangle) in zip(inner_positions, inner_angles):
            inner_theta = np.deg2rad(iangle)
            for vi in range(6):
                va = inner_theta + vi * np.pi / 3.0
                vx = icx + math.cos(va)
                vy = icy + math.sin(va)
                dx, dy = vx - outer_cx, vy - outer_cy
                # Transform to outer hexagon local frame
                cos_t, sin_t = np.cos(outer_theta), np.sin(outer_theta)
                lx = dx * cos_t + dy * sin_t
                ly = -dx * sin_t + dy * cos_t
                for ei in range(6):
                    edge_angle = ei * np.pi / 3.0
                    nx = math.cos(edge_angle)
                    ny = math.sin(edge_angle)
                    dot = lx * nx + ly * ny
                    if dot > max_dot:
                        max_dot = dot
        
        # outer_inradius = max_dot, so outer_side = 2 * max_dot / sqrt(3)
        return 2.0 * max_dot / math.sqrt(3.0)
    
    def generate_hex_lattice_packing(n_hex):
        """Generate hexagonal lattice positions for n_hex unit hexagons."""
        # Hexagonal lattice: positions at (i*sqrt(3), j*1.5) with offset
        # For a hexagon of radius r, positions are:
        # dx = sqrt(3) * (i + j/2), dy = 1.5 * j
        # for i, j in range with |i| + |j| + |i+j| <= 2r ... 
        # Actually for hexagonal numbering:
        # The hexagonal lattice points within distance r from center
        
        positions = []
        # Generate all lattice points in a hexagonal region
        # Using axial coordinates (q, r)
        for q in range(-3, 4):
            for r in range(-3, 4):
                s = -q - r
                if abs(q) <= 3 and abs(r) <= 3 and abs(s) <= 3:
                    # Convert to cartesian
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    positions.append((x, y))
        
        # Sort by distance from center and take first n
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return positions[:n_hex]
    
    # Try different packing configurations
    best_score = float('inf')
    best_result = None
    
    # Configuration 1: Hexagonal lattice, aligned
    lattice_pos = generate_hex_lattice_packing(N)
    inner_angles = [0.0] * N
    
    # Try different outer angles
    for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        side = compute_min_outer_side(lattice_pos, inner_angles, outer_angle)
        if side < best_score:
            best_score = side
            best_result = ([(x, y, 0.0) for x, y in lattice_pos], (0.0, 0.0), side, outer_angle)
    
    # Configuration 2: Try rotating inner hexagons
    for inner_angle in [0.0, 30.0, 60.0]:
        inner_angles = [inner_angle] * N
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_min_outer_side(lattice_pos, inner_angles, outer_angle)
            if side < best_score:
                best_score = side
                best_result = ([(x, y, inner_angle) for x, y in lattice_pos], (0.0, 0.0), side, outer_angle)
    
    # Configuration 3: Different lattice offsets - try shifting the lattice
    for dx_shift in [-0.5, 0.0, 0.5]:
        for dy_shift in [-0.5, 0.0, 0.5]:
            shifted_pos = [(x + dx_shift, y + dy_shift) for x, y in lattice_pos]
            for outer_angle in [0.0, 30.0, 60.0]:
                side = compute_min_outer_side(shifted_pos, inner_angles, outer_angle)
                if side < best_score:
                    best_score = side
                    best_result = ([(x, y, 0.0) for x, y in shifted_pos], (0.0, 0.0), side, outer_angle)
    
    # Check deadline
    if time.time() > deadline:
        if best_result:
            inner, center, side, angle = best_result
            return inner, center, side, angle
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    if best_result:
        inner, center, side, angle = best_result
        return inner, center, side, angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

