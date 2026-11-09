"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.spatial import cKDTree

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Hexagon geometry helpers
    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle."""
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a

        # For a regular hexagon with circumradius R=side, inradius = side * sqrt(3)/2
        # The 6 normals are at angles 30 + 60*k degrees
        inradius = side * math.sqrt(3) / 2
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= inradius + 1e-9

    def outer_hexagon_contains_inner(px, py, angle_deg, ocx, ocy, oangle_deg, oside):
        """Check if inner hexagon at (px,py) with angle is inside outer hexagon."""
        verts = hexagon_vertices(px, py, angle_deg)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, oangle_deg, oside):
                return False
        return True

    def check_valid(inner, ocx, ocy, oside, oangle_deg):
        """Check if all inner hexagons are inside outer and don't overlap."""
        # Check containment
        for x, y, a in inner:
            if not outer_hexagon_contains_inner(x, y, a, ocx, ocy, oangle_deg, oside):
                return False

        # Check non-overlap
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Conservative: require distance >= 2.0 for same orientation
                if dist < 2.0 - 1e-6:
                    return False
        return True

    def compute_min_outer_side(positions, angle_deg=0.0):
        """
        Compute the minimum outer hexagon side length to contain all inner hexagons
        (each of side 1) centered at given positions.
        Uses the fact that a regular hexagon of side S contains a point at distance d
        from center if d <= S (circumradius).
        But we need the inner hexagon vertices to be inside.
        
        For a set of inner hexagons all with same orientation as outer,
        the outer hexagon of side S centered at (cx,cy) contains an inner hexagon
        centered at (px,py) if all 6 vertices of the inner hexagon are inside.
        
        A simpler approach: the outer hexagon must contain the convex hull of all
        inner hexagon vertices. We can compute the minimum enclosing hexagon.
        
        For now, use a practical approach: try different centers and orientations,
        and for each, compute the minimum S needed.
        """
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # The outer hexagon needs to contain all inner hexagon vertices
        # For a given outer center (ocx, ocy) and orientation oangle,
        # the minimum side S is determined by the farthest vertex in each of the 6 directions.
        
        # For efficiency, try a few candidate centers
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # For a regular hexagon with side S, the constraint for a point (px, py) to be inside
        # is: for each of the 6 normal directions theta_k = oangle + 30 + 60*k,
        # (px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k) <= S * sqrt(3)/2
        
        # So S >= max over all points and all k of [(px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k)] / (sqrt(3)/2)
        # where we take the maximum of the absolute values (since both sides matter)
        
        # Actually, the constraint is:
        # (px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k) <= S * sqrt(3)/2  for all k
        # and
        # -(px-ocx)*cos(theta_k) - (py-ocy)*sin(theta_k) <= S * sqrt(3)/2  for all k
        # i.e., |(px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k)| <= S * sqrt(3)/2
        
        # So S >= max over all points p and all k of |dot(p - c, n_k)| / (sqrt(3)/2)
        
        # But we also need to account for the inner hexagons having size.
        # The inner hexagon centered at (px,py) with orientation iangle has vertices
        # at distance 1 from (px,py). So the effective "point" to check is actually
        # the 6 vertices of each inner hexagon.
        
        # For simplicity and speed, let's approximate: the inner hexagon extends to
        # distance 1 from its center. So we can treat each inner hexagon as a circle
        # of radius 1 and compute the minimum enclosing hexagon for these circles.
        
        # The minimum S such that all circles of radius 1 centered at positions are
        # inside the outer hexagon of side S centered at (cx,cy) with orientation oangle:
        # S >= max over all positions p and all k of |dot(p - c, n_k)| / (sqrt(3)/2) + 1
        # Wait, that's not quite right either. The circle of radius 1 is inside the
        # hexagon if the distance from the center of the circle to each side is >= 1.
        # The distance from (px,py) to the side with normal n_k is:
        # S * sqrt(3)/2 - ((px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k))
        # This must be >= 1, so:
        # S * sqrt(3)/2 >= (px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k) + 1
        # Similarly for the other direction:
        # S * sqrt(3)/2 >= -(px-ocx)*cos(theta_k) - (py-ocy)*sin(theta_k) + 1
        # So S >= max over p,k of [(px-ocx)*cos(theta_k) + (py-ocy)*sin(theta_k) + 1] / (sqrt(3)/2)
        # and S >= max over p,k of [-(px-ocx)*cos(theta_k) - (py-ocy)*sin(theta_k) + 1] / (sqrt(3)/2)
        # i.e., S >= max over p,k of |dot(p-c, n_k)| / (sqrt(3)/2) + 1
        
        # But this is a sufficient condition, not necessary. The actual condition is
        # that all 6 vertices of each inner hexagon are inside the outer hexagon.
        
        # Let's use the vertex-based check for accuracy, but optimize the center.
        
        best_s = float('inf')
        best_center = (cx, cy)
        
        # Try a few candidate centers: centroid, and points offset in various directions
        candidate_centers = [(cx, cy)]
        # Add some offsets
        for dx in [-1, -0.5, 0, 0.5, 1]:
            for dy in [-1, -0.5, 0, 0.5, 1]:
                candidate_centers.append((cx + dx, cy + dy))
        
        # Try a few orientations for the outer hexagon
        for oangle in [0, 30, 60]:
            for ocx, ocy in candidate_centers:
                # Compute required S
                required_s = 0
                for px, py in positions:
                    # Get vertices of inner hexagon
                    for vx, vy in hexagon_vertices(px, py, angle_deg):
                        # Check against each side of outer hexagon
                        for k in range(6):
                            theta = math.radians(oangle + 30 + 60 * k)
                            dot = (vx - ocx) * math.cos(theta) + (vy - ocy) * math.sin(theta)
                            # The constraint is dot <= S * sqrt(3)/2
                            # So S >= dot / (sqrt(3)/2)
                            s_needed = dot / (math.sqrt(3) / 2)
                            if s_needed > required_s:
                                required_s = s_needed
                
                if required_s < best_s:
                    best_s = required_s
                    best_center = (ocx, ocy)
        
        return max(best_s, 1.0), best_center

    # Generate candidate positions in a hexagonal grid pattern
    # Triangular lattice with spacing 2.0
    max_radius = 20
    positions = []
    for r in range(-max_radius, max_radius + 1):
        for c in range(-max_radius, max_radius + 1):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * math.sqrt(3)
            if x * x + y * y <= (max_radius * 2) ** 2:
                positions.append((x, y))

    # Sort by distance from origin
    positions.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])

    best_result = None
    best_score = float('inf')

    import random
    random.seed(42)

    # Strategy 1: Greedy placement with randomization
    for attempt in range(500):
        if time.time() > deadline:
            break

        # Create a compact arrangement
        placed = [(0.0, 0.0)]

        # Add neighbors in order of distance, with some randomization
        candidate_positions = positions.copy()
        # Shuffle but keep close ones more likely to be considered first
        random.shuffle(candidate_positions)

        for x, y in candidate_positions:
            if len(placed) >= N:
                break
            # Check if this position is far enough from all placed
            valid = True
            for px, py in placed:
                d = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if d < 2.0 - 1e-6:
                    valid = False
                    break
            if valid:
                placed.append((x, y))

        if len(placed) < N:
            continue

        # Compute outer hexagon
        oside, (ocx, ocy) = compute_min_outer_side(placed, angle_deg=0.0)

        # Verify
        inner = [[x, y, 0.0] for x, y in placed]
        if check_valid(inner, ocx, ocy, oside, 0.0):
            if oside < best_score:
                best_score = oside
                best_result = (inner, (ocx, ocy), oside, 0.0)

    # Strategy 2: Try specific known-good patterns
    # Pattern: 1 center + 6 neighbors + 4 more
    # The 7-hex cluster has centers at:
    # (0,0), (2,0), (1,sqrt(3)), (-1,sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
    # Then add 4 more at distance 2 from existing ones

    sqrt3 = math.sqrt(3)
    base7 = [(0, 0), (2, 0), (1, sqrt3), (-1, sqrt3), (-2, 0), (-1, -sqrt3), (1, -sqrt3)]

    # Try adding 4 more positions
    extra_candidates = []
    for r in range(-5, 6):
        for c in range(-5, 6):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * sqrt3
            # Check if not in base7
            in_base = False
            for bx, by in base7:
                if abs(x - bx) < 1e-9 and abs(y - by) < 1e-9:
                    in_base = True
                    break
            if not in_base:
                # Check if far enough from all base7
                valid = True
                for bx, by in base7:
                    d = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
                    if d < 2.0 - 1e-6:
                        valid = False
                        break
                if valid:
                    extra_candidates.append((x, y))

    extra_candidates.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])

    # Try different combinations of 4 extra positions
    from itertools import combinations
    if len(extra_candidates) >= 4:
        for combo in combinations(extra_candidates[:20], 4):
            if time.time() > deadline:
                break

            placed = base7 + list(combo)
            # Check non-overlap among extras
            valid = True
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    dx = placed[i][0] - placed[j][0]
                    dy = placed[i][1] - placed[j][1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 2.0 - 1e-6:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            oside, (ocx, ocy) = compute_min_outer_side(placed, angle_deg=0.0)
            inner = [[x, y, 0.0] for x, y in placed]
            if check_valid(inner, ocx, ocy, oside, 0.0):
                if oside < best_score:
                    best_score = oside
                    best_result = (inner, (ocx, ocy), oside, 0.0)

    # Strategy 3: Local optimization - perturb positions slightly
    if best_result is not None and time.time() < deadline - 10:
        inner, (ocx, ocy), oside, oangle = best_result
        # Try small perturbations
        for perturb_attempt in range(100):
            if time.time() > deadline:
                break

            new_inner = [[x + random.uniform(-0.1, 0.1), y + random.uniform(-0.1, 0.1), 0.0] for x, y, a in inner]

            # Check non-overlap
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dx = new_inner[i][0] - new_inner[j][0]
                    dy = new_inner[i][1] - new_inner[j][1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < 2.0 - 1e-6:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            positions = [(x, y) for x, y, a in new_inner]
            new_oside, (new_ocx, new_ocy) = compute_min_outer_side(positions, angle_deg=0.0)

            if check_valid(new_inner, new_ocx, new_ocy, new_oside, 0.0):
                if new_oside < best_score:
                    best_score = new_oside
                    best_result = (new_inner, (new_ocx, new_ocy), new_oside, 0.0)

    if best_result is None:
        # Fallback: use a simple arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result
# EVOLVE-BLOCK-END
