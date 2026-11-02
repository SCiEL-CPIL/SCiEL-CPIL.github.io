"""
Kinematic closure strategy:
- Place 12 unit hexagons in a 3x4 rectangular lattice (a well-known dense packing).
- The cluster has a fixed shape; we rotate the entire cluster by angle θ and scale it by factor s.
- The outer hexagon is centered at the centroid of the cluster, with side length L and rotation φ.
- We minimize L over (θ, s) where s is the radial scaling (actually, for fixed unit hexagons,
  we can't scale the inner hexagons—they must be unit. So instead, we fix the inner hexagons
  as unit, and the "scaling" is really about how we arrange them. Let me reconsider.)

Actually, the inner hexagons must be unit (side length 1). So we cannot scale them.
The parameterization should be:
- A fixed arrangement of 12 unit hexagons (e.g., 3x4 lattice with proper spacing)
- Global rotation θ of the entire cluster
- The outer hexagon has center (0,0) [by symmetry, we can center the cluster at origin],
  side length L, and rotation φ.

To minimize L, for a given cluster configuration (positions of 12 unit hexagons, each with its own orientation),
we need to find the minimum L such that all 12 hexagons fit inside the outer hexagon.

For a regular hexagon with center c, side length L, and rotation φ, the interior is the intersection of 6 half-planes.
A point p is inside if for each of the 6 sides, the distance from p to the side (measured inward) is >= 0.
For a hexagon to be inside, all 6 vertices of each inner hexagon must be inside the outer hexagon.

But checking all 72 vertices (12 hexagons × 6 vertices) against 6 half-planes is feasible.

Strategy:
1. Define a base cluster of 12 unit hexagons in a 3×4 grid pattern.
   - In a hexagonal lattice, adjacent hexagons of side 1 are separated by distance √3 in the "flat" direction.
   - For a 3×4 rectangular array: 
     - Horizontal spacing: 3√3 (for flat-top hexagons) or 2 (for pointy-top)? 
     - Let's use flat-top hexagons with side 1.
     - Flat-top hexagon: width = 2, height = √3.
     - Horizontal spacing between centers of adjacent columns: 3/2 * 2 = 3? No.
     - For flat-top: horizontal distance between adjacent hexagons = 3/2 * side * 2 = 3? 
     - Actually, for regular hexagons of side s, the distance between centers of two adjacent hexagons (sharing an edge) is s * √3.
     - In a rectangular grid of flat-top hexagons:
       - Column spacing: s * 3/2 * 2 = 3s? No.
       - Let me compute: a flat-top hexagon has vertices at angles 0°, 60°, 120°, 180°, 240°, 300° from center.
       - Width (x-direction) = 2s, height (y-direction) = s√3.
       - For a grid where hexagons share edges:
         - Horizontal neighbor: shift by (3s/2, s√3/2) or (3s/2, -s√3/2)? 
         - Actually, for flat-top, adjacent hexagons in the same row are spaced by 3s horizontally? No.
         
Let me just use a known dense packing: 12 hexagons in a 3×4 block.
For unit side hexagons in a hexagonal lattice:
- The lattice vectors are a1 = (3/2, √3/2) * 2? 
- Standard: for hexagons of side 1, the center-to-center distance for edge-sharing is √3.
- Lattice vectors: a1 = (3/2, √3/2), a2 = (3/2, -√3/2)? No, that's not right either.

Simpler: I'll place 12 hexagons in a 3×4 grid where:
- Each hexagon is flat-top, side 1.
- Row spacing: √3 (vertical distance between rows in a hex lattice)
- Column spacing: 3 (horizontal distance)
- Odd rows offset by 1.5

Actually, let me just compute positions for a 3×4 rectangular array of flat-top hexagons:
- 3 columns, 4 rows.
- Column i (i=0,1,2): x = i * 3
- Row j (j=0,1,2,3): y = j * √3
- If j is odd, offset x by 1.5

So positions:
(0,0), (3,0), (6,0)
(1.5, √3), (4.5, √3), (7.5, √3)
(0, 2√3), (3, 2√3), (6, 2√3)
(1.5, 3√3), (4.5, 3√3), (7.5, 3√3)

Center of this cluster: x = (0+3+6+1.5+4.5+7.5+0+3+6+1.5+4.5+7.5)/12 = (36+24)/12 = 60/12 = 5
y = (0+0+0+√3+√3+√3+2√3+2√3+2√3+3√3+3√3+3√3)/12 = (0+3√3+6√3+9√3)/12 = 18√3/12 = 1.5√3

So I'll center the cluster at origin by subtracting (5, 1.5√3).

Then rotate the entire cluster by θ, and find the minimum L and φ for the outer hexagon.

For efficiency, I'll do:
- For a given θ, compute the rotated positions of all 12 hexagons (and their vertices).
- Then find the minimum L and φ such that all 72 vertices are inside the outer hexagon.

For a fixed set of points, the minimum enclosing regular hexagon (with arbitrary rotation) is a 2D optimization problem.
I'll use ternary search on φ (rotation of outer hexagon) and for each φ, compute the minimum L needed.

For a given φ, the outer hexagon has 6 sides. The minimum L is determined by the maximum over all 6 directions of the support function of the point set in that direction, divided by the apothem-to-side ratio.

For a regular hexagon with side L, the apothem (distance from center to each side) is L * √3/2.
So for a direction u (unit vector normal to a side), the constraint is: max_i (p_i · u) <= L * √3/2 (assuming center at origin).
Thus L >= max_i (p_i · u) / (√3/2) = max_i (p_i · u) * 2/√3.

For a given φ, the 6 normals are at angles φ, φ+60°, φ+120°, φ+180°, φ+240°, φ+300°.
L(φ) = max over these 6 directions of (max_i (p_i · u_k)) * 2/√3.

We want min_φ L(φ).

This is a 1D optimization problem in φ, which is periodic with period 60°. We can use ternary search on [0, π/3].

For each θ, we do this 1D optimization. Then we optimize over θ as well, but θ is also 1D (period 60° due to hexagonal symmetry of the cluster? Not exactly, since the cluster is 3×4, not symmetric).

Actually, the cluster has some symmetry, but let's just do a 2D grid search or nested ternary search.

Given time constraints, I'll do:
- Coarse grid search over θ in [0, π/3] with step π/36 (10°)
- For each θ, ternary search over φ in [0, π/3] to find min L
- Return the best (θ, φ, L)

Let me implement this.
"""

import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Base cluster: 3x4 grid of flat-top unit hexagons
    # Positions before centering
    positions = []
    for j in range(4):  # rows
        for i in range(3):  # columns
            x = i * 3.0
            y = j * np.sqrt(3)
            if j % 2 == 1:
                x += 1.5
            positions.append([x, y])
    
    positions = np.array(positions)
    
    # Center the cluster
    centroid = positions.mean(axis=0)
    positions -= centroid
    
    # Function to compute vertices of a flat-top unit hexagon at (cx, cy) with rotation angle_deg
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = np.radians(angle_deg)
        # Flat-top hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
        angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + angle_rad
        return np.stack([cx + np.cos(angles), cy + np.sin(angles)], axis=1)
    
    # For a given global rotation theta, compute all 72 vertices
    def get_all_vertices(theta):
        all_verts = []
        for pos in positions:
            # Rotate position by theta
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rx = pos[0] * cos_t - pos[1] * sin_t
            ry = pos[0] * sin_t + pos[1] * cos_t
            # Each inner hexagon is also rotated by theta (since the whole cluster rotates)
            verts = hex_vertices(rx, ry, np.degrees(theta))
            all_verts.append(verts)
        return np.vstack(all_verts)  # 72 x 2
    
    # For a given phi (outer hexagon rotation), compute minimum L needed
    def min_L_for_phi(vertices, phi):
        # 6 normals at phi, phi+60, ..., phi+300
        angles = np.radians(phi + np.array([0, 60, 120, 180, 240, 300]))
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # 6 x 2
        
        # For each normal, compute max dot product with vertices
        # vertices: 72 x 2, normals: 6 x 2
        dots = vertices @ normals.T  # 72 x 6
        max_dots = dots.max(axis=0)  # 6
        
        # L = max(max_dots) * 2/sqrt(3)
        return max_dots.max() * 2.0 / np.sqrt(3)
    
    # Ternary search for phi in [0, pi/3]
    def ternary_search_phi(vertices):
        lo, hi = 0.0, np.pi / 3
        for _ in range(50):  # 50 iterations for good precision
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            f1 = min_L_for_phi(vertices, np.degrees(m1))
            f2 = min_L_for_phi(vertices, np.degrees(m2))
            if f1 < f2:
                hi = m2
            else:
                lo = m1
        phi_opt = (lo + hi) / 2
        return min_L_for_phi(vertices, np.degrees(phi_opt)), np.degrees(phi_opt)
    
    # Grid search over theta
    best_L = float('inf')
    best_theta = 0.0
    best_phi = 0.0
    
    n_theta = 18  # 10-degree steps over 0 to 180 degrees (but period is 360 for the cluster, but hex symmetry might reduce)
    for k in range(n_theta):
        theta = 2 * np.pi * k / n_theta
        vertices = get_all_vertices(theta)
        L, phi = ternary_search_phi(vertices)
        if L < best_L:
            best_L = L
            best_theta = theta
            best_phi = phi
    
    # Construct inner hexagons at optimal theta
    inner = []
    for pos in positions:
        cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)
        rx = pos[0] * cos_t - pos[1] * sin_t
        ry = pos[0] * sin_t + pos[1] * cos_t
        inner.append([rx, ry, best_phi])  # Each inner hexagon has same rotation as outer? 
        # Wait, the inner hexagons are rotated by best_theta (the cluster rotation), not best_phi.
        # Let me fix: inner hexagons are rotated by best_theta.
    
    # Actually, let me redo: the inner hexagons' rotation is best_theta, not best_phi.
    inner = []
    for pos in positions:
        cos_t, sin_t = np.cos(best_theta), np.sin(best_theta)
        rx = pos[0] * cos_t - pos[1] * sin_t
        ry = pos[0] * sin_t + pos[1] * cos_t
        inner.append([rx, ry, np.degrees(best_theta)])
    
    return inner, (0.0, 0.0), best_L, best_phi
# EVOLVE-BLOCK-END
