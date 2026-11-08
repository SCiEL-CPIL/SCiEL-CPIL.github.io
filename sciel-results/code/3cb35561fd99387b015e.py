import numpy as np
from scipy.optimize import minimize_scalar

N = 12

def construct():
    # Define candidate shapes
    # Shape 1: 3x4 Parallelogram
    # Shape 2: Compact Hexagonal Cluster (7 + 5)
    # Shape 3: 2x6 Parallelogram (likely worse)
    # Shape 4: 4x3 Parallelogram (same as 1 but rotated)

    candidates = []

    # Helper to generate 3x4 parallelogram centers
    def get_3x4():
        centers = []
        for i in range(3):
            for j in range(4):
                x = j * np.sqrt(3) + (i % 2) * (np.sqrt(3) / 2)
                y = i * 1.5
                centers.append((x, y))
        return centers

    # Helper to generate compact cluster (1 center, 6 ring1, 5 ring2)
    def get_compact():
        centers = [(0, 0)]
        # Ring 1
        for k in range(6):
            angle = np.pi / 3 * k
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            centers.append((x, y))
        # Ring 2: Add 5 hexagons. 
        # The 12 positions in ring 2 are at angles k*pi/3 and k*pi/3 + pi/6?
        # Actually, ring 2 has 12 spots. 6 are "aligned" with ring 1, 6 are "between".
        # To keep it compact, we should pick the 5 that are closest to the existing shape or fill gaps.
        # Let's just pick 5 consecutive ones in ring 2 to maintain symmetry as much as possible.
        for k in range(5):
            angle = np.pi / 3 * k
            x = 2 * np.sqrt(3) * np.cos(angle)
            y = 2 * np.sqrt(3) * np.sin(angle)
            centers.append((x, y))
        return centers

    candidates.append(get_3x4())
    candidates.append(get_compact())

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for centers in candidates:
        centers = np.array(centers)
        # Center the cluster at origin for easier calculation
        center_of_mass = np.mean(centers, axis=0)
        centers = centers - center_of_mass
        
        # We need to find the minimal enclosing regular hexagon for these centers, 
        # considering the "radius" of the inner hexagons is 1.
        # The condition is that the distance from the outer center to any vertex of any inner hexagon 
        # must be <= the circumradius of the outer hexagon? 
        # No, the inner hexagon must be *inside*.
        # This is equivalent to: The outer hexagon must contain the Minkowski sum of the inner hexagons.
        # Since we can translate the whole cluster, we fix the cluster at origin and find the best outer center/rotation.
        # Actually, the outer center will likely coincide with the cluster center for symmetric shapes.
        
        # Let's optimize the rotation of the outer hexagon.
        # The side length S required is determined by the max projection of the "expanded" cluster onto the 6 normals of the outer hexagon.
        # The "expanded" cluster is the union of inner hexagons.
        # To simplify, we can approximate the inner hexagons as circles of radius 1? 
        # No, that's not tight.
        # Better: The distance from the outer center (0,0) to the farthest point of the inner cluster in direction $\phi$ must be $\le S \cos(\phi - \theta - \pi/6)$? 
        # Let's use the property: A regular hexagon with side S and rotation $\theta$ contains a point $p$ if:
        # $\max_{k=0..5} (\text{dot}(p, n_k)) \le S \frac{\sqrt{3}}{2}$ where $n_k$ are the unit normals to the sides.
        
        # We want to minimize S.
        # S is determined by the max over all points $p$ in the union of inner hexagons, and all normals $n_k$, of dot(p, n_k).
        # So $S(\theta) = \frac{2}{\sqrt{3}} \max_{k} \max_{p \in \text{Union}} \text{dot}(p, n_k(\theta))$.
        
        # The max over $p$ in a convex polygon (inner hexagon) of dot(p, n) is dot(vertex, n).
        # So we just need to check the vertices of all inner hexagons.
        
        # Generate vertices for all inner hexagons
        # Inner hexagons are unit side, rotated 0 (flat top) relative to local frame.
        # We assume all inner hexagons are aligned with each other.
        
        all_vertices = []
        for c in centers:
            # Vertices of unit hexagon at c
            for i in range(6):
                angle = np.pi / 3 * i
                vx = c[0] + np.cos(angle)
                vy = c[1] + np.sin(angle)
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        def cost(theta):
            # Normals of outer hexagon
            # Outer hexagon rotation theta.
            # Normals are at theta + pi/6 + k*pi/3?
            # If flat-top, normals are at 0, 60, 120...
            # Let's say normal k is at angle alpha_k = theta + k * pi / 3.
            # The inradius constraint is: max dot(p, n_k) <= R_in = S * sqrt(3) / 2.
            # So S = 2/sqrt(3) * max_k max_p dot(p, n_k).
            max_proj = 0
            for k in range(6):
                alpha = theta + k * np.pi / 3
                nx = np.cos(alpha)
                ny = np.sin(alpha)
                projs = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
                max_proj = max(max_proj, np.max(projs))
            return (2.0 / np.sqrt(3.0)) * max_proj

        # Optimize theta
        res = minimize_scalar(cost, bounds=(0, np.pi / 3), method='bounded')
        optimal_theta = res.x
        optimal_side = res.fun
        
        if optimal_side < best_side:
            best_side = optimal_side
            best_inner = []
            for c in centers:
                # Return coordinates relative to the final outer center
                # We centered the cluster at origin, so outer center is (0,0)
                # Angle of inner hexagons is 0
                best_inner.append([c[0], c[1], 0.0])
            best_center = (0.0, 0.0)
            best_angle = optimal_theta

    return best_inner, best_center, best_side, best_angle
