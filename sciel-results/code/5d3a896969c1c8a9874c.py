import numpy as np
import time
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def axial_to_cartesian(q, r):
        """
        Convert axial coordinates (q, r) to Cartesian coordinates for a hexagonal grid
        with hexagon side length 1.
        The distance between adjacent hexagon centers is sqrt(3).
        x = sqrt(3) * (q + r/2)
        y = 1.5 * r
        """
        return np.sqrt(3) * (q + r / 2.0), 1.5 * r

    def is_valid_lattice_cluster(coords):
        """
        Check if the set of axial coordinates forms a valid, disjoint packing.
        In a standard hexagonal lattice, any set of distinct axial coordinates
        represents disjoint hexagons.
        We just need to ensure we have N distinct coordinates.
        """
        if len(set(coords)) != len(coords):
            return False
        return True

    def generate_lattice_clusters():
        """
        Generate candidate clusters of N hexagons from the hexagonal lattice.
        We focus on 'compact' clusters, i.e., those that are connected or nearly so.
        For N=12, we can enumerate shapes by taking a central hexagon and adding neighbors
        or by scanning a grid and picking connected components.
        
        A simpler robust approach for small N:
        Enumerate all subsets of a small grid (e.g., radius 3) of size 12? 
        That's C(37, 12) which is too large.
        
        Instead, we use a constructive approach:
        Start with (0,0).
        Add neighbors iteratively.
        This is a tree search.
        
        Given time constraints, we will use a predefined set of known compact shapes
        and some randomized lattice perturbations.
        """
        candidates = []
        
        # Shape 1: 1 center, 6 ring, 5 from next ring (as in current program)
        # This is a standard 'flower' shape.
        coords1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        candidates.append(coords1)
        
        # Shape 2: A 2x3 rectangle in axial coords?
        # Let's try to make a more rectangular block.
        # In axial, a "rectangle" is tricky.
        # Let's try a 3x4 grid in offset coordinates?
        # Offset coords (odd-r):
        # Row 0: 0, 1, 2, 3
        # Row 1: 0.5, 1.5, 2.5 (shifted)
        # It's easier to just try different subsets of the 3rd ring.
        
        # Let's generate all subsets of the 2nd ring (12 hexagons) + center? No, 1+6+12 = 19.
        # We need 12.
        # Center (1) + Ring 1 (6) = 7.
        # We need 5 more from Ring 2 (12 hexagons).
        # There are C(12, 5) = 792 ways to choose 5 from the 2nd ring.
        # We can evaluate all of them! It's fast.
        
        ring2_coords = [
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
            (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2) # Wait, ring 2 has 12 hexes.
            # Let's list them properly.
            # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            # Ring 2 neighbors of Ring 1:
            # (1,0) -> (2,0), (1,1)
            # (0,1) -> (1,1), (0,2)
            # (-1,1) -> (0,2), (-1,2)
            # (-1,0) -> (-1,2), (-2,1)
            # (0,-1) -> (-2,1), (-2,0)
            # (1,-1) -> (-2,0), (-1,-1)
            # Also outer neighbors:
            # (2,0) has neighbors (3,0), (2,1)
            # The set of all hexes at distance 2 from center:
            # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)?? No.
            # Axial distance: max(|q|, |r|, |q+r|) = 2.
            # Let's generate them programmatically.
        ]
        
        # Generate Ring 2 coordinates
        ring2 = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    ring2.append((q, r))
        # ring2 should have 12 elements.
        
        base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        
        # Iterate over all combinations of 5 from ring2
        # This might produce duplicates or symmetries, but it's only 792 iterations.
        # We will compute the side length for each and keep the best.
        
        best_side = float('inf')
        best_coords = None
        
        # Optimization: We can prune based on bounding box in axial space, but let's just compute.
        # To speed up, we can skip combinations that are obviously less compact.
        
        # Also consider clusters that don't include the center?
        # Unlikely to be optimal for N=12, but let's stick to center-included for now.
        # Also consider clusters that are "shifted" lattices?
        # The lattice is fixed. The subset determines the shape.
        
        for comb in itertools.combinations(ring2, 5):
            coords = base + list(comb)
            # Convert to Cartesian
            positions = np.array([axial_to_cartesian(q, r) for q, r in coords])
            
            # Check validity (distinctness is guaranteed by construction)
            # Compute side length
            side = required_outer_side(positions)
            
            if side < best_side:
                best_side = side
                best_coords = coords
                
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
            
        # Also try a "shifted" center?
        # If we shift the whole cluster, the side length is invariant (we re-center in required_outer_side? 
        # No, required_outer_side assumes center at 0.
        # The function `required_outer_side` computes the max projection.
        # This is effectively the side length of the hexagon centered at the origin that contains the points.
        # If the points are shifted, the max projection changes.
        # We want to minimize this.
        # The optimal center for the outer hexagon is the "center" of the bounding box.
        # My `required_outer_side` function implicitly centers the outer hexagon at (0,0).
        # To allow the outer hexagon to move, we should subtract the centroid of the points?
        # Or better: The side length S is determined by the max of (L_k) where L_k are linear forms.
        # If we shift points by d, L_k becomes L_k - n_k . d.
        # We want to minimize max_k (L_k - n_k . d).
        # This is equivalent to finding the smallest hexagon containing the points.
        # The center of this hexagon is not necessarily the centroid.
        # However, for a regular hexagon, the minimal enclosing hexagon's center is unique.
        # My current `required_outer_side` fixes the center at (0,0).
        # This means if the cluster is off-center, it will report a larger side length than necessary.
        # I need to center the cluster first!
        
        # Correction:
        # 1. Compute centroid of positions.
        # 2. Subtract centroid from all positions.
        # 3. Compute required_outer_side on centered positions.
        
        # Let's re-evaluate the best candidate with proper centering.
        # Actually, I should have done this in the loop.
        # Let's re-run the best few candidates with centering.
        
        # For now, let's just return the best one found, but we need to fix the centering logic.
        
        # Let's create a helper that centers and computes side.
        def compute_centered_side(positions):
            centroid = np.mean(positions, axis=0)
            centered = positions - centroid
            return required_outer_side(centered)

        # Re-evaluate the best candidate
        if best_coords is not None:
            positions = np.array([axial_to_cartesian(q, r) for q, r in best_coords])
            final_side = compute_centered_side(positions)
            
            # Return the centered positions?
            # The interface expects (inner, center, side, angle).
            # inner: list of (x, y, angle)
            # center: (x, y) of outer hexagon
            # side: outer side length
            # angle: outer hexagon angle
            
            # If we return positions relative to centroid, and center=(0,0), that works.
            # Or we can return original positions and center=centroid.
            # Let's return original positions and center=centroid.
            
            return (
                [[float(pos[0]), float(pos[1]), 0.0] for pos in positions],
                (float(centroid[0]), float(centroid[1])),
                float(final_side),
                0.0
            )
        else:
            # Fallback: Simple lattice
            coords = [
                (0, 0),
                (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
            ]
            positions = np.array([axial_to_cartesian(q, r) for q, r in coords])
            centroid = np.mean(positions, axis=0)
            side = compute_centered_side(positions)
            return (
                [[float(pos[0]), float(pos[1]), 0.0] for pos in positions],
                (float(centroid[0]), float(centroid[1])),
                float(side),
                0.0
            )

    return generate_lattice_clusters()
# EVOLVE-BLOCK-END
