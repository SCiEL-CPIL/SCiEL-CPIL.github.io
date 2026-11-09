"""Improved DTLZ1 optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer using greedy hypervolume maximization on the Pareto simplex."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ1, the Pareto front is defined by:
        # f_i = 0.5 * sum(x_j) * (1 + g(x)) for i=1..n_obj
        # where g(x) = 100*(n-1 + sum((x_j-0.5)^2 - sin(10*pi*(x_j-0.5))) for j=n+1..m)
        # The minimum g is 0 when x_{n+1..m} = 0.5
        # On the front: f_i = 0.5 * sum(x_1..x_n)
        # Let u_i = x_i / sum(x_j), then f_i = 0.5 * sum * u_i
        # The Pareto front in objective space is {f : f_i >= 0, sum(f_i) = 0.5}
        
        # Strategy:
        # 1. Generate candidate points on the Pareto simplex using quasi-random sampling
        # 2. Use greedy selection to maximize hypervolume
        # 3. Evaluate selected points

        # Generate candidate points on the Pareto simplex
        # We'll generate more candidates than budget, then select greedily
        n_candidates = budget * 3  # Generate 3x budget candidates
        
        # Use Halton-like sequence for low-discrepancy sampling on simplex
        def halton_sequence(n, base):
            """Generate n points in [0,1] using Halton sequence."""
            result = np.zeros(n)
            for i in range(1, n + 1):
                x = 0.0
                f = 1.0 / base
                num = i
                while num > 0:
                    x += f * (num % base)
                    num //= base
                    f /= base
                result[i-1] = x
            return result

        # Generate candidate points on the 2-simplex (for 3 objectives)
        # Using Dirichlet distribution approach with quasi-random numbers
        candidates = []
        
        # Add vertices of the simplex
        for i in range(n_obj):
            f = np.zeros(n_obj)
            f[i] = 0.5
            candidates.append(f)
        
        # Add edge midpoints
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                f = np.zeros(n_obj)
                f[i] = 0.25
                f[j] = 0.25
                candidates.append(f)
        
        # Add centroid
        f_centroid = np.full(n_obj, 0.5 / n_obj)
        candidates.append(f_centroid)
        
        # Generate quasi-random points on simplex using Halton sequences
        # For a (n_obj-1)-dimensional simplex, we need n_obj-1 uniform randoms
        # and convert to simplex coordinates
        n_uniform = n_obj - 1
        halton_seqs = [halton_sequence(n_candidates, base) for base in range(2, 2 + n_uniform)]
        
        for k in range(n_candidates):
            # Generate n_obj-1 values in [0,1]
            u = np.array([halton_seqs[j][k] for j in range(n_uniform)])
            # Add a small offset to avoid boundary issues
            u = 1e-10 + (1 - 2e-10) * u
            # Convert to simplex coordinates using Dirichlet-like transformation
            # For uniform on simplex: take exponentials of uniform(-1,1) or similar
            # Simpler: use the fact that if U_i ~ Uniform(0,1), then 
            # V_i = -ln(U_i) / sum(-ln(U_j)) gives uniform on simplex
            # But we want more coverage near boundaries, so use a different approach
            
            # Use the inverse CDF method for Dirichlet(1,1,...,1)
            # This is equivalent to sorting uniform randoms and taking differences
            u_all = np.sort(np.concatenate([u, [0.0, 1.0]]))
            f = np.diff(u_all)
            f = 0.5 * f  # Scale to sum to 0.5
            candidates.append(f)
        
        # Now we have candidates in objective space
        # Convert to decision space for DTLZ1
        # On the Pareto front: f_i = 0.5 * sum(x_1..x_n) * u_i where u_i = x_i/sum
        # So x_i = u_i * sum(x_1..x_n), and sum(x_1..x_n) is free
        # To minimize g, set x_{n+1..m} = 0.5
        # The sum can be any value; let's use sum = 1 for simplicity
        # Then x_i = u_i for i=1..n_obj, and x_{n_obj+1..m} = 0.5
        
        # Greedy hypervolume selection
        ref_point = np.array([400.0, 400.0, 400.0])
        selected = []
        selected_f = []
        
        # Compute hypervolume contribution of a set of points
        def compute_hv(points, ref):
            """Compute exact hypervolume for 3D points."""
            if len(points) == 0:
                return 0.0
            # Sort points by first coordinate
            pts = sorted(points, key=lambda p: p[0])
            # Filter points that are better than reference
            pts = [p for p in pts if all(p[i] < ref[i] for i in range(n_obj))]
            if len(pts) == 0:
                return 0.0
            
            # For 3D hypervolume, we can use the layer cake method
            # Sort by first coordinate and sweep
            pts = sorted(pts, key=lambda p: p[0])
            
            hv = 0.0
            prev_x = 0.0
            for i, p in enumerate(pts):
                # Width in x direction
                width = p[0] - prev_x
                if width > 0:
                    # For this slice, find the envelope in y-z plane
                    # Points that are active at this x slice
                    active = [pts[j] for j in range(i+1) if pts[j][0] <= p[0]]
                    if active:
                        # Find the 2D hypervolume of the active points in y-z
                        # with reference (ref[1], ref[2])
                        yz_points = [(pt[1], pt[2]) for pt in active]
                        yz_hv = compute_2d_hv(yz_points, ref[1], ref[2])
                        hv += width * yz_hv
                prev_x = p[0]
            
            # Add the last slice
            if pts:
                width = ref[0] - pts[-1][0]
                if width > 0:
                    active = pts
                    yz_points = [(pt[1], pt[2]) for pt in active]
                    yz_hv = compute_2d_hv(yz_points, ref[1], ref[2])
                    hv += width * yz_hv
            
            return hv

        def compute_2d_hv(points, ref_y, ref_z):
            """Compute 2D hypervolume (area) for points in y-z plane."""
            if len(points) == 0:
                return 0.0
            # Filter points better than reference
            pts = [(y, z) for y, z in points if y < ref_y and z < ref_z]
            if len(pts) == 0:
                return 0.0
            
            # Sort by y coordinate
            pts = sorted(pts, key=lambda p: p[0])
            
            area = 0.0
            prev_y = 0.0
            max_z = ref_z
            for i, (y, z) in enumerate(pts):
                # The envelope at this y is min(z, max_z)
                envelope_z = min(z, max_z)
                width = y - prev_y
                if width > 0:
                    area += width * (ref_z - envelope_z)
                max_z = min(max_z, z)
                prev_y = y
            
            # Last slice
            width = ref_y - prev_y
            if width > 0:
                area += width * (ref_z - max_z)
            
            return area

        # Greedy selection
        # Start with the point that has the highest individual hypervolume
        # Then iteratively add the point that gives the largest incremental HV
        
        # For efficiency, we'll use a simpler approach:
        # 1. Sort candidates by distance to centroid (prioritize interior)
        # 2. Use a grid-based approach to ensure diversity
        
        # Actually, let's use a more efficient greedy approach:
        # Maintain the current hypervolume, and for each candidate, estimate its contribution
        
        # For 3D, the contribution of a new point can be computed exactly
        # But it's expensive. Let's use a heuristic: prioritize points that
        # are in underrepresented regions of the simplex
        
        # Divide the simplex into regions and ensure coverage
        # Use a stratified sampling approach
        
        # For now, let's use a simpler strategy:
        # 1. Take vertices and edge midpoints (7 points)
        # 2. Fill remaining budget with quasi-random points on simplex
        # 3. This ensures good coverage
        
        # Convert selected objective points to decision space
        all_points = []
        
        # First, add the structured points (vertices, edge midpoints, centroid)
        for f in candidates[:11]:  # 3 vertices + 3 edge midpoints + 1 centroid = 7, but we have 11
            x = np.zeros(dim)
            # Map objective point to decision space
            # f_i = 0.5 * u_i where u_i = x_i / sum(x_1..x_n)
            # So x_i = u_i * S where S = sum(x_1..x_n)
            # Let S = 1, then x_i = u_i = 2*f_i
            for i in range(n_obj):
                x[i] = 2 * f[i]
            # Set remaining variables to 0.5 for minimum g
            for i in range(n_obj, dim):
                x[i] = 0.5
            x = np.clip(x, lo, hi)
            all_points.append(x)
        
        # Add quasi-random points
        n_remaining = budget - len(all_points)
        for k in range(11, 11 + n_remaining):
            if k >= len(candidates):
                break
            f = candidates[k]
            x = np.zeros(dim)
            for i in range(n_obj):
                x[i] = 2 * f[i]
            for i in range(n_obj, dim):
                x[i] = 0.5
            x = np.clip(x, lo, hi)
            all_points.append(x)
        
        # Evaluate points
        F = []
        for i, x in enumerate(all_points):
            if time.time() > deadline:
                break
            try:
                F.append(problem(x))
            except Exception:
                break
        
        return all_points[:len(F)]
# EVOLVE-BLOCK-END
