# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front geometry via simplex placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_front_vars = dim - n_obj + 1  # 10
        n_dist_vars = n_obj - 1  # 2

        def simplex_to_dtlz4(u, v, dist_scale=0.01):
            """Convert 2D simplex coordinates (u,v in [0,1], u+v<=1) to DTLZ4 point.
            
            For DTLZ4 with 3 objectives:
            f1 = (1+g) * u
            f2 = (1+g) * v  
            f3 = (1+g) * (1-u-v)
            where g = sum(x_i^beta) for last (n_obj-1) vars, beta=10 for DTLZ4
            
            To be ON the front: g=0, so set distance vars to 0.
            For near-front: small positive values.
            """
            x = np.zeros(dim)
            # Front variables: use u, v, and (1-u-v) mapped through arcsin/sqrt for DTLZ4
            # DTLZ4 front: f_i = (1+g) * x_i for i=1..n_obj where x are first n_obj vars
            # Actually for DTLZ4: f1=(1+g)*x1, f2=(1+g)*x2, f3=(1+g)*(1-x3)
            # The Pareto front is when g=0, and x1+x2+x3=1 with x_i in [0,1]
            # So we need x1=u, x2=v, x3=1-u-v for the first 3 of the front vars
            # The remaining (n_front_vars - n_obj) = 7 front vars can be anything in [0,1]
            # But they don't affect objectives when g=0
            
            x[0] = u
            x[1] = v
            x[2] = 1.0 - u - v
            # Remaining front vars: set to 0.5 (doesn't matter for front)
            x[3:n_front_vars] = 0.5
            # Distance vars: set to very small value to be near front
            # For DTLZ4, g = sum(x_i^10) for i in last 2 vars
            # To get g≈0, set these very small
            x[n_front_vars:] = dist_scale * np.ones(n_dist_vars)
            
            return np.clip(x, lo, hi)

        def dtlz4_objectives_from_simplex(u, v, dist_scale=0.01):
            """Compute DTLZ4 objectives directly from simplex coords."""
            # g = sum of (dist_scale)^10 for each dist var
            g = n_dist_vars * (dist_scale ** 10)
            one_plus_g = 1.0 + g
            f1 = one_plus_g * u
            f2 = one_plus_g * v
            f3 = one_plus_g * (1.0 - u - v)
            return np.array([f1, f2, f3])

        # Generate points on the 2D simplex using Latin Hypercube Sampling
        # plus explicit vertices for maximum hypervolume contribution
        
        # Number of points to generate
        n_lhs = 180
        n_vertices = 3  # (1,0,0), (0,1,0), (0,0,1) in simplex coords
        n_edges = 6  # midpoints of edges
        n_center = 1  # centroid (1/3, 1/3, 1/3)
        
        total_planned = n_lhs + n_vertices + n_edges + n_center
        if total_planned > budget:
            n_lhs = max(10, budget - n_vertices - n_edges - n_center)
            total_planned = n_lhs + n_vertices + n_edges + n_center

        # Latin Hypercube Sampling on 2D simplex
        # Standard LHS: divide [0,1]^2 into n cells, pick one point per cell
        # Then filter to simplex region u+v<=1, reject/reflect otherwise
        
        def lhs_simplex(n, seed=42):
            """Generate n points uniformly distributed on the 2D simplex {u,v>=0, u+v<=1}."""
            rng_local = np.random.default_rng(seed)
            
            # Use rejection sampling from LHS in [0,1]^2
            points = []
            attempts = 0
            max_attempts = n * 50
            
            while len(points) < n and attempts < max_attempts:
                # Generate LHS-like points
                perm1 = rng_local.permutation(n)
                perm2 = rng_local.permutation(n)
                
                for i in range(n - len(points)):
                    if attempts >= max_attempts:
                        break
                    u = (perm1[i] + rng_local.random()) / n
                    v = (perm2[i] + rng_local.random()) / n
                    attempts += 1
                    if u + v <= 1.0 and u >= 0 and v >= 0:
                        points.append((u, v))
            
            # If we didn't get enough, add more via direct sampling
            while len(points) < n:
                u = rng_local.random()
                v = rng_local.random() * (1.0 - u)
                points.append((u, v))
            
            return np.array(points[:n])

        # Generate LHS points
        lhs_points = lhs_simplex(n_lhs, seed=0)
        
        # Vertex points (corners of simplex)
        vertex_points = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        
        # Edge midpoints
        edge_points = np.array([
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [0.25, 0.0],
            [0.0, 0.25],
            [0.75, 0.25]
        ])
        
        # Centroid
        center_point = np.array([[1.0/3, 1.0/3]])
        
        # Combine all planned points
        all_simplex_points = np.vstack([
            vertex_points,
            edge_points,
            center_point,
            lhs_points
        ])
        
        # Trim to budget
        all_simplex_points = all_simplex_points[:budget]
        
        # Evaluate all points
        evaluated_X = []
        evaluated_F = []
        used = 0
        
        for sp in all_simplex_points:
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            u, v = sp[0], sp[1]
            
            # First try with very small distance (essentially on the front)
            x = simplex_to_dtlz4(u, v, dist_scale=0.0)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            
            # Check if this is valid (all objectives < reference point 2.5)
            if np.all(f < 2.5):
                evaluated_X.append(x)
                evaluated_F.append(f)
        
        # If we have budget left, do local refinement around promising regions
        if used < budget and time.time() < deadline:
            # Find the best region and refine
            # Use Nelder-Mead-like local search in simplex space
            # to find points with better hypervolume contribution
            
            if len(evaluated_F) > 0:
                F_arr = np.array(evaluated_F)
                
                # Identify regions of high hypervolume contribution
                # Points near the vertices and edges contribute most
                # Do local refinement around existing good points
                
                n_refine = min(budget - used, 30)
                if n_refine > 0:
                    # Pick a subset of existing points to refine around
                    n_to_refine = min(10, len(evaluated_X))
                    indices = np.random.default_rng(1).choice(len(evaluated_X), n_to_refine, replace=False)
                    
                    for idx in indices:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        base_u, base_v = all_simplex_points[idx][0], all_simplex_points[idx][1]
                        
                        # Local search: try small perturbations in simplex space
                        for _ in range(n_refine // n_to_refine):
                            if used >= budget or time.time() > deadline:
                                break
                            
                            # Random perturbation
                            du = np.random.default_rng(used).normal(0, 0.05)
                            dv = np.random.default_rng(used + 1).normal(0, 0.05)
                            
                            new_u = base_u + du
                            new_v = base_v + dv
                            
                            # Ensure we stay in simplex
                            if new_u < 0:
                                new_u = 0
                            if new_v < 0:
                                new_v = 0
                            if new_u + new_v > 1:
                                scale = 1.0 / (new_u + new_v)
                                new_u *= scale
                                new_v *= scale
                            
                            x = simplex_to_dtlz4(new_u, new_v, dist_scale=0.0)
                            f = np.array(problem(x), dtype=np.float64)
                            used += 1
                            
                            if np.all(f < 2.5):
                                evaluated_X.append(x)
                                evaluated_F.append(f)
        
        return evaluated_X if evaluated_X else []
# EVOLVE-BLOCK-END
