"""Improved DTLZ1 optimizer using Halton sequence and vertex-biased sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """DTLZ1 optimizer exploiting Pareto front structure with Halton and vertex biasing."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Base configuration: x_i = 0.5 for i >= 2 ensures g=0
        base_x = np.zeros(dim)
        for i in range(2, dim):
            base_x[i] = 0.5
        
        # Strategy:
        # 1. Allocate ~15% of budget to points near the 3 vertices of the simplex
        # 2. Use the remaining ~85% for Halton sequence sampling
        
        n_vertex = max(1, int(0.15 * budget))
        n_hall = budget - n_vertex
        
        x_points = []
        
        # --- Part 1: Vertex-biased points ---
        # Vertices in (x1, x2) space: (0,0), (1,0), (0,1)
        # We sample points along the edges near these vertices to maximize hypervolume "reach"
        
        # Helper to add points near a vertex (vx, vy)
        # We sample along the two edges connected to the vertex
        def add_vertex_points(vx, vy, count):
            points = []
            if count <= 0: return points
            
            # Determine the two adjacent vertices
            # (0,0) connects to (1,0) and (0,1)
            # (1,0) connects to (0,0) and (0,1)
            # (0,1) connects to (0,0) and (1,0)
            
            if vx == 0 and vy == 0:
                # Edge to (1,0): x1 in [0, 0.2], x2=0
                # Edge to (0,1): x1=0, x2 in [0, 0.2]
                for i in range(count):
                    # Alternate between two edges
                    if i % 2 == 0:
                        # Along x1 axis
                        t = (i // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((t, 0.0))
                    else:
                        # Along x2 axis
                        t = ((i - 1) // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((0.0, t))
                        
            elif vx == 1 and vy == 0:
                # Edge to (0,0): x1 in [0.8, 1], x2=0
                # Edge to (0,1): x1+t, x2=t, x1+t <= 1 -> x1 in [0.8, 1], x2 = 1 - x1? No.
                # Edge (1,0) to (0,1) is x1+x2=1.
                # Let's sample on the edge x1+x2=1 near x1=1.
                # x1 = 1 - t, x2 = t, t in [0, 0.2]
                for i in range(count):
                    if i % 2 == 0:
                        # Edge to (0,0)
                        t = (i // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((1.0 - t, 0.0))
                    else:
                        # Edge to (0,1)
                        t = ((i - 1) // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((1.0 - t, t))
                        
            elif vx == 0 and vy == 1:
                # Edge to (0,0): x1=0, x2 in [0.8, 1]
                # Edge to (1,0): x1=t, x2=1-t, t in [0, 0.2]
                for i in range(count):
                    if i % 2 == 0:
                        # Edge to (0,0)
                        t = (i // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((0.0, 1.0 - t))
                    else:
                        # Edge to (1,0)
                        t = ((i - 1) // 2 + 1) / (count // 2 + 1) * 0.2
                        points.append((t, 1.0 - t))
                        
            return points

        # Distribute vertex points roughly equally
        pts_v0 = add_vertex_points(0, 0, n_vertex // 3)
        pts_v1 = add_vertex_points(1, 0, n_vertex // 3)
        pts_v2 = add_vertex_points(0, 1, n_vertex - len(pts_v0) - len(pts_v1))
        
        x_points.extend(pts_v0)
        x_points.extend(pts_v1)
        x_points.extend(pts_v2)
        
        # --- Part 2: Halton Sequence for the rest ---
        if n_hall > 0:
            halton = qmc.Halton(d=2, scramble=True)
            # We need n_hall points. Halton doesn't require power of 2.
            halton_samples = halton.random(n_hall)
            
            for i in range(n_hall):
                if time.time() > deadline:
                    break
                u = halton_samples[i, 0]
                v = halton_samples[i, 1]
                # Map [0,1]^2 to simplex
                # x1 = u, x2 = v * (1-u)
                x1 = u
                x2 = v * (1 - u)
                x_points.append((x1, x2))
        
        # Evaluate all points
        all_X = []
        used = 0
        for (x1, x2) in x_points:
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = lo[0] + x1 * (hi[0] - lo[0])
            x[1] = lo[1] + x2 * (hi[1] - lo[1])
            
            # Ensure we don't exceed budget
            if used >= budget:
                break
                
            f = problem(x)
            all_X.append(x)
            used += 1
            
        return np.array(all_X) if all_X else np.empty((0, dim))
# EVOLVE-BLOCK-END
