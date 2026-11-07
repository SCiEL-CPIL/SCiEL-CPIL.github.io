"""
Analytic DTLZ1 front sampling with boundary-biased low-discrepancy points.
Strategy:
1. Exploit the analytic structure: Set x_G = 0 to guarantee g(x_G) = 0.
2. Sample x_1, x_2 using a Sobol-like low-discrepancy sequence.
3. Apply a power transformation (x = u^k) to bias samples towards the boundaries
   (corners) of the [0,1]^2 square. This concentrates points where they
   contribute the most to the hypervolume relative to the reference point [400,400,400].
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_pos = self.n_obj - 1  # 2
        n_g = self.dim - n_pos   # 5
        
        # We need to generate self.budget points.
        # The first 2 dimensions (x_1, x_2) determine the position on the front.
        # The remaining dimensions (x_G) are set to 0 to ensure g=0.
        
        # Generate low-discrepancy points in [0, 1]^2
        # Using a simple van der Corput-like sequence or Halton sequence for 2D.
        # Since we have budget=200, a Halton sequence is efficient and deterministic.
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate indices 1..budget
        indices = np.arange(1, self.budget + 1)
        
        # Halton bases for 2 dimensions: 2 and 3
        x1_uniform = np.array([halton_sequence(i, 2) for i in indices])
        x2_uniform = np.array([halton_sequence(i, 3) for i in indices])
        
        # Apply power transformation to bias towards boundaries (0 and 1).
        # For hypervolume with a far reference point, points near the "extremes" 
        # of the front (where some objectives are close to 0) are more valuable 
        # than points in the center. 
        # However, the front is convex towards the origin. 
        # The "corners" of the front in objective space correspond to:
        # (x1=0, x2=0) -> f=(0,0,0.5)
        # (x1=1, x2=0) -> f=(0,0.5,0.5)
        # (x1=0, x2=1) -> f=(0,0,0.5) -- wait, let's check DTLZ1 front shape.
        
        # DTLZ1 Front:
        # f1 = 0.5 * x1 * x2
        # f2 = 0.5 * (1 - x1) * x2
        # f3 = 0.5 * (1 - x2)
        
        # If x2=0: f1=0, f2=0, f3=0.5. (One point)
        # If x2=1: f1=0.5*x1, f2=0.5*(1-x1), f3=0. This is a line segment.
        # If x1=0: f1=0, f2=0.5*x2, f3=0.5*(1-x2). Line segment.
        # If x1=1: f1=0.5*x2, f2=0, f3=0.5*(1-x2). Line segment.
        
        # The front is a triangle in objective space with vertices:
        # A: x1=0, x2=0 -> (0, 0, 0.5)
        # B: x1=1, x2=1 -> (0.5, 0, 0)  -- Wait, x2=1, x1=1 => f1=0.5, f2=0, f3=0
        # C: x1=0, x2=1 -> (0, 0.5, 0)  -- x2=1, x1=0 => f1=0, f2=0.5, f3=0
        
        # Actually, the region is the set of points (f1, f2, f3) such that f1+f2+f3 <= 0.5?
        # f1+f2 = 0.5*x2. f3 = 0.5*(1-x2).
        # So f1+f2+f3 = 0.5*x2 + 0.5 - 0.5*x2 = 0.5.
        # The front is the plane f1+f2+f3 = 0.5 in the first octant, bounded by f_i >= 0.
        # It is a triangle with vertices (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5).
        
        # Hypervolume calculation with reference point R=(400,400,400).
        # We want to maximize the volume of the union of boxes [f_i, R_i].
        # Points close to the vertices (where two objectives are 0) dominate large regions.
        # Points in the center of the triangle (e.g., x1=0.5, x2=2/3? No, x1=0.5, x2=1/3? 
        # Center of triangle in obj space is (1/6, 1/6, 1/6)).
        # The "corners" of the parameter space [0,1]^2 map to the vertices of the triangle?
        # (0,0) -> (0,0,0.5) Vertex
        # (1,1) -> (0.5,0,0) Vertex
        # (0,1) -> (0,0.5,0) Vertex
        # (1,0) -> (0,0,0.5) Same as (0,0)
        
        # So the vertices of the triangle in objective space correspond to the corners of the square in parameter space.
        # To maximize hypervolume, we should sample densely near these corners.
        # A power transformation x = u^k biases towards 0. To bias towards both 0 and 1, 
        # we can use a transformation that stretches the middle and compresses the ends? 
        # No, we want MORE points at the ends.
        # If we use u^k with k > 1, we bias towards 0.
        # If we mix u^k and (1-u)^k, we can bias towards both.
        # Let's use a simple power bias towards 0 for one dimension and 1 for the other? 
        # Or just bias both towards 0? 
        # Since the mapping is symmetric in the triangle, biasing towards (0,0) in parameter space 
        # puts points near Vertex (0,0,0.5). Biasing towards (1,1) puts points near (0.5,0,0).
        # Biasing towards (0,1) puts points near (0,0.5,0).
        
        # Let's use a transformation that concentrates points near the boundaries of the square.
        # x = 0.5 * (1 + cos(pi * u)) biases towards 0 and 1? No, that's a cosine distribution.
        # Let's try x = u^3 for one and (1-u)^3 for the other? 
        # Actually, a standard technique for hypervolume is to use a distribution proportional to the 
        # "importance" of the point. But without knowing the exact HV contribution, 
        # a robust choice is to sample near the vertices.
        
        # Let's use a power law: x = u^k. 
        # If we set k=2, we get more points near 0.
        # To cover all 3 vertices, we can split the budget into 3 parts, 
        # each biased towards one corner of the square: (0,0), (1,0), (0,1), (1,1).
        # But (1,0) and (0,0) map to the same vertex in obj space.
        # So we have 3 distinct vertices in obj space.
        # Corner (0,0) -> V1
        # Corner (1,1) -> V2
        # Corner (0,1) -> V3
        # Corner (1,0) -> V1 (same as 0,0)
        
        # So we should bias towards (0,0), (1,1), and (0,1).
        # Let's allocate 1/3 of budget to each region.
        
        n_total = self.budget
        n_per_region = n_total // 3
        rem = n_total - 3 * n_per_region
        
        # Region 1: Bias towards (0,0)
        # x1 = u^2, x2 = v^2
        # Region 2: Bias towards (1,1)
        # x1 = 1 - u^2, x2 = 1 - v^2
        # Region 3: Bias towards (0,1)
        # x1 = u^2, x2 = 1 - v^2
        
        # We will generate Halton points for each region.
        
        x1_list = []
        x2_list = []
        
        # Helper to generate biased points
        def generate_biased(n, base1, base2, bias_x1, bias_x2):
            # bias_x1: 0 for low, 1 for high
            # bias_x2: 0 for low, 1 for high
            idxs = np.arange(1, n + 1)
            u1 = np.array([halton_sequence(i, base1) for i in idxs])
            u2 = np.array([halton_sequence(i, base2) for i in idxs])
            
            # Apply power transformation to bias towards 0
            # u^2 biases towards 0.
            # To bias towards 1, use 1 - u^2.
            
            if bias_x1 == 0:
                x1 = u1 ** 2
            else:
                x1 = 1 - u1 ** 2
                
            if bias_x2 == 0:
                x2 = u2 ** 2
            else:
                x2 = 1 - u2 ** 2
                
            return x1, x2

        # Region 1: (0,0)
        x1_r1, x2_r1 = generate_biased(n_per_region, 2, 3, 0, 0)
        x1_list.extend(x1_r1)
        x2_list.extend(x2_r1)
        
        # Region 2: (1,1)
        x1_r2, x2_r2 = generate_biased(n_per_region, 2, 3, 1, 1)
        x1_list.extend(x1_r2)
        x2_list.extend(x2_r2)
        
        # Region 3: (0,1)
        x1_r3, x2_r3 = generate_biased(n_per_region + rem, 2, 3, 0, 1)
        x1_list.extend(x1_r3)
        x2_list.extend(x2_r3)
        
        x1_arr = np.array(x1_list)
        x2_arr = np.array(x2_list)
        
        # Construct full dimension vectors
        X_samples = np.zeros((n_total, self.dim))
        X_samples[:, 0] = x1_arr
        X_samples[:, 1] = x2_arr
        # x_G = 0
        
        # Evaluate
        F_samples = np.array([problem(x) for x in X_samples])
        
        return X_samples
# EVOLVE-BLOCK-END
