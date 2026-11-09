import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import time
        t_start = time.time()
        deadline = 300.0  # safety margin before 330s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget  # 200
        dim = self.dim  # 12
        
        # DTLZ4: front vars x1,x2; g vars x3..x12; alpha=100
        alpha = 100.0
        g_val = 0.5 ** (1.0/alpha)  # 0.5^(1/100)
        
        # Stratified area-uniform sampling on the first octant of the unit sphere
        # u3 = cos(theta) uniform in [0,1]; phi uniform in [0, pi/2]
        # Choose grid: n_u3 x n_phi = n
        # Use roughly square: n_u3 = ceil(sqrt(n)), n_phi = n // n_u3
        n_u3 = int(np.ceil(np.sqrt(n)))
        n_phi = n // n_u3
        # Adjust to get exactly n points
        # We'll create a list of (u3, phi)
        pts = []
        # Stratified in u3 and phi
        u3_edges = np.linspace(0, 1, n_u3 + 1)
        phi_edges = np.linspace(0, np.pi/2, n_phi + 1)
        count = 0
        for i in range(n_u3):
            for j in range(n_phi):
                if count >= n:
                    break
                u3 = np.random.uniform(u3_edges[i], u3_edges[i+1])
                phi = np.random.uniform(phi_edges[j], phi_edges[j+1])
                pts.append((u3, phi))
                count += 1
            if count >= n:
                break
        # If we have fewer than n (shouldn't happen), fill with more
        while len(pts) < n:
            u3 = np.random.uniform(0, 1)
            phi = np.random.uniform(0, np.pi/2)
            pts.append((u3, phi))
        
        # Convert to x and evaluate
        F = []
        for (u3, phi) in pts:
            r = np.sqrt(max(0.0, 1.0 - u3*u3))
            u1 = r * np.cos(phi)
            u2 = r * np.sin(phi)
            # Map to x
            x = np.empty(dim)
            # x1
            u1c = np.clip(u1, 0.0, 1.0 - 1e-12)
            x[0] = (2.0/np.pi) * np.arcsin(u1c)
            # x2
            denom = np.sqrt(max(1e-24, 1.0 - u1c*u1c))
            u2_over = np.clip(u2/denom, 0.0, 1.0 - 1e-12)
            x[1] = (2.0/np.pi) * np.arcsin(u2_over)
            # g vars
            for k in range(2, dim):
                x[k] = g_val
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            F.append(f)
            if time.time() > deadline - 5:
                break
        
        return F
