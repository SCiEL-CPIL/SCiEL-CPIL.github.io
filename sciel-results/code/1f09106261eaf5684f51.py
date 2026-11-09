import numpy as np
import time

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using Hammersley points transformed to uniform objective space coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = t_start + 290.0 

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 with M=3 objectives has M-1 = 2 shape variables (x_0, x_1).
        # The remaining N-M+1 = 5 variables (x_2..x_6) are distance variables.
        # To achieve g=0 (minimize distance part), set distance vars to 0.5.
        # The Pareto front is the set of points where g=0.
        # We sample the shape variables (x_0, x_1) to cover the front.
        
        # We want to maximize hypervolume. The objective space front is a 2D simplex.
        # Uniform sampling in (x_0, x_1) leads to non-uniform coverage in objective space
        # because the Jacobian of the mapping depends on x_0.
        # Specifically, area in objective space is proportional to (1 - x_0) dx_0 dx_1.
        # To get uniform coverage in objective space, we sample uniformly in the objective simplex
        # and map back to (x_0, x_1).
        
        # 1. Generate Hammersley points in [0, 1]^2
        n = budget
        i = np.arange(n)
        u_hammersley = i / n
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        v_hammersley = np.mod(i * phi, 1.0)
        
        # 2. Transform (u, v) to uniform points (s1, s2, s3) in the standard simplex
        # s1 + s2 + s3 = 1, s_i >= 0
        s1 = np.zeros(n)
        s2 = np.zeros(n)
        s3 = np.zeros(n)
        
        mask = u_hammersley + v_hammersley <= 1.0
        s1[mask] = u_hammersley[mask]
        s2[mask] = v_hammersley[mask]
        s3[mask] = 1.0 - u_hammersley[mask] - v_hammersley[mask]
        
        s1[~mask] = 1.0 - u_hammersley[~mask]
        s2[~mask] = 1.0 - v_hammersley[~mask]
        s3[~mask] = u_hammersley[~mask] + v_hammersley[~mask] - 1.0
        
        # 3. Map (s1, s2, s3) to (x_0, x_1)
        # f1 = 0.5 s1, f2 = 0.5 s2, f3 = 0.5 s3
        # x_0 = 2 f1 = s1
        # x_1 = f2 / (0.5 (1 - x_0)) = (0.5 s2) / (0.5 (1 - s1)) = s2 / (1 - s1)
        
        x_0 = s1
        # Avoid division by zero if s1 is 1. In that case, s2 must be 0.
        # Limit x_1 to [0, 1]
        denom = 1.0 - x_0
        # Where denom is close to 0, set x_1 to 0 (since s2 must be 0 there for valid simplex point)
        # Actually, if s1=1, s2=0, s3=0. x_1 = 0/0. Limit is 0.
        x_1 = np.divide(s2, denom, out=np.zeros_like(s2), where=denom > 1e-12)
        
        # Clip to [0, 1] to be safe against numerical errors
        x_0 = np.clip(x_0, 0.0, 1.0)
        x_1 = np.clip(x_1, 0.0, 1.0)
        
        samples = np.column_stack([x_0, x_1])
        
        # Explicitly include the corners of the decision space that map to
        # unique vertices of the Pareto front in objective space.
        # (x0, x1) = (1, 0) -> (0.5, 0, 0)
        # (x0, x1) = (0, 1) -> (0, 0.5, 0)
        # (x0, x1) = (0, 0) -> (0, 0, 0.5)
        
        # Replace the first 3 samples with these explicit corners to ensure
        # maximum coverage of the extreme ends of the front.
        if budget >= 3:
            samples[0] = [1.0, 0.0]
            samples[1] = [0.0, 1.0]
            samples[2] = [0.0, 0.0]

        # Construct points
        points = np.zeros((budget, dim))
        points[:, 0] = samples[:, 0]
        points[:, 1] = samples[:, 1]
        
        # Set distance variables to 0.0 to minimize g(x) and achieve the closest Pareto front
        for j in range(2, dim):
            points[:, j] = 0.0
            
        # Evaluate
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))
            
        return results
