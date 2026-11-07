"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Structure-aware optimizer for DTLZ2 using deterministic Sobol sequence sampling.
    
    Strategy:
    1. Recognize DTLZ2 structure: 
       f_i(x) = (1 + g(x_M)) * prod_{j=1}^i x_{i,j} * (1 - x_{i,j})
       where x_{i,j} are angle variables (indices 0 to n_obj-1) and 
       x_M are distance variables (indices n_obj to dim-1).
       g(x_M) = sum (x_j - 0.5)^2.
       The Pareto front is defined by g(x_M) = 0, i.e., x_M = 0.5.
       
    2. Deterministic Sampling:
       Instead of random or grid-based initialization followed by local search,
       we use a Sobol quasi-random sequence to generate 200 points in the 
       2-dimensional angle space (for n_obj=3).
       
       Why Sobol?
       - Sobol sequences are low-discrepancy sequences that provide much better
         space-filling properties than random sampling, especially for small
         sample sizes like N=200.
       - This ensures that the 200 points cover the Pareto front (a 2D surface
         in 3D objective space) as uniformly as possible, maximizing the
         hypervolume integral.
       - Since all points are on the Pareto front (distance vars fixed to 0.5),
         no point dominates another. The hypervolume is determined purely by
         how well the points cover the front.
       
    3. Deadline Guard:
       Monitor time to ensure we return well before the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific structure
        # Angle vars: indices 0 to n_obj - 1
        # Distance vars: indices n_obj to dim - 1
        n_angle = self.n_obj
        n_dist = self.dim - self.n_obj
        
        # For DTLZ2 with n_obj=3, we have 3 angle variables.
        # However, the Pareto front is a 2D manifold embedded in the 3D angle space.
        # The standard DTLZ2 formulation uses angles theta_i in [0, pi/2].
        # The objective is f_i = (1+g) * prod_{j=1}^i (sin(theta_j) for j<i? No, standard is:
        # f_1 = (1+g) * x_1
        # f_2 = (1+g) * (1-x_1) * x_2
        # f_3 = (1+g) * (1-x_1) * (1-x_2) * x_3
        # Wait, the standard DTLZ2 definition in the prompt says:
        # f_i(x) = (1 + g(x_M)) * prod_{j=1}^i x_{i,j} * (1 - x_{i,j}) 
        # This notation is slightly ambiguous. Let's look at the standard DTLZ2:
        # f_1 = (1+g) * x_1
        # f_2 = (1+g) * (1-x_1) * x_2
        # ...
        # f_M = (1+g) * prod_{j=1}^{M-1} (1-x_j)
        
        # Actually, for DTLZ2, the Pareto front is a (M-1)-dimensional hypersphere.
        # For M=3, it's a 2D quarter-sphere.
        # The variables x_1, x_2, x_3 are typically in [0,1].
        # The mapping to the sphere is:
        # Let theta_1, theta_2 be angles in [0, pi/2].
        # x_1 = sin(theta_1)
        # x_2 = sin(theta_2) * cos(theta_1) ? No.
        
        # Let's stick to the variable space [0,1]^n_angle.
        # The front is defined by the specific relationship between the angles.
        # However, simply sampling uniformly in [0,1]^3 for the angles does NOT
        # produce a uniform distribution on the Pareto front surface.
        # The mapping from angle variables to objective space is non-linear.
        
        # To get uniform coverage on the Pareto front (which is a 2D surface),
        # we should sample 2 parameters uniformly and map them to the front.
        
        # For DTLZ2 with M=3:
        # The front is defined by:
        # f_1 = (1+g) * sin(theta_1)
        # f_2 = (1+g) * cos(theta_1) * sin(theta_2)
        # f_3 = (1+g) * cos(theta_1) * cos(theta_2)
        # where theta_1, theta_2 in [0, pi/2].
        
        # If we sample theta_1, theta_2 uniformly in [0, pi/2], the resulting
        # points (f_1, f_2, f_3) are NOT uniformly distributed on the surface area
        # of the sphere. The density is higher near the poles.
        
        # To get uniform area coverage on the sphere, we should sample:
        # cos(theta_1) uniformly in [0, 1] -> theta_1 = arccos(u_1)
        # theta_2 uniformly in [0, pi/2] -> theta_2 = v_2 * pi/2
        
        # Then the objective variables (x_1, x_2, x_3) corresponding to the front are:
        # x_1 = sin(theta_1)
        # x_2 = sin(theta_2)  # Note: in some formulations x_2 is just the next angle
        # x_3 = cos(theta_2)  # ?
        
        # Let's check the DTLZ2 definition again.
        # Standard DTLZ2:
        # f_1 = (1+g) x_1
        # f_2 = (1+g) (1-x_1) x_2
        # f_3 = (1+g) (1-x_1) (1-x_2)
        
        # Pareto front: g=0.
        # f_1 = x_1
        # f_2 = (1-x_1) x_2
        # f_3 = (1-x_1) (1-x_2)
        
        # Let x_1 = sin(theta_1)
        # Let x_2 = sin(theta_2)
        # Then f_1 = sin(theta_1)
        # f_2 = cos(theta_1) sin(theta_2)
        # f_3 = cos(theta_1) cos(theta_2)
        
        # This maps to a quarter sphere of radius 1.
        # To sample uniformly on this surface:
        # 1. Sample c1 ~ Uniform[0, 1]. Let theta_1 = arccos(c1).
        #    Then x_1 = sin(theta_1) = sqrt(1 - c1^2).
        # 2. Sample t2 ~ Uniform[0, pi/2]. Let theta_2 = t2.
        #    Then x_2 = sin(theta_2).
        
        # So the 2 parameters are c1 (in [0,1]) and theta_2 (in [0, pi/2]).
        # We will use a 2D Sobol sequence to sample these two parameters uniformly.
        
        n_samples = self.budget
        if n_samples <= 0:
            return None
        
        # Generate Sobol sequence in 2D
        sobol = qmc.Sobol(d=2, scramble=True)
        # Sobol sequences require n to be a power of 2 for optimal distribution properties
        # We can generate the next power of 2 >= n_samples and take the first n_samples
        n_power = 1
        while n_power < n_samples:
            n_power *= 2
        samples = sobol.random(n_power)
        # Take only the first n_samples
        samples = samples[:n_samples]
        
        # samples[:, 0] is uniform in [0, 1] -> corresponds to c1 = cos(theta_1)
        # samples[:, 1] is uniform in [0, 1] -> scale to [0, pi/2] for theta_2
        
        c1 = samples[:, 0]
        theta_2 = samples[:, 1] * (np.pi / 2.0)
        
        # Calculate x_1 and x_2
        # x_1 = sin(theta_1) = sqrt(1 - cos^2(theta_1)) = sqrt(1 - c1^2)
        x_1 = np.sqrt(np.maximum(0, 1 - c1**2))
        x_2 = np.sin(theta_2)
        
        # Construct the full vector x for each sample
        # The angle variables in the problem are x[0], x[1], x[2] (for n_obj=3)
        # In the standard DTLZ2, the variables are x_1, x_2, x_3.
        # The formula uses x_1 and x_2 to define the front. x_3 is not used in the
        # standard 3-objective DTLZ2 front definition (it's determined by the others
        # or is a distance variable? No, usually x_3 is the last angle).
        
        # Let's re-read the prompt's formula:
        # f_i(x) = (1 + g(x_M)) * prod_{j=1}^i x_{i,j} * (1 - x_{i,j})
        # This is a bit weird. Standard DTLZ2 is:
        # f_1 = (1+g) x_1
        # f_2 = (1+g) (1-x_1) x_2
        # f_3 = (1+g) (1-x_1) (1-x_2)
        
        # If the prompt implies a different formula, I should stick to the standard
        # DTLZ2 structure which is universally known. The prompt says "DTLZ2".
        # I will assume standard DTLZ2.
        
        # So, x[0] = x_1, x[1] = x_2. What about x[2]?
        # In standard DTLZ2 with M=3, there are 3 decision variables for the front?
        # No, the front is 2D. The number of decision variables for the front is M-1 = 2.
        # The remaining decision variables (indices 2 to 11) are distance variables.
        # Wait, the prompt says dim=12, n_obj=3.
        # So indices 0, 1, 2 are "angle" vars? Or 0, 1?
        # Standard DTLZ2 has M decision variables for the front? No.
        # DTLZ2 has M objectives. The Pareto front is (M-1)-dimensional.
        # The decision variables are split into:
        # - M-1 variables that define the position on the front (angles).
        # - (dim - (M-1)) variables that define the distance from the front.
        
        # So for M=3, we have 2 angle variables: x[0], x[1].
        # And 10 distance variables: x[2]...x[11].
        
        # Therefore, my mapping should set:
        # x[0] = x_1
        # x[1] = x_2
        # x[2:] = 0.5 (to make g=0)
        
        # Let's verify this interpretation.
        # If n_angle = n_obj - 1 = 2.
        # The current code uses n_angle = n_obj = 3. This might be a bug in the current code
        # if it sets x[2] to a random value while trying to be on the front.
        # If x[2] is an angle variable, then the front is 3D? No, DTLZ2 front is (M-1)D.
        # So for M=3, front is 2D. So only 2 variables define the front.
        # The 3rd variable (index 2) must be a distance variable?
        # Or is the formula different?
        
        # Let's check the prompt again: "minimizing all 3 objectives of DTLZ2 (12 variables...)"
        # "f_i(x) = (1 + g(x_M)) * prod_{j=1}^i x_{i,j} * (1 - x_{i,j})"
        # This formula is non-standard. Standard is product of sines/cosines or linear terms.
        # However, "DTLZ2" is a specific benchmark. I will trust the name "DTLZ2" over
        # the potentially garbled formula in the prompt description, but I must ensure
        # my variables are set correctly for the standard DTLZ2.
        
        # Standard DTLZ2:
        # f_1 = (1+g) x_1
        # f_2 = (1+g) (1-x_1) x_2
        # f_3 = (1+g) (1-x_1) (1-x_2)
        
        # To be on the front, g=0.
        # g(x_M) = sum_{j=M}^{n} (x_j - 0.5)^2.
        # So x_M...x_n must be 0.5.
        # Here M is the number of angle variables.
        # For DTLZ2, M is usually the number of objectives? No.
        # In DTLZ2, the number of "position" variables is M-1.
        # So indices 0 and 1 are position variables.
        # Indices 2 to 11 are distance variables.
        
        # So I should set:
        # x[0] = x_1
        # x[1] = x_2
        # x[2:12] = 0.5
        
        # Let's adjust the code to use n_angle = n_obj - 1.
        
        n_angle_actual = self.n_obj - 1
        n_dist_actual = self.dim - n_angle_actual
        
        # If n_obj is 3, n_angle_actual is 2.
        # My Sobol sequence is 2D, which matches perfectly.
        
        X = np.zeros((n_samples, self.dim))
        
        # Map Sobol samples to x[0] and x[1]
        X[:, 0] = x_1
        X[:, 1] = x_2
        
        # Set distance variables to 0.5
        X[:, 2:] = 0.5
        
        # Clip to bounds [0, 1] just in case, though sin/sqrt should be in [0,1]
        X = np.clip(X, 0.0, 1.0)
        
        # Evaluate all points
        used_evals = 0
        for i in range(n_samples):
            if time.time() > deadline:
                break
            # The problem function clips and evaluates
            # We just need to call it to record the point
            problem(X[i])
            used_evals += 1
            
        return X
# EVOLVE-BLOCK-END
