"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for DTLZ4 using Direct Analytic Parameterization.
    
    Strategy:
    1. Assume standard DTLZ4 structure:
       - Variables 0 to m-2 (here 0 and 1) are position variables.
       - Variables m-1 to n-1 (here 2 to 11) are distance variables.
    2. The Pareto front for DTLZ4 with m=3 objectives is a quarter unit sphere.
       The objectives are:
       f_0 = 0.5 * (1 + g) * sin(theta_1) * sin(theta_2)
       f_1 = 0.5 * (1 + g) * sin(theta_1) * cos(theta_2)
       f_2 = 0.5 * (1 + g) * cos(theta_1)
       where theta_1, theta_2 are derived from the position variables.
       
       For standard DTLZ4:
       theta_1 = x_0 * pi / 2
       theta_2 = x_1 * pi / 2
       
       To be on the Pareto front, g must be 0.
       For DTLZ4, g=0 is achieved when all distance variables are 0.5.
       
    3. We will generate a set of points on this front.
       To maximize hypervolume, we need to cover the front densely.
       Since the front is 2D (a surface), we need a 2D space-filling curve or grid.
       
       Hypervolume is maximized by points that are spread out.
       A uniform grid in the angle space might not be optimal because the metric 
       on the sphere is non-linear. However, for a quarter sphere, a grid in 
       theta_1, theta_2 is a reasonable first approximation.
       
       Better: Use a Sobol sequence or Latin Hypercube in the angle space to ensure 
       low-discrepancy sampling.
       
       Even better: We can use the fact that the reference point is [2.5, 2.5, 2.5].
       The hypervolume contribution of a point is roughly the volume of the box 
       it "covers". Points near the corners (where objectives are small) contribute 
       more to the union volume if they are isolated, but dense coverage is key.
       
       Let's use a Sobol sequence to generate (u, v) in [0, 1]^2, map to 
       theta_1 = u * pi/2, theta_2 = v * pi/2.
       Then construct x:
       x[0] = u
       x[1] = v
       x[2:] = 0.5  (to make g=0)
       
       Wait, does x[2:]=0.5 guarantee g=0?
       DTLZ4 g function: g(x) = 10 * (n_g) * ( sum_{i in g} x_i^beta + n_g - 1 ) / n_g
       where beta = 100.
       If x_i = 0.5, x_i^100 is extremely small (~7.9e-31).
       So g ~ 10 * n_g * (0 + n_g - 1) / n_g = 10 * (n_g - 1).
       For n_g = 10, g = 90.
       This is NOT zero.
       
       Let's re-read DTLZ4 definition.
       g(x) = 10 * n_g * ( sum_{i=n_m}^{n-1} x_i^beta + n_g - 1 ) / n_g
       To get g=0, we need sum x_i^beta = 1 - n_g? No, that's impossible for x in [0,1] if n_g > 1.
       
       Actually, for DTLZ4, the Pareto front is g=0.
       The condition for g=0 is:
       sum_{i in g} x_i^beta = 1 - n_g ? No.
       
       Let's check the standard definition again.
       DTLZ4:
       g(x) = 10 * (n_g) * ( sum_{i=n_m}^{n-1} x_i^beta + n_g - 1 ) / n_g
       If all x_i = 0.5, sum is n_g * 0.5^beta.
       If beta is large, 0.5^beta is 0.
       So g = 10 * (n_g) * (0 + n_g - 1) / n_g = 10 * (n_g - 1).
       This is not 0.
       
       Wait, the Pareto front is defined by g=0.
       How can g be 0?
       g(x) >= 0.
       g(x) = 0 iff sum_{i in g} x_i^beta = 1 - n_g?
       If n_g = 1, sum x_i^beta = 0 => x_i = 0.
       If n_g > 1, 1 - n_g is negative. Impossible.
       
       I must be misremembering the DTLZ4 g function or the condition.
       
       Let's look up DTLZ4 specifically.
       DTLZ4:
       f_1 = 0.5 * (1+g) * sin(theta_1) * sin(theta_2)
       f_2 = 0.5 * (1+g) * sin(theta_1) * cos(theta_2)
       f_3 = 0.5 * (1+g) * cos(theta_1)
       
       g(x) = 10 * n_g * ( sum_{i=n_m}^{n-1} x_i^beta + n_g - 1 ) / n_g
       
       For g=0, we need sum x_i^beta = 1 - n_g.
       This is only possible if n_g=1 and x_{n_m}=0.
       OR if the formula is different.
       
       Ah, DTLZ2 has g(x) = 1 + 10*n_g*(sum(x_i^2) - x_i)/n_g. g=0 when x_i=0.5.
       DTLZ4 is similar but with x_i^beta.
       g(x) = 1 + 10*n_g*(sum(x_i^beta) - x_i)/n_g ?
       No, standard DTLZ4 definition:
       g(x) = 10 * n_g * ( sum_{i=n_m}^{n-1} x_i^beta + n_g - 1 ) / n_g
       Wait, if x_i=0.5, x_i^beta is tiny.
       Then g = 10 * (n_g - 1).
       
       Is the Pareto front g=0?
       Yes, for DTLZ problems, the Pareto front is typically g=0.
       If g cannot be 0 for n_g > 1, then the "front" is actually at g_min > 0.
       
       Let's check the minimum of g(x).
       Min of sum x_i^beta for x in [0,1] is 0 (when all x_i=0).
       So min g = 10 * n_g * (0 + n_g - 1) / n_g = 10 * (n_g - 1).
       
       So the Pareto front is at g = 10 * (n_g - 1).
       This means the objectives are scaled by (1 + 10*(n_g-1)).
       
       If n_g = 10, scale = 1 + 90 = 91.
       Max objective = 0.5 * 91 * 1 = 45.5.
       Reference point is 2.5.
       All points would be dominated by reference point?
       "Only points strictly better than the reference point in every objective count."
       If min objective is > 2.5, score is 0.
       
       But the current program scores 1.07. So points MUST be better than 2.5.
       This implies g is small, likely close to 0.
       
       Let's re-evaluate the g function for DTLZ4.
       Maybe the definition used in this specific task is different, or I am misinterpreting "n_g".
       n_g = n - (m-1) = 12 - 2 = 10.
       
       Alternative DTLZ4 definition:
       g(x) = 1 + 10 * n_g * ( sum_{i=n_m}^{n-1} (x_i^beta - x_i) ) / n_g ?
       If x_i = 0.5, x_i^beta ~ 0, x_i = 0.5.
       sum (0 - 0.5) = -0.5 * n_g.
       g = 1 + 10 * (-0.5) = 1 - 5 = -4.
       g must be non-negative?
       
       Actually, let's look at the "seed program" strategy which got 1.07.
       It identified g_vars and set them to 0.5.
       It then sampled position vars.
       It got a good score.
       This implies that setting g_vars to 0.5 results in points that are NOT dominated by [2.5, 2.5, 2.5].
       So objectives must be < 2.5.
       
       If f = 0.5 * (1+g) * trig, and f < 2.5, then (1+g) < 5.
       So g < 4.
       
       If my formula g = 10*(n_g-1) = 90 is correct, then f ~ 45. Score would be 0.
       Since score is 1.07, my formula for g is likely wrong for this specific implementation, OR the "g_vars" in the current program are NOT the variables in the standard DTLZ4 g function.
       
       Let's reconsider the variable identification in the current program.
       It found some vars as "g" and some as "pos".
       If the problem is NOT standard DTLZ4, or has a different g definition, the adaptive approach is safer.
       
       However, the directive says "COMPUTE BUDGET... STAGNATION... Make a STRUCTURAL change".
       The current program is adaptive.
       A structural change could be: **Assume the problem IS standard DTLZ4 but fix the g calculation.**
       
       What if the g function is:
       g(x) = 10 * n_g * ( sum_{i in g} x_i^beta ) / n_g ?
       If x_i = 0, g=0.
       If x_i = 0.5, g ~ 0.
       So setting g_vars to 0 might be better than 0.5?
       
       Let's try a strategy that tests a few specific configurations of the g-variables to find the one that minimizes the objectives, then samples the front.
       
       Revised Strategy:
       1. Assume variables 0, 1 are position. Variables 2..11 are distance.
       2. Test a few values for the distance variables to find the configuration that minimizes the sum of objectives (proxy for g).
          - Config A: all 0.5
          - Config B: all 0.0
          - Config C: all 1.0
          - Config D: random
       3. Pick the best config (lowest sum of objectives).
       4. Use that config for the distance variables.
       5. Sample the position variables (0, 1) using a Sobol sequence to cover the front.
       
       This is robust to the specific definition of g, as long as it's separable and the "best" g-value is at the boundary or center.
       
       Let's implement this.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = 0.5 * (lo + hi)
        
        # Assume standard DTLZ4 structure:
        # pos_vars = [0, 1]
        # g_vars = [2, 3, ..., 11]
        
        pos_vars = [0, 1]
        g_vars = list(range(2, self.dim))
        
        # Phase 1: Determine the best setting for g_vars to minimize objectives.
        # We will test a few candidate vectors for the g_vars part.
        candidates = []
        
        # Candidate 1: All mid
        x_g1 = np.full(len(g_vars), 0.5)
        candidates.append(x_g1)
        
        # Candidate 2: All low
        x_g2 = np.full(len(g_vars), 0.0)
        candidates.append(x_g2)
        
        # Candidate 3: All high
        x_g3 = np.full(len(g_vars), 1.0)
        candidates.append(x_g3)
        
        # Candidate 4: Random (seeded for reproducibility in this phase)
        np.random.seed(42)
        x_g4 = np.random.uniform(0, 1, len(g_vars))
        candidates.append(x_g4)
        
        best_g_vec = None
        best_score = np.inf
        
        # Test each candidate by evaluating a point with position vars at 0.5 (center of front)
        # and g vars set to the candidate.
        for g_vec in candidates:
            if time.time() > deadline:
                break
                
            x_test = np.full(self.dim, 0.5) # Default mid
            for i, gv in zip(g_vars, g_vec):
                x_test[i] = gv
                
            # Also set pos vars to 0.5 for a neutral test
            x_test[0] = 0.5
            x_test[1] = 0.5
            
            f_vals = np.array(problem(x_test))
            # Sum of objectives as a proxy for distance to origin (and thus g)
            score = np.sum(f_vals)
            
            if score < best_score:
                best_score = score
                best_g_vec = g_vec
        
        if best_g_vec is None:
            # Fallback if no evals were possible (shouldn't happen with 200 budget)
            best_g_vec = np.full(len(g_vars), 0.5)
        
        # Phase 2: Sample the Pareto front.
        # We have used 4 evaluations so far.
        remaining_budget = self.budget - 4
        
        if remaining_budget <= 0:
            return None
        
        # Generate Sobol sequence for the 2 position variables.
        # We want to cover the quarter circle [0, pi/2] x [0, pi/2].
        # Mapping u, v in [0, 1] to theta1, theta2 in [0, pi/2].
        
        # Use a grid or Sobol?
        # Sobol is better for low-discrepancy.
        # However, for a 2D front, a stratified grid might be simpler and effective.
        # Let's use a Sobol sequence to be safe and "structural".
        
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        n_samples = remaining_budget
        samples = sampler.random(n=n_samples)
        
        # samples are in [0, 1]^2
        # Map to x[0], x[1]
        # x[0] = u * (hi[0] - lo[0]) + lo[0]
        # x[1] = v * (hi[1] - lo[1]) + lo[1]
        
        for u, v in samples:
            if time.time() > deadline:
                break
            
            x = np.full(self.dim, 0.5) # Init with mid, will overwrite pos and g
            
            # Set position variables
            x[0] = lo[0] + u * (hi[0] - lo[0])
            x[1] = lo[1] + v * (hi[1] - lo[1])
            
            # Set g variables to the best found configuration
            for i, gv in zip(g_vars, best_g_vec):
                x[i] = gv
                
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
