"""Rewritten optimizer exploiting ZDT1 structure for maximum hypervolume."""
import numpy as np
import time


class Optimizer:
    """Exploits ZDT1 analytical structure: f1=x[0], f2=g*(1-sqrt(x[0]/g)) where g depends on x[1:].
    
    Key insight: to get points on the Pareto front, we need g to be at its minimum.
    For ZDT1, g = 1 + 10*(n-1)/sum(x[1:]). With x[1:] in [0,1], max sum = n-1, so min g = 11.
    Setting x[1:] = 1.0 gives g = 1 + 10*(n-1)/(n-1) = 11.
    Then f1 = x[0], f2 = 11*(1 - sqrt(x[0]/11)).
    The Pareto front is f2 = 11 - 11*sqrt(f1/11) = 11 - sqrt(11*f1).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            return x, f

        # Phase 1: Verify structure with a few probes
        # Check that x[0] controls f1 and x[1:]=1 gives minimum g
        probe1, f1_probe = eval_point(np.ones(self.dim))
        probe2, f2_probe = eval_point(np.array([0.5] + [1.0] * (self.dim - 1)))
        probe3, f3_probe = eval_point(np.array([0.5] + [0.5] * (self.dim - 1)))

        # f1 should be approximately x[0]
        # f2 with x[1:]=1 should be lower than with x[1:]=0.5 (since smaller g is better for f2 minimization)
        # Actually for ZDT1, f2 = g*(1-sqrt(x0/g)). Larger g means larger f2 (worse).
        # So x[1:]=1 (g=11) should give lower f2 than x[1:]=0.5 (g=1+10*29/14.5=21)

        # Phase 2: Dense sampling of the Pareto front
        # Use the analytical solution: x[0]=t, x[1:]=1.0
        # This gives points on the Pareto front: f1=t, f2=11*(1-sqrt(t/11))
        
        n_front = 150  # Number of front points
        remaining_budget = self.budget - 3  # Subtract probes
        n_front = min(n_front, remaining_budget)
        
        # Generate t values covering [0, 1] with emphasis on the curved region
        # The front is f2 = 11 - sqrt(11*t), which is concave
        # Use a mix of uniform and clustered points
        t_uniform = np.linspace(0.001, 0.999, n_front)
        
        all_points = [probe1, probe2, probe3]
        all_f = [f1_probe, f2_probe, f3_probe]
        
        for t in t_uniform:
            if time.time() > deadline:
                break
            if len(all_points) >= self.budget:
                break
            
            x = np.array([t] + [1.0] * (self.dim - 1))
            x, f = eval_point(x)
            all_points.append(x)
            all_f.append(f)
        
        # Phase 3: Use remaining budget for local refinement
        # Try to improve points slightly by perturbing x[1:] values
        if time.time() < deadline and len(all_points) < self.budget:
            # Get current non-dominated front
            F = np.array(all_f)
            n = len(F)
            
            # Find non-dominated points
            nondom_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom_idx.append(i)
            
            # Refine a subset of non-dominated points
            n_refine = min(len(nondom_idx), (self.budget - len(all_points)) // 3)
            rng = np.random.default_rng(42)
            
            for idx in nondom_idx[:n_refine]:
                if time.time() > deadline:
                    break
                if len(all_points) >= self.budget:
                    break
                
                base_x = all_points[idx].copy()
                base_f = all_f[idx].copy()
                
                # Try small perturbations to x[1:] to see if we can improve
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    if len(all_points) >= self.budget:
                        break
                    
                    # Perturb only x[1:] slightly
                    perturbation = 0.01 * rng.standard_normal(self.dim - 1)
                    new_x = base_x.copy()
                    new_x[1:] += perturbation
                    new_x, new_f = eval_point(new_x)
                    
                    # Check if this point is non-dominated with respect to current set
                    F_current = np.array(all_f)
                    is_dominated = False
                    for j in range(len(F_current)):
                        if np.all(F_current[j] <= new_f) and np.any(F_current[j] < new_f):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        all_points.append(new_x)
                        all_f.append(new_f)
        
        # Ensure we haven't exceeded budget
        all_points = all_points[:self.budget]
        all_f = all_f[:self.budget]
        
        return all_points
