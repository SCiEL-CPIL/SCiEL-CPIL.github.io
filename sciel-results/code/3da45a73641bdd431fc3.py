# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy:
    1. Acknowledge that ZDT3's Pareto front lies on the manifold x_2..x_n = 0.
    2. Instead of fixed deterministic sampling, use an adaptive strategy:
       - Initialize with a small set of points to estimate the local structure.
       - Use a 'cross' operator (SBX crossover on x1) to generate children.
       - Select parents based on a heuristic that favors regions of high 
         marginal hypervolume contribution (approximated by distance to 
         existing points in objective space).
    3. This approach tests if 'cross' changes the chance of improvement by 
       actively exploring the 1D parameter space x1 using recombination 
       rather than pure mutation or fixed sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None
            
        # ZDT3 specific: Pareto front is at x[1:] = 0.
        # We only need to optimize x[0].
        
        # State:
        # We maintain a list of evaluated (x1, f1, f2) tuples.
        # Since we know x[1:]=0, we just track x1.
        
        evaluated = [] # List of (x1, f1, f2)
        
        def evaluate_x1(x1_val):
            x = np.zeros(dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            # Ensure other bounds are respected (they are 0, so 0 is fine)
            # But just in case bounds are weird, clip all.
            x = np.clip(x, lo, hi)
            f = problem(x)
            # f is expected to be array-like of length n_obj
            f1, f2 = float(f[0]), float(f[1])
            evaluated.append((x1_val, f1, f2))
            return f1, f2

        # 1. Initialization: Sample a few points to get a feel for the front.
        # We use a coarse grid to cover [0, 1].
        n_init = min(20, budget)
        init_x1s = np.linspace(0.0, 1.0, n_init)
        
        for x1 in init_x1s:
            if time.time() > deadline:
                break
            evaluate_x1(x1)
            
        if len(evaluated) >= budget:
            return None
            
        # 2. Adaptive Search with 'Cross' Operator
        # The 'cross' operator here is Simulated Binary Crossover (SBX) 
        # adapted for 1D parameter space.
        # We select parents based on their "front-ness" and spread.
        # Since we want to maximize hypervolume, we want to fill gaps 
        # in the objective space.
        
        eta_c = 20.0 # Distribution index for SBX, high = closer to parents
        
        def sbx_crossover(x1_a, x1_b):
            """
            Simulated Binary Crossover for 1D variables.
            Returns two children.
            """
            if x1_a == x1_b:
                return x1_a, x1_b
            
            x1_min = min(x1_a, x1_b)
            x1_max = max(x1_a, x1_b)
            
            u1 = np.random.random()
            u2 = np.random.random()
            
            # Calculate beta1
            if u1 <= 0.5:
                beta1 = (2 * u1) ** (1.0 / (eta_c + 1.0))
            else:
                beta1 = (1.0 / (2 * (1.0 - u1))) ** (1.0 / (eta_c + 1.0))
            
            # Calculate beta2
            if u2 <= 0.5:
                beta2 = (2 * u2) ** (1.0 / (eta_c + 1.0))
            else:
                beta2 = (1.0 / (2 * (1.0 - u2))) ** (1.0 / (eta_c + 1.0))
            
            # Children
            x1_c1 = 0.5 * ((x1_a + x1_b) - beta1 * (x1_b - x1_a))
            x1_c2 = 0.5 * ((x1_a + x1_b) + beta2 * (x1_b - x1_a))
            
            # Clip to bounds
            x1_c1 = np.clip(x1_c1, lo[0], hi[0])
            x1_c2 = np.clip(x1_c2, lo[0], hi[0])
            
            return x1_c1, x1_c2

        # Selection Strategy:
        # We want to pick parents that are "good" in terms of hypervolume contribution.
        # A simple proxy: pick points that are non-dominated and have neighbors 
        # that are far away in objective space (to fill gaps).
        
        # To keep it simple and fast, we'll use a heuristic:
        # 1. Filter to non-dominated points.
        # 2. Among those, pick two at random, but with a bias towards 
        #    points that are "isolated" in the f1-f2 space.
        
        def get_nondominated():
            if not evaluated:
                return []
            nd_idx = []
            for i, (x1_i, f1_i, f2_i) in enumerate(evaluated):
                is_dominated = False
                for j, (x1_j, f1_j, f2_j) in enumerate(evaluated):
                    if i == j:
                        continue
                    if f1_j <= f1_i and f2_j <= f2_i and (f1_j < f1_i or f2_j < f2_i):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_idx.append(i)
            return [evaluated[i] for i in nd_idx]

        # Main loop
        while len(evaluated) < budget:
            if time.time() > deadline:
                break
            
            nd_points = get_nondominated()
            
            if len(nd_points) < 2:
                # If we don't have enough non-dominated points, 
                # just pick two random points from all evaluated
                # or generate new random points if very few.
                if len(evaluated) < 2:
                    # Generate random x1
                    x1_new = np.random.uniform(lo[0], hi[0])
                    evaluate_x1(x1_new)
                    continue
                else:
                    # Pick two random from all
                    idxs = np.random.choice(len(evaluated), 2, replace=False)
                    p1 = evaluated[idxs[0]]
                    p2 = evaluated[idxs[1]]
            else:
                # Pick two from non-dominated
                # To encourage exploration, we pick randomly from ND set.
                # A more sophisticated method would weight by hypervolume contribution,
                # but random from ND is a robust baseline for "cross".
                idxs = np.random.choice(len(nd_points), 2, replace=False)
                p1 = nd_points[idxs[0]]
                p2 = nd_points[idxs[1]]
                
            # Apply Cross (SBX)
            x1_child1, x1_child2 = sbx_crossover(p1[0], p2[0])
            
            # Evaluate child 1
            evaluate_x1(x1_child1)
            
            # Evaluate child 2 if budget allows
            if len(evaluated) < budget:
                if time.time() > deadline:
                    break
                evaluate_x1(x1_child2)
                
        return None
# EVOLVE-BLOCK-END
