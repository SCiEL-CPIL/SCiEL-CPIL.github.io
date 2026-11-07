"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead ensemble for black-box optimization on BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 325s to be safe.
        deadline = start_time + 325.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget or check_time():
                return None
            try:
                f = func(x)
                calls_made += 1
                update_best(x, f)
                return f
            except Exception:
                return None

        # --- Phase 0: Latin Hypercube Sampling (LHS) for diverse starts ---
        # We want 5 independent simplexes. 
        # Each simplex needs dim+1 points. Total points needed: 5 * (dim+1).
        # But we can be smarter: Generate a set of diverse start points, 
        # then build simplexes around them.
        
        # For dim=5, a standard NM simplex has 6 points.
        # Let's pick 5 "anchor" points using LHS to cover the space [-5, 5]^dim.
        n_anchors = 5
        
        # Generate LHS points
        # For each dimension, partition [0,1] into n_anchors intervals, pick one point in each, 
        # then shuffle the assignments.
        u = np.random.rand(n_anchors, dim)
        for j in range(dim):
            # Shuffle the order for dimension j
            perm = np.random.permutation(n_anchors)
            # Map to [0,1] intervals: i/n to (i+1)/n
            # Actually, standard LHS:
            # For dimension j, take u[:, j], sort it, assign ranks 0..n-1, 
            # then randomize the row assignment? 
            # Simpler: 
            # For each dim j, generate n_anchors uniform points in [0,1], 
            # divide [0,1] into n_anchors bins, pick one point per bin (randomly from the generated ones? No, just use the sorted ones mapped to bins).
            
            # Let's use a simple LHS implementation:
            # 1. For each dimension, generate n_anchors uniform random numbers.
            # 2. Sort them.
            # 3. The i-th sorted number is assigned to the i-th bin.
            # 4. Shuffle the rows of the resulting matrix.
            
            # Wait, the standard way:
            # Matrix U of size n x d.
            # For each column j:
            #   Generate n uniform randoms.
            #   Sort them.
            #   The k-th smallest value is assigned to the k-th row (in terms of bin index).
            #   Then, shuffle the rows of the entire matrix.
            
            # Let's just do:
            col = np.random.rand(n_anchors)
            col.sort()
            # Map to bins: bin i is [i/n, (i+1)/n]. 
            # A point in bin i can be i/n + col[i]/n.
            u[:, j] = (np.arange(n_anchors) + col) / n_anchors
            
        # Shuffle rows to break correlation
        perm = np.random.permutation(n_anchors)
        u = u[perm]
        
        # Map to [-5, 5]
        anchors = -5 + 10 * u
        
        # --- Phase 1: Parallel Nelder-Mead Simplexes ---
        # We will maintain 5 simplexes.
        # Each simplex is a list of (point, value) pairs.
        # Initially, each simplex is built around an anchor.
        # To save budget, we don't evaluate all 6 points of 5 simplexes immediately (30 calls).
        # Instead, we evaluate the anchor first. Then we build the simplex by perturbing the anchor.
        
        # However, NM requires the full simplex to start.
        # Budget is 500. 5 simplexes * 6 points = 30 calls. That's 6% of budget. Acceptable.
        # But we can be more efficient: 
        # 1. Evaluate the 5 anchors (5 calls).
        # 2. For each anchor, build a simplex by adding small perturbations.
        #    Perturbation size: start with 0.5 or 1.0.
        
        simplex_size = dim + 1
        step_size = 0.5  # Initial perturbation for simplex construction
        
        simplexes = []
        
        # Evaluate anchors
        anchor_vals = []
        for i in range(n_anchors):
            x = anchors[i]
            f = safe_call(x)
            if f is None:
                break
            anchor_vals.append(f)
        else:
            # All anchors evaluated
            for i in range(n_anchors):
                # Build simplex around anchor i
                simplex = [(anchors[i], anchor_vals[i])]
                for j in range(dim):
                    x_pert = anchors[i].copy()
                    x_pert[j] += step_size
                    x_pert = np.clip(x_pert, -5, 5)
                    f_pert = safe_call(x_pert)
                    if f_pert is None:
                        break
                    simplex.append((x_pert, f_pert))
                else:
                    simplexes.append(simplex)
                if check_time() or calls_made >= budget:
                    break

        if not simplexes:
            return best_x, best_f

        # --- Phase 2: Iterative Nelder-Mead Steps ---
        # We iterate through the simplexes, performing one NM step for each active simplex.
        # This distributes the budget evenly among the 5 search regions.
        # If a simplex converges or stagnates, we can shrink it or discard it.
        
        # NM Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # Main loop
        while not check_time() and calls_made < budget:
            # Check if we have enough budget for at least one more step
            # A step involves up to 1 new function call (reflection, then maybe expansion/contraction/shrink)
            # On average, 1 call per step.
            
            # Shuffle simplexes to avoid biasing one region
            np.random.shuffle(simplexes)
            
            for s_idx in range(len(simplexes)):
                if check_time() or calls_made >= budget:
                    break
                
                simplex = simplexes[s_idx]
                
                # Sort simplex by value
                simplex.sort(key=lambda x: x[1])
                
                # Identify best, worst, second_worst
                x_best, f_best = simplex[0]
                x_worst, f_worst = simplex[-1]
                x_second_worst, f_second_worst = simplex[-2]
                
                # Calculate centroid of all points except the worst
                centroid = np.mean([pt for pt, _ in simplex[:-1]], axis=0)
                
                # 1. Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                
                if f_ref is None:
                    continue
                
                if f_ref < f_second_worst:
                    # Reflection was good, try expansion
                    x_exp = centroid + gamma * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    
                    if f_exp is None:
                        # Out of budget, accept reflection
                        simplex[-1] = (x_ref, f_ref)
                        continue
                    
                    if f_exp < f_ref:
                        simplex[-1] = (x_exp, f_exp)
                    else:
                        simplex[-1] = (x_ref, f_ref)
                else:
                    # Reflection was not good enough, try contraction
                    if f_ref < f_worst:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = safe_call(x_con)
                        
                        if f_con is None:
                            simplex[-1] = (x_ref, f_ref)
                            continue
                        
                        if f_con <= min(f_ref, f_worst):
                            simplex[-1] = (x_con, f_con)
                        else:
                            # Shrink
                            new_simplex = [(x_best, f_best)]
                            for pt, val in simplex[1:-1]:
                                x_shr = x_best + sigma * (pt - x_best)
                                x_shr = np.clip(x_shr, -5, 5)
                                f_shr = safe_call(x_shr)
                                if f_shr is None:
                                    break
                                new_simplex.append((x_shr, f_shr))
                                if check_time() or calls_made >= budget:
                                    break
                            else:
                                simplex = new_simplex
                            simplexes[s_idx] = simplex
                            continue
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - x_worst)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = safe_call(x_con)
                        
                        if f_con is None:
                            simplex[-1] = (x_ref, f_ref)
                            continue
                        
                        if f_con < f_worst:
                            simplex[-1] = (x_con, f_con)
                        else:
                            # Shrink
                            new_simplex = [(x_best, f_best)]
                            for pt, val in simplex[1:-1]:
                                x_shr = x_best + sigma * (pt - x_best)
                                x_shr = np.clip(x_shr, -5, 5)
                                f_shr = safe_call(x_shr)
                                if f_shr is None:
                                    break
                                new_simplex.append((x_shr, f_shr))
                                if check_time() or calls_made >= budget:
                                    break
                            else:
                                simplex = new_simplex
                            simplexes[s_idx] = simplex
                            continue

                # Check for convergence of this simplex
                # If the range of values is very small, the simplex has converged
                f_vals = [val for _, val in simplex]
                if max(f_vals) - min(f_vals) < 1e-8:
                    # Mark as converged, maybe remove it or keep it for safety
                    # For now, we just stop improving it in subsequent iterations by checking this condition
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
