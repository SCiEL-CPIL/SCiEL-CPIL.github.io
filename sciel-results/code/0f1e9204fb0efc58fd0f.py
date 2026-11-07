"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Nelder-Mead Simplex with Global Random Restart.
    
    Strategy:
    1. Global Phase: Random sampling to find a promising basin.
    2. Local Phase: Nelder-Mead Simplex method for fast, gradient-free convergence.
       - NM maintains a simplex of dim+1 points.
       - It reflects, expands, and contracts vertices based on function values.
       - This implicitly adapts to the local landscape geometry (rotation/scale) 
         without explicit covariance estimation, allowing for very fast convergence 
         in the final phase.
    3. Restart: If NM stagnates or budget allows, restart from best point or new random point.
    
    Why this differs from 1+1 ES:
    - NM uses a structured set of points (simplex) rather than a single trajectory.
    - NM's reflection/expansion/contraction is deterministic given the function values,
      allowing it to "rotate" the search direction to align with valley floors more efficiently
      than random 1+1 steps in the final exploitation phase.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Helper to check time and budget
        def check_limit():
            nonlocal calls_made
            return calls_made >= budget or time.time() > deadline

        # Nelder-Mead implementation
        def nelder_mead(x0, fx0, remaining_budget, rng):
            nonlocal best_x, best_fx, calls_made
            
            # Initialize simplex
            # Start with a small simplex around x0
            alpha = 0.1
            gamma = 0.5
            simplex = [x0.copy()]
            fvals = [fx0]
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += alpha
                if xi[i] > 5.0: xi[i] = 5.0
                if xi[i] < -5.0: xi[i] = -5.0
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                fvals.append(fi)
                
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Nelder-Mead parameters
            alpha_reflect = 1.0
            gamma_expand = 2.0
            rho_contract = 0.5
            sigma_shrink = 0.5
            
            while not check_limit():
                # Sort again to ensure order
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect the worst point
                worst_idx = -1
                x_worst = simplex[worst_idx]
                f_worst = fvals[worst_idx]
                
                x_reflect = centroid + alpha_reflect * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                f_reflect = func(x_reflect)
                calls_made += 1
                
                if f_reflect < best_fx:
                    best_fx = f_reflect
                    best_x = x_reflect.copy()

                # Check if reflection is good enough
                if fvals[0] <= f_reflect < fvals[-2]:
                    # Replace worst with reflected
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                elif f_reflect < fvals[0]:
                    # Try expansion
                    x_expand = centroid + gamma_expand * (centroid - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    calls_made += 1
                    
                    if f_expand < best_fx:
                        best_fx = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                else:
                    # Reflection is not good, try contraction
                    if f_reflect < f_worst:
                        # Outside contraction
                        x_contract = centroid + rho_contract * (x_reflect - centroid)
                        x_contract = np.clip(x_contract, -5, 5)
                        f_contract = func(x_contract)
                        calls_made += 1
                        
                        if f_contract < best_fx:
                            best_fx = f_contract
                            best_x = x_contract.copy()
                        
                        if f_contract <= f_reflect:
                            simplex[-1] = x_contract
                            fvals[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                calls_made += 1
                                if fvals[i] < best_fx:
                                    best_fx = fvals[i]
                                    best_x = simplex[i].copy()
                    else:
                        # Inside contraction
                        x_contract = centroid - rho_contract * (x_worst - centroid)
                        x_contract = np.clip(x_contract, -5, 5)
                        f_contract = func(x_contract)
                        calls_made += 1
                        
                        if f_contract < best_fx:
                            best_fx = f_contract
                            best_x = x_contract.copy()
                        
                        if f_contract < f_worst:
                            simplex[-1] = x_contract
                            fvals[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                fvals[i] = func(simplex[i])
                                calls_made += 1
                                if fvals[i] < best_fx:
                                    best_fx = fvals[i]
                                    best_x = simplex[i].copy()
                
                # Check for convergence: if simplex is very small, break
                spread = np.max(np.abs(np.array(simplex) - simplex[0]))
                if spread < 1e-4:
                    break

            return best_x, best_fx

        # Phase 1: Global Search / Initialization
        # Use a portion of the budget to find a good starting point
        # Then use NM for refinement.
        
        # We will alternate:
        # 1. Random sample
        # 2. Run NM for a limited number of steps
        # 3. If improved, continue NM or restart
        # 4. If not, new random sample
        
        # Let's do 5 major restarts.
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if check_limit():
                break
            
            # Choose starting point
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
            else:
                # Mix of random and perturbation of best
                if rng.random() < 0.5:
                    x0 = rng.uniform(-5, 5, dim)
                else:
                    if best_x is not None:
                        x0 = best_x + rng.normal(0, 1.0, dim)
                        x0 = np.clip(x0, -5, 5)
                    else:
                        x0 = rng.uniform(-5, 5, dim)
            
            fx0 = func(x0)
            calls_made += 1
            
            if fx0 < best_fx:
                best_fx = fx0
                best_x = x0.copy()
            
            # Run Nelder-Mead from this point
            # Limit the NM run to avoid getting stuck in a local min for too long
            # but allow it to converge if it's good.
            # We pass the remaining budget, but NM will stop if converged or time/budget hits.
            # To ensure diversity, we might want to cap NM steps per restart?
            # Actually, NM is efficient. Let it run until it converges or budget is low.
            # But if we use all budget in restart 1, we can't explore others.
            # So, let's limit NM to a fraction of the restart budget?
            # No, NM is fast. Let's just run it. If it converges, we break out.
            # If it doesn't, it will stop when budget is exhausted.
            
            # To balance exploration, let's cap the NM steps per restart to budget_per_restart
            # But NM doesn't have a simple "step" counter in the same way.
            # We'll just rely on the budget check inside NM.
            # To prevent one NM from eating all budget, we can pass a "soft" limit.
            # However, the global `calls_made` check in `check_limit` is hard.
            # Let's modify NM to accept a max_steps?
            # Simpler: Just run NM. If it finds a global min early, great.
            # If it finds a local min, it will converge and break out of the while loop in NM.
            # Then we go to next restart.
            
            nelder_mead(x0, fx0, budget_per_restart, rng)
            
            # If we are still far from convergence (simplex spread large), 
            # maybe we should have restarted sooner? 
            # The NM function breaks if spread < 1e-4.
            # If it doesn't break, it means it's still moving or budget/time hit.
            # If budget/time hit, the outer loop will break too.

        return best_x, best_fx
# EVOLVE-BLOCK-END
