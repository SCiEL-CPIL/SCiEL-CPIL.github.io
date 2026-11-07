"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (JADE) with Nelder-Mead polish for Step Ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls += 1

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        def _eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            # Clip to box before calling func to ensure validity and consistency
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Differential Evolution (JADE) ---
        
        # JADE parameters
        Np = max(10, 5 * dim)  # Population size, at least 10
        F = 0.5  # Scale factor
        CR = 0.5  # Crossover rate
        tau1 = 0.1
        tau2 = 0.1
        
        # Initialize population
        # Start with a mix of random points and points near origin
        pop = []
        fpop = []
        
        # First individual is the initial best
        x0 = np.zeros(dim)
        f0 = best_f
        pop.append(x0)
        fpop.append(f0)
        
        for _ in range(Np - 1):
            x = rng.uniform(-5, 5, dim)
            fx = _eval(x)
            if fx is None:
                # If budget ran out, break
                break
            pop.append(x)
            fpop.append(fx)
            
        # If we couldn't fill the population, adjust Np
        Np = len(pop)
        if Np < 4:
             # Fallback to local search if DE can't start
             pass
        else:
             # Sort population initially
             indices = np.argsort(fpop)
             pop = [pop[i] for i in indices]
             fpop = [fpop[i] for i in indices]
             
             # JADE loop
             # We reserve some budget for Nelder-Mead
             nm_budget = 100
             de_budget = budget - nm_budget - calls
             
             while calls < de_budget and time.time() < deadline:
                 # Generate new population
                 new_pop = []
                 new_fpop = []
                 
                 # Keep the best individual (elitism)
                 new_pop.append(pop[0].copy())
                 new_fpop.append(fpop[0])
                 
                 # Collect successful F and CR values for adaptation
                 F_succ = []
                 CR_succ = []
                 
                 for i in range(1, Np):
                     # Mutation: DE/rand/1
                     # Choose 3 distinct indices different from i
                     r1, r2, r3 = rng.choice([j for j in range(Np) if j != i], size=3, replace=False)
                     
                     v = pop[r1] + F * (pop[r2] - pop[r3])
                     
                     # Adapt CR for this trial
                     cr_trial = rng.uniform(0, 1)
                     
                     # Crossover: Binomial
                     j_rand = rng.integers(0, dim)
                     u = np.empty(dim)
                     for j in range(dim):
                         if rng.uniform() <= cr_trial or j == j_rand:
                             u[j] = v[j]
                         else:
                             u[j] = pop[i][j]
                     
                     u = np.clip(u, -5, 5)
                     
                     fu = _eval(u)
                     if fu is None:
                         break
                     
                     # Selection
                     if fu <= fpop[i]:
                         new_pop.append(u)
                         new_fpop.append(fu)
                         F_succ.append(F)
                         CR_succ.append(cr_trial)
                     else:
                         new_pop.append(pop[i])
                         new_fpop.append(fpop[i])
                         
                 if calls >= de_budget or time.time() > deadline:
                     break
                     
                 # Update population
                 pop = new_pop
                 fpop = new_fpop
                 
                 # Sort population
                 indices = np.argsort(fpop)
                 pop = [pop[i] for i in indices]
                 fpop = [fpop[i] for i in indices]
                 
                 # Update best
                 if fpop[0] < best_f:
                     best_f = fpop[0]
                     best_x = pop[0].copy()
                     
                 # Adapt F and CR (JADE)
                 if F_succ:
                     F_mean = np.mean(F_succ)
                     F_std = np.std(F_succ)
                     # Cauchy distribution for F, Normal for CR (simplified here to uniform/normal mix)
                     # Standard JADE uses Cauchy for F. Let's use a simpler adaptation to avoid complexity/instability.
                     F = np.clip(F + tau1 * (F_mean - F), 0.1, 0.9)
                     
                     if CR_succ:
                         CR_mean = np.mean(CR_succ)
                         CR = np.clip(CR + tau2 * (CR_mean - CR), 0.0, 1.0)

        # --- Phase 2: Nelder-Mead Polish ---
        
        if calls < budget - 5 and time.time() < deadline - 5:
            # Initialize simplex around best_x
            # Simplex size
            alpha = 0.5
            beta = 0.5
            
            simplex = []
            f_simplex = []
            
            # Vertex 0: best_x
            x0 = best_x.copy()
            f0 = best_f
            simplex.append(x0)
            f_simplex.append(f0)
            
            # Other vertices
            for i in range(dim):
                xi = best_x.copy()
                if abs(best_x[i]) > 1e-3:
                    xi[i] = xi[i] * (1 + alpha)
                else:
                    xi[i] = alpha
                simplex.append(xi)
                fx = _eval(xi)
                if fx is None:
                    break
                f_simplex.append(fx)
                
            # If we didn't get enough vertices, skip NM or restart with smaller step
            if len(simplex) < dim + 1:
                # Try to fill remaining vertices with smaller steps
                for i in range(len(simplex) - 1, dim):
                     xi = best_x.copy()
                     xi[i] = best_x[i] + 0.1
                     simplex.append(xi)
                     fx = _eval(xi)
                     if fx is None:
                         break
                     f_simplex.append(fx)
                     if len(simplex) > dim:
                         break
            
            if len(simplex) >= dim + 1:
                # Nelder-Mead Iteration
                max_nm_calls = budget - calls
                nm_iter = 0
                while nm_iter < max_nm_calls and calls < budget and time.time() < deadline:
                    # Sort simplex by function value
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    x_worst = simplex[-1]
                    f_worst = f_simplex[-1]
                    
                    # Reflection
                    gamma = 1.0
                    x_r = centroid + gamma * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = _eval(x_r)
                    if f_r is None:
                        break
                    
                    if f_simplex[0] <= f_r < f_simplex[-2]:
                        # Accept reflection
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    elif f_r < f_simplex[0]:
                        # Expansion
                        gamma_e = 2.0
                        x_e = centroid + gamma_e * (centroid - x_worst)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = _eval(x_e)
                        if f_e is None:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    else:
                        # Contraction
                        # Outside contraction if f_worst < f_r, else inside
                        gamma_c = 0.5
                        if f_r < f_worst:
                            x_c = centroid + gamma_c * (x_r - centroid)
                        else:
                            x_c = centroid - gamma_c * (x_worst - centroid)
                        
                        x_c = np.clip(x_c, -5, 5)
                        f_c = _eval(x_c)
                        if f_c is None:
                            break
                        
                        if f_c < min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = _eval(simplex[i])
                                if f_simplex[i] is None:
                                    break
                            if f_simplex[1] is None:
                                break
                    
                    # Update global best
                    if f_simplex[0] < best_f:
                        best_f = f_simplex[0]
                        best_x = simplex[0].copy()
                        
                    nm_iter += 1
                    # Check convergence (optional, but helps stop early if stuck)
                    # If simplex is very small, stop
                    spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                    if np.max(spread) < 1e-6:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
