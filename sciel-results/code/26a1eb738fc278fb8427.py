"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for Büche-Rastrigin (BBOB f4) using Halton initialization 
    and multi-start Nelder-Mead refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        rng = np.random.default_rng(12345)
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or check_deadline():
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Global Exploration using Halton Sequence ---
        # Halton sequences provide low-discrepancy coverage, which is excellent
        # for finding promising basins in multimodal functions like Rastrigin.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def generate_halton_points(n, dim):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, primes[j])
            return points

        # Use 60 calls for initial exploration
        n_initial = 60
        halton_points = generate_halton_points(n_initial, dim)
        # Map [0, 1]^d to [-5, 5]^d
        halton_points = halton_points * 10.0 - 5.0
        
        # Add origin and a few random points for extra diversity
        extra_points = [np.zeros(dim)]
        extra_points.extend(rng.uniform(-5, 5, size=(5, dim)))
        
        all_initial_points = np.vstack([halton_points, np.array(extra_points)])
        # Shuffle to avoid bias in order (though Halton is already well-distributed)
        rng.shuffle(all_initial_points)

        candidates = []
        
        for x in all_initial_points:
            fx = evaluate(x)
            if fx is None:
                break
            candidates.append((fx, x.copy()))

        if best_x is None:
            return None, np.inf

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # --- Phase 2: Local Refinement using Nelder-Mead ---
        # Select top K candidates for deep optimization
        # With 500 budget and 65 used, we have ~435 left.
        # Nelder-Mead in 5D uses (dim+1) points in simplex.
        # Each iteration uses 1-3 function calls.
        # We can afford to optimize top 3-5 candidates.
        
        top_k = 5
        top_candidates = candidates[:top_k]
        
        # Allocate remaining budget
        remaining_budget = budget - calls_made
        # Reserve some budget for potential random restarts if NM fails
        reserve_calls = 20
        nm_budget = max(0, remaining_budget - reserve_calls)
        
        # Distribute budget among top candidates.
        # Give more budget to the best candidate.
        # Weights: 50%, 30%, 10%, 5%, 5%
        weights = [0.5, 0.3, 0.1, 0.05, 0.05]
        if len(top_candidates) < 5:
             # Adjust weights if fewer candidates
             weights = [1.0 / len(top_candidates)] * len(top_candidates)

        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x, deadline
            
            if check_deadline() or calls_made >= budget:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = []
            
            fx0 = evaluate(x0)
            if fx0 is None:
                return
            simplex_f.append(fx0)
            
            # Add perturbed points to build simplex
            # Perturb each dimension
            for i in range(dim):
                if check_deadline() or calls_made >= budget:
                    return
                perturbation = x0.copy()
                # If x0[i] is 0, step size 0.5, else 0.5 * sign(x0[i])
                step = 0.5 if x0[i] >= 0 else -0.5
                perturbation[i] += step
                perturbation = np.clip(perturbation, -5, 5)
                
                fp = evaluate(perturbation)
                if fp is None:
                    return
                simplex.append(perturbation)
                simplex_f.append(fp)
                
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            # Main NM loop
            while calls_made < budget and not check_deadline():
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Reflection
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if check_deadline() or calls_made >= budget:
                    break
                f_reflect = evaluate(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()
                
                if simplex_f[0] <= f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_expand = evaluate(x_expand)
                    if f_expand is None:
                        break
                    
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    # Inside contraction if f_reflect < f_worst, else outside
                    if f_reflect < f_worst:
                        x_contract = centroid + 0.5 * (x_reflect - centroid)
                    else:
                        x_contract = centroid - 0.5 * (x_worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_contract = evaluate(x_contract)
                    if f_contract is None:
                        break
                    
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = x_contract.copy()
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if check_deadline() or calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                break
                            
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                            
                            simplex_f[i] = f_shrink
                        
                        if check_deadline() or calls_made >= budget:
                            break

        # Run NM on top candidates
        for i, (f_init, x_init) in enumerate(top_candidates):
            if check_deadline() or calls_made >= budget:
                break
                
            # Allocate budget for this candidate
            alloc = int(nm_budget * weights[i])
            # Ensure we don't use more than available
            alloc = min(alloc, budget - calls_made)
            
            if alloc < 10: # Not enough for a meaningful NM run
                continue

            # We don't pass max_calls directly to the while loop condition 
            # but we track calls_made. To strictly limit per-candidate, 
            # we can just let it run until global budget/deadline or 
            # we add a local counter. 
            # Given the small number of candidates, letting it run until 
            # global budget is hit is fine, but we might want to stop 
            # if no improvement is made for a while.
            
            # Simple stagnation check: save best_f before run
            f_before = best_f
            n_iter_start = calls_made
            
            # Run NM
            # The NM function above runs until global budget is hit.
            # We need a way to stop it early if it's not making progress 
            # to save budget for other candidates.
            # However, implementing a strict "stop after X calls" in the 
            # inner loop is tricky with the nonlocal state.
            # Alternative: Just run the best candidate first with most budget.
            # If it improves, great. If not, the next candidate gets less.
            
            # Let's modify the approach: 
            # We will run NM on the *current* best_x (which might have been 
            # improved by previous NM runs). 
            # Actually, it's better to run NM on the specific x_init.
            
            # To enforce budget per candidate, we can wrap the NM logic 
            # or just rely on the fact that NM converges quickly or stalls.
            # If it stalls, it wastes calls. 
            # Let's add a simple "no improvement" check in the main loop 
            # of the Optimizer, not inside NM, to switch candidates.
            
            # For now, run NM. It will stop if global budget is hit.
            # To prevent one bad candidate from eating all budget, 
            # we can limit the number of iterations in NM.
            # Let's create a version of NM that respects a local call limit.
            
            local_calls = 0
            local_limit = alloc
            
            # Re-implement NM loop inline to respect local_limit? 
            # That's too much code duplication.
            # Instead, we can just run NM. If the first candidate uses 
            # all budget, that's a risk. 
            # Better: Run NM with a strict local limit by checking 
            # calls_made - n_iter_start < local_limit.
            
            # We need to modify the evaluate function or pass a limit.
            # Let's just run the standard NM but break out if 
            # (calls_made - n_iter_start) >= local_limit.
            # This requires modifying the NM function to accept a limit.
            
            # Since I can't easily modify the inner function definition 
            # to take a limit without restructuring, I will rely on 
            # the fact that NM in 5D usually converges in < 50-100 calls.
            # If it doesn't, it's likely stuck. 
            # To be safe, I will manually break out of the NM loop 
            # by checking the global budget, but that doesn't help 
            # with per-candidate limits.
            
            # Let's stick to the previous plan: 
            # 1. Halton (60 calls)
            # 2. NM on best point (remaining budget)
            # If the best point is in a local minimum, NM will fail.
            # To mitigate, we should pick the top 3 and run NM on them sequentially, 
            # but stop early if no improvement.
            
            # Let's refine the NM call to include a "max_iter" or "max_calls" 
            # check inside the loop.
            
            # I will rewrite the NM part to be a helper that takes a max_calls.
            
            pass # Placeholder for logic restructuring

        # Since I can't easily insert the "local limit" check into the 
        # existing `nelder_mead` function defined above without 
        # significant refactoring, I will use a simpler strategy:
        # Run NM on the global best point found in Phase 1.
        # If it improves, great.
        # If it doesn't improve significantly after, say, 50 calls, 
        # we stop and try the next best candidate.
        
        # Let's implement a wrapper that runs NM but stops if 
        # (calls_made - start_calls) > limit.
        
        def run_nm_limited(x0, limit):
            nonlocal calls_made, best_f, best_x
            
            start_calls = calls_made
            
            if check_deadline() or calls_made >= budget:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = []
            
            fx0 = evaluate(x0)
            if fx0 is None:
                return
            simplex_f.append(fx0)
            
            for i in range(dim):
                if check_deadline() or calls_made >= budget or (calls_made - start_calls) >= limit:
                    return
                perturbation = x0.copy()
                step = 0.5 if x0[i] >= 0 else -0.5
                perturbation[i] += step
                perturbation = np.clip(perturbation, -5, 5)
                
                fp = evaluate(perturbation)
                if fp is None:
                    return
                simplex.append(perturbation)
                simplex_f.append(fp)
                
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls_made < budget and not check_deadline():
                if (calls_made - start_calls) >= limit:
                    break
                    
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                x_reflect = centroid + alpha * (centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                
                if check_deadline() or calls_made >= budget:
                    break
                f_reflect = evaluate(x_reflect)
                if f_reflect is None:
                    break
                
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()
                
                if simplex_f[0] <= f_reflect < simplex_f[-2]:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[0]:
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_expand = evaluate(x_expand)
                    if f_expand is None:
                        break
                    
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = x_expand.copy()
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                else:
                    if f_reflect < f_worst:
                        x_contract = centroid + 0.5 * (x_reflect - centroid)
                    else:
                        x_contract = centroid - 0.5 * (x_worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    
                    if check_deadline() or calls_made >= budget:
                        break
                    f_contract = evaluate(x_contract)
                    if f_contract is None:
                        break
                    
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = x_contract.copy()
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        for i in range(1, len(simplex)):
                            if check_deadline() or calls_made >= budget or (calls_made - start_calls) >= limit:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            f_shrink = evaluate(simplex[i])
                            if f_shrink is None:
                                break
                            
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = simplex[i].copy()
                            
                            simplex_f[i] = f_shrink
                        
                        if check_deadline() or calls_made >= budget:
                            break

        # Execute NM on top candidates with limited budget
        for i, (f_init, x_init) in enumerate(top_candidates):
            if check_deadline() or calls_made >= budget:
                break
                
            alloc = int(nm_budget * weights[i])
            alloc = min(alloc, budget - calls_made)
            
            if alloc < 10:
                continue

            run_nm_limited(x_init, alloc)

        # Final check: if we have spare budget, do a few random jumps 
        # from the best point to see if we can escape a local minimum
        if calls_made < budget and not check_deadline():
            for _ in range(min(10, budget - calls_made)):
                if check_deadline() or calls_made >= budget:
                    break
                x_jump = best_x + rng.standard_normal(dim) * 0.5
                x_jump = np.clip(x_jump, -5, 5)
                evaluate(x_jump)

        return best_x, best_f

# EVOLVE-BLOCK-END
