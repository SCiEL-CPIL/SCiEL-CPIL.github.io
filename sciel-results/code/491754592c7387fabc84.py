"""Multi-objective optimizer for ZDT4 using Memetic Strategy (Global + Local Search)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Memetic Multi-Objective Optimizer.
    
    Strategy:
    1. Global Search: Use a small population to find diverse points on the Pareto front.
       We use a simple random initialization followed by a few iterations of a lightweight
       evolutionary step (or just random sampling if budget is tight) to get 'seeds'.
    2. Local Search: For each seed found, apply a Nelder-Mead simplex method to minimize
       a scalarized objective (Tchebycheff or weighted sum) or directly minimize the 
       secondary objective f2 while keeping f1 close to the seed's f1. 
       
       For ZDT4, f1 depends only on x1, and f2 depends on x2..x10.
       The Pareto front is defined by x1 in [0,1] and x2..x10 in {0.5} (since sin(10*0.5)=1 is max? 
       Wait, ZDT4 f2 = 1 + 10(n-1) - (sum xi^2 - 2 prod cos(20 pi xi)).
       Min f2 requires sum xi^2 - 2 prod cos(...) to be max.
       Max is achieved when cos(20 pi xi) = 1 => 20 pi xi = 2 pi k => xi = k/10.
       In [0,1], xi=0.5 gives cos(pi)=-1? No.
       Let's check ZDT4 definition:
       f1(x) = 1 + 10(n-1) - (sum_{i=2}^n (x_i^2 - 2 cos(20 pi x_i)))
       Wait, standard ZDT4:
       f1(x) = x1
       g(x) = 1 + 10(n-1) - (sum_{i=2}^n (x_i^2 - 2 cos(20 pi x_i)))
       f2(x) = g(x) (1 - sqrt(f1(x)/g(x)))
       
       To minimize f2, we need to minimize g(x).
       g(x) is minimized when the term (sum ... - 2 prod ...) is maximized.
       The term is sum (x_i^2 - 2 cos(20 pi x_i)).
       To maximize this, we want x_i^2 large and cos(20 pi x_i) small (negative).
       In [0,1], x_i^2 is max at 1 or 0? 1^2=1, 0^2=0.
       cos(20 pi x_i) is min (-1) when 20 pi x_i = pi, 3pi, ... => x_i = 1/20, 3/20, ..., 19/20.
       At x_i=0.5, cos(10 pi) = 1. So 0.5 is a local MAX of cos, thus local MIN of -2cos.
       Wait, we want to MAXIMIZE (x_i^2 - 2 cos(...)).
       So we want cos to be -1.
       x_i = 0.5 => cos(10 pi) = 1. Term = 0.25 - 2(1) = -1.75.
       x_i = 0.05 => cos(pi) = -1. Term = 0.0025 - 2(-1) = 2.0025.
       So local minima of g (which are local maxima of the subtracted term) are at odd multiples of 0.05.
       The global minimum of g is at x_i = 0.5? No, at x_i where cos is -1.
       Actually, the global minimum of g is 1. This happens when x_i = 0.5 for all i?
       Let's re-evaluate.
       Standard ZDT4 Pareto front: f1 = x1, f2 = (1+x1^2)(1 - sqrt(x1/(1+x1^2)))? No, that's ZDT1/2/3.
       ZDT4: f1 = x1. g = 1 + 10(n-1) - (sum (x_i^2 - 2 cos(20 pi x_i))).
       Min g = 1. This requires sum (x_i^2 - 2 cos(20 pi x_i)) = 10(n-1).
       Max value of (x_i^2 - 2 cos(20 pi x_i)) for x in [0,1]:
       x=1: 1 - 2(1) = -1.
       x=0.5: 0.25 - 2(1) = -1.75.
       x=0: 0 - 2(1) = -2.
       Wait, if x_i are in [0,1], the max of x_i^2 is 1. The min of -2cos is -2.
       So max term is 1 - 2 = -1?
       If so, sum is -10(n-1). g = 1 + 10(n-1) - (-10(n-1)) = 1 + 20(n-1). That's huge.
       
       Let's check the standard definition again.
       ZDT4:
       f1(x) = x1
       g(x) = 1 + 10(n-1) - (sum_{i=2}^n (x_i^2 - 2 cos(20 pi x_i)))
       f2(x) = g(x) (1 - sqrt(f1(x)/g(x)))
       
       Bounds: x1 in [0,1], xi in [0,1] for i>1? Or [-5,5]?
       Usually ZDT4 has x1 in [0,1] and xi in [-5,5] for i>1?
       No, the task says "bounds given per variable". The prompt says "10 variables, bounds given per variable".
       Standard ZDT4 often has x1 in [0,1] and others in [-5,5].
       If bounds are [-5,5], then x_i^2 can be 25.
       cos(20 pi x_i) is in [-1, 1].
       Term x_i^2 - 2 cos(...).
       To maximize this, we want x_i^2 large and cos small (-1).
       If x_i = 5, x_i^2 = 25. cos(100 pi) = 1. Term = 25 - 2 = 23.
       If x_i = 4.9, x_i^2 ~ 24. cos(98 pi) = 1.
       The function has many local optima.
       
       However, the Pareto front for ZDT4 is typically found when the secondary variables are set to their "optimal" values that minimize g.
       The global minimum of g is 1.
       This implies sum (x_i^2 - 2 cos(20 pi x_i)) = 10(n-1).
       If n=10, sum = 90.
       Average term = 10.
       If x_i in [-5,5], can we get 10?
       x_i^2 - 2 cos(...).
       If x_i=0, term = -2.
       If x_i=1, term = 1 - 2(1) = -1.
       If x_i=0.5, term = 0.25 - 2(1) = -1.75.
       If x_i=1.5, term = 2.25 - 2(1) = 0.25.
       If x_i=2.5, term = 6.25 - 2(1) = 4.25.
       If x_i=3.5, term = 12.25 - 2(1) = 10.25.
       If x_i=4.5, term = 20.25 - 2(1) = 18.25.
       So to get sum 90 with 9 variables, average 10.
       We can mix values.
       
       Key Insight for ZDT4: The problem is highly deceptive. The global Pareto front is isolated from the majority of the search space. Random search is very unlikely to hit the specific valley where g is minimized. DE might struggle to jump between the many local optima.
       
       Memetic approach:
       1. Initialize a set of random points.
       2. For each point, perform a local search to minimize g(x) (which is f2/g? No, minimize g to get close to PF).
          Actually, minimizing f2 directly is hard because f2 depends on f1.
          Better: Minimize g(x) for fixed x1.
          If we can find the global minimum of g(x) for a given x1, we get a point on the Pareto Front (or very close).
          Since g does not depend on x1, we can find the global minimum of g(x_{2:10}) once, and then vary x1 to sweep the front.
          
          Strategy:
          1. Find the global minimum of g(x_{2:10}) using a robust local search (Nelder-Mead) from multiple random starts.
          2. Once we have the optimal x_{2:10} (let's call it x_opt_sec), we know the minimum value g_min.
          3. The Pareto front is parameterized by x1 in [0, 1] (assuming bound is [0,1] for x1).
          4. For a range of x1 values, evaluate f(x1, x_opt_sec).
          5. This gives us a dense set of points on the true Pareto Front.
          
          This is extremely efficient for ZDT4 because it decouples the problem.
          The "deception" is in the secondary variables. Once we solve for the secondary variables, the front is trivial to sweep.
          
          Implementation:
          1. Use ~50% of budget to find the best x_{2:10} that minimizes g.
             - Use multiple random starts for Nelder-Mead.
             - Keep the best g found.
          2. Use remaining budget to sweep x1.
             - Pick several x1 values (e.g., 10-20 points) in [lo[0], hi[0]].
             - Evaluate problem(x1, best_x_sec).
             - Also, maybe refine x_sec for each x1? No, g is independent of x1. So x_opt_sec is constant.
             
          Wait, is g independent of x1?
          ZDT4: g(x) = 1 + 10(n-1) - (sum_{i=2}^n ...). Yes, sum is from i=2 to n. So g depends only on x2..xn.
          f1 = x1.
          f2 = g (1 - sqrt(f1/g)).
          
          So yes, the Pareto Front is exactly the curve obtained by fixing x2..xn at the global minimizers of g and varying x1.
          
          This is a "cheat" specific to ZDT4 structure, but it is a valid optimization strategy: exploit problem structure.
          
          Steps:
          1. Parse bounds. Assume x1 is the first variable.
          2. Define local function g_only(x_sec) which calls problem to get f2 and f1? 
             No, problem(x) returns [f1, f2].
             We can compute g from f1, f2?
             f2 = g (1 - sqrt(f1/g))
             Let s = sqrt(f1/g). f2 = g - g*s = g - sqrt(f1*g).
             f2^2 = g^2 - 2g sqrt(f1*g) + f1*g
             This is messy to invert.
             Easier: Just evaluate problem(x1, x_sec) and look at f2?
             No, we want to minimize g.
             We can construct a dummy x with x1 fixed to a value (e.g., 0.5) and vary x_sec.
             But f1 changes if we change x1.
             We can just call problem(x) with x1 fixed and x_sec varying.
             We want to minimize g.
             Note: f2 = g (1 - sqrt(x1/g)).
             If x1=0, f2 = g. So minimizing f2 at x1=0 is equivalent to minimizing g.
             So: Fix x1 = lo[0] (or any valid value, preferably 0 if in bounds, else lo).
             Minimize f2(x) subject to x1 fixed.
             This finds the global minimum of g.
             
          3. Algorithm:
             - Budget B.
             - Phase 1: Find x_sec_opt.
               - Use Nelder-Mead on the 9-dimensional subspace x2..x9.
               - Objective: f2 evaluated with x1 fixed to lo[0].
               - Use multiple random starts (e.g., 10 starts).
               - Allocate B/2 evaluations to this phase.
               - Keep the best x_sec found (min f2).
             - Phase 2: Sweep x1.
               - Use the remaining B/2 evaluations.
               - Select K points for x1 in [lo[0], hi[0]].
               - For each x1_k, evaluate problem(x1_k, x_sec_opt).
               - Record these points.
               - Also, maybe do a few local refinements of x_sec for different x1? 
                 Not necessary if g is independent. But numerical noise or bounds clipping might matter.
                 The problem clips x to bounds. So if x_sec_opt is slightly out of bounds, it gets clipped.
                 Nelder-Mead should respect bounds if we implement a bounded version or use penalty.
                 Simpler: Use scipy.optimize.minimize with method='Nelder-Mead' and bounds?
                 Nelder-Mead in scipy does not support bounds directly in older versions, but we can use 'Powell' or 'L-BFGS-B' or just penalize.
                 Actually, `scipy.optimize.minimize` with method='Nelder-Mead' ignores bounds.
                 We can use a penalty method or just clip inside the objective function (but that creates flat regions).
                 Better: Use `scipy.optimize.minimize` with method='Powell' or 'L-BFGS-B' if we have gradients? No gradients.
                 Let's use a simple bounded Nelder-Mead implementation or just rely on the fact that the optimum is likely inside.
                 Given the deceptive nature, the global min of g is likely at specific points like 0.5, 1.5, etc.
                 
                 Let's stick to a robust local search.
                 I will implement a simple bounded Nelder-Mead or use `scipy.optimize.minimize` with `method='Nelder-Mead'` and hope the bounds are not tight, or implement a simple penalty.
                 Actually, the task says "problem(x) clips... to bounds". So the objective function is flat outside bounds.
                 Nelder-Mead might get stuck on the flat boundary.
                 To mitigate, initialize simplex strictly inside bounds.
                 
             - Phase 3: Refine.
               - If budget allows, do a few more random sweeps or local searches.
               
          4. Return the archive of points.

    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        used = 0
        archive_F = []
        archive_X = []

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def update_archive(f_new, x_new):
            nonlocal archive_F, archive_X
            # Check domination
            for i in range(len(archive_F)):
                if dominates(archive_F[i], f_new):
                    return
            # Remove dominated
            new_F = []
            new_X = []
            for i in range(len(archive_F)):
                if not dominates(f_new, archive_F[i]):
                    new_F.append(archive_F[i])
                    new_X.append(archive_X[i])
            new_F.append(f_new)
            new_X.append(x_new)
            archive_F = new_F
            archive_X = new_X

        # Phase 1: Find optimal secondary variables (x2..xn)
        # We assume x1 is the first variable.
        # Fix x1 to lo[0] to minimize f2 (which equals g at x1=0).
        # If lo[0] is not 0, f2 = g(1 - sqrt(x1/g)). Minimizing f2 still correlates with minimizing g.
        # Let's fix x1 to a value in the middle of its range to be safe?
        # No, x1=0 is best for isolating g. If 0 is not in bounds, use lo[0].
        x1_fix = lo[0]
        
        def objective_sec(x_sec):
            nonlocal used
            if is_deadline_exceeded():
                return 1e10
            x_full = np.concatenate(([x1_fix], x_sec))
            # Clip to bounds to be safe, though problem does it
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            return f[1]  # Minimize f2

        # Use multiple random starts for local search
        # Budget allocation: 60% for local search, 40% for sweeping
        n_starts = 10
        evals_per_start = max(50, (budget * 0.6) // n_starts)
        
        best_x_sec = None
        best_f2 = np.inf

        for i in range(n_starts):
            if is_deadline_exceeded():
                break
            if used >= budget * 0.6:
                break

            # Random start inside bounds
            x_sec_start = lo[1:] + span[1:] * np.random.random(dim - 1)
            
            # Use scipy.optimize.minimize with Nelder-Mead
            # We need to handle bounds. Nelder-Mead doesn't support bounds.
            # We will use a penalty function or just rely on clipping in objective_sec.
            # Clipping in objective_sec makes the function flat outside bounds, which can trap NM.
            # To avoid this, we can use a soft penalty.
            
            def penalized_obj(x_sec):
                # x_sec is the variable
                # Check bounds
                pen = 0.0
                for j in range(len(x_sec)):
                    if x_sec[j] < lo[j+1]:
                        pen += (lo[j+1] - x_sec[j])**2 * 100
                    if x_sec[j] > hi[j+1]:
                        pen += (x_sec[j] - hi[j+1])**2 * 100
                if pen > 0:
                    return objective_sec(np.clip(x_sec, lo[1:], hi[1:])) + pen
                return objective_sec(x_sec)

            try:
                from scipy.optimize import minimize
                res = minimize(penalized_obj, x_sec_start, method='Nelder-Mead', 
                               options={'maxiter': evals_per_start, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_f2:
                    best_f2 = res.fun
                    best_x_sec = res.x
            except Exception:
                pass

        if best_x_sec is None:
            # Fallback to random if local search failed
            best_x_sec = lo[1:] + span[1:] * np.random.random(dim - 1)
            f = problem(np.concatenate(([x1_fix], best_x_sec)))
            used += 1
            best_f2 = f[1]

        # Phase 2: Sweep x1
        # We have best_x_sec which minimizes g.
        # Now we vary x1 to generate the Pareto front.
        # Number of points to sweep: use remaining budget.
        remaining_budget = budget - used
        n_sweep = max(10, remaining_budget // 2)
        
        # Create a grid of x1 values
        x1_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        for x1 in x1_vals:
            if is_deadline_exceeded():
                break
            if used >= budget:
                break
            
            x_full = np.concatenate(([x1], best_x_sec))
            x_full = np.clip(x_full, lo, hi)
            f = problem(x_full)
            used += 1
            update_archive(f, x_full)

        # Phase 3: If we have budget left, do a few random explorations to ensure we didn't miss a different branch
        # (Though for ZDT4, the global min of g is unique in terms of the front shape)
        # Just add a few random points to be safe.
        while used < budget and not is_deadline_exceeded():
            x_rand = lo + span * np.random.random(dim)
            f = problem(x_rand)
            used += 1
            update_archive(f, x_rand)

        return archive_X
# EVOLVE-BLOCK-END
