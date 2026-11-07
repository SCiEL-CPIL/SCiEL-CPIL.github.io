"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Global + local hybrid optimizer using multi-start L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We will use a portion of the budget for global sampling and the rest for local refinement.
        # Strategy:
        # 1. Sample several random points.
        # 2. Pick the best one.
        # 3. Use L-BFGS-B to refine from that point.
        # 4. If budget remains, sample more points and refine the best new one, keeping the overall best.
        
        # To strictly control the number of function evaluations, we can't just call scipy.optimize.minimize
        # without tracking calls, because it might use more than expected or we might exceed budget.
        # However, L-BFGS-B is very efficient. 
        # Let's implement a controlled loop.
        
        # Since we don't know f_opt, we just minimize.
        # We need to be careful with the budget. 
        # Let's divide the budget into phases.
        # Phase 1: Global exploration. Evaluate N random points.
        # Phase 2: Local exploitation. Run L-BFGS-B from the best point found so far.
        
        # A simpler and more robust approach for black-box with limited budget:
        # Use a population-based method or a smarter local search.
        # Given dim=5, L-BFGS-B from a good start is very strong.
        # The main risk is getting stuck in a local minimum.
        # Büche-Rastrigin is multimodal.
        
        # Let's do a multi-start L-BFGS-B.
        # 1. Generate K initial points.
        # 2. For each point, run a short L-BFGS-B optimization.
        # 3. Keep the best result.
        
        # How to manage budget?
        # We can wrap the function to count calls.
        
        calls_made = 0
        max_calls = budget
        
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                # Return a large value to stop optimization if budget exceeded
                return 1e10
            calls_made += 1
            val = func(x)
            return val
            
        # We need to ensure we don't exceed max_calls.
        # scipy's minimize doesn't easily accept a "max calls" limit that strictly stops and returns current best 
        # without potentially making one extra call or not respecting the limit perfectly in all edge cases, 
        # but we can check after.
        # Actually, if we exceed, the result is invalid. So we must be conservative.
        
        # Let's reserve a small buffer.
        # Let's try 10 restarts.
        # Each L-BFGS-B run might take ~20-50 function evaluations for dim=5.
        # 10 restarts * 50 = 500. This fits perfectly.
        
        num_restarts = 10
        best_x = None
        best_f = np.inf
        
        # We will manually manage the restarts to ensure we stay within budget.
        # Instead of relying on scipy's internal loop which might be hard to budget precisely,
        # we can run scipy with a tight maxiter and check calls.
        
        # Let's use a simpler, more controllable approach:
        # 1. Sample 20 random points. (20 calls)
        # 2. Pick the best.
        # 3. Run L-BFGS-B from that best point with a limit on iterations.
        # 4. If budget remains, sample 20 more, pick best, run L-BFGS-B, etc.
        
        # However, L-BFGS-B can be very efficient. 
        # Let's try a different strategy: CMA-ES is great for black-box multimodal, but not in stdlib/scipy.
        # Nelder-Mead is in scipy. It's derivative-free.
        
        # Let's stick to L-BFGS-B but be very careful.
        # We will create a wrapper that raises an exception or returns inf if budget is hit.
        
        class BudgetExceeded(Exception):
            pass

        def strict_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                raise BudgetExceeded()
            calls_made += 1
            return func(x)

        # We will perform several restarts.
        # For each restart:
        # 1. Sample a random point x0.
        # 2. Evaluate it.
        # 3. Run minimize(method='L-BFGS-B', x0=x0, bounds=[(-5,5)]*dim)
        # 4. Catch BudgetExceeded. If caught, the last valid point from minimize is not easily accessible if it fails mid-way.
        #    Actually, if an exception is raised inside the function, minimize will propagate it.
        #    We need the best point found *before* the exception.
        #    This is tricky with scipy.optimize.minimize.
        
        # Alternative: Implement a simple (1+1)-ES or Nelder-Mead ourselves with strict budget control?
        # The seed program uses (1+1)-ES. The directive says "do not reuse the current designs".
        # So (1+1)-ES is out.
        
        # Let's use Nelder-Mead implemented manually? Or use scipy.optimize.minimize with Nelder-Mead?
        # Nelder-Mead is derivative-free and robust.
        # But controlling the exact number of function evaluations with scipy is still tricky.
        
        # Let's try a different approach: **Simulated Annealing** or **Particle Swarm Optimization (PSO)**.
        # PSO is population-based, easy to control budget, and good for multimodal.
        # Or **Differential Evolution**.
        
        # Let's implement a simple **Differential Evolution** (DE) or **PSO**.
        # PSO is simpler to implement with strict budget control.
        
        # PSO Parameters:
        # Population size: 20
        # Iterations: (budget - pop_size) / pop_size
        # For budget 500, pop 20:
        # Init: 20 calls.
        # Remaining: 480.
        # Iterations: 480 / 20 = 24 iterations.
        # In each iteration, we evaluate 20 new positions.
        
        def pso_optimize():
            nonlocal calls_made
            n_particles = 20
            # Check if we have enough budget for at least initialization
            if calls_made + n_particles > max_calls:
                # Not enough for full PSO, fallback to random search
                # Just sample remaining budget
                while calls_made < max_calls:
                    x = rng.uniform(-5, 5, dim)
                    f = func(x)
                    calls_made += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                return best_x, best_f

            # Initialize particles
            particles = rng.uniform(-5, 5, (n_particles, dim))
            velocities = rng.uniform(-1, 1, (n_particles, dim))
            
            pbest_pos = particles.copy()
            pbest_val = np.inf
            gbest_pos = np.zeros(dim)
            gbest_val = np.inf
            
            # Evaluate initial positions
            for i in range(n_particles):
                if calls_made >= max_calls:
                    break
                f = func(particles[i])
                calls_made += 1
                if f < pbest_val:
                    pbest_val = f
                    pbest_pos = particles[i].copy()
                if f < gbest_val:
                    gbest_val = f
                    gbest_pos = particles[i].copy()
            
            # Update global best
            if gbest_val < best_f:
                best_f = gbest_val
                best_x = gbest_pos.copy()
                
            # PSO Parameters
            w = 0.7  # inertia
            c1 = 1.5  # cognitive
            c2 = 1.5  # social
            
            # Run PSO iterations
            while calls_made < max_calls:
                # Check if we can evaluate at least one particle
                if calls_made + 1 > max_calls:
                    break
                    
                # We will update particles one by one to strictly control budget?
                # No, standard PSO updates all. If we run out of budget mid-iteration, we just stop.
                # But we need to ensure we don't exceed.
                
                for i in range(n_particles):
                    if calls_made >= max_calls:
                        break
                    
                    r1 = rng.random()
                    r2 = rng.random()
                    
                    v = w * velocities[i] + c1 * r1 * (pbest_pos[i] - particles[i]) + c2 * r2 * (gbest_pos - particles[i])
                    velocities[i] = v
                    particles[i] = particles[i] + v
                    particles[i] = np.clip(particles[i], -5, 5)
                    
                    f = func(particles[i])
                    calls_made += 1
                    
                    if f < pbest_val:
                        pbest_val = f
                        pbest_pos[i] = particles[i].copy()
                    if f < gbest_val:
                        gbest_val = f
                        gbest_pos = particles[i].copy()
                        
                # Update global best after each full iteration (or continuously)
                if gbest_val < best_f:
                    best_f = gbest_val
                    best_x = gbest_pos.copy()
                    
            return best_x, best_f

        # Start with a few random samples to get a decent start?
        # PSO initializes randomly, so it covers the space.
        # Let's just run PSO.
        
        # One issue: PSO might be slow to converge if the landscape is rugged (Büche-Rastrigin is rugged).
        # Maybe a hybrid: Random search first to find a good basin, then local search?
        # The directive says "new program... from an approach family that none of the current programs use".
        # Current is (1+1)-ES. PSO is a different family (swarm intelligence).
        
        # Let's refine the PSO parameters.
        # Büche-Rastrigin has many local minima. PSO with a decent population should explore well.
        
        # Let's also consider that the "minimizer lies in [-4, 4]^5".
        # We can sample in [-4, 4] initially to be closer?
        # The box is [-5, 5]. The minimizer is in [-4, 4].
        # Sampling in [-5, 5] is fine, but maybe focusing on [-4, 4] is slightly better?
        # Let's stick to [-5, 5] as per the problem statement, but the shift/rotation means the "center" is unknown.
        
        # Let's try a different swarm: **Differential Evolution** is often stronger than PSO for these benchmarks.
        # But PSO is easier to implement correctly with strict budget.
        
        # Let's stick with PSO but tune it.
        # Population 20, w=0.7, c1=1.5, c2=1.5 is standard.
        
        # Let's execute PSO.
        pso_optimize()
        
        # If PSO didn't use all budget (e.g., due to break conditions), we can do a final polish?
        # No, PSO loop runs until calls_made >= max_calls.
        
        return best_x, best_f

# EVOLVE-BLOCK-END
