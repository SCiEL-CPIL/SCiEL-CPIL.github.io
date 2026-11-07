"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2."""
import numpy as np
import time
from scipy.optimize import linprog

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    # Simple projection for sanity checks or fallback
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Solves the problem as a Linear Program.
    
    We want to minimize t such that:
    1. h_i in [0, 1]
    2. sum(h_i) = N/2
    3. For all lags k, correlate(h, 1-h)[k] <= t * N / 2
    
    Note: The evaluator computes value = max(c) / N * 2.
    So minimizing value is equivalent to minimizing max(c).
    Let M = max(c). We minimize M.
    Constraints: c_k <= M for all k.
    c_k = sum_{i} h_i * (1-h_{i+k})  (with appropriate indexing for 'full' mode)
    
    Since c_k is quadratic (h_i * h_j), this is NOT a linear program directly.
    Wait. correlate(h, 1-h) = sum h_i (1-h_{i+k}) = sum h_i - sum h_i h_{i+k}.
    This is a Quadratic Program (QP) because of the h_i h_j terms.
    
    However, we can reframe.
    Let x_i = h_i.
    We want to minimize t.
    Constraints:
    sum_i x_i (1 - x_{i+k}) <= t * N / 2  for all k.
    sum_i x_i = N/2
    0 <= x_i <= 1
    
    This is a Quadratic Constraint Program (QCP).
    scipy.optimize.minimize can handle QPs if we use methods like SLSQP or trust-constr,
    but they are slow and prone to local minima for non-convex problems.
    Is the problem convex?
    The constraint is sum x_i - sum x_i x_{i+k} <= t N/2.
    The term -sum x_i x_{i+k} is concave (Hessian has negative eigenvalues for the cross terms).
    So the constraint is of the form: (linear) + (concave) <= linear.
    This means the feasible region is NOT convex. Standard convex solvers won't work directly.
    
    Alternative Strategy:
    The current program uses L-BFGS-B on the non-smooth objective.
    A different strategy: **Simulated Annealing** or **Basin Hopping** with a robust objective.
    Or, exploit the structure.
    
    Let's look at the objective again:
    value = max_k ( sum_i h_i - sum_i h_i h_{i+k} ) / (N/2)
    Since sum_i h_i = N/2 is fixed, minimizing max_k (N/2 - sum_i h_i h_{i+k}) is equivalent to
    MAXIMIZING min_k ( sum_i h_i h_{i+k} ).
    
    So the problem is:
    Maximize m
    Subject to:
    sum_i h_i h_{i+k} >= m  for all k
    sum_i h_i = N/2
    0 <= h_i <= 1
    
    This is a Max-Min problem.
    Let f_k(h) = sum_i h_i h_{i+k}.
    We want to maximize min_k f_k(h).
    
    This is still non-convex because f_k is quadratic.
    However, if we fix the values of h to be binary (0 or 1), the problem becomes combinatorial.
    But h is continuous.
    
    Let's try a different angle. The function g(h) = min_k f_k(h) is the minimum of quadratic forms.
    It is not concave.
    
    Let's reconsider the "Different Strategy" directive.
    The current program uses `scipy.optimize.minimize` with L-BFGS-B.
    A materially different strategy is **Genetic Algorithm** or **Particle Swarm Optimization** (PSO).
    These are global search methods that do not rely on gradients, which are undefined or misleading at the kinks of the max function.
    
    Let's implement a simple, fast **Particle Swarm Optimization (PSO)** tailored for this problem.
    It explores the space stochastically and is robust to non-convexity.
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # 1.0
    
    # Time check
    if remaining_time() < 10:
        return project(best_h)
    
    # PSO Parameters
    n_particles = 50
    max_iters = 100
    w = 0.8  # inertia
    c1 = 1.5 # cognitive
    c2 = 1.5 # social
    
    # Initialize particles
    # We need to ensure sum(h) = N/2 and h in [0,1].
    # Strategy: Generate random h, then project.
    # But projection changes the position.
    # Better: Parameterize h such that constraints are satisfied.
    # Or just use the project function in the loop.
    
    particles = np.zeros((n_particles, N))
    velocities = np.zeros((n_particles, N))
    
    for i in range(n_particles):
        # Random init
        h_init = np.random.uniform(0, 1, N)
        h_init = project(h_init)
        particles[i] = h_init
    
    pbest = particles.copy()
    pbest_scores = np.array([objective(p) for p in pbest])
    
    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)
    
    if gbest_score < best_val:
        best_val = gbest_score
        best_h = gbest.copy()
    
    for it in range(max_iters):
        if remaining_time() < 5:
            break
            
        for i in range(n_particles):
            r1 = np.random.rand(N)
            r2 = np.random.rand(N)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Project to feasible set
            particles[i] = project(particles[i])
            
            # Evaluate
            score = objective(particles[i])
            
            if score < pbest_scores[i]:
                pbest[i] = particles[i]
                pbest_scores[i] = score
                
            if score < gbest_score:
                gbest = particles[i].copy()
                gbest_score = score
                
                if gbest_score < best_val:
                    best_val = gbest_score
                    best_h = gbest.copy()
                    
    # Final local refinement using Nelder-Mead on the best particle found
    if remaining_time() > 10:
        from scipy.optimize import minimize
        
        def obj_scipy(x):
            h_p = project(x)
            return objective(h_p)
        
        # Start from the best PSO result
        res = minimize(
            obj_scipy,
            best_h,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        h_res = project(res.x)
        val_res = objective(h_res)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
            
    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
