"""
Strategy: Priority-Based Dispatching with CMA-ES Optimization.

Instead of directly optimizing the permutation (combinatorial space), we optimize
a continuous vector of weights (one per job). The permutation is derived by
sorting jobs by these weights. We use CMA-ES (a state-of-the-art derivative-free
optimizer) to find weights that minimize the makespan of the resulting priority
list. This approach can escape local minima of the permutation space by moving
through the continuous weight space, where the objective function (makespan of
the sorted list) is piecewise constant but allows for "smoother" navigation via
the CMA-ES distribution updates.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _weights_to_order(weights, P):
    """
    Convert a weight vector to a job order.
    Higher weight means earlier in the sequence (higher priority).
    Ties are broken by job index for determinism.
    """
    # Sort indices by weight descending
    # np.argsort returns indices that would sort the array.
    # We want descending order of weights.
    order = np.argsort(-weights)
    return list(order)

def _evaluate_weights(weights, P):
    """
    Objective function for CMA-ES: returns makespan for the order derived from weights.
    """
    order = _weights_to_order(weights, P)
    return _makespan(order, P)

def _cmaes_optimize(objective, x0, sigma, max_iter, time_limit, start_time):
    """
    Simple implementation of CMA-ES.
    x0: initial mean (weight vector)
    sigma: initial step size
    max_iter: maximum number of generations
    """
    n = len(x0)
    pop_size = int(4 + 3 * np.log(n))
    if pop_size < 4:
        pop_size = 4
        
    # CMA-ES Parameters
    mu = pop_size // 2
    # Weights for selection
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / np.sum(w)
    mu_eff = 1.0 / np.sum(w**2)
    
    # Learning rates
    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((n + 2)**2 + mu_eff))
    
    # State variables
    mean = np.array(x0, dtype=float)
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
    
    best_x = mean.copy()
    best_f = objective(mean)
    
    for g in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Sample population
        # Use standard normal and transform via Cholesky of C
        L = np.linalg.cholesky(C)
        z = np.random.randn(pop_size, n)
        # y = L @ z.T -> (pop_size, n)
        y = z @ L.T
        x = np.array([mean + sigma * yi for yi in y])
        
        # Evaluate
        f = np.array([objective(xi) for xi in x])
        
        # Sort by fitness
        order = np.argsort(f)
        x_sorted = x[order]
        f_sorted = f[order]
        
        # Update best
        if f_sorted[0] < best_f:
            best_f = f_sorted[0]
            best_x = x_sorted[0].copy()
            
        # Update mean
        mean_new = mean + sigma * np.sum(w[:, None] * x_sorted[:mu], axis=0)
        
        # Update path evolution strategies
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * L.T @ ((mean_new - mean) / sigma)
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (g + 1)))) / chi_n < (1.4 + 2 / (n + 1))
        
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean_new - mean) / sigma
        
        # Update Covariance Matrix
        delta_hsig = (1 - hsig) * cc * (2 - cc)
        C = (1 - c1 - cmu + delta_hsig) * C
        C += c1 * (np.outer(pc, pc) - C)
        C += cmu * np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', x_sorted[:mu] - mean, x_sorted[:mu] - mean) / sigma**2, axis=0)
        
        # Ensure symmetry and positive definiteness
        C = (C + C.T) / 2.0
        
        # Adaptive step size (Cumulative Step Size Adaptation)
        sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
        
        # Prevent sigma from becoming too small or too large
        if sigma < 1e-10:
            break
        if sigma > 1e10:
            break
            
        mean = mean_new
        
    return best_x, best_f

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using CMA-ES on a priority-list representation.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # 1. Initialize weights
    # Start with random weights, or perhaps total processing time as a heuristic seed?
    # Using total processing time as initial weights aligns with NEH intuition.
    totals = P.sum(axis=1)
    x0 = totals + np.random.randn(n_jobs) * 1e-3
    
    # Initial sigma
    sigma = 1.0
    
    # Determine max iterations based on n_jobs
    # CMA-ES is typically O(10 * n^1.5) or similar for good convergence.
    # We have a time limit, so we run until time runs out or convergence.
    max_iter = 10000  # Upper bound, will be cut by time
    
    # Run CMA-ES
    best_weights, best_makespan = _cmaes_optimize(
        lambda w: _evaluate_weights(w, P), 
        x0, 
        sigma, 
        max_iter, 
        time_limit, 
        start_time
    )
    
    # Convert best weights to order
    best_order = _weights_to_order(best_weights, P)
    
    # 2. Local Search Refinement
    # The CMA-ES result might be stuck in a local optimum of the piecewise constant function.
    # Perform a quick insertion local search on the resulting permutation to polish it.
    
    def _local_search(order, P, time_limit, start_time):
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(order)
            for i in range(n):
                if time.time() - start_time > time_limit:
                    return order
                job_i = order[i]
                for j in range(n):
                    if i == j:
                        continue
                    new_order = order[:]
                    new_order.pop(i)
                    new_order.insert(j, job_i)
                    ms = _makespan(new_order, P)
                    if ms < best_makespan:
                        best_makespan = ms
                        order = new_order
                        improved = True
                        break
                if improved:
                    break
        return order

    # Run local search with a tighter time limit to ensure we return
    # We've already used some time in CMA-ES. Let's give it the remaining time.
    final_order = _local_search(best_order, P, time_limit, start_time)
    
    return final_order
# EVOLVE-BLOCK-END
