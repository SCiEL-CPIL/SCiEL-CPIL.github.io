"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    c_prev = [0.0] * n_machines
    c_curr = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[-1]

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _soft_assignment_makespan(processing_times, z, tau=1.0):
    """
    Compute a soft makespan using Gumbel-Softmax relaxation.
    
    z: shape (n_jobs, n_positions) - continuous latent vector
    tau: temperature for softmax relaxation
    
    Returns: (soft_makespan, hard_order)
    """
    n_jobs, n_machines = processing_times.shape
    n_positions = n_jobs
    
    # Apply Gumbel-Softmax to get soft permutation matrix
    # z is (n_jobs, n_positions), we want P[position, job] = probability job at position
    # Softmax over jobs for each position
    # To make it a proper relaxation, we use the straight-through estimator
    
    # Temperature-scaled softmax: P[pos, job] = softmax(z[job, pos] / tau) over job
    z_scaled = z / max(tau, 1e-6)
    # Subtract max for numerical stability
    z_max = np.max(z_scaled, axis=0, keepdims=True)
    z_exp = np.exp(z_scaled - z_max)
    z_sum = np.sum(z_exp, axis=0, keepdims=True)
    P = z_exp / z_sum  # P[pos, job]
    
    # Expected processing time at each position: E[p[pos, m]] = sum_j P[pos, j] * p[j, m]
    # P is (n_positions, n_jobs), processing_times is (n_jobs, n_machines)
    # E_p[pos, m] = P[pos, :] @ p[:, m]
    E_p = P @ processing_times  # (n_positions, n_machines)
    
    # Compute makespan using the expected processing times
    # This is a surrogate for the makespan
    if n_positions == 0:
        return 0.0, list(range(n_jobs))
    
    c_prev = np.zeros(n_machines)
    c_curr = np.zeros(n_machines)
    
    for i in range(n_positions):
        p_row = E_p[i]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    soft_ms = c_prev[-1]
    
    # Get hard assignment: for each position, pick job with highest z
    hard_order = []
    used_jobs = set()
    for pos in range(n_positions):
        # Pick job with highest z[job, pos] not yet used
        best_job = -1
        best_val = -float('inf')
        for job in range(n_jobs):
            if job not in used_jobs and z[job, pos] > best_val:
                best_val = z[job, pos]
                best_job = job
        if best_job == -1:
            # Fallback: pick any unused job
            for job in range(n_jobs):
                if job not in used_jobs:
                    best_job = job
                    break
        hard_order.append(best_job)
        used_jobs.add(best_job)
    
    return soft_ms, hard_order

def _gradient_step(processing_times, z, lr, tau):
    """
    Compute approximate gradient of soft makespan w.r.t. z and take a step.
    Uses finite differences on a random subset of coordinates to reduce cost.
    """
    n_jobs, n_machines = processing_times.shape
    n_positions = n_jobs
    
    # Compute gradient via finite differences on a subset
    eps = 1e-3
    grad = np.zeros_like(z)
    
    # Select a random subset of coordinates to update (e.g., 10% of the matrix)
    num_updates = max(1, int(0.1 * n_jobs * n_positions))
    indices = np.random.choice(n_jobs * n_positions, num_updates, replace=False)
    
    for idx in indices:
        i, j = divmod(idx, n_positions)
        
        z_plus = z.copy()
        z_plus[i, j] += eps
        ms_plus, _ = _soft_assignment_makespan(processing_times, z_plus, tau)
        
        z_minus = z.copy()
        z_minus[i, j] -= eps
        ms_minus, _ = _soft_assignment_makespan(processing_times, z_minus, tau)
        
        grad[i, j] = (ms_plus - ms_minus) / (2 * eps)
    
    # Take gradient step
    z_new = z - lr * grad
    
    return z_new

def _anneal_temperature(tau_start, tau_end, t, total_time):
    """Linear annealing schedule."""
    if total_time <= 0:
        return tau_end
    progress = min(t / total_time, 1.0)
    return tau_start * (tau_end / tau_start) ** progress

def _continuous_optimization(processing_times, z_init, time_limit):
    """
    Run gradient-based optimization with Gumbel-Softmax relaxation.
    """
    n_jobs, n_machines = processing_times.shape
    n_positions = n_jobs
    
    start_time = time.time()
    
    tau_start = 1.0
    tau_end = 0.1
    total_time = time_limit
    
    z = z_init.copy()
    best_z = z.copy()
    best_soft_ms, best_hard_order = _soft_assignment_makespan(processing_times, z, tau_start)
    best_ms = _makespan(processing_times, best_hard_order)
    
    lr = 0.1
    step = 0
    max_steps = 50  # Hard limit on steps to prevent timeout
    
    while (time.time() - start_time) < time_limit and step < max_steps:
        tau = _anneal_temperature(tau_start, tau_end, time.time() - start_time, total_time)
        
        # Gradient step
        z = _gradient_step(processing_times, z, lr, tau)
        
        # Evaluate
        soft_ms, hard_order = _soft_assignment_makespan(processing_times, z, tau)
        hard_ms = _makespan(processing_times, hard_order)
        
        if hard_ms < best_ms:
            best_ms = hard_ms
            best_z = z.copy()
            best_hard_order = hard_order
        
        # Occasionally reset to best to prevent drift
        if step % 10 == 0 and step > 0:
            if _makespan(processing_times, best_hard_order) < best_ms:
                z = best_z.copy()
        
        step += 1
        
        # Reduce learning rate
        lr = max(0.01, lr * 0.95)
    
    return best_hard_order, best_ms, best_z

def _local_search(processing_times, order, time_limit):
    """Local search with swap moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                return current_order, current_ms
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    return current_order, current_ms
                
                # Swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_ms = _makespan(processing_times, current_order)
                
                if new_ms < current_ms:
                    current_ms = new_ms
                    improved = True
                else:
                    # Revert
                    current_order[i], current_order[j] = current_order[j], current_order[i]
    
    return current_order, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses Gumbel-Softmax continuous optimization followed by local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    
    # Create initial z from NEH order
    z = np.random.randn(n_jobs, n_jobs) * 0.1
    # Bias z so that the NEH order is favored
    for pos, job in enumerate(initial_order):
        z[job, pos] += 5.0
    
    # Run continuous optimization
    remaining_time = deadline - time.time()
    if remaining_time < 10.0:
        # Not enough time, just do local search
        best_order, best_ms = _local_search(processing_times, initial_order, time_limit=remaining_time * 0.9)
        return best_order
    
    # Split time: 60% continuous optimization, 40% local search
    cont_time = remaining_time * 0.6
    search_time = remaining_time * 0.35
    
    # Run continuous optimization
    cont_order, cont_ms, best_z = _continuous_optimization(processing_times, z, cont_time)
    
    # Get the best order from continuous optimization
    # Also try the NEH order
    neh_ms = _makespan(processing_times, initial_order)
    
    if cont_ms < neh_ms:
        best_order = cont_order
        best_ms = cont_ms
    else:
        best_order = initial_order
        best_ms = neh_ms
    
    # Run local search on the best order
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_order, best_ms = _local_search(processing_times, best_order, time_limit=remaining_time * 0.95)
    
    return best_order
# EVOLVE-BLOCK-END
