"""Assembly line balancing with simulated annealing priority optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    
    while eligible:
        # Find the highest-priority task that fits
        best_task = -1
        best_pri = -np.inf
        remaining = cycle_time - current_load
        
        for task in eligible:
            if times[task] <= remaining:
                if priority_scores[task] > best_pri:
                    best_pri = priority_scores[task]
                    best_task = task
        
        if best_task >= 0:
            eligible.discard(best_task)
            current_load += times[best_task]
            
            for succ in successors[best_task]:
                unassigned_preds[succ] -= 1
                if unassigned_preds[succ] == 0:
                    eligible.add(succ)
        else:
            # No eligible task fits; start new station
            station_count += 1
            current_load = 0
            # Check if any eligible task exceeds cycle time
            for t in eligible:
                if times[t] > cycle_time:
                    return 10**9
    
    return station_count


def _initial_priority(times, predecessors, cycle_time):
    """Generate a reasonable initial priority using critical path and other features."""
    n = len(times)
    if n == 0:
        return np.array([])
    
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Topological sort
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    topo_order = []
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return np.random.RandomState(42).rand(n)
    
    # Critical path (longest path from task to end)
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    # Position in topological order
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(topo_order):
        topo_pos[u] = idx
    
    # Number of direct successors
    num_direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine features
    # Higher priority for: longer critical path, earlier in topo order, more successors
    priority = 0.5 * cp / (max(cp) + 1e-9) + 0.3 * (1 - topo_pos / (n + 1)) + 0.2 * num_direct_succ / (n + 1)
    
    return priority


def _simulated_annealing(times, predecessors, cycle_time, deadline):
    """
    Use simulated annealing to optimize priority scores.
    """
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    
    # Initial solution
    current_priority = _initial_priority(times, predecessors, cycle_time)
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    
    best_priority = current_priority.copy()
    best_score = current_score
    
    # Annealing parameters
    T = 1.0  # Initial temperature
    cooling_rate = 0.9995
    T_min = 1e-6
    
    # We'll perturb the priority by adding noise to random tasks
    rng = np.random.RandomState(42)
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5:
        iteration += 1
        
        # Check for stagnation
        if no_improve_count > max_no_improve:
            # Restart with a new random perturbation of the best
            current_priority = best_priority + rng.randn(n) * 0.1
            current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
            no_improve_count = 0
            T = 0.5  # Reset temperature to medium
            continue
        
        # Generate a neighbor by perturbing priority of 1-3 random tasks
        num_perturb = rng.choice([1, 2, 3])
        indices = rng.choice(n, size=num_perturb, replace=False)
        magnitudes = rng.choice([0.05, 0.1, 0.2, 0.3], size=num_perturb)
        
        new_priority = current_priority.copy()
        for idx, mag in zip(indices, magnitudes):
            new_priority[idx] += rng.randn() * mag
        
        new_score = _greedy_balance(times, predecessors, cycle_time, new_priority)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Better solution
            current_priority = new_priority
            current_score = new_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_priority = current_priority.copy()
                if best_score == 1:
                    break
        else:
            # Worse solution; accept with probability exp(-delta/T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_priority = new_priority
                    current_score = new_score
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # Adapt cooling rate based on progress
        if iteration % 1000 == 0 and no_improve_count > 500:
            cooling_rate = 0.999
            T = max(T, 0.1)
    
    return best_priority


def _local_search(times, predecessors, cycle_time, start_priority, start_score, deadline):
    """
    Local search to refine the priority scores.
    """
    n = len(times)
    if n == 0:
        return np.array([]), start_score
    
    current_priority = start_priority.copy()
    current_score = start_score
    best_priority = current_priority.copy()
    best_score = current_score
    
    rng = np.random.RandomState(123)
    
    # Try perturbations of increasing specificity
    step_sizes = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    
    for step in step_sizes:
        improved = False
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            
            # Perturb 1-2 random tasks
            num_perturb = rng.choice([1, 2])
            indices = rng.choice(n, size=num_perturb, replace=False)
            magnitudes = rng.choice([-1, 1], size=num_perturb) * step
            
            new_priority = current_priority.copy()
            for idx, mag in zip(indices, magnitudes):
                new_priority[idx] += mag
            
            new_score = _greedy_balance(times, predecessors, cycle_time, new_priority)
            
            if new_score < current_score:
                current_priority = new_priority
                current_score = new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_priority = current_priority.copy()
                    if best_score == 1:
                        return best_priority, best_score
        
        if not improved and step < 0.1:
            break
    
    return best_priority, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Simulated annealing (main optimization)
    sa_priority = _simulated_annealing(times, predecessors, cycle_time, deadline - 30)
    sa_score = _greedy_balance(times, predecessors, cycle_time, sa_priority)
    
    # Phase 2: Local search to refine
    if time.time() < deadline - 10:
        ls_priority, ls_score = _local_search(times, predecessors, cycle_time, sa_priority, sa_score, deadline - 5)
        
        if ls_score < sa_score:
            sa_priority = ls_priority
            sa_score = ls_score
    
    # Phase 3: Try a few random restarts if time remains
    if time.time() < deadline - 5 and sa_score > 1:
        rng = np.random.RandomState(42 + int(time.time() % 1000))
        
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            
            # Random restart with small perturbation of best
            trial_priority = sa_priority + rng.randn(n) * 0.05
            trial_score = _greedy_balance(times, predecessors, cycle_time, trial_priority)
            
            if trial_score < sa_score:
                sa_priority = trial_priority
                sa_score = trial_score
    
    return sa_priority
# EVOLVE-BLOCK-END
