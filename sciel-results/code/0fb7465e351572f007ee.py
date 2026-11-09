"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Module-level deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 110.0  # seconds, safety margin before 120s kill

def _elapsed_ok():
    """Check if we're still within the time budget."""
    return (_time.time() - _START_TIME) < _TIME_LIMIT

def dispatch(ready_operations, time):
    """
    Select the next operation to start and optionally the machine.
    
    Strategy: Dynamic bottleneck optimization.
    We model the dispatch decision by prioritizing the operation that maximizes
    the ratio of its job's remaining critical path length to the estimated
    cumulative load of its eligible machines. This shifts focus to real-time
    machine congestion and avoids exacerbating load on already busy machines.
    
    Key improvements over static heuristics:
    1. Machine contestation awareness: We estimate machine load by considering
       which other ready operations compete for the same machines.
    2. Smooth surrogate for min-duration: We use a soft-min (log-sum-exp) to
       evaluate the effective processing time, accounting for machine availability.
    3. Urgency density: We weight by the ratio of remaining work to remaining ops,
       which captures the average workload per operation and helps balance tail effects.
    
    We return (index, machine) to allow the simulator to pick the machine.
    """
    if not ready_operations:
        return 0  # Should not happen, but safe fallback
    
    if len(ready_operations) == 1:
        op = ready_operations[0]
        # Pick the machine with the shortest duration
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    # Check time budget early
    if not _elapsed_ok():
        # Fallback: return the operation with the most remaining work
        best_idx = max(range(len(ready_operations)), key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best_idx]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (best_idx, best_machine)
    
    n_ops = len(ready_operations)
    
    # Pre-compute machine contestation: for each machine, how many ready ops can use it?
    # This helps us identify congested machines.
    machine_contestation = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contestation[m] = machine_contestation.get(m, 0) + 1
    
    # For each operation, compute a composite score
    scores = np.zeros(n_ops)
    best_machines = [None] * n_ops
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            scores[i] = -1e9
            continue
        
        remaining_work = op["remaining_work"]
        remaining_ops = max(op["remaining_ops"], 1)
        
        # Urgency density: average work per remaining op
        urgency_density = remaining_work / remaining_ops
        
        # For each eligible machine, compute a "load-adjusted" duration
        # We penalize machines with high contestation (many ready ops can use them)
        # and use a soft-min to combine durations across machines.
        
        machine_scores = []
        machine_keys = []
        
        for m, duration in options.items():
            # Contestation penalty: machines with more competing ops get a higher effective cost
            contestation = machine_contestation.get(m, 1)
            # Use a smooth penalty: effective_duration = duration * (1 + 0.3 * (contestation - 1))
            # This is a heuristic; the factor 0.3 balances between avoiding congestion
            # and not over-penalizing shared machines.
            effective_duration = duration * (1.0 + 0.3 * (contestation - 1))
            machine_scores.append(effective_duration)
            machine_keys.append(m)
        
        machine_scores = np.array(machine_scores)
        
        # Soft-min: approximate min with a smooth function
        # soft_min(x) = -log(sum(exp(-x_i / tau))) * tau
        # For small tau, this approaches the true min.
        tau = 1.0  # temperature parameter
        soft_min_duration = -tau * np.log(np.sum(np.exp(-machine_scores / tau)))
        
        # The machine that achieves the soft-min is the one with the lowest effective duration
        best_machine_idx = int(np.argmin(machine_scores))
        best_machine = machine_keys[best_machine_idx]
        
        # Composite score:
        # We want to maximize:
        # - urgency_density (jobs with more average work per op should go first)
        # - remaining_work (jobs with more total work should go first to reduce tail)
        # - 1/soft_min_duration (faster processing is better)
        # We use a weighted sum with a multiplicative urgency term.
        
        # Normalize roughly to avoid scale issues
        # score = urgency_density * (0.5 * remaining_work + 0.5 * (1.0 / max(soft_min_duration, 0.1)))
        
        w1 = 1.0  # weight for urgency density
        w2 = 0.5  # weight for remaining work
        w3 = 0.5  # weight for inverse duration
        
        score = w1 * urgency_density * (w2 * remaining_work + w3 * (1.0 / max(soft_min_duration, 0.1)))
        
        scores[i] = score
        best_machines[i] = best_machine
    
    # Check time budget before final selection
    if not _elapsed_ok():
        best_idx = int(np.argmax(scores[:len(ready_operations)//2])) if len(ready_operations) > 1 else 0
        op = ready_operations[best_idx]
        best_machine = best_machines[best_idx] or min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (best_idx, best_machine)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    # If best_machine is None (shouldn't happen), fallback
    if best_machine is None:
        op = ready_operations[best_idx]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
    
    return (best_idx, best_machine)

# EVOLVE-BLOCK-END
