"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: we must finish well before 120s.
_START_TIME = _time.monotonic()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 2.0  # seconds before the hard limit

def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (_time.monotonic() - _START_TIME)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: Greedy load-balancing on the bipartite (ready_op, machine) graph.
    For each (op, machine) pair, estimate the resulting machine load and select
    the pair that minimizes the increase in the maximum machine load. This avoids
    bottlenecks that purely job-centric heuristics ignore.
    
    We combine:
      1. Machine load balancing: prefer assigning to the machine with least current load.
      2. Job urgency: prefer operations with more remaining work / fewer remaining ops.
      3. Duration: prefer shorter durations to reduce blocking.
    
    The dispatch function returns an index (or (index, machine)) into ready_operations.
    """
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time_remaining() <= 0:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    
    # Collect machine options for each operation
    # machine_options: {machine: duration}
    
    # Estimate current machine loads. Since we don't have direct access to machine
    # state, we infer load from the time parameter and the fact that machines process
    # one op at a time. A better proxy: use the minimum duration per machine across
    # all ready ops as a lower bound, and weight by how many ops want that machine.
    
    # Build a score for each (op_idx, machine) pair
    # We want to minimize: resulting max machine load increase
    
    # First, compute for each machine the total "demand" from ready ops that can use it
    machine_demand = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            if m not in machine_demand:
                machine_demand[m] = 0.0
            machine_demand[m] += dur
    
    # Normalize demand: machines with higher demand are more loaded
    if machine_demand:
        max_demand = max(machine_demand.values())
        if max_demand > 0:
            machine_load_norm = {m: d / max_demand for m, d in machine_demand.items()}
        else:
            machine_load_norm = {m: 0.0 for m in machine_demand}
    else:
        machine_load_norm = {}
    
    # Score each operation: combine urgency and best-machine assignment
    scores = []
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        machine_options = op.get("machine_options", {})
        
        # Urgency: more remaining work / fewer remaining ops = more urgent
        if remaining_ops > 0:
            urgency = remaining_work / remaining_ops
        else:
            urgency = remaining_work
        
        # Best machine for this op: minimize (machine_load_norm + duration)
        best_machine_score = float('inf')
        for m, dur in machine_options.items():
            load = machine_load_norm.get(m, 0.0)
            # Weight: we want low load and low duration
            pair_score = load * 10.0 + dur  # weight load heavily
            if pair_score < best_machine_score:
                best_machine_score = pair_score
        
        if best_machine_score == float('inf'):
            best_machine_score = 0.0
        
        # Combined score: higher is better for dispatch
        # We want high urgency AND low best_machine_score (i.e., good machine availability)
        # Normalize urgency
        score = urgency - 0.5 * best_machine_score
        scores.append(score)
    
    # Also consider: pick the op that, if scheduled, leads to the smallest increase
    # in the makespan proxy. A simple proxy: for each op, the earliest it can finish
    # on its best machine.
    
    # Alternative: use a composite that favors ops that can start soon on underloaded machines
    final_scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Minimum duration across machines
        min_dur = min(machine_options.values()) if machine_options else 0
        max_dur = max(machine_options.values()) if machine_options else 0
        
        # Urgency component
        if remaining_ops > 0:
            urgency = remaining_work / remaining_ops
        else:
            urgency = remaining_work
        
        # Speed component: shorter min duration is better
        speed = 1.0 / (min_dur + 1e-6) if min_dur > 0 else 0.0
        
        # Load balance: prefer ops that have access to less-demanded machines
        min_load = min(machine_load_norm.get(m, 0.0) for m in machine_options) if machine_options else 0.0
        
        # Composite: urgency * speed * (1 + (1 - min_load))
        composite = urgency * speed * (1.0 + (1.0 - min_load))
        final_scores.append(composite)
    
    # Deadline check
    if _time_remaining() <= 0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Pick the operation with the highest composite score
    best_idx = int(np.argmax(final_scores))
    return best_idx
# EVOLVE-BLOCK-END
