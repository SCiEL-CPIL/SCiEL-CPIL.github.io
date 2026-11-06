"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Module-level cache for machine load estimates to avoid recomputation
# We use a simple approach: track the sum of durations assigned to each machine
# Since we don't have direct access to the simulation state, we estimate
# machine load from the ready_operations themselves.

def dispatch(ready_operations, time):
    """
    Dispatch rule using a regret-based heuristic with WSJF fallback.
    
    Strategy:
    1. For each ready operation, compute the "regret" of not assigning it to its best machine.
       Regret = (best_machine_cost) - (second_best_machine_cost) where cost includes
       estimated future load on that machine.
    2. Also compute urgency based on remaining operations and remaining work.
    3. Combine regret and urgency to select the operation with highest priority.
    4. Return the index of the selected operation (and optionally the best machine).
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Estimate machine load from ready operations: sum of minimum durations
    # for each machine across all ready operations
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0.0
            # Use the minimum duration as a proxy for how much this machine is "busy"
            # Actually, we should sum up all durations that could be assigned to this machine
            # But since we don't know which will be assigned, use a weighted estimate
            machine_load[m] += d / len(op["machine_options"])
    
    # For each operation, compute:
    # - best_cost: min over machines of (duration + alpha * machine_load[machine])
    # - second_best_cost: second min
    # - regret = second_best_cost - best_cost (how much we lose if we don't use best machine)
    # - urgency: based on remaining_ops and remaining_work
    
    alpha = 0.5  # Weight for machine load in cost
    
    best_costs = []
    second_best_costs = []
    urgencies = []
    best_machines = []
    
    for op in ready_operations:
        options = op["machine_options"]
        if not options:
            best_costs.append(float('inf'))
            second_best_costs.append(float('inf'))
            urgencies.append(0.0)
            best_machines.append(None)
            continue
        
        # Compute cost for each machine: duration + alpha * estimated load
        machine_costs = {}
        for m, d in options.items():
            load = machine_load.get(m, 0.0)
            cost = d + alpha * load
            machine_costs[m] = cost
        
        # Sort machines by cost
        sorted_machines = sorted(machine_costs.items(), key=lambda x: x[1])
        best_m = sorted_machines[0][0]
        best_cost = sorted_machines[0][1]
        
        if len(sorted_machines) >= 2:
            second_best_cost = sorted_machines[1][1]
        else:
            second_best_cost = float('inf')
        
        regret = second_best_cost - best_cost
        
        # Urgency: higher remaining_ops means more urgent (more work to do)
        # Also consider remaining_work
        rem_ops = op.get("remaining_ops", 1)
        rem_work = op.get("remaining_work", 0)
        
        # Urgency score: prioritize jobs with more remaining operations
        # and those with high remaining work
        urgency = rem_ops * 1.0 + rem_work * 0.1
        
        best_costs.append(best_cost)
        second_best_costs.append(second_best_cost)
        urgencies.append(urgency)
        best_machines.append(best_m)
    
    # Combine regret and urgency
    # Normalize regret and urgency to [0, 1] range for fair combination
    max_regret = max(second_best_costs[i] - best_costs[i] for i in range(n_ops)) if n_ops > 0 else 1.0
    max_urgency = max(urgencies) if urgencies else 1.0
    
    if max_regret < 1e-9:
        max_regret = 1.0
    if max_urgency < 1e-9:
        max_urgency = 1.0
    
    # Priority = w1 * normalized_regret + w2 * normalized_urgency
    w1 = 0.6  # Weight for regret
    w2 = 0.4  # Weight for urgency
    
    priorities = []
    for i in range(n_ops):
        norm_regret = (second_best_costs[i] - best_costs[i]) / max_regret
        norm_urgency = urgencies[i] / max_urgency
        priority = w1 * norm_regret + w2 * norm_urgency
        priorities.append(priority)
    
    # Select operation with highest priority
    best_idx = int(np.argmax(priorities))
    
    # Return just the index as per the original interface
    return best_idx

# EVOLVE-BLOCK-END
