# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a contention-based heuristic
    inspired by bipartite assignment principles.
    
    Strategy:
    1. For each ready operation, determine its 'best' machines (minimum duration).
    2. For each (operation, machine) pair, calculate a score based on:
       - Processing time on that machine (shorter is better).
       - Machine contention: how many other ready operations also consider this machine 'best'.
         (Higher contention is worse, as it blocks others).
       - Job urgency: remaining work of the job (higher is better to prioritize).
    3. Select the (operation, machine) pair with the lowest score.
    4. Return the index of the operation and the chosen machine.
    """
    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Precompute best duration and remaining work for each op
    best_durs = []
    remaining_works = []
    
    for i in range(n_ops):
        op = ready_operations[i]
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        best_durs.append(min(durations))
        remaining_works.append(op["remaining_work"])
        
    best_durs = np.array(best_durs)
    remaining_works = np.array(remaining_works)
    
    # Collect all unique machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    all_machines = sorted(all_machines)
    
    best_score = float('inf')
    best_idx = 0
    best_machine = None
    
    # Weights
    w_time = 1.0
    w_contention = 2.0
    w_urgency = 0.5
    
    for i in range(n_ops):
        op = ready_operations[i]
        machine_options = op["machine_options"]
        r_j = remaining_works[i]
        
        for m, d in machine_options.items():
            # Calculate contention: number of OTHER ops that also have m as a best machine
            contention = 0
            for k in range(n_ops):
                if k == i:
                    continue
                other_op = ready_operations[k]
                if m in other_op["machine_options"]:
                    other_d = other_op["machine_options"][m]
                    if other_d == best_durs[k]:
                        contention += 1
            
            # Score: Lower is better
            # We want low d, low contention, high r_j
            score = w_time * d + w_contention * contention - w_urgency * r_j
            
            if score < best_score:
                best_score = score
                best_idx = i
                best_machine = m
                
    # Return tuple (index, machine) to allow explicit machine assignment
    # If the evaluator doesn't support tuples, this might fail, but the contract says "optionally"
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
