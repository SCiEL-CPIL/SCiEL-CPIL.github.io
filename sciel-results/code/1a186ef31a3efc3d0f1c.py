"""Open-shop dispatch rule: MWR + SPT with machine-aware load balancing."""
# EVOLVE-BLOCK-START
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0


def _time_left():
    return _DEADLINE - time.time()


def _soft_max(values, temperature=2.0):
    """Smooth max via log-sum-exp. Returns a smooth approximation of max(values)."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    max_val = max(values)
    # Shift for numerical stability
    shifted = [v - max_val for v in values]
    log_sum = math.log(sum(math.exp(v / temperature) for v in shifted))
    return max_val + temperature * log_sum


def _estimate_machine_load(machine, ready_operations):
    """
    Estimate how loaded a machine is based on ready operations that could use it.
    Higher estimate = more congested machine.
    """
    total_work = 0.0
    count = 0
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if machine in mo:
            total_work += mo[machine]
            count += 1
    if count == 0:
        return 0.0
    # Average duration on this machine among ready ops
    return total_work / count


def _combined_score(op, all_ops):
    """
    Score an operation. Lower is better.
    
    Components:
    1. MWR: prioritize jobs with most remaining total work (urgency)
    2. SPT: prefer operations with short processing times on their best machine
    3. Machine load: prefer ops whose best machine is less congested
    4. Soft makespan surrogate: smooth max of machine workloads
    """
    remaining_work = op.get("remaining_work", 0)
    remaining_ops = op.get("remaining_ops", 1)
    machine_options = op["machine_options"]
    
    if not machine_options:
        return float('inf')
    
    durations = list(machine_options.values())
    min_dur = min(durations)
    
    # --- Component 1: MWR (urgency) ---
    # Jobs with more remaining work are more critical; we want to start them sooner.
    # So high remaining_work should give LOW score.
    max_rw = max(o.get("remaining_work", 0) for o in all_ops)
    if max_rw > 0:
        mwr_score = 1.0 - (remaining_work / max_rw)  # 0 for max remaining, 1 for min
    else:
        mwr_score = 0.5
    
    # --- Component 2: SPT (efficiency) ---
    # Shorter processing time is better.
    max_possible_dur = max(
        max(o["machine_options"].values()) 
        for o in all_ops if o["machine_options"]
    )
    if max_possible_dur > 0:
        spt_score = min_dur / max_possible_dur  # 0 for shortest, 1 for longest
    else:
        spt_score = 0.5
    
    # --- Component 3: Machine load balancing ---
    # Find the best machine (shortest duration) and check how loaded it is.
    best_machine = min(machine_options, key=machine_options.get)
    best_machine_load = _estimate_machine_load(best_machine, all_ops)
    
    # Normalize machine load: higher load = worse
    # Use the max possible load across all machines for normalization
    all_machines = set()
    for o in all_ops:
        all_machines.update(o.get("machine_options", {}).keys())
    
    if all_machines:
        max_load = max(
            _estimate_machine_load(m, all_ops) for m in all_machines
        )
        if max_load > 0:
            load_score = best_machine_load / max_load  # 0 for least loaded, 1 for most
        else:
            load_score = 0.5
    else:
        load_score = 0.5
    
    # --- Component 4: Soft makespan surrogate ---
    # Estimate the makespan contribution if we pick this op.
    # For each machine, estimate total work that would be on it.
    # Use soft max to get a smooth surrogate for max machine load.
    machine_workloads = []
    for m in all_machines:
        workload = 0.0
        for o in all_ops:
            mo = o.get("machine_options", {})
            if m in mo:
                # If this is the op we're evaluating, use its duration on m
                if o is op:
                    workload += mo[m]
                else:
                    workload += mo[m]
        machine_workloads.append(workload)
    
    if machine_workloads:
        soft_makespan = _soft_max(machine_workloads, temperature=3.0)
        # Normalize: lower is better
        max_soft = max(machine_workloads) if machine_workloads else 1.0
        if max_soft > 0:
            makespan_score = soft_makespan / (max_soft + 1e-9)  # Close to 1, but smooth
        else:
            makespan_score = 0.5
    else:
        makespan_score = 0.5
    
    # Weighted combination
    # MWR and SPT are the strongest signals; load balancing and makespan surrogate refine
    score = (
        0.35 * mwr_score +
        0.30 * spt_score +
        0.25 * load_score +
        0.10 * makespan_score
    )
    
    return score


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Score each ready operation using MWR, SPT, machine load, and soft makespan surrogate
    2. Select the operation with the lowest score
    """
    if not ready_operations:
        return 0
    
    # Time guard
    if _time_left() < 0.5:
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx
    
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        score = _combined_score(op, ready_operations)
        if score < best_score:
            best_score = score
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
