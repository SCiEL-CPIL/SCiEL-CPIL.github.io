"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = None
_START = None
# Track estimated machine loads to guide dispatch toward relieving bottlenecks
_MACHINE_LOAD = {}
_MACHINE_LOAD_VERSION = 0

def _ensure_deadline():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _START = _time.monotonic()
        _DEADLINE = _START + 145.0  # safety margin before 150s kill
    return _DEADLINE

def _time_left():
    return max(0.0, _DEADLINE - _time.monotonic())

def _soft_min(values, temperature=0.1):
    """Smooth surrogate for min using log-sum-exp (negative log-sum-exp)."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    # soft-min = -log-sum-exp(-x / T) * T
    neg_scaled = -arr / temperature
    max_neg = np.max(neg_scaled)
    lse = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    return float(-lse * temperature)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Multi-criteria weighted heuristic balancing:
    1. Remaining work (longest job first to reduce tail)
    2. Machine scarcity (prefer operations on machines with fewer alternatives)
    3. Soft-min processing time (smooth surrogate for best-case duration)
    4. Machine load awareness (prefer ops that relieve most-loaded machines)
    
    Returns: index of the ready operation to start
    """
    global _MACHINE_LOAD, _MACHINE_LOAD_VERSION
    deadline = _ensure_deadline()
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Fast path: if we're close to deadline, use simple heuristic
    if _time_left() < 0.5:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Collect all machine IDs to build a load snapshot
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    # Normalize current machine loads for relative comparison
    if all_machines:
        max_load = max(_MACHINE_LOAD.get(m, 0.0) for m in all_machines)
        min_load = min(_MACHINE_LOAD.get(m, 0.0) for m in all_machines)
        load_range = max_load - min_load if max_load > min_load else 1.0
    else:
        load_range = 1.0
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # 1. Remaining work of the job (higher = more urgent to finish)
        rem_work = op["remaining_work"]
        
        # 2. Machine scarcity: how many machine options does this op have?
        #    Fewer options = more constrained = higher priority
        num_options = len(op["machine_options"])
        scarcity = 1.0 / max(num_options, 1)
        
        # 3. Soft-min processing time (smooth surrogate for min duration)
        durations = list(op["machine_options"].values())
        soft_min_dur = _soft_min(durations, temperature=0.15)
        
        # 4. Average processing time
        avg_duration = sum(durations) / len(durations)
        
        # 5. Number of remaining operations for this job
        rem_ops = op["remaining_ops"]
        
        # 6. Machine load factor: prefer ops whose best machine is less loaded
        #    Compute soft-max over (load / duration) to identify which machine
        #    this op would likely use, then check its relative load
        machine_ids = list(op["machine_options"].keys())
        loads = np.array([_MACHINE_LOAD.get(m, 0.0) for m in machine_ids], dtype=np.float64)
        durs = np.array(durations, dtype=np.float64)
        # Score per machine: normalized load relative to duration
        # Higher means "this machine is more congested relative to the work it would do"
        if len(machine_ids) > 1:
            # Use soft-max to find the "most relevant" machine (highest load/duration ratio)
            ratios = loads / np.maximum(durs, 1e-6)
            temp = 0.2
            scaled = ratios / temp
            max_scaled = np.max(scaled)
            exp_scaled = np.exp(scaled - max_scaled)
            probs = exp_scaled / np.sum(exp_scaled)
            # Expected relative load of the machine this op would likely use
            expected_rel_load = float(np.sum(probs * loads)) / load_range
        else:
            expected_rel_load = loads[0] / load_range if len(loads) == 1 else 0.0
        
        # Composite score:
        # - Weight remaining work heavily (dominant factor for makespan)
        # - Add scarcity bonus (constrained operations should go first)
        # - Add soft-min duration bonus (longer best-case = more impactful)
        # - Penalize high expected machine load (prefer relieving less-loaded machines)
        # - Small penalty for many remaining ops
        
        score = (
            1.0 * rem_work +
            0.35 * scarcity * rem_work +
            0.12 * soft_min_dur +
            0.08 * avg_duration -
            0.15 * expected_rel_load * rem_work -
            0.02 * rem_ops
        )
        scores[i] = score
    
    best_idx = int(np.argmax(scores))
    
    # Update machine load estimate: assume the chosen op will go to the machine
    # with the shortest duration (greedy assignment for load tracking)
    chosen_op = ready_operations[best_idx]
    best_machine = min(chosen_op["machine_options"], key=lambda m: chosen_op["machine_options"][m])
    best_duration = chosen_op["machine_options"][best_machine]
    _MACHINE_LOAD[best_machine] = _MACHINE_LOAD.get(best_machine, 0.0) + best_duration
    _MACHINE_LOAD_VERSION += 1
    
    return best_idx
# EVOLVE-BLOCK-END
