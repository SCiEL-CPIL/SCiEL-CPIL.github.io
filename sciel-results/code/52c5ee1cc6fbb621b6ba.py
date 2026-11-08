"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# --- Global deadline guard ---
_START_TIME = _time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

# --- Parameterized dispatch rule weights ---
# Weights for the multi-criteria scoring of ready operations.
# Based on established job-shop heuristics:
#   MWRF (most work remaining): proven effective for makespan
#   Fewer remaining ops: helps finish jobs sooner (SPT-like at job level)
#   Shortest processing time on chosen machine: minimizes current bottleneck
W_REMAINING_WORK = 1.0
W_REMAINING_OPS = 0.2
W_SPT = 0.8


def _pick_machine(op):
    """Choose the machine with shortest processing time for this operation."""
    machine_options = op["machine_options"]
    if not machine_options:
        return None
    best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
    return best_machine


def dispatch(ready_operations, time):
    """Dispatch rule: weighted combination of scheduling heuristics.
    
    Returns (index, machine) to start the best operation on the best machine.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        idx = 0
        m = _pick_machine(ready_operations[0])
        if m is not None:
            return (idx, m)
        return idx
    
    # Normalize features across ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # SPT: shortest processing time among eligible machines
    spt = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    
    def _normalize(a):
        """Normalize to [0, 1] range; higher is better for the objective."""
        rng = a.max() - a.min()
        if rng < 1e-12:
            return np.ones_like(a) * 0.5
        return (a - a.min()) / rng
    
    # For makespan minimization:
    # - Higher remaining_work is generally good (MWRF)
    # - Lower remaining_ops is good (finish soon)
    # - Lower SPT is good (less time on machine)
    
    norm_work = _normalize(remaining_works)
    norm_ops = _normalize(remaining_ops)
    norm_spt = _normalize(spt)
    
    # Score: higher is better
    # Increased SPT weight relative to remaining_ops based on inspiration 1's success
    # Use inverse SPT (longer processing time = lower score) for better discrimination
    inv_spt = 1.0 / (spt + 1e-6)  # Inverse to make longer processing time lower priority
    norm_inv_spt = _normalize(inv_spt)
    
    scores = (W_REMAINING_WORK * norm_work +
              W_REMAINING_OPS * (1.0 - norm_ops) +
              W_SPT * (1.0 - norm_inv_spt))
    
    idx = int(np.argmax(scores))
    
    # Choose machine: shortest processing time
    m = _pick_machine(ready_operations[idx])
    if m is not None:
        return (idx, m)
    return idx
# EVOLVE-BLOCK-END
