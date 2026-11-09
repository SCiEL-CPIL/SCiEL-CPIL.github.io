"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard to ensure we finish well before the 150s limit
_DEADLINE = 145.0
_START_TIME = _time.time()

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            best_machine = min(opts, key=opts.get)
            return (0, best_machine)
        return 0

    # Check deadline
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    # Extract metrics
    works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    num_opts = np.zeros(n, dtype=np.float64)
    min_durs = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            num_opts[i] = 0.0
            min_durs[i] = 0.0
        else:
            vals = np.array(list(opts.values()), dtype=np.float64)
            num_opts[i] = float(len(vals))
            min_durs[i] = np.min(vals)
            
    # Handle edge case where all works are 0
    if works.max() == 0:
        return 0

    # Calculate Criticality: High work and Low flexibility is most urgent
    # Avoid division by zero
    safe_opts = np.where(num_opts > 0, num_opts, 1.0)
    criticality = works / safe_opts
    
    # Z-Score Normalization
    def z_score(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
        
    z_work = z_score(works)
    z_crit = z_score(criticality)
    z_ops = z_score(ops_counts)
    
    # Weights:
    # MWRF is the primary driver for Open Shop Makespan.
    # Criticality (Work/Flexibility) adds urgency for bottleneck jobs.
    # Remaining Ops is a minor tie-breaker.
    
    w_work = 0.70
    w_crit = 0.25
    w_ops = 0.05
    
    scores = (w_work * z_work 
              + w_crit * z_crit 
              + w_ops * z_ops)
    
    best_idx = int(np.argmax(scores))
    
    # Return (index, machine) to let the simulator place on the fastest machine
    best_op = ready_operations[best_idx]
    opts = best_op.get("machine_options", {})
    if opts:
        # Pick the machine with the shortest duration to minimize local congestion
        best_machine = min(opts, key=opts.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
