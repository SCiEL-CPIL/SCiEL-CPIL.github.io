"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard to ensure we finish well before the 150s limit
_DEADLINE = 145.0
_START_TIME = _time.time()

def _soft_min(vals, temp=1.0):
    """Smooth approximation of min using log-sum-exp."""
    if len(vals) == 0:
        return 0.0
    vals = np.array(vals, dtype=np.float64)
    if len(vals) == 1:
        return vals[0]
    # Soft min: -log(sum(exp(-x/temp)))/n * temp
    # Use log-sum-exp for numerical stability
    shifted = -vals / temp
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum_exp * temp / len(vals)

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
    
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    mean_durs = np.zeros(n, dtype=np.float64)
    num_opts = np.zeros(n, dtype=np.float64)
    # Store duration values for soft-min computation
    dur_vals_list = []
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durs[i] = 0.0
            max_durs[i] = 0.0
            mean_durs[i] = 0.0
            num_opts[i] = 0.0
            dur_vals_list.append(np.array([0.0]))
        else:
            vals = np.array(list(opts.values()), dtype=np.float64)
            min_durs[i] = np.min(vals)
            max_durs[i] = np.max(vals)
            mean_durs[i] = np.mean(vals)
            num_opts[i] = float(len(vals))
            dur_vals_list.append(vals)
    
    # Handle edge case where all works are 0
    if works.max() == 0:
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            best_machine = min(opts, key=opts.get)
            return (0, best_machine)
        return 0

    # Log transform to handle skewness in processing times
    log_works = np.log(works + 1.0)
    log_min_durs = np.log(min_durs + 1.0)
    log_mean_durs = np.log(mean_durs + 1.0)
    
    # Z-Score Normalization for robustness against outliers
    def z_score(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
        
    z_work = z_score(log_works)
    z_min_dur = z_score(log_min_durs)
    z_mean_dur = z_score(log_mean_durs)
    z_ops = z_score(ops_counts)
    
    # Calculate Coefficient of Variation for works (original scale)
    mean_work = works.mean()
    if mean_work > 0:
        std_work = works.std()
        cv = std_work / mean_work
    else:
        cv = 0.0
        
    # Adaptive weights
    # If CV is high (large disparity in work), prioritize MWRF
    # If CV is low (similar work), prioritize SPT
    cv_clamped = np.clip(cv, 0.0, 1.0)
    
    # Weight for Work (MWRF)
    w_work = 0.35 + 0.35 * cv_clamped
    # Weight for SPT (Shortest Processing Time) - use both min and mean
    w_spt = 1.0 - w_work
    # Small weight for remaining ops to break ties and favor complex jobs
    w_ops = 0.05
    # Small weight for flexibility (more machine options is better)
    w_flex = 0.05
    
    # Total score
    # Higher score is better
    # We want high work, low duration (so -z_dur), high ops, high flexibility
    scores = (w_work * z_work 
              - w_spt * 0.5 * (z_min_dur + z_mean_dur) 
              + w_ops * z_ops 
              + w_flex * z_score(num_opts))
    
    best_idx = int(np.argmax(scores))
    
    # Load-aware machine assignment: instead of always picking the shortest
    # duration machine, consider the congestion implied by how many other
    # ready operations also want that machine. A machine that many other
    # high-work ops also need will be a bottleneck.
    #
    # Build a machine -> (count of ready ops wanting it, sum of their work)
    # map to estimate relative load.
    machine_load = {}
    for op in ready_operations:
        for m in op.get("machine_options", {}):
            if m not in machine_load:
                machine_load[m] = [0, 0.0]
            machine_load[m][0] += 1
            machine_load[m][1] += op.get("remaining_work", 0.0)
    
    best_op = ready_operations[best_idx]
    opts = best_op.get("machine_options", {})
    if opts:
        # Score each candidate machine: prefer shorter duration, but penalize
        # machines that are heavily contended (high count * high work of other ops).
        # Use a smooth penalty based on the fraction of total ready work that
        # wants this machine.
        total_ready_work = sum(op.get("remaining_work", 0.0) for op in ready_operations)
        if total_ready_work <= 0:
            total_ready_work = 1.0
        
        best_machine = None
        best_score = -float('inf')
        for m, dur in opts.items():
            load_info = machine_load.get(m, [1, 0.0])
            # Fraction of total ready work that also wants this machine
            # (excluding this op's own contribution is a refinement, but
            #  using total is simpler and still captures contention)
            contention = load_info[1] / total_ready_work
            # Normalize duration by the max duration across all opts for this op
            max_dur = max(opts.values()) if opts else 1.0
            if max_dur <= 0:
                max_dur = 1.0
            norm_dur = dur / max_dur
            # Combine: minimize duration, penalize contention
            # Use a weighted combination; weight on contention scales with
            # how many ops are competing (more competition -> bigger penalty)
            n_competitors = load_info[0]
            contention_weight = 0.3 + 0.1 * (n_competitors - 1)  # grows with competition
            m_score = -norm_dur - contention_weight * contention
            if m_score > best_score:
                best_score = m_score
                best_machine = m
        
        if best_machine is None:
            best_machine = min(opts, key=opts.get)
        return (best_idx, best_machine)
    
    return best_idx
# EVOLVE-BLOCK-END
