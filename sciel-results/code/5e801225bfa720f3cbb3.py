"""Job-Shop Scheduling dispatcher with dynamic bottleneck identification and multi-criteria scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast simple rule
    if _deadline_exceeded():
        # Fast fallback: pick operation with smallest min duration (SPT)
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(1.0)
        return int(np.argmin(min_durs))

    # Extract features
    remaining_work = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    machine_options_list = []

    for i, op in enumerate(ready_operations):
        remaining_work[i] = op.get("remaining_work", 0.0)
        remaining_ops[i] = op.get("remaining_ops", 1.0)
        mo = op.get("machine_options", {})
        if mo:
            durs = list(mo.values())
            min_durations[i] = min(durs)
            machine_options_list.append(mo)
        else:
            min_durations[i] = 1.0
            machine_options_list.append({0: 1.0})

    # --- Structural Change: Instantaneous Load/Urgency Balance ---
    # Instead of simulating makespan, we balance two immediate factors:
    # 1. Job Urgency (WSPT-like): Jobs with more remaining work relative to current op duration are more urgent.
    # 2. Machine Load: Prefer machines with less current load to avoid bottlenecks.
    
    # Calculate current machine loads from the ready set
    # Note: This is an approximation of the queue length at this specific event time.
    # In a true event-driven simulation, 'time' is the current time, and these ops are ready.
    # The 'load' here represents the contention if we were to schedule all of them.
    
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    
    if not all_machines:
        all_machines = {0}
    
    # machine_load[m] = total work queued on machine m among ready ops
    machine_load = {m: 0.0 for m in all_machines}
    for mo in machine_options_list:
        for m, dur in mo.items():
            machine_load[m] += dur

    # Calculate score for each operation
    scores = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        mo = machine_options_list[i]
        
        # Find the best machine (least loaded) for this operation
        best_m = None
        best_load = float('inf')
        best_dur = float('inf')
        
        for m, dur in mo.items():
            # We prefer the machine with the least load.
            # If loads are similar, prefer shorter duration?
            # Let's use a composite key: (load, duration)
            if best_m is None or machine_load[m] < best_load or (machine_load[m] == best_load and dur < best_dur):
                best_load = machine_load[m]
                best_m = m
                best_dur = dur
        
        # If no machine found (should not happen), default
        if best_m is None:
            best_m = 0
            best_dur = min_durations[i]
            best_load = machine_load.get(0, 0.0)

        # 1. Urgency Component: WSPT (Work / Time)
        # Higher WSPT means more work per unit time, which is often desirable to clear large jobs?
        # Actually, SPT (Shortest Processing Time) minimizes mean flow time.
        # WSPT minimizes weighted completion time.
        # Let's use Inverse WSPT as a proxy for "shortness" or urgency?
        # If remaining_work is high and dur is low, it's a "big job with small next step" -> High priority?
        # Let's try: Score = Load / (Remaining_Work / Dur) = Load * Dur / Remaining_Work
        # Low Score = Good.
        
        rw = remaining_work[i]
        if rw < 1e-6:
            rw = 1.0
            
        urgency = rw / best_dur  # Higher is more urgent (more work remaining per unit time of this step)
        
        # 2. Load Component:
        # We want to avoid the most loaded machine.
        # Normalized load: best_load / total_load? Or just best_load?
        # Let's use best_load directly.
        
        # Combine:
        # We want high Urgency and Low Load.
        # Score = Load / Urgency ?
        # If Urgency is high (big job), we tolerate higher load?
        # Or if Load is high, we prefer high Urgency to clear the queue?
        
        # Let's try a simple additive score with normalization.
        # Normalize Load and Urgency across the ready set.
        
        scores[i] = 0.0 # Placeholder
        
    # We need to normalize scores across operations to combine them.
    # Let's extract arrays for vectorization or simple loops.
    
    loads = np.empty(n)
    urgencies = np.empty(n)
    
    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_load = float('inf')
        best_dur = float('inf')
        
        for m, dur in mo.items():
            if best_m is None or machine_load[m] < best_load or (machine_load[m] == best_load and dur < best_dur):
                best_load = machine_load[m]
                best_m = m
                best_dur = dur
        
        if best_m is None:
            best_m = 0
            best_dur = min_durations[i]
            best_load = machine_load.get(0, 0.0)
            
        rw = remaining_work[i]
        if rw < 1e-6:
            rw = 1.0
            
        loads[i] = best_load
        urgencies[i] = rw / best_dur

    # Normalize
    def _norm(arr):
        mn = arr.min()
        mx = arr.max()
        if mx <= mn + 1e-12:
            return np.full_like(arr, 0.5, dtype=np.float64)
        return (arr - mn) / (mx - mn)

    n_load = _norm(loads)
    n_urgency = _norm(urgencies)
    
    # We want to MINIMIZE load and MAXIMIZE urgency.
    # Score = w1 * n_load - w2 * n_urgency
    # Let's use equal weights.
    final_scores = 0.5 * n_load - 0.5 * n_urgency
    
    best_idx = int(np.argmin(final_scores))
    
    return best_idx
# EVOLVE-BLOCK-END
