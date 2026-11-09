"""Job-Shop Scheduling Dispatch Rule.
Structural approach: lookahead-based critical path with job urgency ratio
as the dominant scheduling signal, combined with machine load balancing
and completion proximity.
"""
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Deadline guard: fallback to LPT if time is running out
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(0.0)
        idx = int(np.argmax(min_durs))
        op = ready_operations[idx]
        mo = op.get("machine_options", {})
        if mo:
            return (idx, min(mo, key=mo.get))
        return idx

    # Extract features
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=np.float64)
    
    min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_m, min_d = min(machine_options.items(), key=lambda item: item[1])
            min_durations[i] = min_d
            best_machines[i] = best_m
        else:
            min_durations[i] = 0.0
            best_machines[i] = None

    eps = 1e-9
    
    # 1. Job Urgency Ratio (dominant signal):
    #    remaining_work / (min_duration + eps)
    #    High ratio = job has substantial remaining work relative to current op
    #    => prioritize to keep pipeline flowing, avoid stalling
    urgency_ratio = remaining_works / (min_durations + eps)
    # Log-compress to handle wide dynamic range
    log_urgency = np.log1p(urgency_ratio)
    max_log_urgency = np.max(log_urgency)
    if max_log_urgency > 0:
        urgency_score = log_urgency / max_log_urgency
    else:
        urgency_score = np.ones(n)
    
    # 2. Machine Congestion: prefer ops whose best machine has fewer competing ready ops
    machine_counts = {}
    for i in range(n):
        bm = best_machines[i]
        if bm is not None:
            machine_counts[bm] = machine_counts.get(bm, 0) + 1
    
    congestion = np.ones(n, dtype=np.float64)  # default 1 if no machine
    for i in range(n):
        bm = best_machines[i]
        if bm is not None:
            congestion[i] = machine_counts.get(bm, 1)
    
    max_cong = np.max(congestion)
    if max_cong > 1:
        # Invert: less congested = higher score
        congestion_score = 1.0 - (congestion - 1.0) / (max_cong - 1.0)
    else:
        congestion_score = np.ones(n)
    
    # 3. Completion Proximity: fewer remaining ops = closer to done
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)
    max_ops = np.max(remaining_ops_safe)
    if max_ops > 1:
        ops_score = 1.0 - (remaining_ops_safe - 1.0) / (max_ops - 1.0)
    else:
        ops_score = np.ones(n)
    
    # 4. LPT (Longest Processing Time first): longer ops are more inflexible
    max_min_dur = np.max(min_durations)
    if max_min_dur > 0:
        lpt_score = min_durations / max_min_dur
    else:
        lpt_score = np.ones(n)
    
    # Weighted combination
    w_urgency = 0.45
    w_congestion = 0.20
    w_ops = 0.15
    w_lpt = 0.20
    
    scores = (w_urgency * urgency_score +
              w_congestion * congestion_score +
              w_ops * ops_score +
              w_lpt * lpt_score)
    
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
