"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time_mod
import numpy as np
import math

_DEADLINE = _time_mod.monotonic() + 145.0  # safety margin before 150s kill


def _soft_min(values, temperature=1.0):
    """Smooth min using log-sum-exp trick. Returns a value close to min(values)."""
    if not values:
        return float('inf')
    arr = np.array(values, dtype=np.float64)
    m = np.min(arr)
    # For soft-min: -logsumexp(-x)/T, as T->0 approaches min
    # Using: min approx = m - T * log(sum(exp(-(x-m)/T)))
    if temperature <= 1e-12:
        return float(m)
    scaled = -(arr - m) / temperature
    # clip to avoid overflow
    scaled = np.clip(scaled, -700, 700)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return float(m - temperature * log_sum)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next in an open-shop simulation.
    
    Strategy (MWR + SPT + machine load balancing):
    1. Primary: job with most remaining work (MWR).
    2. Secondary: shortest effective duration considering machine load.
    3. Tertiary: prefer machines that reduce load imbalance.
    
    We use a weighted composite score:
      score = w1 * normalized_remaining_work 
            - w2 * normalized_min_duration 
            - w3 * normalized_machine_load_penalty
    """
    # Deadline guard
    if _time_mod.monotonic() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n_ops = len(ready_operations)
    
    # Pre-compute per-operation features
    works = []
    min_durs = []
    machine_load_penalties = []
    
    # Track machine loads from remaining work across all ready ops
    # In open-shop, we approximate machine load by summing durations of ready ops per machine
    machine_loads = {}
    
    for op in ready_operations:
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            for m, dur in machine_options.items():
                if m not in machine_loads:
                    machine_loads[m] = 0.0
                machine_loads[m] += dur
        else:
            # If machine_options is a list, assume single machine
            for dur in machine_options:
                pass  # can't track machine identity
    
    # Compute features for each operation
    for i, op in enumerate(ready_operations):
        work = op["remaining_work"]
        works.append(work)
        
        machine_options = op["machine_options"]
        if isinstance(machine_options, dict):
            durations = list(machine_options.values())
            machines = list(machine_options.keys())
        else:
            durations = list(machine_options)
            machines = [0] * len(durations)
        
        if not durations:
            min_dur = float('inf')
            load_penalty = 0.0
        else:
            min_dur = min(durations)
            # Machine load penalty: prefer machines with lower current load
            # Use soft-min of (load + duration) across machines, or just min load
            # Penalty = min over machines of (machine_load[m] + duration) - min_dur
            # This captures: if we pick this op, what's the earliest finish considering load
            if machines:
                finish_times = [machine_loads.get(m, 0.0) + d for m, d in zip(machines, durations)]
                # Use soft-min for smoothness
                soft_finish = _soft_min(finish_times, temperature=1.0)
                load_penalty = soft_finish - min_dur
            else:
                load_penalty = 0.0
        
        min_durs.append(min_dur)
        machine_load_penalties.append(load_penalty)
    
    # Normalize features to [0, 1] range for weighting
    works_arr = np.array(works, dtype=np.float64)
    min_durs_arr = np.array(min_durs, dtype=np.float64)
    penalties_arr = np.array(machine_load_penalties, dtype=np.float64)
    
    def normalize(arr):
        if len(arr) == 0:
            return arr
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    norm_work = normalize(works_arr)
    norm_dur = normalize(min_durs_arr)
    norm_pen = normalize(penalties_arr)
    
    # Weights: emphasize remaining work most, then duration, then load balance
    w1 = 0.5   # remaining work (higher is better)
    w2 = 0.35  # duration (lower is better)
    w3 = 0.15  # machine load penalty (lower is better)
    
    # Composite score: higher is better
    scores = w1 * norm_work - w2 * norm_dur - w3 * norm_pen
    
    # Pick the operation with the highest composite score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
